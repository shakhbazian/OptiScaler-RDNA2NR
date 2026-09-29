#pragma once
#include "NativeHookHub.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <vector>

namespace DlssNr::Native {
using Microsoft::WRL::ComPtr;
inline void Check(HRESULT result) { if(FAILED(result))throw std::runtime_error("native D3D12 operation failed"); }
struct Commands {
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    explicit Commands(ID3D12Device* device) {
        NativeHooks::OriginScope scope(NativeHooks::Origin::Private);
        Check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)));
        Check(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)));
    }
};
// Only GPU-retired recordings may return their pair. The pool is device-local
// and weakly registered: it cannot keep a removed device alive indefinitely.
class ShadowPool {
public:
    struct Pair {std::unique_ptr<Commands> before,after;};
    static std::shared_ptr<ShadowPool> ForDevice(ID3D12Device* device){
        static std::mutex registryMutex;
        static std::map<ID3D12Device*,std::weak_ptr<ShadowPool>> registry;
        std::lock_guard lock(registryMutex);
        for(auto it=registry.begin();it!=registry.end();)it=it->second.expired()?registry.erase(it):++it;
        auto& weak=registry[device];auto result=weak.lock();
        if(!result){result=std::make_shared<ShadowPool>();weak=result;}
        return result;
    }
    Pair Acquire(ID3D12Device* device){
        Pair result;
        {std::lock_guard lock(mutex);for(auto& slot:idle)if(slot.before){result=std::move(slot);break;}}
        NativeHooks::OriginScope scope(NativeHooks::Origin::Private);
        if(result.before){
            Check(result.before->allocator->Reset());Check(result.after->allocator->Reset());
            Check(result.before->list->Reset(result.before->allocator.Get(),nullptr));
            Check(result.after->list->Reset(result.after->allocator.Get(),nullptr));
            ++reused;
        }else{result.before=std::make_unique<Commands>(device);result.after=std::make_unique<Commands>(device);++created;}
        return result;
    }
    void Return(Pair pair)noexcept{
        std::lock_guard lock(mutex);
        for(auto& slot:idle)if(!slot.before){slot=std::move(pair);return;}
    }
    std::atomic<std::uint64_t> created{0},reused{0};
private:
    std::mutex mutex;
    std::array<Pair,8> idle;
};
// Bounded speculative command capture; native GPU lists are allocated only at
// a qualified marker. State restoration is canonical, never a setter journal.
class Recording {
public:
    using Call=std::function<void(ID3D12GraphicsCommandList*)>;
    static constexpr std::size_t MaxCommands=65536, MaxBytes=8*1024*1024, MaxObjects=4096;
    ComPtr<ID3D12Device> device;
    const std::uint64_t generation;
    std::atomic<bool> valid{true},closed{false},marked{false},submitted{false};
    const char* reason="unmarked";
    std::unique_ptr<Commands> prefix,suffix,abort;
    std::shared_ptr<ShadowPool> shadowPool;
    std::atomic<bool> gpuRetired{false};
    std::size_t commandCount=0,bytes=0;
    explicit Recording(ID3D12Device* d,std::uint64_t value,ID3D12PipelineState* initial=nullptr)
        :device(d),generation(value),shadowPool(ShadowPool::ForDevice(d)){
        if(live.fetch_add(1)>=64){--live;throw std::runtime_error("live recording budget");}
        try{if(initial)SetPipelineState(initial);}catch(...){--live;throw;}
    }
    ~Recording(){
        if(gpuRetired&&prefix&&suffix)shadowPool->Return({std::move(prefix),std::move(suffix)});
        if(ownsSplitBudget)--liveSplits;--live;
    }
    void Reject(const char* why) noexcept {valid=false;reason=why;}
    void Reserve(std::size_t n=64) {
        if(n>MaxBytes||bytes>MaxBytes-n||commandCount==MaxCommands)throw std::runtime_error("recording budget");
        bytes+=n;++commandCount;
    }
    template<class T> std::vector<T> Copy(const T* p,std::size_t n,std::size_t maximum=65536) {
        if(n>maximum||n>MaxBytes/sizeof(T)||(!p&&n))throw std::runtime_error("recording array budget");
        Reserve(n*sizeof(T));return n?std::vector<T>(p,p+n):std::vector<T>{};
    }
    void Keep(IUnknown* object) {
        if(!object||kept.count(object))return;
        if(kept.size()==MaxObjects)throw std::runtime_error("recording resource budget");
        kept.emplace(object,ComPtr<IUnknown>(object));
    }
    void Command(Call call) {
        Reserve();
        if(suffix){NativeHooks::OriginScope scope(NativeHooks::Origin::Private);call(suffix->list.Get());}
        else journal.push_back(std::move(call));
    }
    void State(unsigned slot,unsigned index,Call call) {
        states[slot*4096+index]=call;Command(std::move(call));
    }
    void Root(bool graphics,unsigned index,unsigned kind,Call call,unsigned offset=0) {
        if(index>=64||offset>=64)throw std::runtime_error("root parameter budget");
        const unsigned first=1000000+unsigned(graphics)*10000+index*100;
        auto& old=rootKinds[graphics][index];
        if(kind!=2||old!=kind){auto it=states.lower_bound(first);
            while(it!=states.end()&&it->first<first+100)it=states.erase(it);}
        old=kind;states[first+(kind==2?offset+1:0)]=call;Command(std::move(call));
    }
    void Signature(bool graphics,ID3D12RootSignature* signature) {
        Keep(signature);
        if(signatures[graphics]!=signature){
            const unsigned first=1000000+unsigned(graphics)*10000;
            auto it=states.lower_bound(first);while(it!=states.end()&&it->first<first+10000)it=states.erase(it);
            rootKinds[graphics].fill(0);signatures[graphics]=signature;
        }
        State(graphics?30:29,0,[=](auto* p){if(graphics)p->SetGraphicsRootSignature(signature);else p->SetComputeRootSignature(signature);});
    }
    D3D12_CPU_DESCRIPTOR_HANDLE Descriptor(D3D12_CPU_DESCRIPTOR_HANDLE source,D3D12_DESCRIPTOR_HEAP_TYPE type) {
        const auto index=static_cast<unsigned>(type);
        if(!source.ptr)return source;
        if(index>=heaps.size()||heapUsed[index]==512)throw std::runtime_error("descriptor snapshot budget");
        if(!heaps[index]){
            D3D12_DESCRIPTOR_HEAP_DESC desc{};desc.Type=type;desc.NumDescriptors=512;
            Check(device->CreateDescriptorHeap(&desc,IID_PPV_ARGS(&heaps[index])));
        }
        auto target=heaps[index]->GetCPUDescriptorHandleForHeapStart();
        target.ptr+=SIZE_T(heapUsed[index]++)*device->GetDescriptorHandleIncrementSize(type);
        device->CopyDescriptorsSimple(1,target,source,type);return target;
    }
    bool Cut(bool restoreState=true) {
        if(!valid||closed||marked||predicated||!queries.empty()||events||!splits.empty()){
            Reject("marker inside unsupported scope or repeated marker");return false;
        }
        NativeHooks::OriginScope scope(NativeHooks::Origin::Private);
        if(liveSplits.fetch_add(1)>=16){--liveSplits;Reject("live shadow budget");return false;}
        ownsSplitBudget=true;
        auto pair=shadowPool->Acquire(device.Get());
        auto before=std::move(pair.before),after=std::move(pair.after);
        for(const auto& call:journal)call(before->list.Get());
        Check(before->list->Close());
        if(restoreState)for(const auto& state:states)state.second(after->list.Get());
        journal.clear();prefix=std::move(before);suffix=std::move(after);marked=true;reason="marked";
        return true;
    }
    void Close(bool success) {
        closed=success;
        if(!success){Reject("original Close failed");return;}
        if(valid&&suffix){NativeHooks::OriginScope scope(NativeHooks::Origin::Private);Check(suffix->list->Close());}
        if(!marked){journal.clear();states.clear();kept.clear();}
    }
    bool Ready()const noexcept{return valid&&closed&&marked&&!submitted&&prefix&&suffix;}
    void ClearState(ID3D12PipelineState* pso) {
        states.clear();signatures.fill(nullptr);for(auto& k:rootKinds)k.fill(0);
        boundHeaps.fill(nullptr);predicated=false;Keep(pso);
        Command([=](auto* p){p->ClearState(pso);});
        // A new private list starts with defaults; only non-default PSO needs replay.
        if(pso)states[25*4096]=[=](auto* p){p->SetPipelineState(pso);};
    }
    void SetComputeRootSignature(ID3D12RootSignature* signature){Signature(false,signature);}
    void SetGraphicsRootSignature(ID3D12RootSignature* signature){Signature(true,signature);}
    void SetDescriptorHeaps(UINT n,ID3D12DescriptorHeap* const* pointers) {
        if(n>2||(!pointers&&n))throw std::runtime_error("descriptor heap count");
        std::array<ID3D12DescriptorHeap*,2> copy{};
        for(UINT i=0;i<n;++i){copy[i]=pointers[i];Keep(copy[i]);}
        if(copy!=boundHeaps){
            for(unsigned g=0;g<2;++g)for(unsigned r=0;r<64;++r)
                if(rootKinds[g][r]==1){states.erase(1000000+g*10000+r*100);rootKinds[g][r]=0;}
            boundHeaps=copy;
        }
        State(28,0,[=](auto* p){p->SetDescriptorHeaps(n,copy.data());});
    }
    void ResourceBarrier(UINT n,const D3D12_RESOURCE_BARRIER* barriers) {
        auto copy=Copy(barriers,n);
        for(const auto& b:copy){
            if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_TRANSITION){
                Keep(b.Transition.pResource);
                const auto key=std::make_pair(b.Transition.pResource,b.Transition.Subresource);
                if(b.Flags&D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY)splits.insert(key);
                else{splits.erase(key);resourceStates[key]=b.Transition.StateAfter;
                    if(b.Transition.Subresource!=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES)
                        resourceStates.erase({b.Transition.pResource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES});}
            }else if(b.Type==D3D12_RESOURCE_BARRIER_TYPE_UAV)Keep(b.UAV.pResource);
            else {Keep(b.Aliasing.pResourceBefore);Keep(b.Aliasing.pResourceAfter);}
        }
        Command([=](auto* p){p->ResourceBarrier(n,copy.data());});
    }
    bool EntryState(ID3D12Resource* resource,D3D12_RESOURCE_STATES& out)const {
        const auto found=resourceStates.find({resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES});
        if(found==resourceStates.end())return false;out=found->second;return true;
    }
    void OMSetRenderTargets(UINT n,const D3D12_CPU_DESCRIPTOR_HANDLE* rt,BOOL contiguous,
                            const D3D12_CPU_DESCRIPTOR_HANDLE* depth) {
        if(n>8||(!rt&&n))throw std::runtime_error("RTV count");
        std::array<D3D12_CPU_DESCRIPTOR_HANDLE,8> targets{};
        for(UINT i=0;i<n;++i){auto source=rt[contiguous?0:i];
            if(contiguous)source.ptr+=SIZE_T(i)*device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            targets[i]=Descriptor(source,D3D12_DESCRIPTOR_HEAP_TYPE_RTV);}
        const bool hasDepth=depth!=nullptr;
        const auto dsv=hasDepth?Descriptor(*depth,D3D12_DESCRIPTOR_HEAP_TYPE_DSV):D3D12_CPU_DESCRIPTOR_HANDLE{};
        State(46,0,[=](auto* p){p->OMSetRenderTargets(n,targets.data(),FALSE,hasDepth?&dsv:nullptr);});
    }
    void BeginQuery(ID3D12QueryHeap* heap,D3D12_QUERY_TYPE type,UINT index) {
        Keep(heap);queries.emplace(heap,type,index);Command([=](auto* p){p->BeginQuery(heap,type,index);});
    }
    void EndQuery(ID3D12QueryHeap* heap,D3D12_QUERY_TYPE type,UINT index) {
        Keep(heap);if(type!=D3D12_QUERY_TYPE_TIMESTAMP&&queries.erase({heap,type,index})!=1)Reject("unpaired query");
        Command([=](auto* p){p->EndQuery(heap,type,index);});
    }
    void SetPredication(ID3D12Resource* buffer,UINT64 offset,D3D12_PREDICATION_OP op) {
        Keep(buffer);predicated=buffer!=nullptr;
        State(55,0,[=](auto* p){p->SetPredication(buffer,offset,op);});
    }
    void BeginEvent(UINT metadata,const void* data,UINT size) {
        auto copy=Copy(static_cast<const unsigned char*>(data),size);++events;
        Command([=](auto* p){p->BeginEvent(metadata,copy.data(),size);});
    }
    void EndEvent(){if(events)--events;else Reject("unpaired event");Command([](auto* p){p->EndEvent();});}
    void ExecuteBundle(ID3D12GraphicsCommandList*){Reject("bundle state requires qualification");}
    void ExecuteIndirect(ID3D12CommandSignature*,UINT,ID3D12Resource*,UINT64,ID3D12Resource*,UINT64){Reject("indirect state requires qualification");}
    template<class T,class F> static void Extended(ID3D12GraphicsCommandList* list,F&& fn) {
        ComPtr<T> ptr;Check(list->QueryInterface(IID_PPV_ARGS(&ptr)));fn(ptr.Get());
    }
#include "NativeRecordingMethods.inl"
private:
    static inline std::atomic<unsigned> live{0},liveSplits{0};
    bool ownsSplitBudget=false;
    std::vector<Call> journal;
    std::map<unsigned,Call> states;
    std::map<IUnknown*,ComPtr<IUnknown>> kept;
    std::array<ID3D12RootSignature*,2> signatures{};
    std::array<std::array<unsigned,64>,2> rootKinds{};
    std::array<ID3D12DescriptorHeap*,2> boundHeaps{};
    std::array<ComPtr<ID3D12DescriptorHeap>,4> heaps{};
    std::array<UINT,4> heapUsed{};
    std::map<std::pair<ID3D12Resource*,UINT>,D3D12_RESOURCE_STATES> resourceStates;
    std::set<std::pair<ID3D12Resource*,UINT>> splits;
    std::set<std::tuple<ID3D12QueryHeap*,D3D12_QUERY_TYPE,UINT>> queries;
    unsigned events=0;
    bool predicated=false;
};
} // namespace DlssNr::Native
