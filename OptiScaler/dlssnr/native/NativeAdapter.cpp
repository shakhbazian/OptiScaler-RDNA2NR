#include "pch.h"
#include "NativeAdapter.h"
#include <dlssnr/NrBackendSelection.h>
#include "NativeMethods.h"
#include "NativeSession.h"
#include "NativeMotion.h"
#include "../../../rdna2_nr/include/nr_extent.h"
#include <Config.h>
#include <Util.h>
#include <resource_tracking/ResTrack_dx12.h>
#include <dlssnr/HipFrameSettings.h>
#include <dlssnr/WorkingResolution.h>
#include <shared_mutex>
#include <unordered_map>
#include <chrono>

namespace DlssNr::Native {
namespace {
std::atomic<bool> enabled{false};
bool CpuTraceEnabled(){static const bool on=GetEnvironmentVariableW(L"NR_NATIVE_CPU_TRACE",nullptr,0)>0;return on;}
std::atomic<std::uint64_t> markCount{0},markTotalUs{0},markCutUs{0};
std::atomic<std::uint64_t> nextCookie{1};
struct Feature {
    const std::uint64_t cookie=nextCookie++;
    std::uint64_t frame=0;
    std::shared_ptr<std::atomic<bool>> live=std::make_shared<std::atomic<bool>>(true);
};
struct Evaluation {std::shared_ptr<Feature> feature;std::uint64_t frame=0;bool hdr=false;};
struct Features {std::mutex mutex;std::map<std::uint64_t,std::shared_ptr<Feature>> items;};
Features& FeatureRegistry(){static auto* value=new Features;return *value;}
thread_local Evaluation* currentEvaluation=nullptr;
struct Sidecar {
    std::atomic<bool> live{true};const std::uint64_t cookie=nextCookie++;
    std::mutex mutex;ComPtr<ID3D12Device> device;
    std::shared_ptr<Recording> recording;std::shared_ptr<const Intent> intent;
    std::uint64_t generation=0;
};
struct Registry {
    std::shared_mutex mutex;
    std::unordered_map<const void*,std::shared_ptr<Sidecar>> aliases;
    std::unordered_map<IUnknown*,std::weak_ptr<Sidecar>> anchors;
};
Registry& Lists(){static auto* value=new Registry;return *value;}
std::shared_ptr<Sidecar> Find(const void* list) {
    auto& registry=Lists();std::shared_lock lock(registry.mutex);
    const auto it=registry.aliases.find(list);
    return it!=registry.aliases.end()&&it->second->live.load()?it->second:nullptr;
}
bool SameObject(const void* a,const void* b)noexcept{
    if(a==b)return true;try{auto left=Find(a),right=Find(b);return left&&left==right;}catch(...){return false;}
}
// The parent list owns the sentinel, not vice versa. Driver destruction merely
// flips liveness; registry/COM cleanup happens later outside that callback.
const GUID SentinelId={0xb86490b4,0x927a,0x48c2,{0xa0,0xd2,0x4d,0xc7,0x82,0x2d,0xea,0x51}};
class Sentinel final:public IUnknown {
    std::atomic<ULONG> refs{1};std::shared_ptr<Sidecar> side;
public:
    explicit Sentinel(std::shared_ptr<Sidecar> value):side(std::move(value)){}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid,void** out)override {
        if(!out)return E_POINTER;*out=nullptr;if(iid!=__uuidof(IUnknown))return E_NOINTERFACE;
        *out=static_cast<IUnknown*>(this);AddRef();return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef()override{return ++refs;}
    ULONG STDMETHODCALLTYPE Release()override{auto left=--refs;if(!left)delete this;return left;}
    ~Sentinel(){side->live=false;}
};
// Some COM wrappers preserve private data but have their own identity/vtable.
// Resolve only our live anchor; commands are still observed on the underlying
// native list. Never register unobserved wrapper methods as covered aliases.
std::shared_ptr<Sidecar> FindMarkerList(ID3D12GraphicsCommandList* list){
    if(auto side=Find(list))return side;
    ComPtr<IUnknown> anchor;UINT size=sizeof(IUnknown*);
    if(FAILED(list->GetPrivateData(SentinelId,&size,anchor.GetAddressOf()))||
        size!=sizeof(IUnknown*)||!anchor)return nullptr;
    auto& registry=Lists();std::shared_lock lock(registry.mutex);
    const auto it=registry.anchors.find(anchor.Get());
    if(it==registry.anchors.end())return nullptr;
    auto side=it->second.lock();return side&&side->live?side:nullptr;
}
template<class F>void Record(ID3D12GraphicsCommandList* list,F&& fn)noexcept{
    if(!enabled.load(std::memory_order_acquire))return;
    try{auto side=Find(list);if(!side)return;std::lock_guard lock(side->mutex);
        const auto& r=side->recording;if(!r||!r->valid||r->closed||r->submitted)return;
        if(!Config::Instance()->DlssNrEnabled.value_or_default()){
            r->Reject("NR disabled during recording");return;}
        try{fn(*r);}catch(...){r->Reject("recording budget or copy failure");}
    }catch(...){HistoryDiscontinuity();}
}
void OnReset(ID3D12GraphicsCommandList* list,ID3D12PipelineState* pso,bool success)noexcept{
    if(!enabled.load())return;
    try{auto side=Find(list);if(!side)return;std::lock_guard lock(side->mutex);
        side->intent.reset();
        if(!Config::Instance()->DlssNrEnabled.value_or_default())side->recording.reset();
        else if(success)side->recording=std::make_shared<Recording>(side->device.Get(),++side->generation,pso);
        else if(side->recording)side->recording->Reject("original Reset failed");
    }catch(...){if(auto side=Find(list)){std::lock_guard lock(side->mutex);side->recording.reset();}HistoryDiscontinuity();}
}
void OnClose(ID3D12GraphicsCommandList* list,bool success)noexcept{
    if(!enabled.load())return;
    try{auto side=Find(list);if(!side)return;std::lock_guard lock(side->mutex);
        if(side->recording){side->recording->Close(success);
            if(!side->recording->marked)side->recording.reset();}
    }catch(...){if(auto side=Find(list)){std::lock_guard lock(side->mutex);if(side->recording)side->recording->Reject("shadow Close failed");}}
}
#include "NativeRecorderMethods.inl"
using QueryFn=HRESULT(STDMETHODCALLTYPE*)(IUnknown*,REFIID,void**);
using QueryMethod=NativeHooks::Method<0,QueryFn>;
void ObserveQuery(IUnknown* self,REFIID iid,void** result,HRESULT status)noexcept{
    if(FAILED(status)||!result||!*result||!enabled.load())return;
    try{auto side=Find(self);if(!side)return;
        // Lifetime callbacks cannot record GPU commands. This sibling interface
        // must not be rejected or registered as a command-list alias.
        if(iid==__uuidof(ID3DDestructionNotifier))return;
        if(!KnownListInterface(iid)){
            wchar_t text[40]{};StringFromGUID2(iid,text,40);
            LOG_DEBUG("Native NR unobserved command-list IID {}",wstring_to_string(text));
            std::lock_guard lock(side->mutex);
            if(side->recording)side->recording->Reject("unobserved private command-list interface");return;}
        // A recognized IID can still expose a wrapper with different physical
        // methods. Validate coverage before trusting the new alias.
        NativeHooks::OriginScope privateCalls(NativeHooks::Origin::Private);
        ComPtr<ID3D12GraphicsCommandList> list;std::vector<void*> aliases;
        if(FAILED(static_cast<IUnknown*>(*result)->QueryInterface(IID_PPV_ARGS(&list)))||
           !SurveyList(list.Get(),aliases,false)){
            std::lock_guard lock(side->mutex);if(side->recording)side->recording->Reject("uncovered command-list alias");return;}
        auto& registry=Lists();std::unique_lock lock(registry.mutex);
        if(registry.aliases.size()<1024)registry.aliases[*result]=side;
        else{lock.unlock();std::lock_guard guard(side->mutex);if(side->recording)side->recording->Reject("alias budget");}
    }catch(...){HistoryDiscontinuity();}
}
void Watch(ID3D12GraphicsCommandList* list,ID3D12Device* device,ID3D12PipelineState* initial,bool closed,bool explicitOwned=false) {
    if(!list||!enabled.load()||(!explicitOwned&&NativeHooks::origin==NativeHooks::Origin::Private)||list->GetType()!=D3D12_COMMAND_LIST_TYPE_DIRECT)return;
    if(Find(list))return;
    NativeHooks::OriginScope privateCalls(NativeHooks::Origin::Private);
    std::vector<void*> aliases;
    if(!SurveyList(list,aliases,false))return;
    auto side=std::make_shared<Sidecar>();side->device=device;
    side->recording=std::make_shared<Recording>(device,++side->generation,initial);side->recording->closed=closed;
    ComPtr<Sentinel> sentinel;sentinel.Attach(new Sentinel(side));
    if(FAILED(list->SetPrivateDataInterface(SentinelId,sentinel.Get())))return;
    std::vector<std::shared_ptr<Sidecar>> retired;
    auto& registry=Lists();std::unique_lock lock(registry.mutex);
    static std::atomic<unsigned> creations{0};
    if((++creations%16)==0||registry.aliases.size()+aliases.size()>1024){
        for(auto it=registry.aliases.begin();it!=registry.aliases.end();)
            if(!it->second->live){retired.push_back(std::move(it->second));it=registry.aliases.erase(it);}else ++it;
        for(auto it=registry.anchors.begin();it!=registry.anchors.end();)
            it=it->second.expired()?registry.anchors.erase(it):++it;
    }
    if(registry.aliases.size()+aliases.size()<=1024){
        for(void* alias:aliases)registry.aliases[alias]=side;
        registry.anchors[sentinel.Get()]=side;
    }
    else side->live=false;
    lock.unlock();retired.clear();
}
using CreateFn=HRESULT(STDMETHODCALLTYPE*)(ID3D12Device*,UINT,D3D12_COMMAND_LIST_TYPE,ID3D12CommandAllocator*,ID3D12PipelineState*,REFIID,void**);
using CreateMethod=NativeHooks::Method<2012,CreateFn>;
void ObserveCreate(ID3D12Device* device,UINT,D3D12_COMMAND_LIST_TYPE type,ID3D12CommandAllocator*,ID3D12PipelineState* pso,
                    REFIID,void** out,HRESULT result)noexcept{
    if(FAILED(result)||type!=D3D12_COMMAND_LIST_TYPE_DIRECT||!out||!*out||!enabled.load())return;
    try{ComPtr<ID3D12GraphicsCommandList> list;auto* object=static_cast<IUnknown*>(*out);
        if(SUCCEEDED(object->QueryInterface(IID_PPV_ARGS(&list))))Watch(list.Get(),device,pso,false);
    }catch(...){HistoryDiscontinuity();}
}
using Create1Fn=HRESULT(STDMETHODCALLTYPE*)(ID3D12Device4*,UINT,D3D12_COMMAND_LIST_TYPE,D3D12_COMMAND_LIST_FLAGS,REFIID,void**);
using Create1Method=NativeHooks::Method<2051,Create1Fn>;
void ObserveCreate1(ID3D12Device4* device,UINT,D3D12_COMMAND_LIST_TYPE type,D3D12_COMMAND_LIST_FLAGS,REFIID,void** out,HRESULT result)noexcept{
    if(FAILED(result)||type!=D3D12_COMMAND_LIST_TYPE_DIRECT||!out||!*out||!enabled.load())return;
    try{ComPtr<ID3D12GraphicsCommandList> list;auto* object=static_cast<IUnknown*>(*out);
        if(SUCCEEDED(object->QueryInterface(IID_PPV_ARGS(&list))))Watch(list.Get(),device,nullptr,true);
    }catch(...){HistoryDiscontinuity();}
}
NativeQueue::FenceOp originalSignal=nullptr,originalWait=nullptr;
HRESULT STDMETHODCALLTYPE Signal(ID3D12CommandQueue* q,ID3D12Fence* f,UINT64 value){return NativeQueue::Signal(q,f,value,originalSignal);}
HRESULT STDMETHODCALLTYPE Wait(ID3D12CommandQueue* q,ID3D12Fence* f,UINT64 value){return NativeQueue::Wait(q,f,value,originalWait);}
using UpdateFn=void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*,ID3D12Resource*,UINT,const D3D12_TILED_RESOURCE_COORDINATE*,
    const D3D12_TILE_REGION_SIZE*,ID3D12Heap*,UINT,const D3D12_TILE_RANGE_FLAGS*,const UINT*,const UINT*,D3D12_TILE_MAPPING_FLAGS);
using CopyFn=void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*,ID3D12Resource*,const D3D12_TILED_RESOURCE_COORDINATE*,ID3D12Resource*,
    const D3D12_TILED_RESOURCE_COORDINATE*,const D3D12_TILE_REGION_SIZE*,D3D12_TILE_MAPPING_FLAGS);
UpdateFn originalUpdate=nullptr;CopyFn originalCopy=nullptr;
using AnnotationFn=void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*,UINT,const void*,UINT);
using EndAnnotationFn=void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*);
AnnotationFn originalMarker=nullptr,originalBegin=nullptr;EndAnnotationFn originalEnd=nullptr;
void STDMETHODCALLTYPE QueueMarker(ID3D12CommandQueue* q,UINT metadata,const void* data,UINT size){
    NativeQueue::Mutation(q,[&]{originalMarker(q,metadata,data,size);});}
void STDMETHODCALLTYPE QueueBegin(ID3D12CommandQueue* q,UINT metadata,const void* data,UINT size){
    NativeQueue::Mutation(q,[&]{originalBegin(q,metadata,data,size);});}
void STDMETHODCALLTYPE QueueEnd(ID3D12CommandQueue* q){NativeQueue::Mutation(q,[&]{originalEnd(q);});}
void STDMETHODCALLTYPE UpdateTiles(ID3D12CommandQueue* q,ID3D12Resource* r,UINT n,const D3D12_TILED_RESOURCE_COORDINATE* c,
    const D3D12_TILE_REGION_SIZE* size,ID3D12Heap* heap,UINT ranges,const D3D12_TILE_RANGE_FLAGS* flags,const UINT* offsets,
    const UINT* counts,D3D12_TILE_MAPPING_FLAGS mapping){NativeQueue::Mutation(q,[&]{originalUpdate(q,r,n,c,size,heap,ranges,flags,offsets,counts,mapping);});}
void STDMETHODCALLTYPE CopyTiles(ID3D12CommandQueue* q,ID3D12Resource* dst,const D3D12_TILED_RESOURCE_COORDINATE* dc,
    ID3D12Resource* src,const D3D12_TILED_RESOURCE_COORDINATE* sc,const D3D12_TILE_REGION_SIZE* size,D3D12_TILE_MAPPING_FLAGS flags){
    NativeQueue::Mutation(q,[&]{originalCopy(q,dst,dc,src,sc,size,flags);});}
std::shared_ptr<NativeQueue::Prepared> PrepareBatch(ID3D12CommandQueue* queue,UINT count,ID3D12CommandList* const* lists)noexcept{
    if(!enabled.load()||queue->GetDesc().Type!=D3D12_COMMAND_LIST_TYPE_DIRECT||!lists||!count||count>256)return nullptr;
    if(!Config::Instance()->DlssNrEnabled.value_or_default()){
        HistoryDiscontinuity();return nullptr;}
    ComPtr<ID3D12Device> queueDevice;
    if(FAILED(queue->GetDevice(IID_PPV_ARGS(&queueDevice)))||
       DlssNr::SelectNrBackend(queueDevice.Get(),Config::Instance()->DlssNrBackend.value_or_default())!=
           DlssNr::NrBackendSelection::AmdHip){HistoryDiscontinuity();return nullptr;}
    try{
        auto** table=*reinterpret_cast<void***>(queue);
        if(!NativeHooks::Method<1014,NativeQueue::FenceOp>::Covered(table[14])||
            !NativeHooks::Method<1015,NativeQueue::FenceOp>::Covered(table[15])||
            !NativeHooks::Method<1008,UpdateFn>::Covered(table[8])||
            !NativeHooks::Method<1009,CopyFn>::Covered(table[9])||
            !NativeHooks::Method<1011,AnnotationFn>::Covered(table[11])||
            !NativeHooks::Method<1012,AnnotationFn>::Covered(table[12])||
            !NativeHooks::Method<1013,EndAnnotationFn>::Covered(table[13])){HistoryDiscontinuity();return nullptr;}
        std::shared_ptr<NativeBatch> result;
        bool multiple=false;
        for(UINT i=0;i<count;++i){auto side=Find(lists[i]);if(!side)continue;
            std::lock_guard lock(side->mutex);if(!side->intent)continue;
            if(result){multiple=true;
                if(result->recording)result->recording->submitted=true;
                if(side->recording)side->recording->submitted=true;
                continue;}
            result=std::make_shared<NativeBatch>();result->recording=side->recording;result->intent=std::move(side->intent);
            result->markerIndex=i;result->repeated=!side->recording||side->recording->submitted;
            if(side->recording&&!side->recording->Ready()&&!result->repeated){
                static std::atomic<unsigned> warnings{0};
                if(warnings++<8)LOG_WARN("Native NR unqualified submission: {} closed={} commands={}",
                    side->recording->reason,side->recording->closed.load(),side->recording->commandCount);
            }
        }
        if(multiple){HistoryDiscontinuity();return nullptr;}
        if(result){static std::atomic<unsigned> prepared{0};
            if(++prepared<=3)LOG_INFO("Native NR prepared game submission: marker={}, recording ready={}",
                result->markerIndex,result->recording&&result->recording->Ready());}
        return result;
    }catch(...){HistoryDiscontinuity();return nullptr;}
}
struct ExternalState {ComPtr<ID3D12Resource> resource;D3D12_RESOURCE_STATES state;};
struct FrontendContext {
    bool admitting=true,complete=false;
    std::shared_ptr<Sidecar> game;
    std::shared_ptr<Intent> intent;
    std::vector<ExternalState> external;
};
// These lists are ours and never escape recording. Retire their observation
// immediately, rather than waiting for the game-list registry's lazy sweep.
// The immutable Recording itself is retained by Intent until GPU retirement.
struct ForgetOwnedObservation {
    std::shared_ptr<Sidecar> side;
    ~ForgetOwnedObservation(){
        side->live=false;
        auto& registry=Lists();std::unique_lock lock(registry.mutex);
        for(auto it=registry.aliases.begin();it!=registry.aliases.end();)
            it=it->second==side?registry.aliases.erase(it):++it;
        for(auto it=registry.anchors.begin();it!=registry.anchors.end();)
            it=it->second.expired()||it->second.lock()==side?registry.anchors.erase(it):++it;
    }
};
thread_local FrontendContext* frontend=nullptr;
void TransitionFrontend(ID3D12GraphicsCommandList* list,ID3D12Resource* resource,
                        D3D12_RESOURCE_STATES from,D3D12_RESOURCE_STATES to){
    if(!resource||from==to)return;
    D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,from,to};list->ResourceBarrier(1,&b);
}
}

EvaluationScope::EvaluationScope(std::uint64_t feature,bool hdr):previous(currentEvaluation){
    auto value=std::make_unique<Evaluation>();value->hdr=hdr;
    if(feature){auto& registry=FeatureRegistry();std::lock_guard lock(registry.mutex);
        auto& state=registry.items[feature];if(!state)state=std::make_shared<Feature>();
        value->feature=state;value->frame=++state->frame;}
    currentEvaluation=value.get();context=value.release();
}
EvaluationScope::~EvaluationScope(){currentEvaluation=static_cast<Evaluation*>(previous);delete static_cast<Evaluation*>(context);}
bool FrontendSessionReady(NVSDK_NGX_Parameter* params,bool before,
                          ID3D12GraphicsCommandList* list)noexcept{
    try{
    if(!params||!currentEvaluation||!currentEvaluation->feature||!list)return false;
    auto side=FindMarkerList(list);
    if(!side)return false;
    ID3D12Resource* target=nullptr;
    params->Get(before?NVSDK_NGX_Parameter_Color:NVSDK_NGX_Parameter_Output,&target);
    if(!target)return false;
    const auto desc=target->GetDesc();
    if(desc.Width>UINT_MAX||desc.Height>UINT_MAX)return false;
    // The upscaler's timing queue can be the swapchain/FG queue. It does not
    // identify the queue that will submit this list. Admit recording on the
    // observed native device; Submit checks the actual queue before GPU work
    // and warms a replacement session if it changed. This also avoids comparing
    // a game's COM device wrapper against the recorder's native device pointer.
    const auto work=ModelResolution(static_cast<unsigned>(desc.Width),desc.Height,
                                    Config::Instance()->DlssNrWorkingScale.value_or_default());
    return SessionReadyFor(side->device.Get(),nullptr,work.width,work.height,
                           currentEvaluation->feature->cookie,before);
    }catch(...){return false;}
}
bool PrepareOwned(std::uint64_t id,ID3D12Device* device,ID3D12CommandQueue* queue,
                  unsigned width,unsigned height,bool before)noexcept{
    try{
        std::uint64_t cookie=0;
        {auto& registry=FeatureRegistry();std::lock_guard lock(registry.mutex);
            auto& feature=registry.items[id];if(!feature)feature=std::make_shared<Feature>();cookie=feature->cookie;}
        const auto& cfg=*Config::Instance();
        if(cfg.DlssNrSpatialCompression.value_or_default())return false;
        const auto work=ModelResolution(width,height,cfg.DlssNrWorkingScale.value_or_default());
        if(!NrV2::SupportedLogicalExtent(work.width,work.height)||
           !NrV2::SupportedGraphExtent(NrV2::AlignNetworkExtent(work.width),NrV2::AlignNetworkExtent(work.height)))return false;
        return WarmSession(device,queue,work.width,work.height,cookie,before);
    }catch(...){return false;}
}

bool Initialize(ID3D12Device* device)noexcept{
    static std::mutex setup;std::lock_guard lock(setup);
    if(enabled.load())return true;
    try{
        NativeHooks::OriginScope privateCalls(NativeHooks::Origin::Private);
        Commands dummy(device);std::vector<void*> aliases;
        if(NativeHooks::Begin()!=NO_ERROR)return false;
        bool ok=SurveyList(dummy.list.Get(),aliases,true);
        auto** table=*reinterpret_cast<void***>(dummy.list.Get());
        ok=(QueryMethod::Ensure(table[0])==NO_ERROR)&&ok;
        auto** deviceTable=*reinterpret_cast<void***>(device);
        ok=(CreateMethod::Ensure(deviceTable[12])==NO_ERROR)&&ok;
        ComPtr<ID3D12Device4> device4;
        if(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device4)))){
            auto** t=*reinterpret_cast<void***>(device4.Get());ok=(Create1Method::Ensure(t[51])==NO_ERROR)&&ok;}
        ok=(NativeHooks::Commit()==NO_ERROR)&&ok;
        Check(dummy.list->Close());if(!ok)return false;
        // The existing OptiScaler queue listener remains the owner of logical
        // FG bookkeeping; queue mutation listeners share its physical endpoint.
        ResTrack_Dx12::HookLateNrQueue(device);
        D3D12_COMMAND_QUEUE_DESC desc{};ComPtr<ID3D12CommandQueue> queue;
        Check(device->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue)));
        auto** qt=*reinterpret_cast<void***>(queue.Get());
        if(!originalSignal){
            if(NativeHooks::Begin()!=NO_ERROR)return false;
            originalSignal=reinterpret_cast<NativeQueue::FenceOp>(qt[14]);originalWait=reinterpret_cast<NativeQueue::FenceOp>(qt[15]);
            originalUpdate=reinterpret_cast<UpdateFn>(qt[8]);originalCopy=reinterpret_cast<CopyFn>(qt[9]);
            originalMarker=reinterpret_cast<AnnotationFn>(qt[11]);originalBegin=reinterpret_cast<AnnotationFn>(qt[12]);
            originalEnd=reinterpret_cast<EndAnnotationFn>(qt[13]);
            NativeHooks::Attach<1014>(&originalSignal,Signal);NativeHooks::Attach<1015>(&originalWait,Wait);
            NativeHooks::Attach<1008>(&originalUpdate,UpdateTiles);NativeHooks::Attach<1009>(&originalCopy,CopyTiles);
            NativeHooks::Attach<1011>(&originalMarker,QueueMarker);NativeHooks::Attach<1012>(&originalBegin,QueueBegin);
            NativeHooks::Attach<1013>(&originalEnd,QueueEnd);
            if(NativeHooks::Commit()!=NO_ERROR){originalSignal=nullptr;originalWait=nullptr;originalUpdate=nullptr;originalCopy=nullptr;return false;}
        }
        InstallObservers();QueryMethod::observer.store(ObserveQuery);CreateMethod::observer.store(ObserveCreate);
        Create1Method::observer.store(ObserveCreate1);NativeHooks::sameObject.store(SameObject);
        const auto assets=Util::DllPath().parent_path();
        std::filesystem::path package;
        const auto configured=Config::Instance()->DlssNrModelPath.value_or_default();
        if(!configured.empty())package=std::filesystem::path(string_to_wstring(configured));
        else {
            const DWORD length=GetEnvironmentVariableW(L"LOCALAPPDATA",nullptr,0);
            if(length>1){
                std::wstring local(length,L'\0');
                if(GetEnvironmentVariableW(L"LOCALAPPDATA",local.data(),length)==length-1){
                    local.resize(length-1);
                    package=std::filesystem::path(local)/L"OptiScaler-RDNA2NR"/L"models"/L"1"/
                        L"E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E"/
                        L"mixed-v5-gfx1030.nrwgt";
                }
            }
        }
        SetAssetPaths((assets/L"dlssnr_hip_scheduled_bridge.dll").wstring(),package.wstring());
        NativeQueue::SetPrepare(PrepareBatch);enabled=true;
        LOG_INFO("Native NR recorder installed; runtime will warm on the first observed queue");return true;
    }catch(...){NativeHooks::Abort();enabled=false;return false;}
}
const char* Mark(ID3D12GraphicsCommandList* list,NVSDK_NGX_Parameter* params,bool before,bool warmOnly,bool interop)noexcept{
    const bool trace=CpuTraceEnabled();
    const auto markStart=trace?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
    try{
        const auto& cfg=*Config::Instance();
        if(!cfg.DlssNrEnabled.value_or_default()||before!=cfg.DlssNrRunBeforeSr.value_or_default())return nullptr;
        if(!enabled.load())return "native recorder unavailable";
        const auto* evaluation=currentEvaluation;
        if(!evaluation||!evaluation->feature)return "native NGX feature context unavailable";
        if(evaluation->hdr&&!frontend&&!warmOnly)return "native first profile requires SDR";
        if(cfg.DlssNrPasses.value_or_default()!=1||cfg.DlssNrFinishedPicture.value_or_default()||
            cfg.DlssNrDeferredDlss.value_or_default()||cfg.DlssNrResidualAcrossRr.value_or_default()||
            cfg.DlssNrHoldFrame.value_or_default())return "native profile requires one ordinary pre/post pass without frame hold";
        auto side=FindMarkerList(list);if(!side){
            ComPtr<ID3D12Device> device;if(SUCCEEDED(list->GetDevice(IID_PPV_ARGS(&device)))){
                Watch(list,device.Get(),nullptr,false);side=Find(list);
                if(side){std::lock_guard lock(side->mutex);side->recording->Reject("late observation; waiting for Reset");}
            }
            return "list was not observed from creation or Reset";
        }
        std::lock_guard lock(side->mutex);auto& r=side->recording;
        if(!r||!r->valid||r->closed)return r?r->reason:"list generation unavailable";
        auto intent=std::make_shared<Intent>();intent->before=before;intent->warmOnly=warmOnly;
        intent->feature=evaluation->feature->cookie;
        intent->sourceFrame=evaluation->frame;intent->featureLive=evaluation->feature->live;
        intent->temporalAccumulation=DlssNr::HipFrameSettings::TemporalAccumulation(cfg);
        ID3D12Resource* target=nullptr;ID3D12Resource* motion=nullptr;
        params->Get(before?NVSDK_NGX_Parameter_Color:NVSDK_NGX_Parameter_Output,&target);
        params->Get(NVSDK_NGX_Parameter_MotionVectors,&motion);
        if(!target)return "native color unavailable";
        const auto d=target->GetDesc();
        if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||d.DepthOrArraySize!=1||d.MipLevels!=1||
            d.SampleDesc.Count!=1||d.Width>UINT_MAX)return "native profile requires a single-sample 2D texture";
        const bool colorFormat=d.Format==DXGI_FORMAT_R16G16B16A16_FLOAT||
            ((frontend||warmOnly)&&(d.Format==DXGI_FORMAT_R11G11B10_FLOAT||d.Format==DXGI_FORMAT_R8G8B8A8_UNORM||
                        d.Format==DXGI_FORMAT_R32G32B32A32_FLOAT));
        if(!colorFormat)return "native color format is not qualified";
        if(d.Flags&D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE)return "native color denies shader reads";
        intent->color=target;intent->width=static_cast<unsigned>(d.Width);intent->height=d.Height;
        unsigned baseX=0,baseY=0,motionX=0,motionY=0,renderW=0,renderH=0;
        params->Get(before?NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X:NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X,&baseX);
        params->Get(before?NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y:NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y,&baseY);
        params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,&motionX);
        params->Get(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y,&motionY);
        params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,&renderW);
        params->Get(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,&renderH);
        if(baseX||baseY||motionX||motionY||(before&&((renderW&&renderW!=intent->width)||(renderH&&renderH!=intent->height))))
            return "native first profile requires zero-offset full-frame resources";
        if(!before){unsigned outW=0,outH=0;
            params->Get(NVSDK_NGX_Parameter_OutWidth,&outW);params->Get(NVSDK_NGX_Parameter_OutHeight,&outH);
            if((outW&&outW!=intent->width)||(outH&&outH!=intent->height))
                return "native output subrect is not supported";}
        if(!NrV2::SupportedLogicalExtent(intent->width,intent->height))return "native extent exceeds checked runtime budget";
        const auto& colorBarrier=before?cfg.ColorResourceBarrier:cfg.OutputResourceBarrier;
        if(!r->EntryState(target,intent->colorState)){
            if(interop)intent->colorState=before?D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            else if(colorBarrier.has_value())intent->colorState=static_cast<D3D12_RESOURCE_STATES>(colorBarrier.value());
            else return "native color entry state is not established";
        }
        if(motion){const auto md=motion->GetDesc();
            if(md.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||md.DepthOrArraySize!=1||md.MipLevels!=1||
                md.SampleDesc.Count!=1||md.Format!=DXGI_FORMAT_R16G16_FLOAT||md.Width>UINT_MAX)return "native motion format unsupported";
            intent->motion=motion;intent->motionWidth=static_cast<unsigned>(md.Width);intent->motionHeight=md.Height;
            if((renderW&&renderW!=intent->motionWidth)||(renderH&&renderH!=intent->motionHeight))
                return "native motion subrect or high-resolution convention requires qualification";
            if(!r->EntryState(motion,intent->motionState)){
                if(interop)intent->motionState=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                else if(cfg.MVResourceBarrier.has_value())intent->motionState=
                    static_cast<D3D12_RESOURCE_STATES>(cfg.MVResourceBarrier.value());
                else return "native motion entry state is not established";
            }
        }
        intent->controls=DlssNr::HipFrameSettings::Controls(cfg);
        auto& mv=intent->motionParameters;mv.scaleX=mv.scaleY=1;
        params->Get(NVSDK_NGX_Parameter_MV_Scale_X,&mv.scaleX);params->Get(NVSDK_NGX_Parameter_MV_Scale_Y,&mv.scaleY);
        params->Get(NVSDK_NGX_Parameter_Jitter_Offset_X,&mv.currentJitterX);params->Get(NVSDK_NGX_Parameter_Jitter_Offset_Y,&mv.currentJitterY);
        mv.effectiveWidth=intent->motionWidth;mv.effectiveHeight=intent->motionHeight;mv.jitterMode=NrV2::JitterMode::AddPreviousMinusCurrent;
        unsigned reset=0;params->Get(NVSDK_NGX_Parameter_Reset,&reset);intent->reset=reset!=0;
        if(warmOnly){
            if(cfg.DlssNrSpatialCompression.value_or_default())return "HIP peripheral compression is not supported";
            const auto work=ModelResolution(intent->width,intent->height,cfg.DlssNrWorkingScale.value_or_default());
            if(!NrV2::SupportedLogicalExtent(work.width,work.height)||
               !NrV2::SupportedGraphExtent(NrV2::AlignNetworkExtent(work.width),NrV2::AlignNetworkExtent(work.height)))
                return "HIP working extent exceeds runtime limits";
            // This marker only warms the session. Its game texture is never
            // submitted as model input; the common frontend builds that later.
            intent->width=work.width;intent->height=work.height;
        }
        if(frontend&&frontend->admitting){frontend->game=side;frontend->intent=std::move(intent);return nullptr;}
        const auto cutStart=trace?std::chrono::steady_clock::now():std::chrono::steady_clock::time_point{};
        if(!r->Cut())return r->reason;
        if(trace){
            const auto end=std::chrono::steady_clock::now();
            markTotalUs.fetch_add(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(end-markStart).count()));
            markCutUs.fetch_add(static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(end-cutStart).count()));
            markCount.fetch_add(1);
        }
        side->intent=std::move(intent);return nullptr;
    }catch(...){HistoryDiscontinuity();return "native marker allocation or recording failed";}
}
bool FrontendActive()noexcept{return frontend&&!frontend->admitting;}
D3D12_RESOURCE_STATES FrontendColorState()noexcept{return frontend?frontend->external.front().state:D3D12_RESOURCE_STATE_COMMON;}
void CompleteFrontend()noexcept{if(FrontendActive())frontend->complete=true;}

class ModelBackend final:public INrBackend {
    ComPtr<ID3D12Device> device;
    MotionReducer motionReducer;
public:
    const char* Name()const noexcept override{return "NativeMixedModelPass";}
    bool Initialize(ID3D12Device* d)override{device=d;return d!=nullptr;}
    bool Resize(const NrBackendSize& s)override{
        return s.format==DXGI_FORMAT_R16G16B16A16_FLOAT&&NrV2::SupportedLogicalExtent(s.width,s.height)&&
            NrV2::SupportedGraphExtent(NrV2::AlignNetworkExtent(s.width),NrV2::AlignNetworkExtent(s.height));
    }
    void Shutdown()noexcept override{}
    NrBackendEvaluation Evaluate(const NrBackendFrame& frame)override{
        if(!FrontendActive()||!frame.input||!frame.output||frame.input==frame.output||
           frame.motionBaseX||frame.motionBaseY)return {NrBackendResult::Unsupported,0,false};
        auto side=FindMarkerList(frame.commands);if(!side)return {NrBackendResult::NotInitialized,0,false};
        auto& v=*frontend->intent;
        v.color=frame.input;v.output=frame.output;v.motion=frame.motion;
        v.width=frame.size.width;v.height=frame.size.height;
        v.motionWidth=frame.motionWidth;v.motionHeight=frame.motionHeight;
        v.colorState=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        v.outputState=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        v.motionState=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        // The runtime consumes normalized motion. Keep the game's original
        // units and effective extent; RunPass' NGX working-size scale would
        // otherwise multiply normalized motion by WorkingScale a second time.
        ID3D12Resource* reducedMotion=nullptr;
        if(v.motion&&(v.motionWidth>v.width||v.motionHeight>v.height)){
            motionReducer.Open(device.Get());
            reducedMotion=motionReducer.Record(frame.commands,v.motion.Get(),v.width,v.height);
            TransitionFrontend(frame.commands,reducedMotion,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            v.motion=reducedMotion;v.motionWidth=v.width;v.motionHeight=v.height;
        }
        v.reset=v.reset||frame.reset;
        std::unique_lock lock(side->mutex);auto& r=side->recording;
        if(!r||!r->valid||r->closed)return {NrBackendResult::NotInitialized,0,false};
        // Restore external states if HIP fails after the codec prefix was submitted.
        r->abort=std::make_unique<Commands>(r->device.Get());
        for(const auto& external:frontend->external){
            auto atCut=external.state;r->EntryState(external.resource.Get(),atCut);
            TransitionFrontend(r->abort->list.Get(),external.resource.Get(),atCut,external.state);
        }
        Check(r->abort->list->Close());
        // Resolve binds its complete state. Replaying Encode's descriptor table
        // here would reference the game target after it has become a UAV.
        if(!r->Cut(false)){
            LOG_WARN("Native common model boundary rejected: {}",r->reason);
            return {NrBackendResult::Unsupported,0,false};
        }
        lock.unlock();
        if(reducedMotion)TransitionFrontend(frame.commands,reducedMotion,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        return {NrBackendResult::Success,frame.submissionEpoch,false};
    }
};
std::unique_ptr<INrBackend> MakeModelBackend(){return std::make_unique<ModelBackend>();}

const char* RecordFrontend(ID3D12GraphicsCommandList* game,NVSDK_NGX_Parameter* params,bool before,
                          std::shared_ptr<void> lease,const std::function<void(ID3D12GraphicsCommandList*)>& record,
                          bool interop)noexcept{
    if(frontend)return "nested common frontend is not supported";
    FrontendContext context;frontend=&context;
    struct Clear {~Clear(){frontend=nullptr;}} clear;
    try{
        if(const auto* reason=Mark(game,params,before,false,interop))return reason;
        if(!context.intent)return nullptr; // NR disabled or the other placement.
        auto& v=*context.intent;context.external.push_back({v.color,v.colorState});
        if(v.motion)context.external.push_back({v.motion,v.motionState});
        ID3D12Resource* exposure=nullptr;
        if(Config::Instance()->DlssNrWhitePointSource.value_or_default()==1)
            params->Get(NVSDK_NGX_Parameter_ExposureTexture,&exposure);
        if(exposure){
            const auto desc=exposure->GetDesc();
            const bool readable=desc.Format==DXGI_FORMAT_R32_FLOAT||desc.Format==DXGI_FORMAT_R16_FLOAT||
                desc.Format==DXGI_FORMAT_R16G16_FLOAT||desc.Format==DXGI_FORMAT_R32G32_FLOAT||
                desc.Format==DXGI_FORMAT_R16G16B16A16_FLOAT||desc.Format==DXGI_FORMAT_R32G32B32A32_FLOAT;
            if(exposure==v.color.Get()||exposure==v.motion.Get()||!readable||
               desc.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||desc.DepthOrArraySize!=1||
               desc.SampleDesc.Count!=1||desc.MipLevels!=1||(desc.Flags&D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE))
                return "native exposure requires an independent typed readable float texture";
            D3D12_RESOURCE_STATES state{};
            std::lock_guard lock(context.game->mutex);
            if(!context.game->recording->EntryState(exposure,state)){
                const auto& configured=Config::Instance()->ExposureResourceBarrier;
                if(!configured.has_value())return "native exposure entry state is not established";
                state=static_cast<D3D12_RESOURCE_STATES>(configured.value());
            }
            context.external.push_back({exposure,state});
        }
        Commands commands(context.game->device.Get());
        Watch(commands.list.Get(),context.game->device.Get(),nullptr,false,true);
        auto side=Find(commands.list.Get());if(!side)return "common frontend observation failed";
        ForgetOwnedObservation forget{side};
        context.admitting=false;
        {
            // Observe owned commands without invoking game/FG state listeners.
            NativeHooks::OriginScope scope(NativeHooks::Origin::Restore);
            for(std::size_t i=1;i<context.external.size();++i){const auto& ext=context.external[i];
                TransitionFrontend(commands.list.Get(),ext.resource.Get(),ext.state,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);}
            record(commands.list.Get());
            for(std::size_t i=1;i<context.external.size();++i){const auto& ext=context.external[i];
                TransitionFrontend(commands.list.Get(),ext.resource.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,ext.state);}
            Check(commands.list->Close());
        }
        if(!context.complete)return "common frontend did not record a complete resolve";
        if(!side->recording||!side->recording->Ready())
            return side->recording?side->recording->reason:"common frontend was not completed";
        std::lock_guard lock(context.game->mutex);
        if(!context.game->recording||!context.game->recording->Cut())return "game marker could not accept the common frontend";
        v.frontend=side->recording;v.frontendLease=std::move(lease);
        context.game->intent=std::move(context.intent);
        static std::atomic<unsigned> attached{0};
        if(++attached<=3)LOG_INFO("Native NR common frontend attached to game frame: {}x{} before SR={}",
            v.width,v.height,before);
        return nullptr;
    }catch(...){HistoryDiscontinuity();return "common frontend recording failed";}
}
void ReleaseFeature(std::uint64_t feature)noexcept{
    if(CpuTraceEnabled()){
        const auto count=markCount.exchange(0);
        if(count)LOG_INFO("NativeCPU mark n={} total={} cut={} us",count,
            markTotalUs.exchange(0),markCutUs.exchange(0));
    }
    auto& registry=FeatureRegistry();std::uint64_t cookie=0;
    {std::lock_guard lock(registry.mutex);auto it=registry.items.find(feature);
        if(it!=registry.items.end()){cookie=it->second->cookie;*it->second->live=false;registry.items.erase(it);}}
    if(cookie){HistoryDiscontinuity();StopSession(cookie);}
}
void Shutdown()noexcept{enabled=false;NativeQueue::SetPrepare(nullptr);HistoryDiscontinuity();StopSession();}
#ifdef NR_NATIVE_TEST_HOOKS
extern "C" __declspec(dllexport) void DlssNrNativeTestSetTimingQueue(ID3D12CommandQueue* queue){
    State::Instance().currentCommandQueue=queue;
}
// Diagnostic builds only: emulate the menu's configuration assignment between
// headless Evaluate calls. No failure switch or toggle export ships in Release.
extern "C" __declspec(dllexport) void DlssNrNativeTestSetEnabled(unsigned value){
    Config::Instance()->DlssNrEnabled=value!=0;
}
extern "C" __declspec(dllexport) void DlssNrNativeTestSetControls(unsigned preset){
    auto& cfg=*Config::Instance();
    const float values[3][6]={{2,1,1,-1,0,1},{0,.25f,.5f,0,1,.75f},{1,1.5f,1.5f,.5f,0,.5f}};
    const auto& v=values[preset%3];
    cfg.DlssNrStyle=static_cast<unsigned>(v[0]);cfg.DlssNrLocalTone=v[1];
    cfg.DlssNrLocalStructure=v[2];cfg.DlssNrSkinStructure=v[3];
    cfg.DlssNrAutoMask=v[4]!=0;cfg.DlssNrIntensity=v[5];
}
extern "C" __declspec(dllexport) void DlssNrNativeTestSetWorkingScale(float scale){
    Config::Instance()->DlssNrWorkingScale=scale;
}
extern "C" __declspec(dllexport) void DlssNrNativeTestSetTemporal(unsigned enabled){
    Config::Instance()->DlssNrTemporalAccumulation=enabled!=0;
}
#endif
} // namespace DlssNr::Native
