#include "pch.h"
#include "NativeSession.h"
#include "NativeAdapter.h"
#include <dlssnr/DlssNr_Status.h>
#include "../../../rdna2_nr/include/nr_execution_profile.h"
#include "../../../rdna2_nr/include/nr_runtime_v3_validation.h"
#include <array>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <thread>

namespace DlssNr::Native {
namespace {
using NrV3::Status;
using CpuClock=std::chrono::steady_clock;
// This first HIP profile accepts one frozen package; its digest is checked by the companion.
constexpr std::streamoff AcceptedModelBytes=291595458;
bool CpuTraceEnabled(){static const bool on=GetEnvironmentVariableW(L"NR_NATIVE_CPU_TRACE",nullptr,0)>0;return on;}
struct CpuTraceTotals {
    std::atomic<std::uint64_t> count{0};
    std::array<std::atomic<std::uint64_t>,9> us{};
    CpuTraceTotals(){for(auto& value:us)value.store(0);}
    void Add(const std::array<CpuClock::time_point,10>& t){
        for(unsigned i=0;i<9;++i)us[i].fetch_add(static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(t[i+1]-t[i]).count()),std::memory_order_relaxed);
        count.fetch_add(1,std::memory_order_relaxed);
    }
    void Report(){
        const auto n=count.exchange(0);if(!n)return;
        std::array<std::uint64_t,9> values{};
        for(unsigned i=0;i<9;++i)values[i]=us[i].exchange(0);
        LOG_INFO("NativeCPU submit n={} select={} prepare={} input={} output={} prefix={} notify={} enqueue={} arm={} suffix={} us",
            n,values[0],values[1],values[2],values[3],values[4],values[5],values[6],values[7],values[8]);
    }
};
CpuTraceTotals& TraceTotals(){static auto* totals=new CpuTraceTotals;return *totals;}
NrV3::TokenRequest Request(const NrV3::Token& token){return {{sizeof(NrV3::TokenRequest),NrV3::Version},token};}
NrV2::Hash256 Hash(const char* hex){NrV2::Hash256 result{};for(unsigned i=0;i<32;++i){unsigned value=0;
    for(unsigned j=0;j<2;++j){const char c=hex[i*2+j];value=value*16+unsigned(c<='9'?c-'0':c-'A'+10);}
    result.bytes[i]=static_cast<std::uint8_t>(value);}return result;}
NrV2::Texture Texture(ID3D12Resource* resource,unsigned w,unsigned h,NrV2::Format format){return {resource,{0,0,w,h,w,h},format,0};}
void Transition(ID3D12GraphicsCommandList* list,ID3D12Resource* resource,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){
    if(!resource||a==b)return;D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,a,b};list->ResourceBarrier(1,&barrier);
}
struct Shape {
    ComPtr<ID3D12Device> device;ComPtr<ID3D12CommandQueue> queue;
    unsigned width=0,height=0;std::uint64_t feature=0;bool before=false;
    bool Same(const Shape& b)const{return device==b.device&&queue==b.queue&&width==b.width&&height==b.height&&feature==b.feature&&before==b.before;}
};
struct Flight {
    std::unique_ptr<Commands> ingress,egress;
    std::shared_ptr<Recording> recording;
    std::shared_ptr<const Intent> intent;
    NrV3::Token token{};
    UINT64 retire=0;
    bool tokenLive=false;
};
struct Owner {
    std::shared_ptr<ShadowPool> shadowPool;
    Shape shape;HMODULE module=nullptr;NrSubmissionApi::Api api{};void* runtime=nullptr;
    std::array<Flight,3> flights;
    ComPtr<ID3D12Fence> fence;UINT64 serial=0,epoch=0,discontinuity=0,lastFrame=0;
    float jitterX=0,jitterY=0;bool fault=false,temporalAccumulation=true;
    std::string failureReason;
    bool Poll(){
        NrV3::Snapshot snapshot{};
        if(api.poll(runtime,&snapshot,sizeof(snapshot))!=Status::Ok)return false;
        const auto complete=fence->GetCompletedValue();if(complete==UINT64_MAX)return false;
        for(auto& f:flights){
            if(f.tokenLive)for(const auto& slot:snapshot.slots)
                if(NrV3::Same(slot.token,f.token)&&slot.phase==NrV3::Phase::Terminal){
                    auto request=Request(f.token);
                    if(api.acknowledgeTerminal(runtime,&request,sizeof(request))!=Status::Ok)return false;
                    f.tokenLive=false;break;
                }
            if(f.recording&&f.retire&&complete>=f.retire&&!f.tokenLive){
                f.recording->gpuRetired=true;
                if(f.intent&&f.intent->frontend)f.intent->frontend->gpuRetired=true;
                f.recording.reset();f.intent.reset();f.retire=0;
            }
        }
        return true;
    }
    bool Prepare(const std::wstring& dll,const std::wstring& weights){
        NativeHooks::OriginScope privateCalls(NativeHooks::Origin::Private);
        shadowPool=ShadowPool::ForDevice(shape.device.Get());
        module=LoadLibraryExW(dll.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if(!module){failureReason="HIP companion could not load (Windows error "+std::to_string(GetLastError())+")";
            LOG_ERROR("Native NR companion could not load (Windows error {}): {}",GetLastError(),
            wstring_to_string(dll));return false;}
        auto get=reinterpret_cast<NrSubmissionApi::GetApi>(GetProcAddress(module,"DlssNrHipBackendGetScheduledApiV2"));
        if(!get||get(NrExecution::AcceptedMixedId,NrSubmissionApi::Version,&api,sizeof(api))!=Status::Ok){
            failureReason="HIP companion API or mixed-v5 contract is incompatible";
            LOG_ERROR("Native NR companion API or mixed-v5 contract is incompatible");return false;}
        if(api.create(shape.device.Get(),shape.queue.Get(),&runtime)!=Status::Ok){
            failureReason="HIP runtime could not start on the selected device or queue";
            LOG_ERROR("Native NR HIP runtime could not start on the selected device/queue");return false;}
        std::ifstream input(std::filesystem::path(weights),std::ios::binary|std::ios::ate);
        if(!input){failureReason="Model package missing or unreadable: "+wstring_to_string(weights);
            LOG_ERROR("Native NR model package missing or unreadable: {}",wstring_to_string(weights));return false;}
        const auto size=input.tellg();
        if(size!=AcceptedModelBytes){failureReason="Model package size invalid for mixed-v5";
            LOG_ERROR("Native NR model package size invalid: {}",wstring_to_string(weights));return false;}
        std::vector<unsigned char> package(static_cast<std::size_t>(size));input.seekg(0);
        if(!input.read(reinterpret_cast<char*>(package.data()),size)){
            failureReason="Model package read failed";
            LOG_ERROR("Native NR model package read failed: {}",wstring_to_string(weights));return false;}
        NrV3::Config config{};config.prefix={sizeof(config),NrV3::Version};
        config.sourceHash=Hash("E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E");
        config.packageHash=Hash("A7E6EE38172A81E12D613FA9A2F57E32AA1944908E56CD2A33E1F6C94369E3CB");
        config.graphVersion=1;config.mathContract=NrV2::MathContract;
        config.logicalWidth=shape.width;config.logicalHeight=shape.height;
        config.networkWidth=NrV2::AlignNetworkExtent(shape.width);config.networkHeight=NrV2::AlignNetworkExtent(shape.height);
        const auto modelStatus=api.configure(runtime,&config,sizeof(config),package.data(),package.size());
        if(modelStatus!=Status::Ok){failureReason=modelStatus==Status::Failed?
            "Model validation or GPU memory setup failed; check runtime log":
            "Model package rejected (status "+std::to_string(static_cast<unsigned>(modelStatus))+")";
            LOG_ERROR("Native NR model setup failed (status {}): {}",
            static_cast<unsigned>(modelStatus),wstring_to_string(weights));return false;}
        Check(shape.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)));
        for(auto& flight:flights){flight.ingress=std::make_unique<Commands>(shape.device.Get());
            flight.egress=std::make_unique<Commands>(shape.device.Get());
            Check(flight.ingress->list->Close());Check(flight.egress->list->Close());}
        return true;
    }
    bool Close(){
        NativeHooks::OriginScope privateCalls(NativeHooks::Origin::Private);
        if(runtime){
            if(fault||api.beginDrain(runtime)!=Status::Ok)return false;
            const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            for(;;){
                if(fence&&!Poll())return false;
                bool live=false;for(auto& f:flights)live|=bool(f.recording)||f.tokenLive;
                const auto status=api.shutdown(runtime);
                if(status==Status::Ok&&!live)break;
                if((status!=Status::Busy&&status!=Status::Ok)||std::chrono::steady_clock::now()>=end)return false;
                Sleep(1);
            }
            if(api.destroy(&runtime)!=Status::Ok)return false;
        }
        if(module){FreeLibrary(module);module=nullptr;}
        if(CpuTraceEnabled()&&shadowPool)LOG_INFO("NativeCPU shadow created={} reused={}",
            shadowPool->created.load(),shadowPool->reused.load());
        return true;
    }
};
struct Session {
    std::mutex mutex;std::condition_variable condition;
    std::shared_ptr<Owner> owner;
    std::unique_ptr<Shape> desired;
    std::wstring dll,package;
    std::shared_ptr<Owner> quarantine;
    bool workerStarted=false,request=false,working=false,failed=false;
    std::string failureReason;
    std::atomic<unsigned> lastOutcome{0};
    std::atomic<UINT64> discontinuity{1},applied{0},raw{0};
    std::atomic<unsigned> ready{0},faults{0},pending{0};
};
Session& Get(){static auto* session=new Session;return *session;}
#ifdef NR_NATIVE_TEST_HOOKS
// Only the separately compiled diagnostic host defines this macro. The
// OptiScaler release DLL has neither a failure switch nor this export.
std::atomic<unsigned> testFailure{0};
extern "C" __declspec(dllexport) void DlssNrNativeTestFailure(unsigned value){testFailure=value;}
#endif
void RetainFault(Session& s,std::shared_ptr<Owner> owner){
    if(owner&&owner->runtime)owner->api.quarantine(owner->runtime);
    if(owner){owner->fault=true;s.quarantine=std::move(owner);}
    s.failed=true;s.ready=0;++s.faults;
    if(s.failureReason.empty())
        s.failureReason="HIP submission or drain failed; the NR session was quarantined";
}
void Worker(){
    auto& s=Get();
    for(;;){
        std::unique_ptr<Shape> desired;std::shared_ptr<Owner> old;std::wstring dll,package;
        {std::unique_lock lock(s.mutex);s.condition.wait(lock,[&]{return s.request;});
            s.request=false;s.working=true;s.ready=0;old=std::move(s.owner);
            if(s.desired)desired=std::make_unique<Shape>(*s.desired);
            dll=s.dll;package=s.package;}
        bool good=true;
        try{if(old&&!old->Close())good=false;}catch(...){good=false;}
        std::shared_ptr<Owner> created;
        if(good&&desired){
            created=std::make_shared<Owner>();created->shape=*desired;
            try{good=created->Prepare(dll,package);}catch(...){good=false;}
            if(!good){try{if(!created->Close())created->fault=true;}catch(...){created->fault=true;}}
        }
        {std::lock_guard lock(s.mutex);s.working=false;
            if(!good){
                const auto faultsBefore=s.faults.load();
                if(old&&old->runtime)RetainFault(s,old);
                if(created&&created->runtime)RetainFault(s,created);
                if(s.faults.load()==faultsBefore)++s.faults;
                s.failed=true;s.pending=0;
                s.failureReason=created&&!created->failureReason.empty()?created->failureReason:
                    "HIP session warmup or drain failed";
                LOG_ERROR("Native NR warmup/drain failed; original submissions remain active");
            }else{
                s.failureReason.clear();s.lastOutcome=0;
                s.owner=std::move(created);s.ready=s.owner&&!s.request?1:0;s.pending=s.owner||s.request?1:0;
                if(s.owner)LOG_INFO("Native NR ready on observed queue: {}x{}",s.owner->shape.width,s.owner->shape.height);
            }
        }
    }
}
}
void SetAssetPaths(std::wstring dll,std::wstring package){
    auto& s=Get();std::lock_guard lock(s.mutex);s.dll=std::move(dll);s.package=std::move(package);
    if(!s.workerStarted){s.workerStarted=true;std::thread(Worker).detach();}
}
void HistoryDiscontinuity()noexcept{++Get().discontinuity;}
std::uint64_t AppliedFrames()noexcept{return Get().applied.load();}
std::uint64_t BypassedFrames()noexcept{return Get().raw.load();}
bool SessionReady()noexcept{return Get().ready.load()!=0;}
RuntimeStatus ReadRuntimeStatus()noexcept{
    auto& s=Get();std::lock_guard lock(s.mutex);
    return {s.ready.load()!=0,s.pending.load()!=0,s.failed,s.lastOutcome.load(),
            s.applied.load(),s.raw.load(),s.failureReason};
}
bool SessionReadyFor(ID3D12Device* device,ID3D12CommandQueue* queue,unsigned width,unsigned height,
                     std::uint64_t feature,bool before)noexcept{
    auto& s=Get();std::lock_guard lock(s.mutex);
    if(!s.owner||!s.ready.load()||s.request||s.working||s.owner->fault)return false;
    const auto& shape=s.owner->shape;
    return shape.device.Get()==device&&(!queue||shape.queue.Get()==queue)&&shape.width==width&&
           shape.height==height&&shape.feature==feature&&shape.before==before;
}
bool WarmSession(ID3D12Device* device,ID3D12CommandQueue* queue,unsigned width,unsigned height,
                 std::uint64_t feature,bool before)noexcept{
    try{
        auto& s=Get();std::unique_lock lock(s.mutex);
        if(!s.workerStarted||s.failed)return false;
        Shape shape{device,queue,width,height,feature,before};
        if(!s.desired||!s.desired->Same(shape)){
            s.desired=std::make_unique<Shape>(shape);s.request=true;s.pending=1;s.ready=0;s.condition.notify_one();
        }
        const auto end=CpuClock::now()+std::chrono::seconds(30);
        while(!s.failed&&CpuClock::now()<end){
            if(!s.working&&!s.request&&s.owner&&s.owner->shape.Same(shape))return true;
            lock.unlock();Sleep(2);lock.lock();
        }
        return false;
    }catch(...){return false;}
}
void StopSession(std::uint64_t feature)noexcept{
    if(CpuTraceEnabled())TraceTotals().Report();
    auto& s=Get();std::lock_guard lock(s.mutex);
    if(!s.workerStarted)return;
    if(feature&&(!s.desired||s.desired->feature!=feature)&&(!s.owner||s.owner->shape.feature!=feature))return;
    s.desired.reset();s.request=true;s.ready=0;s.pending=1;++s.discontinuity;s.condition.notify_one();
}
extern "C" __declspec(dllexport) unsigned DlssNrNativeReady(){return Get().ready.load();}
extern "C" __declspec(dllexport) unsigned DlssNrNativePending(){return Get().pending.load();}
extern "C" __declspec(dllexport) unsigned DlssNrNativeFaults(){return Get().faults.load();}
extern "C" __declspec(dllexport) UINT64 DlssNrNativeApplied(){return Get().applied.load();}

NativeQueue::Result NativeBatch::Submit(ID3D12CommandQueue* queue,UINT count,
    ID3D12CommandList* const* lists,NativeQueue::Execute execute)noexcept {
    const bool trace=CpuTraceEnabled();std::array<CpuClock::time_point,10> times{};
    if(trace)times[0]=CpuClock::now();
    auto& session=Get();
    // A warm runtime says nothing about whether any game submission actually
    // reached it. Report each distinct bypass cause a few times so the game
    // log can separate a recorded pass from a raw upscaler submission.
    enum class Bypass : unsigned { Reused, LockBusy, Invalid, RuntimePending, PollFailed, FlightsBusy, WarmOnly, Count };
    static std::array<std::atomic<unsigned>,static_cast<unsigned>(Bypass::Count)> bypassReports{};
    auto raw=[&](Bypass cause){
        const auto count=++session.raw;
        ++session.discontinuity;
        session.lastOutcome=2;
        // The codec may already have recorded transitions on a private list.
        // A raw game submission discards that list, so retire its CPU-side
        // scratch-state assumptions before another frame can record.
        if(intent&&intent->frontend)DlssNr::RetryAfterFailure();
        const auto n=++bypassReports[static_cast<unsigned>(cause)];
        static constexpr const char* labels[]={"reused list", "queue lock busy", "invalid recording or feature",
            "runtime pending or shape changed", "runtime poll failed", "all flights busy", "warmup-only frame"};
        if(n<=3)LOG_INFO("Native NR bypass #{}: {} (ready={}, applied={})",count,
            labels[static_cast<unsigned>(cause)],session.ready.load(),session.applied.load());
        return NativeQueue::Result::Unclaimed;
    };
    // Consume even a bypassed generation: a later resubmission must never be
    // reinterpreted as a new frame merely because warmup has since completed.
    const bool usable=recording&&recording->Ready();
    if(recording&&recording->submitted.exchange(true))return raw(Bypass::Reused);
    std::unique_lock lock(session.mutex,std::try_to_lock);
    if(!lock.owns_lock())return raw(Bypass::LockBusy);
    if(repeated||!usable||!intent||!intent->featureLive||!intent->featureLive->load()||session.failed)
        return raw(Bypass::Invalid);
    Shape shape{recording->device,queue,intent->width,intent->height,intent->feature,intent->before};
    if(!session.owner||!session.owner->shape.Same(shape)||session.request||session.working){
        if(!session.desired||!session.desired->Same(shape)){
            try{session.desired=std::make_unique<Shape>(shape);}catch(...){return raw(Bypass::RuntimePending);}
            session.request=true;session.pending=1;session.ready=0;
            session.condition.notify_one();
        }
        return raw(Bypass::RuntimePending);
    }
    if(intent->warmOnly)return raw(Bypass::WarmOnly);
    auto owner=session.owner;auto& p=*owner;
    if(p.fault||!p.Poll()){RetainFault(session,owner);return raw(Bypass::PollFailed);}
    Flight* flight=nullptr;for(auto& f:p.flights)if(!f.recording&&!f.tokenLive){flight=&f;break;}
    if(!flight)return raw(Bypass::FlightsBusy);
    if(trace)times[1]=CpuClock::now();
    auto& f=*flight;bool prefixSubmitted=false,waitAttempted=false,recorded=false;
#ifdef NR_NATIVE_TEST_HOOKS
    const auto injection=testFailure.exchange(0);
#else
    constexpr unsigned injection=0;
#endif
    try{
        NativeHooks::OriginScope privateCalls(NativeHooks::Origin::Private);
        Check(f.ingress->allocator->Reset());Check(f.egress->allocator->Reset());
        Check(f.ingress->list->Reset(f.ingress->allocator.Get(),nullptr));
        Check(f.egress->list->Reset(f.egress->allocator.Get(),nullptr));
        if(injection==1)throw std::runtime_error("diagnostic before reservation");
        const auto& v=*intent;
        Transition(f.ingress->list.Get(),v.color.Get(),v.colorState,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        Transition(f.ingress->list.Get(),v.motion.Get(),v.motionState,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        NrV3::Snapshot snapshot{};if(p.api.poll(p.runtime,&snapshot,sizeof(snapshot))!=Status::Ok)throw std::runtime_error("snapshot");
        NrV3::Input input{};input.prefix={sizeof(input),NrV3::Version};
        input.key={v.feature,v.sourceFrame,++p.epoch};input.generation=snapshot.generation;
        input.commands=f.ingress->list.Get();input.queue=queue;
        input.color=Texture(v.color.Get(),v.width,v.height,NrV2::Format::Rgba16Float);
        if(v.motion)input.motion=Texture(v.motion.Get(),v.motionWidth,v.motionHeight,NrV2::Format::Rg16Float);
        input.controls=v.controls;input.motionParameters=v.motionParameters;
        input.reserved=v.temporalAccumulation?0:NrV3::DisableTemporalAccumulation;
        input.motionParameters.previousJitterX=p.jitterX;input.motionParameters.previousJitterY=p.jitterY;
        const auto discontinuity=session.discontinuity.load();
        input.resetReasons=(v.reset||p.discontinuity!=discontinuity||v.sourceFrame!=p.lastFrame+1||
            v.temporalAccumulation!=p.temporalAccumulation)?NrV2::Explicit:0;
        NrV3::RecordResult result{};
        if(trace)times[2]=CpuClock::now();
        const auto status=p.api.recordInput(p.runtime,&input,sizeof(input),&result,sizeof(result));
        if(trace)times[3]=CpuClock::now();
        recorded=result.recorded!=0;f.token=result.token;f.tokenLive=recorded;
        if(status!=Status::Ok||!recorded)
            throw std::runtime_error("record input status "+std::to_string(static_cast<unsigned>(status)));
        if(injection==2)throw std::runtime_error("diagnostic after reservation");
        Transition(f.ingress->list.Get(),v.color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,v.colorState);
        Transition(f.ingress->list.Get(),v.motion.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,v.motionState);
        auto* outputResource=v.output?v.output.Get():v.color.Get();
        const auto outputState=v.output?v.outputState:v.colorState;
        const auto publicationState=outputResource==v.color.Get()?D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        Transition(f.egress->list.Get(),outputResource,outputState,publicationState);
        NrV3::Output output{};output.prefix={sizeof(output),NrV3::Version};output.token=f.token;
        output.commands=f.egress->list.Get();output.queue=queue;
        output.target=Texture(outputResource,v.width,v.height,NrV2::Format::Rgba16Float);
        NrV3::RecordResult out{};
        const auto outputStatus=p.api.recordOutputDeferred(p.runtime,&output,sizeof(output),&out,sizeof(out));
        if(outputStatus!=Status::Ok||!out.recorded)
            throw std::runtime_error("record output status "+std::to_string(static_cast<unsigned>(outputStatus)));
        Transition(f.egress->list.Get(),outputResource,publicationState,outputState);
        Check(f.ingress->list->Close());Check(f.egress->list->Close());
        // Allocate both arrays before any GPU operation: exceptions cannot
        // interrupt assembly of the forward-raw continuation after a prefix.
        std::vector<ID3D12CommandList*> before,after;
        before.reserve(markerIndex+3);after.reserve(count-markerIndex+2);
        before.insert(before.end(),lists,lists+markerIndex);before.push_back(recording->prefix->list.Get());
        if(v.frontend)before.push_back(v.frontend->prefix->list.Get());
        before.push_back(f.ingress->list.Get());
        after.push_back(f.egress->list.Get());
        if(v.frontend)after.push_back(v.frontend->suffix->list.Get());
        const auto rawSuffixOffset=after.size();
        after.push_back(recording->suffix->list.Get());
        after.insert(after.end(),lists+markerIndex+1,lists+count);
        f.recording=recording;f.intent=intent;recording->submitted=true;
        if(trace)times[4]=CpuClock::now();
        execute(queue,static_cast<UINT>(before.size()),before.data());prefixSubmitted=true;
        if(trace)times[5]=CpuClock::now();
        const auto token=Request(f.token);
        const bool notifyOk=p.api.notifyInputSubmitted(p.runtime,&token,sizeof(token))==Status::Ok;
        if(trace)times[6]=CpuClock::now();
        const bool enqueueOk=notifyOk&&injection!=3&&p.api.enqueue(p.runtime,&token,sizeof(token))==Status::Ok;
        if(trace)times[7]=CpuClock::now();
        if(!notifyOk||!enqueueOk||injection==4){
            if(SUCCEEDED(p.shape.device->GetDeviceRemovedReason())){
                // A common-color prefix may have transitioned game resources.
                // Its abort continuation restores them without Resolve.
                if(v.frontend&&v.frontend->abort) {
                    ID3D12CommandList* restore=v.frontend->abort->list.Get();execute(queue,1,&restore);
                }
                execute(queue,static_cast<UINT>(after.size()-rawSuffixOffset),after.data()+rawSuffixOffset);
                // Retain the entire faulted owner. HIP may have started partial
                // work on its private buffers; the game color was not published.
                RetainFault(session,owner);++session.discontinuity;return NativeQueue::Result::Submitted;
            }
            throw std::runtime_error("prefix failed with lost device");
        }
        waitAttempted=true;
        if(p.api.armOutput(p.runtime,&token,sizeof(token))!=Status::Ok)throw std::runtime_error("arm output");
        if(injection==5)throw std::runtime_error("diagnostic after accepted wait");
        if(trace)times[8]=CpuClock::now();
        execute(queue,static_cast<UINT>(after.size()),after.data());
        if(injection==6)throw std::runtime_error("diagnostic after output publication");
        if(p.api.notifyOutputSubmitted(p.runtime,&token,sizeof(token))!=Status::Ok)throw std::runtime_error("output submit");
        f.retire=++p.serial;Check(queue->Signal(p.fence.Get(),f.retire));
        if(trace){times[9]=CpuClock::now();TraceTotals().Add(times);}
        p.discontinuity=discontinuity;p.lastFrame=v.sourceFrame;
        p.temporalAccumulation=v.temporalAccumulation;
        p.jitterX=v.motionParameters.currentJitterX;p.jitterY=v.motionParameters.currentJitterY;
        ++session.applied;session.lastOutcome=1;return NativeQueue::Result::Submitted;
    }catch(const std::exception& error){
        static std::atomic<unsigned> warnings{0};
        if(warnings++<8)LOG_WARN("Native NR submission failed at {} (prefix={}, wait={})",error.what(),prefixSubmitted,waitAttempted);
        if(prefixSubmitted||waitAttempted){RetainFault(session,owner);return NativeQueue::Result::Quarantined;}
        if(recorded){auto token=Request(f.token);
            if(p.api.drop(p.runtime,&token,sizeof(token))!=Status::Ok){RetainFault(session,owner);return raw(Bypass::PollFailed);}
            f.tokenLive=false;}
        // Neither list was submitted; close/discard before its allocator is reused.
        f.ingress->list->Close();f.egress->list->Close();
        if(!p.Poll())RetainFault(session,owner);
        return raw(Bypass::Invalid);
    }
}
} // namespace DlssNr::Native
