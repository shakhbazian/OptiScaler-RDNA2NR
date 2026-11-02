#include "pch.h"
#include "NativeSession.h"
#include "NativeAdapter.h"
#include <dlssnr/DlssNr_Status.h>
#include "../../../rdna2_nr/include/nr_execution_profile.h"
#include "../../../rdna2_nr/include/nr_model_catalog.h"
#include "../../../rdna2_nr/include/nr_model_spec.h"
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
void __cdecl RuntimeLog(const char* text){if(text)LOG_INFO("Native NR HIP: {}",text);}
// The catalog pins each source/graph pair; the companion checks the full digest.
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
    unsigned width=0,height=0;std::uint64_t feature=0;bool before=false;std::uint64_t modelRevision=0;unsigned modelGraph=1;
    bool Same(const Shape& b)const{return device==b.device&&queue==b.queue&&width==b.width&&height==b.height&&feature==b.feature&&before==b.before&&modelRevision==b.modelRevision&&modelGraph==b.modelGraph;}
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
        if(!get){failureReason="HIP companion API is missing";return false;}
        // Prefer the combined A/B executor. Older companions remain usable for A.
        auto profile=NrExecution::ModelABProfileId;
        auto status=get(profile,NrSubmissionApi::Version,&api,sizeof(api));
        if(status==Status::Unsupported){
            profile=NrExecution::OptimizedCheckpointId;
            status=get(profile,NrSubmissionApi::Version,&api,sizeof(api));
        }
        if(status==Status::Unsupported){
            profile=NrExecution::AcceptedMixedId;
            status=get(profile,NrSubmissionApi::Version,&api,sizeof(api));
        }
        if(status!=Status::Ok){
            failureReason="HIP companion execution profile is incompatible";
            LOG_ERROR("Native NR companion execution profile is incompatible");return false;}
        // HIP registers GPU code at module load. Keep that registration alive
        // across resizes instead of unloading/reinitializing HIP inside a game.
        // Per-session GPU resources are still drained and destroyed in Close.
        HMODULE pinned=nullptr;
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_PIN,
                              reinterpret_cast<LPCWSTR>(get),&pinned)){
            failureReason="Could not retain the HIP companion for this process";return false;}
        using SetLogSink=void (__cdecl*)(void (__cdecl*)(const char*));
        if(auto setLog=reinterpret_cast<SetLogSink>(GetProcAddress(module,"DlssNrHipBackendSetLogSink")))setLog(RuntimeLog);
        LOG_INFO("Native NR executor: {}",profile==NrExecution::ModelABProfileId?NrExecution::ModelABProfileName:profile==NrExecution::OptimizedCheckpointId?
            NrExecution::OptimizedCheckpointName:NrExecution::AcceptedMixedName);
        LOG_INFO("Native NR creating runtime for {}x{} on queue {}",shape.width,shape.height,static_cast<void*>(shape.queue.Get()));
        if(api.create(shape.device.Get(),shape.queue.Get(),&runtime)!=Status::Ok){
            failureReason="HIP runtime could not start on the selected device or queue";
            LOG_ERROR("Native NR HIP runtime could not start on the selected device/queue");return false;}
        LOG_INFO("Native NR runtime created; reading model package");
        std::ifstream input(std::filesystem::path(weights),std::ios::binary|std::ios::ate);
        if(!input){failureReason="Model package missing or unreadable: "+wstring_to_string(weights);
            LOG_ERROR("Native NR model package missing or unreadable: {}",wstring_to_string(weights));return false;}
        const auto size=input.tellg();
        if(size<128||size>512ll*1024*1024){failureReason="Model package size is outside the supported range";return false;}
        std::array<unsigned char,128> header{};input.seekg(0);input.read(reinterpret_cast<char*>(header.data()),header.size());
        const auto* identity=NrModel::Identify(header.data(),static_cast<std::uint64_t>(size));
        if(!identity||identity->graph!=shape.modelGraph){failureReason="Model package header does not match the selected A/B model";return false;}
        if(identity->graph==2&&profile!=NrExecution::ModelABProfileId){failureReason="Model B needs the A/B HIP companion from this build";return false;}
        input.clear();
        std::vector<unsigned char> package(static_cast<std::size_t>(size));input.seekg(0);
        if(!input.read(reinterpret_cast<char*>(package.data()),size)){
            failureReason="Model package read failed";
            LOG_ERROR("Native NR model package read failed: {}",wstring_to_string(weights));return false;}
        NrV3::Config config{};config.prefix={sizeof(config),NrV3::Version};
        config.sourceHash=Hash(identity->sourceHash);
        config.packageHash=Hash(identity->packageHash);
        config.graphVersion=identity->graph;config.mathContract=NrV2::MathContract;
        config.logicalWidth=shape.width;config.logicalHeight=shape.height;
        const auto extent=NrModel::WorkingExtent(identity->graph,shape.width,shape.height);
        config.networkWidth=extent.width;config.networkHeight=extent.height;
        LOG_INFO("Native NR model {}: {}x{} (network {}x{})",identity->graph==2?"B balanced mixed":"A release mixed",shape.width,shape.height,extent.width,extent.height);
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
            LOG_INFO("Native NR draining {}x{}",shape.width,shape.height);
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
            LOG_INFO("Native NR session destroyed; companion remains loaded");
        }
        if(module){FreeLibrary(module);module=nullptr;}
        return true;
    }
};
struct Session {
    std::mutex mutex;std::condition_variable condition;
    std::shared_ptr<Owner> owner;
    std::unique_ptr<Shape> desired;
    std::wstring dll,package;
    std::uint64_t assetRevision=0;unsigned assetGraph=1;
    std::shared_ptr<Owner> quarantine;
    bool workerStarted=false,request=false,working=false,failed=false;
    std::string failureReason;
    std::atomic<unsigned> lastOutcome{0};
    std::atomic<UINT64> discontinuity{1},applied{0},raw{0};
    std::atomic<unsigned> ready{0},faults{0},pending{0};
};
Session& Get(){static auto* session=new Session;return *session;}
#ifdef NR_NATIVE_TEST_HOOKS
// Failure injection is a standalone regression seam, absent from release DLLs.
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
            try{good=created->Prepare(dll,package);}
            catch(const std::exception& error){created->failureReason=error.what();good=false;}
            catch(...){created->failureReason="Unknown exception during HIP session preparation";good=false;}
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
                LOG_ERROR("Native NR warmup/drain failed: {}; original submissions remain active",s.failureReason);
            }else{
                s.failed=false;s.failureReason.clear();s.lastOutcome=0;
                s.owner=std::move(created);s.ready=s.owner&&!s.request?1:0;s.pending=s.owner||s.request?1:0;
                if(s.owner)LOG_INFO("Native NR ready on observed queue: {}x{}",s.owner->shape.width,s.owner->shape.height);
            }
        }
    }
}
}
void SetAssetPaths(std::wstring dll,std::wstring package,unsigned graph){
    auto& s=Get();std::lock_guard lock(s.mutex);
    if(s.quarantine)return; // A device/submission fault still requires a fresh process.
    if(s.dll==dll&&s.package==package&&s.assetGraph==graph&&s.workerStarted)return;
    s.dll=std::move(dll);s.package=std::move(package);s.assetGraph=graph;++s.assetRevision;++s.discontinuity;
    // A model change retires the previous owner before uploading new weights.
    // Recorded frames carry the revision so they cannot consume another model.
    s.desired.reset();s.request=true;s.ready=0;s.pending=1;
    if(!s.quarantine){s.failed=false;s.failureReason.clear();}
    if(!s.workerStarted){s.workerStarted=true;std::thread(Worker).detach();}
    s.condition.notify_one();
}
std::uint64_t AssetRevision()noexcept{auto& s=Get();std::lock_guard lock(s.mutex);return s.assetRevision;}
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
           shape.height==height&&shape.feature==feature&&shape.before==before&&shape.modelRevision==s.assetRevision;
}
bool WarmSession(ID3D12Device* device,ID3D12CommandQueue* queue,unsigned width,unsigned height,
                 std::uint64_t feature,bool before)noexcept{
    try{
        auto& s=Get();std::unique_lock lock(s.mutex);
        if(!s.workerStarted||s.failed)return false;
        Shape shape{device,queue,width,height,feature,before,s.assetRevision,s.assetGraph};
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
    auto& session=Get();
    auto raw=[&](){
        ++session.raw;
        ++session.discontinuity;
        session.lastOutcome=2;
        // The codec may already have recorded transitions on a private list.
        // A raw game submission discards that list, so retire its CPU-side
        // scratch-state assumptions before another frame can record.
        if(intent&&intent->frontend)DlssNr::RetryAfterFailure();
        return NativeQueue::Result::Unclaimed;
    };
    // Consume even a bypassed generation: a later resubmission must never be
    // reinterpreted as a new frame merely because warmup has since completed.
    const bool usable=recording&&recording->Ready();
    if(recording&&recording->submitted.exchange(true))return raw();
    std::unique_lock lock(session.mutex,std::try_to_lock);
    if(!lock.owns_lock())return raw();
    if(repeated||!usable||!intent||intent->modelRevision!=session.assetRevision||!intent->featureLive||!intent->featureLive->load()||session.failed)
        return raw();
    Shape shape{recording->device,queue,intent->width,intent->height,intent->feature,intent->before,intent->modelRevision,session.assetGraph};
    if(!session.owner||!session.owner->shape.Same(shape)||session.request||session.working){
        if(!session.desired||!session.desired->Same(shape)){
            try{session.desired=std::make_unique<Shape>(shape);}catch(...){return raw();}
            session.request=true;session.pending=1;session.ready=0;
            session.condition.notify_one();
        }
        return raw();
    }
    if(intent->warmOnly)return raw();
    auto owner=session.owner;auto& p=*owner;
    if(p.fault||!p.Poll()){RetainFault(session,owner);return raw();}
    Flight* flight=nullptr;for(auto& f:p.flights)if(!f.recording&&!f.tokenLive){flight=&f;break;}
    if(!flight)return raw();
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
        const auto status=p.api.recordInput(p.runtime,&input,sizeof(input),&result,sizeof(result));
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
        execute(queue,static_cast<UINT>(before.size()),before.data());prefixSubmitted=true;
        const auto token=Request(f.token);
        const bool notifyOk=p.api.notifyInputSubmitted(p.runtime,&token,sizeof(token))==Status::Ok;
        const bool enqueueOk=notifyOk&&injection!=3&&p.api.enqueue(p.runtime,&token,sizeof(token))==Status::Ok;
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
        execute(queue,static_cast<UINT>(after.size()),after.data());
        if(injection==6)throw std::runtime_error("diagnostic after output publication");
        if(p.api.notifyOutputSubmitted(p.runtime,&token,sizeof(token))!=Status::Ok)throw std::runtime_error("output submit");
        f.retire=++p.serial;Check(queue->Signal(p.fence.Get(),f.retire));
        p.discontinuity=discontinuity;p.lastFrame=v.sourceFrame;
        p.temporalAccumulation=v.temporalAccumulation;
        p.jitterX=v.motionParameters.currentJitterX;p.jitterY=v.motionParameters.currentJitterY;
        ++session.applied;session.lastOutcome=1;
        return NativeQueue::Result::Submitted;
    }catch(const std::exception& error){
        static std::atomic<unsigned> warnings{0};
        if(warnings++<8)LOG_WARN("Native NR submission failed at {} (prefix={}, wait={})",error.what(),prefixSubmitted,waitAttempted);
        if(prefixSubmitted||waitAttempted){RetainFault(session,owner);return NativeQueue::Result::Quarantined;}
        if(recorded){auto token=Request(f.token);
            if(p.api.drop(p.runtime,&token,sizeof(token))!=Status::Ok){RetainFault(session,owner);return raw();}
            f.tokenLive=false;}
        // Neither list was submitted; close/discard before its allocator is reused.
        f.ingress->list->Close();f.egress->list->Close();
        if(!p.Poll())RetainFault(session,owner);
        return raw();
    }
}
} // namespace DlssNr::Native
