#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <hip/hip_runtime_api.h>

#include "../include/nr_runtime_v3_dx12.h"
#include "../include/nr_runtime_v3_validation.h"
#include "../include/nr_runtime_v3_session.h"
#include "../include/nr_weight_package.h"
#include "../include/nr_gpu_model.h"
#include "../include/nr_spatial_graph.h"
#ifdef NR_SCHEDULED_MIXED
#include "../include/nr_gpu_scheduled_state.h"
#endif

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;
extern "C" hipError_t launch_v2_prepare_features(
    const void*,std::size_t,const void*,std::size_t,const void*,std::size_t,const float*,void*,
    unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,bool,std::uint32_t,bool,
    unsigned,unsigned,unsigned,unsigned,float,float,float,float,float,float,float,float,float,hipStream_t);
extern "C" hipError_t launch_v2_compose(
    const void*,std::size_t,const void*,std::size_t,const void*,const void*,const void*,void*,
    std::size_t,float*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,bool,bool,float,hipStream_t);
extern "C" hipError_t launch_v2_validate_output(
    const void*,std::size_t,unsigned,unsigned,std::uint32_t*,hipStream_t);
extern "C" hipError_t launch_v2_validate_neural(
    const void*,const void*,const float*,unsigned,unsigned,std::uint32_t*,hipStream_t);
extern "C" hipError_t launch_v2_inject_nonfinite(void*,hipStream_t);
extern "C" hipError_t launch_v3_inject_head_nonfinite(void*,std::size_t,hipStream_t);
#ifdef NR_SCHEDULED_MIXED
extern "C" hipError_t launch_scheduled_begin(NrScheduled::State*,NrScheduled::Frame,
    NrScheduled::Decision*,hipStream_t);
extern "C" hipError_t launch_scheduled_resolve(const void*,std::size_t,unsigned,unsigned,
    void*,std::size_t,const float*,float*,unsigned,unsigned,const std::uint32_t*,
    NrScheduled::State*,NrScheduled::Frame,hipStream_t);
extern "C" hipError_t launch_scheduled_prepare_features(
    const void*,std::size_t,const void*,std::size_t,const void*,std::size_t,const float*,void*,
    unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,bool,
    const NrScheduled::Decision*,unsigned,unsigned,unsigned,unsigned,float,float,float,float,
    float,float,float,float,float,hipStream_t);
extern "C" hipError_t launch_scheduled_compose(
    const void*,std::size_t,const void*,std::size_t,const void*,const void*,const void*,void*,
    std::size_t,float*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,bool,
    const NrScheduled::Decision*,float,hipStream_t);
#endif

namespace {
void Require(bool v,const char* what){if(!v)throw std::runtime_error(what);}
struct CleanupError:std::runtime_error{using std::runtime_error::runtime_error;};
void Hr(HRESULT v,const char* what){if(FAILED(v)){char b[128]{};std::snprintf(b,sizeof(b),"%s: 0x%08lx",what,static_cast<unsigned long>(v));throw std::runtime_error(b);}}
void Hip(hipError_t v,const char* what){if(v!=hipSuccess)throw std::runtime_error(std::string(what)+": "+hipGetErrorString(v));}
// HIP's current device is thread-local. The scheduled owner can be configured on a
// warm-up thread and used later from the render/queue thread, so the stream's
// device must be selected for every entry point that touches HIP.
struct ScopedHipDevice {
    int previous=-1;
    bool changed=false;
    explicit ScopedHipDevice(int selected){
        Hip(hipGetDevice(&previous),"hipGetDevice");
        if(previous!=selected){Hip(hipSetDevice(selected),"hipSetDevice");changed=true;}
    }
    ~ScopedHipDevice(){if(changed)static_cast<void>(hipSetDevice(previous));}
};
D3D12_RESOURCE_DESC BufferDesc(std::uint64_t bytes){D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=bytes;d.Height=1;d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;return d;}
D3D12_RESOURCE_DESC TextureDesc(unsigned w,unsigned h,DXGI_FORMAT f){D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=w;d.Height=h;d.DepthOrArraySize=1;d.MipLevels=1;d.Format=f;d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_UNKNOWN;return d;}
D3D12_RESOURCE_BARRIER Barrier(ID3D12Resource* r,D3D12_RESOURCE_STATES a,D3D12_RESOURCE_STATES b){D3D12_RESOURCE_BARRIER v{};v.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;v.Transition.pResource=r;v.Transition.StateBefore=a;v.Transition.StateAfter=b;v.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;return v;}
NrV2::Sha256Digest Digest(const NrV2::Hash256& h){NrV2::Sha256Digest d{};std::memcpy(d.data(),h.bytes,d.size());return d;}
}

namespace NrV3 {
struct RuntimeDx12::Impl {
    struct Shared {
        ComPtr<ID3D12Resource> resource;
        hipExternalMemory_t memory=nullptr;void* pointer=nullptr;HANDLE handle=nullptr;
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};std::uint64_t bytes=0;
        bool Release() noexcept {
            if(pointer){if(hipFree(pointer)!=hipSuccess)return false;pointer=nullptr;}
            if(memory){if(hipDestroyExternalMemory(memory)!=hipSuccess)return false;memory=nullptr;}
            if(handle){if(!CloseHandle(handle))return false;handle=nullptr;}
            resource.Reset();footprint={};bytes=0;return true;
        }
    };
    struct Timeline {
        ComPtr<ID3D12Fence> fence;hipExternalSemaphore_t semaphore=nullptr;
        bool Release() noexcept {if(semaphore){if(hipDestroyExternalSemaphore(semaphore)!=hipSuccess)return false;semaphore=nullptr;}fence.Reset();return true;}
    };
    struct TransportSlot {
        Shared color,motion,mask,output;
        Input staged{};Token token{};
        ComPtr<ID3D12GraphicsCommandList> inputList,outputList;
        ComPtr<ID3D12Resource> inputColor,inputMotion,inputMask,outputTarget;
        bool hasInput=false,hasOutput=false;
#ifdef NR_SCHEDULED_MIXED
        Phase scheduledPhase=Phase::Free;
        std::uint64_t scheduledInputSerial=0,scheduledRetireSerial=0;
        bool scheduledHipTouched=false;
        bool telemetryReported=false;
        bool transferReported=false;
        std::array<hipEvent_t,5> telemetry{};
#endif
        bool ReleaseInterop() noexcept {return output.Release()&&mask.Release()&&motion.Release()&&color.Release();}
        void ReleaseRefs(){inputList.Reset();outputList.Reset();inputColor.Reset();inputMotion.Reset();inputMask.Reset();outputTarget.Reset();staged={};token={};hasInput=false;hasOutput=false;
#ifdef NR_SCHEDULED_MIXED
            scheduledPhase=Phase::Free;scheduledInputSerial=0;scheduledRetireSerial=0;scheduledHipTouched=false;
            telemetryReported=false;
            transferReported=false;
#endif
        }
    };

    ComPtr<ID3D12Device> device;ComPtr<ID3D12CommandQueue> queue;
    int hipDevice=-1;
    Timeline d3d,hip;
    hipStream_t stream=nullptr;hipEvent_t completion=nullptr;bool eventRecorded=false;
    std::array<TransportSlot,Capacity> slots{};
    std::unique_ptr<NrV2::WeightPackage> package;
    std::unique_ptr<NrV2::GpuModel> model;
    std::unique_ptr<NrV2::SpatialGraphRuntime> graph;
    void* prepared=nullptr;void* head=nullptr;float* committed=nullptr;float* candidate=nullptr;
    const void* blendScale=nullptr;std::uint32_t* finiteDevice=nullptr;std::uint32_t* finiteHost=nullptr;
    Config config{};Session session;Token running{};
    std::uint64_t nextD3d=1,modelUploads=0,dispatches=0,swaps=0;
    std::uint64_t lastRecordUs=0,lastDispatchUs=0,lastPollUs=0;
    bool configured=false,closed=false,quarantined=false;
#ifdef NR_SCHEDULED_MIXED
    bool scheduled=false,scheduledDraining=false,scheduledFaulted=false,scheduledDestroyed=false;
    bool scheduledTelemetry=false;
    bool scheduledHipTelemetry=false;
    bool scheduledReplay=false;
    hipGraph_t networkReplay=nullptr;
    hipGraphExec_t networkExecutable=nullptr;
    ComPtr<ID3D12QueryHeap> transferQueries;
    ComPtr<ID3D12Resource> transferReadback;
    std::uint64_t transferFrequency=0;
    std::uint64_t scheduledGeneration=1,scheduledLastEpoch=0;
    NrScheduled::State* gpuState=nullptr;
    NrScheduled::Decision* gpuDecisions=nullptr;
#endif
#ifdef NR_RUNTIME_V3_TEST_HOOKS
    bool failNext=false,injectHeadNext=false,injectOutputNext=false,failCleanup=false,failSignal=false,failEvent=false;
#endif

    explicit Impl(ID3D12Device* d,ID3D12CommandQueue* q
#ifdef NR_SCHEDULED_MIXED
        ,bool scheduledMode=false
#endif
        ):device(d),queue(q)
#ifdef NR_SCHEDULED_MIXED
        ,scheduled(scheduledMode)
#endif
    {
        Require(d&&q,"device and queue required");ComPtr<ID3D12Device> owner;
        Hr(q->GetDevice(IID_PPV_ARGS(&owner)),"queue GetDevice");
        Require(owner.Get()==d&&q->GetDesc().Type==D3D12_COMMAND_LIST_TYPE_DIRECT,"direct bound queue required");
        const LUID luid=d->GetAdapterLuid();int count=0,selected=-1;Hip(hipGetDeviceCount(&count),"hipGetDeviceCount");
        for(int i=0;i<count;++i){hipDeviceProp_t p{};Hip(hipGetDeviceProperties(&p,i),"hipGetDeviceProperties");if(!std::memcmp(&luid,p.luid,sizeof(luid))&&!std::strncmp(p.gcnArchName,"gfx1030",7)){selected=i;break;}}
        Require(selected>=0,"matching gfx1030 HIP device required");
        hipDevice=selected;
        ScopedHipDevice boundDevice(hipDevice);
        try{
            char priorityFlag[16]{};
            const bool tryHigh=scheduled&&GetEnvironmentVariableA("DLSSNR_HIP_PRIORITY",priorityFlag,sizeof(priorityFlag))>0&&
                std::strcmp(priorityFlag,"high")==0;
            int least=0,greatest=0;
            const bool priorityRangeAvailable=scheduled&&hipDeviceGetStreamPriorityRange(&least,&greatest)==hipSuccess;
            if(tryHigh&&priorityRangeAvailable&&greatest<least){
                const auto priorityStatus=hipStreamCreateWithPriority(&stream,hipStreamNonBlocking,greatest);
                if(priorityStatus!=hipSuccess){
                    std::fprintf(stderr,"ScheduledPriority requested=high unavailable=%s; using default\n",hipGetErrorString(priorityStatus));
                    stream=nullptr;
                }
            }
            if(!stream)Hip(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"hipStreamCreate");
            if(scheduled){int actual=0;
                const auto readStatus=hipStreamGetPriority(stream,&actual);
                if(priorityRangeAvailable)
                    std::fprintf(stderr,"ScheduledPriority requested=%s range=%d:%d actual=%s%d\n",
                        tryHigh?"high":"default",greatest,least,readStatus==hipSuccess?"":"unavailable/",actual);
                else
                    std::fprintf(stderr,"ScheduledPriority requested=%s range=unavailable actual=%s%d\n",
                        tryHigh?"high":"default",readStatus==hipSuccess?"":"unavailable/",actual);
            }
#ifdef NR_SCHEDULED_MIXED
            char telemetryFlag[8]{};
            char replayFlag[8]{};
            scheduledReplay=scheduled&&!(GetEnvironmentVariableA("DLSSNR_NETWORK_REPLAY",replayFlag,sizeof(replayFlag))>0&&replayFlag[0]=='0');
            scheduledTelemetry=scheduled&&GetEnvironmentVariableA("DLSSNR_GPU_TIMELINE",telemetryFlag,sizeof(telemetryFlag))>0&&
                telemetryFlag[0]=='1';
            char hipTimingFlag[8]{};
            scheduledHipTelemetry=scheduledTelemetry&&GetEnvironmentVariableA("DLSSNR_HIP_EVENT_TIMELINE",hipTimingFlag,sizeof(hipTimingFlag))>0&&hipTimingFlag[0]=='1';
            if(scheduledTelemetry)std::fprintf(stderr,"ScheduledGpu telemetry=d3d12 hip_events=%s\n",scheduledHipTelemetry?"unqualified":"disabled");
            if(scheduledHipTelemetry)for(auto& slot:slots)for(auto& event:slot.telemetry)
                Hip(hipEventCreate(&event),"scheduled telemetry event");
            if(scheduledTelemetry){
                D3D12_QUERY_HEAP_DESC queries{};queries.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
                queries.Count=Capacity*4;
                Hr(device->CreateQueryHeap(&queries,IID_PPV_ARGS(&transferQueries)),"transfer query heap");
                D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_READBACK;
                const auto desc=BufferDesc(Capacity*4*sizeof(std::uint64_t));
                Hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&transferReadback)),"transfer readback");
                Hr(queue->GetTimestampFrequency(&transferFrequency),"transfer timestamp frequency");
                Require(transferFrequency>0,"zero transfer timestamp frequency");
            }
#endif
            Hip(hipEventCreateWithFlags(&completion,hipEventDisableTiming),"hipEventCreate");
            Hip(hipMalloc(&finiteDevice,sizeof(*finiteDevice)),"hipMalloc finite");
            Hip(hipHostMalloc(&finiteHost,sizeof(*finiteHost),hipHostMallocDefault),"hipHostMalloc finite");*finiteHost=0;
            CreateTimeline(d3d);CreateTimeline(hip);
        }catch(...){CleanupPartial();throw;}
    }
    void CleanupPartial()noexcept{
#ifdef NR_SCHEDULED_MIXED
        for(auto& slot:slots)for(auto& event:slot.telemetry)
            if(event){static_cast<void>(hipEventDestroy(event));event=nullptr;}
#endif
        for(auto& s:slots)static_cast<void>(s.ReleaseInterop());
        if(finiteHost){static_cast<void>(hipHostFree(finiteHost));finiteHost=nullptr;}
        if(finiteDevice){static_cast<void>(hipFree(finiteDevice));finiteDevice=nullptr;}
        if(completion){static_cast<void>(hipEventDestroy(completion));completion=nullptr;}
        static_cast<void>(hip.Release());static_cast<void>(d3d.Release());
        if(stream){static_cast<void>(hipStreamDestroy(stream));stream=nullptr;}
    }
    void CreateTimeline(Timeline& t){
        Hr(device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&t.fence)),"CreateFence");
        HANDLE h=nullptr;Hr(device->CreateSharedHandle(t.fence.Get(),nullptr,GENERIC_ALL,nullptr,&h),"CreateSharedHandle fence");
        hipExternalSemaphoreHandleDesc x{};x.type=hipExternalSemaphoreHandleTypeD3D12Fence;x.handle.win32.handle=h;
        const auto s=hipImportExternalSemaphore(&t.semaphore,&x);CloseHandle(h);Hip(s,"hipImportExternalSemaphore");
    }
    void CreateShared(Shared& out,const D3D12_RESOURCE_DESC& texture){
        Require(!out.resource&&!out.pointer&&!out.memory,"occupied shared slot mutation");
        UINT rows=0;UINT64 rowBytes=0;device->GetCopyableFootprints(&texture,0,1,0,&out.footprint,&rows,&rowBytes,&out.bytes);
        Require(rows==texture.Height,"shared footprint rows");D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;auto desc=BufferDesc(out.bytes);
        Hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_SHARED,&desc,D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&out.resource)),"Create shared buffer");
        Hr(device->CreateSharedHandle(out.resource.Get(),nullptr,GENERIC_ALL,nullptr,&out.handle),"Create shared buffer handle");
        hipExternalMemoryHandleDesc x{};x.type=hipExternalMemoryHandleTypeD3D12Resource;x.handle.win32.handle=out.handle;x.size=out.bytes;x.flags=hipExternalMemoryDedicated;
        Hip(hipImportExternalMemory(&out.memory,&x),"hipImportExternalMemory");hipExternalMemoryBufferDesc m{};m.size=out.bytes;
        Hip(hipExternalMemoryGetMappedBuffer(&out.pointer,out.memory,&m),"hipExternalMemoryGetMappedBuffer");
    }
    static bool Match(const Shared& s,unsigned w,unsigned h){return s.resource&&s.footprint.Footprint.Width==w&&s.footprint.Footprint.Height==h;}
    void Ensure(Shared& s,unsigned w,unsigned h,DXGI_FORMAT f){if(Match(s,w,h))return;Require(!s.resource,"fixed-geometry slot changed before reconfigure");CreateShared(s,TextureDesc(w,h,f));}
    static bool TextureMatches(const NrV2::Texture& t,DXGI_FORMAT f){if(!t.resource)return false;const auto d=t.resource->GetDesc();return d.Dimension==D3D12_RESOURCE_DIMENSION_TEXTURE2D&&d.Width==t.transform.backingWidth&&d.Height==t.transform.backingHeight&&d.DepthOrArraySize==1&&d.MipLevels==1&&d.SampleDesc.Count==1&&d.Format==f;}
    bool RuntimeDevice(ID3D12Resource* r)const{ComPtr<ID3D12Device> d;return r&&SUCCEEDED(r->GetDevice(IID_PPV_ARGS(&d)))&&d.Get()==device.Get();}
    bool RuntimeList(ID3D12GraphicsCommandList* l)const{ComPtr<ID3D12Device> d;return l&&l->GetType()==D3D12_COMMAND_LIST_TYPE_DIRECT&&SUCCEEDED(l->GetDevice(IID_PPV_ARGS(&d)))&&d.Get()==device.Get();}
    static bool DistinctLists(ID3D12GraphicsCommandList* a,ID3D12GraphicsCommandList* b){
        if(!a||!b||a==b)return false;
        ComPtr<IUnknown> left,right;
        return SUCCEEDED(a->QueryInterface(IID_PPV_ARGS(&left)))&&
            SUCCEEDED(b->QueryInterface(IID_PPV_ARGS(&right)))&&left.Get()!=right.Get();
    }
    bool Conflicts(ID3D12Resource* r,int except=-1)const{
        if(!r)return false;
        for(unsigned i=0;i<Capacity;++i){if(static_cast<int>(i)==except)continue;const auto& s=slots[i];if(!s.hasInput)continue;
            if(r==s.inputColor.Get()||r==s.inputMotion.Get()||r==s.inputMask.Get()||r==s.outputTarget.Get()||
                r==s.color.resource.Get()||r==s.motion.resource.Get()||r==s.mask.resource.Get()||r==s.output.resource.Get())return true;}
        return false;
    }
#ifdef NR_SCHEDULED_MIXED
    bool ConflictsExceptOrderedOutput(ID3D12Resource* r,int except)const{
        if(!r)return false;
        for(unsigned i=0;i<Capacity;++i){
            if(static_cast<int>(i)==except)continue;
            const auto& s=slots[i];if(!s.hasInput)continue;
            // The native frontend reuses one output surface. Every accepted
            // frame publishes and consumes it on this registered queue before
            // the next frame's prefix, so an older output target is ordered.
            if(r==s.outputTarget.Get()&&s.hasOutput&&r!=s.inputColor.Get()&&
               r!=s.inputMotion.Get()&&r!=s.inputMask.Get())continue;
            if(r==s.inputColor.Get()||r==s.inputMotion.Get()||r==s.inputMask.Get()||
               r==s.outputTarget.Get()||r==s.color.resource.Get()||r==s.motion.resource.Get()||
               r==s.mask.resource.Get()||r==s.output.resource.Get())return true;
        }
        return false;
    }
#endif
    // Ingress reads are serialized on the registered D3D12 queue. The DX11
    // bridge deliberately reuses its color/motion cache across frames and
    // orders each new DX11 copy after the previous D3D12 work. Retaining a
    // read-only source in another slot therefore is not a write conflict.
#ifdef NR_SCHEDULED_MIXED
    bool ConflictsWithSlotWrite(ID3D12Resource* r)const{
        if(!r)return false;
        for(const auto& s:slots){if(!s.hasInput)continue;
            if(r==s.outputTarget.Get()||r==s.color.resource.Get()||r==s.motion.resource.Get()||
                r==s.mask.resource.Get()||r==s.output.resource.Get())return true;}
        return false;
    }
#endif
    Status ValidateInputResources(const Input& f)const{
        if(f.queue!=queue.Get()||!RuntimeList(f.commands)||!TextureMatches(f.color,DXGI_FORMAT_R16G16B16A16_FLOAT))return Status::InvalidResource;
        const bool mv=!NrV2::Empty(f.motion),mask=!NrV2::Empty(f.controlMask);
        if((mv&&!TextureMatches(f.motion,DXGI_FORMAT_R16G16_FLOAT))||(mask&&!TextureMatches(f.controlMask,DXGI_FORMAT_R16G16B16A16_FLOAT)))return Status::InvalidResource;
        if(!RuntimeDevice(f.color.resource)||(mv&&!RuntimeDevice(f.motion.resource))||(mask&&!RuntimeDevice(f.controlMask.resource)))return Status::InvalidResource;
#ifdef NR_SCHEDULED_MIXED
        if(scheduled){
            if((mv&&f.color.resource==f.motion.resource)||(mask&&
                (f.color.resource==f.controlMask.resource||(mv&&f.motion.resource==f.controlMask.resource)))||
                ConflictsWithSlotWrite(f.color.resource)||(mv&&ConflictsWithSlotWrite(f.motion.resource))||
                (mask&&ConflictsWithSlotWrite(f.controlMask.resource)))return Status::InvalidResource;
        }else
#endif
        if(Conflicts(f.color.resource)||(mv&&Conflicts(f.motion.resource))||(mask&&Conflicts(f.controlMask.resource)))return Status::InvalidResource;
        return Status::Ok;
    }
    Status ValidateOutputResources(const Output& o,int index,bool deferred=false)const{
        bool conflicts=Conflicts(o.target.resource,index);
#ifdef NR_SCHEDULED_MIXED
        if(deferred&&scheduled)conflicts=ConflictsExceptOrderedOutput(o.target.resource,index);
#endif
        if(o.queue!=queue.Get()||!RuntimeList(o.commands)||!TextureMatches(o.target,DXGI_FORMAT_R16G16B16A16_FLOAT)||!RuntimeDevice(o.target.resource)||conflicts)return Status::InvalidResource;
        const auto& s=slots[static_cast<unsigned>(index)];
        // A deferred owned output may publish into its previously captured
        // color. Distinct lists plus the queue/HIP fences enforce read-before-
        // write; legacy paths and an overlapping same-list recording stay off.
        bool inPlace=false;
#ifdef NR_SCHEDULED_MIXED
        inPlace=deferred&&scheduled&&o.target.resource==s.inputColor.Get()&&
            DistinctLists(o.commands,s.inputList.Get());
#else
        (void)deferred;
#endif
        return (o.target.resource==s.inputColor.Get()&&!inPlace)||o.target.resource==s.inputMotion.Get()||o.target.resource==s.inputMask.Get()||
            o.target.resource==s.color.resource.Get()||o.target.resource==s.motion.resource.Get()||o.target.resource==s.mask.resource.Get()||o.target.resource==s.output.resource.Get()?Status::InvalidResource:Status::Ok;
    }
    int Index(const Token& t)const{for(unsigned i=0;i<Capacity;++i)if(slots[i].hasInput&&Same(slots[i].token,t))return static_cast<int>(i);return -1;}
    int FreeIndex()const{const auto v=session.View();for(unsigned i=0;i<Capacity;++i)if(v.slots[i].phase==Phase::Free&&!slots[i].hasInput)return static_cast<int>(i);return -1;}
#ifdef NR_SCHEDULED_MIXED
    std::uint32_t ScheduledOccupied()const{std::uint32_t n=0;for(const auto& slot:slots)if(slot.scheduledPhase!=Phase::Free)++n;return n;}
    int ScheduledFreeIndex()const{for(unsigned i=0;i<Capacity;++i)if(slots[i].scheduledPhase==Phase::Free&&!slots[i].hasInput)return static_cast<int>(i);return -1;}
    Status ScheduledLive()const{if(quarantined)return Status::Quarantined;if(closed||scheduledDraining)return Status::Closed;if(scheduledFaulted)return Status::Failed;return configured?Status::Ok:Status::NotConfigured;}
#endif

    // Replay owns pointers into prepared/head and the spatial workspace. Destroy
    // it only after the owning stream is idle, before any of those allocations.
#ifdef NR_SCHEDULED_MIXED
    bool ReleaseReplay()noexcept{
        if(networkExecutable){if(hipGraphExecDestroy(networkExecutable)!=hipSuccess)return false;networkExecutable=nullptr;}
        if(networkReplay){if(hipGraphDestroy(networkReplay)!=hipSuccess)return false;networkReplay=nullptr;}
        return true;
    }
    void CaptureNetwork(const Config& next){
        if(!scheduledReplay)return;
        Hip(hipStreamBeginCapture(stream,hipStreamCaptureModeThreadLocal),"network capture begin");
        try{graph->Enqueue(prepared,next.networkHeight,next.networkWidth,head);}
        catch(...){
            // End even an invalidated capture; otherwise normal shutdown would
            // operate on a stream that is still in capture mode.
            static_cast<void>(hipStreamEndCapture(stream,&networkReplay));
            throw;
        }
        Hip(hipStreamEndCapture(stream,&networkReplay),"network capture end");
        Hip(hipGraphInstantiate(&networkExecutable,networkReplay,nullptr,nullptr,0),"network instantiate");
        std::size_t nodes=0;Hip(hipGraphGetNodes(networkReplay,nullptr,&nodes),"network node count");
        std::fprintf(stderr,"ScheduledReplay configured=%ux%u nodes=%zu\n",next.networkWidth,next.networkHeight,nodes);
    }
#endif
    bool ReleaseNeural()noexcept{
#ifdef NR_SCHEDULED_MIXED
        if(!ReleaseReplay())return false;
#endif
        if(prepared){if(hipFree(prepared)!=hipSuccess)return false;prepared=nullptr;}
        if(head){if(hipFree(head)!=hipSuccess)return false;head=nullptr;}
        if(committed){if(hipFree(committed)!=hipSuccess)return false;committed=nullptr;}
        if(candidate){if(hipFree(candidate)!=hipSuccess)return false;candidate=nullptr;}
#ifdef NR_SCHEDULED_MIXED
        if(gpuDecisions){if(hipFree(gpuDecisions)!=hipSuccess)return false;gpuDecisions=nullptr;}
        if(gpuState){if(hipFree(gpuState)!=hipSuccess)return false;gpuState=nullptr;}
#endif
        blendScale=nullptr;
        if(graph&&!graph->Release())return false;graph.reset();
        if(model&&!model->Release())return false;model.reset();package.reset();return true;
    }
    Status Configure(const Config& next,const void* bytes,std::uint64_t count){
        if(quarantined)return Status::Quarantined;if(closed)return Status::Closed;
#ifdef NR_SCHEDULED_MIXED
        std::size_t freeBefore=0,totalBefore=0;
        if(scheduledTelemetry)static_cast<void>(hipMemGetInfo(&freeBefore,&totalBefore));
#endif
        auto s=Validate(next);if(s!=Status::Ok)return s;if(!bytes||!count||count>std::numeric_limits<std::size_t>::max())return Status::InvalidSize;
#ifdef NR_SCHEDULED_MIXED
        if(scheduled){if(scheduledDraining||scheduledFaulted)return Status::Closed;if(ScheduledOccupied())return Status::Busy;}
        else
#endif
        {const auto before=session.View();
            if(before.lifecycle!=Lifecycle::Created&&before.lifecycle!=Lifecycle::Live)return Status::Closed;
            if(before.occupied)return Status::Busy;}
        const bool sameModel=configured&&package&&model&&graph&&NrV2::Same(config.sourceHash,next.sourceHash)&&NrV2::Same(config.packageHash,next.packageHash)&&config.graphVersion==next.graphVersion&&config.mathContract==next.mathContract;
        try{
            std::unique_ptr<NrV2::WeightPackage> replacement;
            if(sameModel){if(count!=package->ByteLength()||std::memcmp(bytes,package->Data(),package->ByteLength()))return Status::Failed;}
            else{
                std::vector<std::uint8_t> owned(static_cast<std::size_t>(count));std::memcpy(owned.data(),bytes,owned.size());
                replacement=std::make_unique<NrV2::WeightPackage>(NrV2::WeightPackage::LoadMemory(std::move(owned),Digest(next.packageHash),Digest(next.sourceHash)));
            }
            if(prepared||head||committed||candidate){if(hipStreamSynchronize(stream)!=hipSuccess)throw CleanupError("reconfigure idle");FreeNeuralBuffersChecked();}
            if(!sameModel){
                if(graph&&!graph->Release())throw CleanupError("graph release failed");graph.reset();
                if(model&&!model->Release())throw CleanupError("model release failed");model.reset();package.reset();
                package=std::move(replacement);model=std::make_unique<NrV2::GpuModel>(*package);graph=std::make_unique<NrV2::SpatialGraphRuntime>(*model,stream);++modelUploads;
            }
            const auto featureBytes=std::size_t(next.networkWidth)*next.networkHeight*16*sizeof(std::uint16_t);
            const auto headBytes=std::size_t(next.networkWidth)*next.networkHeight*4*sizeof(std::uint16_t);
            const auto historyBytes=std::size_t(next.logicalWidth)*next.logicalHeight*3*sizeof(float);
            Hip(hipMalloc(&prepared,featureBytes),"hipMalloc prepared");Hip(hipMalloc(&head,headBytes),"hipMalloc head");
            Hip(hipMalloc(&committed,historyBytes),"hipMalloc committed");Hip(hipMalloc(&candidate,historyBytes),"hipMalloc candidate");
#ifdef NR_SCHEDULED_MIXED
            if(scheduled){Hip(hipMalloc(&gpuState,sizeof(*gpuState)),"hipMalloc scheduled state");
                Hip(hipMalloc(&gpuDecisions,Capacity*sizeof(*gpuDecisions)),"hipMalloc decisions");
                Hip(hipMemsetAsync(gpuState,0,sizeof(*gpuState),stream),"clear scheduled state");
                Hip(hipMemsetAsync(gpuDecisions,0,Capacity*sizeof(*gpuDecisions),stream),"clear decisions");}
#endif
            Hip(hipMemsetAsync(prepared,0,featureBytes,stream),"memset prepared");Hip(hipMemsetAsync(committed,0,historyBytes,stream),"memset committed");Hip(hipMemsetAsync(candidate,0,historyBytes,stream),"memset candidate");
            graph->Prepare(prepared,next.networkHeight,next.networkWidth,head);blendScale=model->TensorData(70,"temporal_blend_scale",1);
#ifdef NR_SCHEDULED_MIXED
            CaptureNetwork(next);
            if(scheduled){if(configured){if(scheduledGeneration==ReservedSerial-1)throw std::runtime_error("scheduled generation overflow");++scheduledGeneration;}
                scheduledLastEpoch=0;}
            else
#endif
            {s=session.Configure();if(s!=Status::Ok)throw std::runtime_error("session configure failed");}
            for(auto& slot:slots){if(!slot.ReleaseInterop())throw CleanupError("slot release failed");slot.ReleaseRefs();
                CreateShared(slot.output,TextureDesc(next.logicalWidth,next.logicalHeight,DXGI_FORMAT_R16G16B16A16_FLOAT));
#ifdef NR_SCHEDULED_MIXED
                if(scheduled){CreateShared(slot.color,TextureDesc(next.logicalWidth,next.logicalHeight,DXGI_FORMAT_R16G16B16A16_FLOAT));
                    CreateShared(slot.motion,TextureDesc(next.logicalWidth,next.logicalHeight,DXGI_FORMAT_R16G16_FLOAT));
                    CreateShared(slot.mask,TextureDesc(next.logicalWidth,next.logicalHeight,DXGI_FORMAT_R16G16B16A16_FLOAT));}
#endif
            }
            config=next;configured=true;
#ifdef NR_SCHEDULED_MIXED
            if(scheduledTelemetry){std::size_t freeAfter=0,totalAfter=0;
                if(hipMemGetInfo(&freeAfter,&totalAfter)==hipSuccess)
                    std::fprintf(stderr,"ScheduledMemory extent=%ux%u free_before=%zu free_after=%zu consumed_delta=%zu total=%zu\n",
                        next.logicalWidth,next.logicalHeight,freeBefore,freeAfter,
                        freeBefore>=freeAfter?freeBefore-freeAfter:0,totalAfter);}
#endif
            return Status::Ok;
        }catch(const CleanupError& e){std::fprintf(stderr,"v3 configure cleanup: %s\n",e.what());Quarantine();return Status::Quarantined;
        }catch(const std::exception& e){std::fprintf(stderr,"v3 configure: %s\n",e.what());
#ifdef NR_SCHEDULED_MIXED
            if(scheduled){scheduledFaulted=true;return Status::Failed;}
#endif
            session.Fault(nullptr);return Status::Failed;}
    }
    void FreeNeuralBuffersChecked(){
#ifdef NR_SCHEDULED_MIXED
        if(!ReleaseReplay())throw CleanupError("free network replay");
#endif
        if(prepared){if(hipFree(prepared)!=hipSuccess)throw CleanupError("free prepared");prepared=nullptr;}if(head){if(hipFree(head)!=hipSuccess)throw CleanupError("free head");head=nullptr;}
        if(committed){if(hipFree(committed)!=hipSuccess)throw CleanupError("free committed");committed=nullptr;}if(candidate){if(hipFree(candidate)!=hipSuccess)throw CleanupError("free candidate");candidate=nullptr;}
#ifdef NR_SCHEDULED_MIXED
        if(gpuState){if(hipFree(gpuState)!=hipSuccess)throw CleanupError("free scheduled state");gpuState=nullptr;}
        if(gpuDecisions){if(hipFree(gpuDecisions)!=hipSuccess)throw CleanupError("free decisions");gpuDecisions=nullptr;}
#endif
    }

    Status RecordInput(const Input& f,RecordResult& out){
        const auto start=std::chrono::steady_clock::now();
        out.status=Status::Failed;if(!configured)return out.status=Status::NotConfigured;
        auto s=Validate(f,config);if(s!=Status::Ok)return out.status=s;s=ValidateInputResources(f);if(s!=Status::Ok)return out.status=s;
        const int index=FreeIndex();if(index<0)return out.status=Status::Busy;Token t{f.key,f.generation};
        auto& slot=slots[static_cast<unsigned>(index)];const bool mv=!NrV2::Empty(f.motion),mask=!NrV2::Empty(f.controlMask);
        try{
            Ensure(slot.color,f.color.transform.extentWidth,f.color.transform.extentHeight,DXGI_FORMAT_R16G16B16A16_FLOAT);
            if(mv)Ensure(slot.motion,f.motion.transform.extentWidth,f.motion.transform.extentHeight,DXGI_FORMAT_R16G16_FLOAT);
            if(mask)Ensure(slot.mask,f.controlMask.transform.extentWidth,f.controlMask.transform.extentHeight,DXGI_FORMAT_R16G16B16A16_FLOAT);
            s=session.Record(t,f.resetReasons!=0,mv);if(s!=Status::Ok)return out.status=s;
            slot.staged=f;slot.token=t;slot.hasInput=true;slot.inputList=f.commands;slot.inputColor=f.color.resource;
            if(mv)slot.inputMotion=f.motion.resource;if(mask)slot.inputMask=f.controlMask.resource;
            std::array<D3D12_RESOURCE_BARRIER,8> before{},after{};UINT n=0;
            before[n++]=Barrier(f.color.resource,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
            before[n++]=Barrier(slot.color.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
            if(mv){before[n++]=Barrier(f.motion.resource,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);before[n++]=Barrier(slot.motion.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);}
            if(mask){before[n++]=Barrier(f.controlMask.resource,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);before[n++]=Barrier(slot.mask.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);}
            f.commands->ResourceBarrier(n,before.data());
            CopyIn(f.commands,slot.color,f.color);if(mv)CopyIn(f.commands,slot.motion,f.motion);if(mask)CopyIn(f.commands,slot.mask,f.controlMask);
            n=0;after[n++]=Barrier(f.color.resource,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);after[n++]=Barrier(slot.color.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
            if(mv){after[n++]=Barrier(f.motion.resource,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);after[n++]=Barrier(slot.motion.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);}
            if(mask){after[n++]=Barrier(f.controlMask.resource,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);after[n++]=Barrier(slot.mask.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);}
            f.commands->ResourceBarrier(n,after.data());out.recorded=1;out.token=t;lastRecordUs=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count());return out.status=Status::Ok;
        }catch(...){if(slot.hasInput){out.recorded=1;out.token=t;}session.Fault(nullptr);return out.status=Status::Failed;}
    }
    static void CopyIn(ID3D12GraphicsCommandList* list,Shared& dst,const NrV2::Texture& src){D3D12_TEXTURE_COPY_LOCATION a{};a.pResource=dst.resource.Get();a.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;a.PlacedFootprint=dst.footprint;D3D12_TEXTURE_COPY_LOCATION b{};b.pResource=src.resource;b.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;const auto& t=src.transform;const D3D12_BOX box{t.baseX,t.baseY,0,t.baseX+t.extentWidth,t.baseY+t.extentHeight,1};list->CopyTextureRegion(&a,0,0,0,&b,&box);}
    Status NotifyInput(const Token& t){
        auto s=session.CheckInputAck(t);if(s!=Status::Ok)return s;const int i=Index(t);if(i<0)return Status::StaleFrame;
#ifdef NR_RUNTIME_V3_TEST_HOOKS
        if(failSignal){failSignal=false;Quarantine();return Status::Quarantined;}
#endif
        const auto serial=nextD3d++;if(serial==ReservedSerial)return Status::Failed;const auto hr=queue->Signal(d3d.fence.Get(),serial);
        if(FAILED(hr)){Quarantine();return Status::Quarantined;}s=session.InputSubmitted(t,serial);return s;
    }
    Status Dispatch(){
        if(eventRecorded)return Status::Busy;const auto done=d3d.fence->GetCompletedValue();if(done==ReservedSerial){Quarantine();return Status::Quarantined;}
        Session::Decision d{};auto s=session.Dispatch(done,d);if(s!=Status::Ok)return s;Token t{d.key,session.View().generation};const int i=Index(t);if(i<0){session.Fault(&t);return Status::Failed;}
        auto& slot=slots[static_cast<unsigned>(i)];const auto* state=FindSlot(t);if(!state){session.Fault(&t);return Status::Failed;}
        try{
            hipExternalSemaphoreWaitParams wait{};wait.params.fence.value=state->inputSerial;Hip(hipWaitExternalSemaphoresAsync(&d3d.semaphore,&wait,1,stream),"HIP input wait");
            const bool hasMask=!NrV2::Empty(slot.staged.controlMask);float local=slot.staged.controls.structure,skin=-1,automatic=-1;
            if(!hasMask&&(slot.staged.controls.skinStrength>=0||slot.staged.controls.autoStrength>=0)){local=1;skin=slot.staged.controls.skinStrength>=0?slot.staged.controls.skinStrength:slot.staged.controls.structure;automatic=slot.staged.controls.autoStrength>=0?slot.staged.controls.autoStrength:slot.staged.controls.structure;}
            float jx=0,jy=0;if(slot.staged.motionParameters.jitterMode==NrV2::JitterMode::AddPreviousMinusCurrent){jx=slot.staged.motionParameters.previousJitterX-slot.staged.motionParameters.currentJitterX;jy=slot.staged.motionParameters.previousJitterY-slot.staged.motionParameters.currentJitterY;}
            const auto start=std::chrono::steady_clock::now();
            Hip(launch_v2_prepare_features(slot.color.pointer,slot.color.footprint.Footprint.RowPitch,slot.motion.pointer,slot.motion.resource?slot.motion.footprint.Footprint.RowPitch:0,hasMask?slot.mask.pointer:nullptr,hasMask?slot.mask.footprint.Footprint.RowPitch:0,committed,prepared,config.logicalHeight,config.logicalWidth,config.networkHeight,config.networkWidth,slot.color.footprint.Footprint.Width,slot.color.footprint.Footprint.Height,hasMask?slot.mask.footprint.Footprint.Width:1,hasMask?slot.mask.footprint.Footprint.Height:1,hasMask,d.noise,d.useHistory,slot.motion.resource?slot.motion.footprint.Footprint.Width:1,slot.motion.resource?slot.motion.footprint.Footprint.Height:1,slot.staged.motionParameters.effectiveWidth,slot.staged.motionParameters.effectiveHeight,slot.staged.motionParameters.scaleX,slot.staged.motionParameters.scaleY,jx,jy,slot.staged.controls.styleIndex,slot.staged.controls.tone,local,skin,automatic,stream),"prepare features");
            graph->Enqueue(prepared,config.networkHeight,config.networkWidth,head);
#ifdef NR_RUNTIME_V3_TEST_HOOKS
            if(injectHeadNext){injectHeadNext=false;Hip(launch_v3_inject_head_nonfinite(head,std::size_t(config.networkWidth)*config.networkHeight*4*sizeof(std::uint16_t),stream),"inject head nonfinite");}
#endif
            Hip(launch_v2_compose(slot.color.pointer,slot.color.footprint.Footprint.RowPitch,hasMask?slot.mask.pointer:nullptr,hasMask?slot.mask.footprint.Footprint.RowPitch:0,head,prepared,blendScale,slot.output.pointer,slot.output.footprint.Footprint.RowPitch,candidate,config.logicalHeight,config.logicalWidth,config.networkWidth,slot.color.footprint.Footprint.Width,slot.color.footprint.Footprint.Height,hasMask?slot.mask.footprint.Footprint.Width:1,hasMask?slot.mask.footprint.Footprint.Height:1,hasMask,d.useHistory,slot.staged.controls.intensity,stream),"compose");
#ifdef NR_RUNTIME_V3_TEST_HOOKS
            if(injectOutputNext){injectOutputNext=false;Hip(launch_v2_inject_nonfinite(slot.output.pointer,stream),"inject output nonfinite");}
            if(failNext){failNext=false;throw std::runtime_error("injected partial dispatch failure");}
#endif
            Hip(hipMemsetAsync(finiteDevice,0,sizeof(*finiteDevice),stream),"clear finite");Hip(launch_v2_validate_output(slot.output.pointer,slot.output.footprint.Footprint.RowPitch,config.logicalWidth,config.logicalHeight,finiteDevice,stream),"validate output");
            Hip(launch_v2_validate_neural(prepared,head,candidate,config.networkWidth*config.networkHeight,config.logicalWidth*config.logicalHeight,finiteDevice,stream),"validate neural");
            Hip(hipMemcpyAsync(finiteHost,finiteDevice,sizeof(*finiteHost),hipMemcpyDeviceToHost,stream),"copy finite");hipExternalSemaphoreSignalParams signal{};signal.params.fence.value=t.key.submissionEpoch;
            Hip(hipSignalExternalSemaphoresAsync(&hip.semaphore,&signal,1,stream),"HIP output signal");
#ifdef NR_RUNTIME_V3_TEST_HOOKS
            if(failEvent){failEvent=false;Quarantine();return Status::Quarantined;}
#endif
            Hip(hipEventRecord(completion,stream),"record completion");
            lastDispatchUs=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count());running=t;eventRecorded=true;++dispatches;return Status::Ok;
        }catch(...){session.Fault(&t);return Status::Failed;}
    }
    const Slot* FindSlot(const Token& t)const{const auto v=session.View();for(const auto& s:v.slots)if(s.phase!=Phase::Free&&Same(s.token,t))return &s;return nullptr;}
    Status Poll(Snapshot& out){
        const auto start=std::chrono::steady_clock::now();if(quarantined){out=session.View();return out.status=Status::Quarantined;}
        auto d=d3d.fence->GetCompletedValue();if(d==ReservedSerial){Quarantine();out=session.View();return out.status=Status::Quarantined;}
        bool idle=false;const auto life=session.View().lifecycle;if(life==Lifecycle::Draining||life==Lifecycle::Faulted){const auto q=hipStreamQuery(stream);if(q==hipSuccess)idle=true;else if(q!=hipErrorNotReady){Quarantine();out=session.View();return out.status=Status::Quarantined;}}
        static_cast<void>(session.Retire(d,idle));
        if(eventRecorded){const auto e=hipEventQuery(completion);if(e==hipSuccess){const auto hv=hip.fence->GetCompletedValue();if(hv==ReservedSerial){Quarantine();out=session.View();return out.status=Status::Quarantined;}if(hv>=running.key.submissionEpoch){if(*finiteHost){session.Fault(&running);}else{auto s=session.Complete(running,NrAsync::Completion::Success,hv);if(s==Status::Ok){std::swap(committed,candidate);++swaps;}}eventRecorded=false;running={};}}else if(e!=hipErrorNotReady){Quarantine();out=session.View();return out.status=Status::Quarantined;}}
        out=session.View();lastPollUs=static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now()-start).count());return out.status;
    }
    Status RecordOutput(const Output& o,RecordResult& out){
        out.status=Status::Failed;auto s=Validate(o,config);if(s!=Status::Ok)return out.status=s;
        const int i=Index(o.token);if(i<0)return out.status=Status::StaleFrame;s=ValidateOutputResources(o,i);if(s!=Status::Ok)return out.status=s;const auto* state=FindSlot(o.token);if(!state||state->phase!=Phase::Ready)return out.status=Status::StaleFrame;
        if(hip.fence->GetCompletedValue()<o.token.key.submissionEpoch)return out.status=Status::Busy;
        s=session.Lease(o.token);if(s!=Status::Ok)return out.status=s;auto& slot=slots[static_cast<unsigned>(i)];
        if(FAILED(queue->Wait(hip.fence.Get(),o.token.key.submissionEpoch))){Quarantine();return out.status=Status::Quarantined;}
        slot.outputList=o.commands;slot.outputTarget=o.target.resource;slot.hasOutput=true;
        std::array<D3D12_RESOURCE_BARRIER,2> before{Barrier(slot.output.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE),Barrier(o.target.resource,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST)};o.commands->ResourceBarrier(2,before.data());
        D3D12_TEXTURE_COPY_LOCATION dst{};dst.pResource=o.target.resource;dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;D3D12_TEXTURE_COPY_LOCATION src{};src.pResource=slot.output.resource.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=slot.output.footprint;o.commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        std::array<D3D12_RESOURCE_BARRIER,2> after{Barrier(slot.output.resource.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON),Barrier(o.target.resource,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_UNORDERED_ACCESS)};o.commands->ResourceBarrier(2,after.data());out.recorded=1;out.token=o.token;return out.status=Status::Ok;
    }
    Status NotifyOutput(const Token& t){auto s=session.CheckOutputAck(t);if(s!=Status::Ok)return s;const auto serial=nextD3d++;if(serial==ReservedSerial)return Status::Failed;if(FAILED(queue->Signal(d3d.fence.Get(),serial))){Quarantine();return Status::Quarantined;}return session.OutputSubmitted(t,serial);}
    Status Drop(const Token& t){return session.Drop(t);}
    Status Ack(const Token& t){auto s=session.AcknowledgeTerminal(t);if(s==Status::Ok){const int i=Index(t);if(i>=0)slots[static_cast<unsigned>(i)].ReleaseRefs();}return s;}
    Status BeginDrain(){return session.BeginDrain();}
    Status Shutdown(){
        if(quarantined)return Status::Quarantined;if(closed)return Status::Ok;static_cast<void>(session.BeginDrain());Snapshot v{};Poll(v);auto s=session.Close();if(s!=Status::Ok)return s;closed=true;return Status::Ok;
    }
    Status Quarantine(){quarantined=true;return session.Quarantine();}
    Status Destroy(){
        if(quarantined)return Status::Quarantined;if(!closed)return Status::Busy;
#ifdef NR_RUNTIME_V3_TEST_HOOKS
        if(failCleanup){failCleanup=false;Quarantine();return Status::Quarantined;}
#endif
        for(auto& s:slots){if(!s.ReleaseInterop()){Quarantine();return Status::Quarantined;}s.ReleaseRefs();}
        if(!ReleaseNeural()){Quarantine();return Status::Quarantined;}
        if(finiteDevice){if(hipFree(finiteDevice)!=hipSuccess){Quarantine();return Status::Quarantined;}finiteDevice=nullptr;}
        if(finiteHost){if(hipHostFree(finiteHost)!=hipSuccess){Quarantine();return Status::Quarantined;}finiteHost=nullptr;}
        if(completion){if(hipEventDestroy(completion)!=hipSuccess){Quarantine();return Status::Quarantined;}completion=nullptr;}
        if(!hip.Release()||!d3d.Release()){Quarantine();return Status::Quarantined;}
        if(stream){if(hipStreamDestroy(stream)!=hipSuccess){Quarantine();return Status::Quarantined;}stream=nullptr;}return Status::Ok;
    }
#ifdef NR_SCHEDULED_MIXED
    Status ScheduledRecordInput(const Input& f,RecordResult& out){
        out.status=ScheduledLive();if(out.status!=Status::Ok)return out.status;
        auto s=Validate(f,config,true);if(s!=Status::Ok)return out.status=s;
        s=ValidateInputResources(f);if(s!=Status::Ok)return out.status=s;
        if(f.generation!=scheduledGeneration||f.key.submissionEpoch<=scheduledLastEpoch)return out.status=Status::StaleFrame;
        const bool mv=!NrV2::Empty(f.motion),mask=!NrV2::Empty(f.controlMask);
        // Slots reserve full color capacity off the submission thread. Smaller
        // motion/mask rectangles fit without allocation; kernels must sample
        // the recorded rectangle, never the unused portion of that capacity.
        if((mv&&(f.motion.transform.extentWidth>config.logicalWidth||f.motion.transform.extentHeight>config.logicalHeight))||
            (mask&&(f.controlMask.transform.extentWidth>config.logicalWidth||f.controlMask.transform.extentHeight>config.logicalHeight)))
            return out.status=Status::Unsupported;
        const int index=ScheduledFreeIndex();if(index<0)return out.status=Status::Busy;
        auto& slot=slots[static_cast<unsigned>(index)];const Token token{f.key,f.generation};
        slot.staged=f;slot.token=token;slot.hasInput=true;slot.inputList=f.commands;slot.inputColor=f.color.resource;
        if(mv)slot.inputMotion=f.motion.resource;if(mask)slot.inputMask=f.controlMask.resource;
        slot.scheduledPhase=Phase::Recorded;scheduledLastEpoch=f.key.submissionEpoch;
        try{
            if(scheduledTelemetry)f.commands->EndQuery(transferQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,
                static_cast<unsigned>(index)*4);
            std::array<D3D12_RESOURCE_BARRIER,8> before{},after{};UINT n=0;
            before[n++]=Barrier(f.color.resource,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
            before[n++]=Barrier(slot.color.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);
            if(mv){before[n++]=Barrier(f.motion.resource,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);before[n++]=Barrier(slot.motion.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);}
            if(mask){before[n++]=Barrier(f.controlMask.resource,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);before[n++]=Barrier(slot.mask.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST);}
            f.commands->ResourceBarrier(n,before.data());
            CopyIn(f.commands,slot.color,f.color);if(mv)CopyIn(f.commands,slot.motion,f.motion);if(mask)CopyIn(f.commands,slot.mask,f.controlMask);
            n=0;after[n++]=Barrier(f.color.resource,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            after[n++]=Barrier(slot.color.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);
            if(mv){after[n++]=Barrier(f.motion.resource,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);after[n++]=Barrier(slot.motion.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);}
            if(mask){after[n++]=Barrier(f.controlMask.resource,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);after[n++]=Barrier(slot.mask.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COMMON);}
            f.commands->ResourceBarrier(n,after.data());
            if(scheduledTelemetry){
                const unsigned query=static_cast<unsigned>(index)*4;
                f.commands->EndQuery(transferQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,query+1);
                f.commands->ResolveQueryData(transferQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,query,2,
                    transferReadback.Get(),query*sizeof(std::uint64_t));
            }
            out.recorded=1;out.token=token;return out.status=Status::Ok;
        }catch(...){out.recorded=1;out.token=token;scheduledFaulted=true;return out.status=Status::Failed;}
    }
    Status ScheduledInputSubmitted(const Token& token){
        const auto live=ScheduledLive();if(live!=Status::Ok)return live;const int i=Index(token);
        if(i<0||slots[static_cast<unsigned>(i)].scheduledPhase!=Phase::Recorded)return Status::StaleFrame;
        if(nextD3d==ReservedSerial)return Status::Failed;
        const auto value=nextD3d++;if(FAILED(queue->Signal(d3d.fence.Get(),value))){quarantined=true;return Status::Quarantined;}
        auto& slot=slots[static_cast<unsigned>(i)];slot.scheduledInputSerial=value;slot.scheduledPhase=Phase::Queued;return Status::Ok;
    }
    Status ScheduledEnqueue(const Token& token){
        const auto live=ScheduledLive();if(live!=Status::Ok)return live;const int i=Index(token);
        if(i<0||slots[static_cast<unsigned>(i)].scheduledPhase!=Phase::Queued)return Status::StaleFrame;
        auto& slot=slots[static_cast<unsigned>(i)];
        try{
            if(scheduledHipTelemetry)Hip(hipEventRecord(slot.telemetry[0],stream),"telemetry input start");
            hipExternalSemaphoreWaitParams wait{};wait.params.fence.value=slot.scheduledInputSerial;
            slot.scheduledHipTouched=true;
            Hip(hipWaitExternalSemaphoresAsync(&d3d.semaphore,&wait,1,stream),"scheduled HIP input wait");
            if(scheduledHipTelemetry)Hip(hipEventRecord(slot.telemetry[1],stream),"telemetry input ready");
            const bool hasMask=!NrV2::Empty(slot.staged.controlMask);
            const bool hasMotion=!NrV2::Empty(slot.staged.motion);
            const NrScheduled::Frame frame{token.key.streamId,token.key.frameIndex,
                slot.staged.resetReasons?1u:0u,hasMotion?1u:0u,
                (slot.staged.reserved&DisableTemporalAccumulation)?1u:0u};
            auto* decision=gpuDecisions+i;
            Hip(launch_scheduled_begin(gpuState,frame,decision,stream),"scheduled frame begin");
            float local=slot.staged.controls.structure,skin=-1,automatic=-1;
            if(!hasMask&&(slot.staged.controls.skinStrength>=0||slot.staged.controls.autoStrength>=0)){
                local=1;skin=slot.staged.controls.skinStrength>=0?slot.staged.controls.skinStrength:slot.staged.controls.structure;
                automatic=slot.staged.controls.autoStrength>=0?slot.staged.controls.autoStrength:slot.staged.controls.structure;}
            float jx=0,jy=0;if(slot.staged.motionParameters.jitterMode==NrV2::JitterMode::AddPreviousMinusCurrent){
                jx=slot.staged.motionParameters.previousJitterX-slot.staged.motionParameters.currentJitterX;
                jy=slot.staged.motionParameters.previousJitterY-slot.staged.motionParameters.currentJitterY;}
            Hip(launch_scheduled_prepare_features(slot.color.pointer,slot.color.footprint.Footprint.RowPitch,
                hasMotion?slot.motion.pointer:nullptr,hasMotion?slot.motion.footprint.Footprint.RowPitch:0,
                hasMask?slot.mask.pointer:nullptr,hasMask?slot.mask.footprint.Footprint.RowPitch:0,
                committed,prepared,config.logicalHeight,config.logicalWidth,config.networkHeight,config.networkWidth,
                slot.color.footprint.Footprint.Width,slot.color.footprint.Footprint.Height,
                hasMask?slot.staged.controlMask.transform.extentWidth:1,hasMask?slot.staged.controlMask.transform.extentHeight:1,
                hasMask,decision,hasMotion?slot.staged.motion.transform.extentWidth:1,
                hasMotion?slot.staged.motion.transform.extentHeight:1,
                slot.staged.motionParameters.effectiveWidth,slot.staged.motionParameters.effectiveHeight,
                slot.staged.motionParameters.scaleX,slot.staged.motionParameters.scaleY,jx,jy,
                slot.staged.controls.styleIndex,slot.staged.controls.tone,local,skin,automatic,stream),"scheduled prepare");
            if(scheduledHipTelemetry)Hip(hipEventRecord(slot.telemetry[2],stream),"telemetry graph start");
            if(networkExecutable)Hip(hipGraphLaunch(networkExecutable,stream),"network replay");
            else graph->Enqueue(prepared,config.networkHeight,config.networkWidth,head);
            if(scheduledHipTelemetry)Hip(hipEventRecord(slot.telemetry[3],stream),"telemetry graph end");
            Hip(launch_scheduled_compose(slot.color.pointer,slot.color.footprint.Footprint.RowPitch,
                hasMask?slot.mask.pointer:nullptr,hasMask?slot.mask.footprint.Footprint.RowPitch:0,
                head,prepared,blendScale,slot.output.pointer,slot.output.footprint.Footprint.RowPitch,candidate,
                config.logicalHeight,config.logicalWidth,config.networkWidth,
                slot.color.footprint.Footprint.Width,slot.color.footprint.Footprint.Height,
                hasMask?slot.staged.controlMask.transform.extentWidth:1,hasMask?slot.staged.controlMask.transform.extentHeight:1,
                hasMask,decision,slot.staged.controls.intensity,stream),"scheduled compose");
            Hip(hipMemsetAsync(finiteDevice,0,sizeof(*finiteDevice),stream),"clear scheduled finite");
            Hip(launch_v2_validate_output(slot.output.pointer,slot.output.footprint.Footprint.RowPitch,
                config.logicalWidth,config.logicalHeight,finiteDevice,stream),"scheduled validate output");
            Hip(launch_v2_validate_neural(prepared,head,candidate,
                config.networkWidth*config.networkHeight,config.logicalWidth*config.logicalHeight,
                finiteDevice,stream),"scheduled validate neural");
            Hip(launch_scheduled_resolve(slot.color.pointer,slot.color.footprint.Footprint.RowPitch,
                slot.color.footprint.Footprint.Width,slot.color.footprint.Footprint.Height,
                slot.output.pointer,slot.output.footprint.Footprint.RowPitch,candidate,committed,
                config.logicalWidth,config.logicalHeight,finiteDevice,gpuState,frame,stream),"scheduled resolve");
            if(scheduledHipTelemetry)Hip(hipEventRecord(slot.telemetry[4],stream),"telemetry HIP end");
            hipExternalSemaphoreSignalParams done{};done.params.fence.value=token.key.submissionEpoch;
            Hip(hipSignalExternalSemaphoresAsync(&hip.semaphore,&done,1,stream),"scheduled output signal");
            slot.scheduledPhase=Phase::Running;++dispatches;return Status::Ok;
        }catch(const std::exception& e){std::fprintf(stderr,"scheduled enqueue: %s\n",e.what());
            quarantined=true;return Status::Quarantined;}
    }
    Status ScheduledRecordOutput(const Output& o,RecordResult& out,bool deferred=false){
        out.status=ScheduledLive();if(out.status!=Status::Ok)return out.status;
        auto s=Validate(o,config);if(s!=Status::Ok)return out.status=s;
        const int i=Index(o.token);if(i<0)return out.status=Status::StaleFrame;
        auto& slot=slots[static_cast<unsigned>(i)];
        if(slot.scheduledPhase!=(deferred?Phase::Recorded:Phase::Running)||slot.hasOutput)
            return out.status=Status::StaleFrame;
        s=ValidateOutputResources(o,i,deferred);if(s!=Status::Ok)return out.status=s;
        slot.outputList=o.commands;slot.outputTarget=o.target.resource;slot.hasOutput=true;
        if(scheduledTelemetry)o.commands->EndQuery(transferQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,
            static_cast<unsigned>(i)*4+2);
        const auto targetState=o.target.resource==slot.inputColor.Get()?
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        std::array<D3D12_RESOURCE_BARRIER,2> before{
            Barrier(slot.output.resource.Get(),D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_SOURCE),
            Barrier(o.target.resource,targetState,D3D12_RESOURCE_STATE_COPY_DEST)};
        o.commands->ResourceBarrier(2,before.data());
        D3D12_TEXTURE_COPY_LOCATION dst{};dst.pResource=o.target.resource;dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION src{};src.pResource=slot.output.resource.Get();src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=slot.output.footprint;
        o.commands->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        std::array<D3D12_RESOURCE_BARRIER,2> after{
            Barrier(slot.output.resource.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_COMMON),
            Barrier(o.target.resource,D3D12_RESOURCE_STATE_COPY_DEST,targetState)};
        o.commands->ResourceBarrier(2,after.data());
        if(scheduledTelemetry){
            const unsigned query=static_cast<unsigned>(i)*4+2;
            o.commands->EndQuery(transferQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,query+1);
            o.commands->ResolveQueryData(transferQueries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,query,2,
                transferReadback.Get(),query*sizeof(std::uint64_t));
        }
        out.recorded=1;out.token=o.token;
        if(!deferred){
            // Legacy owned-list path: producer Signal was already accepted.
            if(FAILED(queue->Wait(hip.fence.Get(),o.token.key.submissionEpoch))){quarantined=true;return out.status=Status::Quarantined;}
            slot.scheduledPhase=Phase::CopyRecorded;
        }
        return out.status=Status::Ok;
    }
    Status ScheduledArmOutput(const Token& token){
        const auto live=ScheduledLive();if(live!=Status::Ok)return live;
        const int i=Index(token);if(i<0)return Status::StaleFrame;
        auto& slot=slots[static_cast<unsigned>(i)];
        if(slot.scheduledPhase!=Phase::Running||!slot.hasOutput)return Status::StaleFrame;
        if(FAILED(queue->Wait(hip.fence.Get(),token.key.submissionEpoch))){quarantined=true;return Status::Quarantined;}
        slot.scheduledPhase=Phase::CopyRecorded;return Status::Ok;
    }
    Status ScheduledOutputSubmitted(const Token& token){
        const auto live=ScheduledLive();if(live!=Status::Ok)return live;const int i=Index(token);
        if(i<0||slots[static_cast<unsigned>(i)].scheduledPhase!=Phase::CopyRecorded)return Status::StaleFrame;
        if(nextD3d==ReservedSerial)return Status::Failed;
        const auto value=nextD3d++;
        if(FAILED(queue->Signal(d3d.fence.Get(),value))){quarantined=true;return Status::Quarantined;}
        auto& slot=slots[static_cast<unsigned>(i)];slot.scheduledRetireSerial=value;slot.scheduledPhase=Phase::Retiring;
        return Status::Ok;
    }
    Status ScheduledDrop(const Token& token){
        const int i=Index(token);if(i<0)return Status::StaleFrame;
        auto& slot=slots[static_cast<unsigned>(i)];
        if(slot.scheduledPhase==Phase::Recorded){slot.ReleaseRefs();return Status::Ok;}
        if(slot.scheduledPhase==Phase::Queued||slot.scheduledPhase==Phase::Running||
            slot.scheduledPhase==Phase::CopyRecorded){slot.scheduledPhase=Phase::Cancelling;return Status::Ok;}
        return Status::StaleFrame;
    }
    Status ScheduledPoll(Snapshot& out){
        out={};out.prefix={sizeof(out),Version};
        if(quarantined)return out.status=Status::Quarantined;
        const auto d=d3d.fence->GetCompletedValue(),h=hip.fence->GetCompletedValue();
        if(d==ReservedSerial||h==ReservedSerial){quarantined=true;return out.status=Status::Quarantined;}
        for(unsigned slotIndex=0;slotIndex<Capacity;++slotIndex){
            auto& slot=slots[slotIndex];
            if(slot.scheduledPhase==Phase::Retiring&&d>=slot.scheduledRetireSerial)slot.scheduledPhase=Phase::Terminal;
            else if(slot.scheduledPhase==Phase::Cancelling&&
                (slot.scheduledHipTouched?h>=slot.token.key.submissionEpoch:d>=slot.scheduledInputSerial))
                slot.scheduledPhase=Phase::Terminal;
            if(scheduledHipTelemetry&&slot.scheduledPhase==Phase::Terminal&&!slot.telemetryReported&&
                slot.scheduledHipTouched&&slot.scheduledRetireSerial&&slot.hasOutput){
                // The imported D3D12 fence may become visible before HIP's
                // timing event query. Synchronize only in this explicit
                // diagnostic mode; production never pays this per-frame cost.
                const auto timingReady=hipEventSynchronize(slot.telemetry[4]);
                if(timingReady!=hipSuccess){
                    std::fprintf(stderr,"ScheduledGpuUnqualified frame=%llu event_sync=%d\n",
                        static_cast<unsigned long long>(slot.token.key.frameIndex),static_cast<int>(timingReady));
                    continue;
                }
                float ms[4]{};bool valid=true;
                for(unsigned j=0;j<4;++j)if(hipEventElapsedTime(&ms[j],slot.telemetry[j],slot.telemetry[j+1])!=hipSuccess)
                    valid=false;
                if(valid)std::fprintf(stderr,"ScheduledGpuUnqualified frame=%llu wait_ms=%.3f prepare_ms=%.3f graph_ms=%.3f post_ms=%.3f total_ms=%.3f\n",
                    static_cast<unsigned long long>(slot.token.key.frameIndex),ms[0],ms[1],ms[2],ms[3],
                    ms[0]+ms[1]+ms[2]+ms[3]);
                slot.telemetryReported=true;
            }
            if(scheduledTelemetry&&slot.scheduledPhase==Phase::Terminal&&!slot.transferReported&&slot.hasOutput&&
                slot.scheduledRetireSerial&&transferReadback){
                const unsigned index=slotIndex*4;
                const D3D12_RANGE range{index*sizeof(std::uint64_t),(index+4)*sizeof(std::uint64_t)};
                void* mapped=nullptr;
                if(SUCCEEDED(transferReadback->Map(0,&range,&mapped))&&mapped){
                    const auto* ticks=static_cast<const std::uint64_t*>(mapped)+index;
                    if(ticks[1]>=ticks[0]&&ticks[3]>=ticks[2])
                        std::fprintf(stderr,"ScheduledTransfer frame=%llu ingress_gpu_ms=%.3f publish_gpu_ms=%.3f\n",
                            static_cast<unsigned long long>(slot.token.key.frameIndex),
                            1000.0*double(ticks[1]-ticks[0])/double(transferFrequency),
                            1000.0*double(ticks[3]-ticks[2])/double(transferFrequency));
                    // All four timestamps belong to the same D3D12 queue.
                    // The middle interval includes HIP and its queue wait;
                    // it is not advertised as pure kernel execution time.
                    if(ticks[0]<=ticks[1]&&ticks[1]<=ticks[2]&&ticks[2]<=ticks[3])
                        std::fprintf(stderr,"ScheduledBoundary frame=%llu ingress_ms=%.3f neural_wait_ms=%.3f publish_ms=%.3f total_ms=%.3f\n",
                            static_cast<unsigned long long>(slot.token.key.frameIndex),
                            1000.0*double(ticks[1]-ticks[0])/double(transferFrequency),
                            1000.0*double(ticks[2]-ticks[1])/double(transferFrequency),
                            1000.0*double(ticks[3]-ticks[2])/double(transferFrequency),
                            1000.0*double(ticks[3]-ticks[0])/double(transferFrequency));
                    const D3D12_RANGE empty{0,0};transferReadback->Unmap(0,&empty);
                }
                slot.transferReported=true;
            }
        }
        out.lifecycle=closed?Lifecycle::Closed:scheduledFaulted?Lifecycle::Faulted:
            scheduledDraining?Lifecycle::Draining:Lifecycle::Live;
        out.generation=scheduledGeneration;out.capacity=Capacity;out.lastAcceptedEpoch=scheduledLastEpoch;
        for(unsigned i=0;i<Capacity;++i){const auto& slot=slots[i];
            out.slots[i].token=slot.token;out.slots[i].phase=slot.scheduledPhase;
            out.slots[i].inputSerial=slot.scheduledInputSerial;out.slots[i].outputSerial=slot.scheduledRetireSerial;
            if(slot.scheduledPhase!=Phase::Free)++out.occupied;}
        return out.status=Status::Ok;
    }
    Status ScheduledAck(const Token& token){const int i=Index(token);if(i<0)return Status::StaleFrame;
        auto& slot=slots[static_cast<unsigned>(i)];if(slot.scheduledPhase!=Phase::Terminal)return Status::Busy;
        slot.ReleaseRefs();return Status::Ok;}
    Status ScheduledDrain(){if(quarantined)return Status::Quarantined;scheduledDraining=true;return Status::Ok;}
    Status ScheduledShutdown(){
        if(quarantined)return Status::Quarantined;if(closed)return Status::Ok;
        scheduledDraining=true;Snapshot snapshot{};ScheduledPoll(snapshot);
        if(ScheduledOccupied())return Status::Busy;
        const auto idle=hipStreamQuery(stream);if(idle==hipErrorNotReady)return Status::Busy;
        if(idle!=hipSuccess){quarantined=true;return Status::Quarantined;}
        closed=true;return Status::Ok;
    }
    Status ScheduledDestroy(){
        if(scheduledDestroyed)return Status::Ok;
        if(quarantined)return Status::Quarantined;if(!closed)return Status::Busy;
        for(auto& slot:slots){if(!slot.ReleaseInterop()){quarantined=true;return Status::Quarantined;}slot.ReleaseRefs();}
        if(!ReleaseNeural()){quarantined=true;return Status::Quarantined;}
        if(finiteDevice){if(hipFree(finiteDevice)!=hipSuccess){quarantined=true;return Status::Quarantined;}finiteDevice=nullptr;}
        if(finiteHost){if(hipHostFree(finiteHost)!=hipSuccess){quarantined=true;return Status::Quarantined;}finiteHost=nullptr;}
        if(completion){if(hipEventDestroy(completion)!=hipSuccess){quarantined=true;return Status::Quarantined;}completion=nullptr;}
        for(auto& slot:slots)for(auto& event:slot.telemetry)
            if(event){if(hipEventDestroy(event)!=hipSuccess){quarantined=true;return Status::Quarantined;}event=nullptr;}
        if(!hip.Release()||!d3d.Release()){quarantined=true;return Status::Quarantined;}
        if(stream){if(hipStreamDestroy(stream)!=hipSuccess){quarantined=true;return Status::Quarantined;}stream=nullptr;}
        scheduledDestroyed=true;
        return Status::Ok;
    }
#endif
#ifdef NR_RUNTIME_V3_TEST_HOOKS
    Status Arm(unsigned kind){if(quarantined)return Status::Quarantined;if(closed)return Status::Closed;if(failNext||injectHeadNext||injectOutputNext||failCleanup||failSignal||failEvent)return Status::Busy;if(kind==0)failNext=true;else if(kind==1)injectHeadNext=true;else if(kind==2)injectOutputNext=true;else if(kind==3)failCleanup=true;else if(kind==4)failSignal=true;else failEvent=true;return Status::Ok;}
    Status GetState(NrV3Test::State& s)const{s={};s.structSize=sizeof(s);s.version=NrV3Test::Version;s.snapshot=session.View();s.modelUploads=modelUploads;s.graphDispatches=dispatches;s.historySwaps=swaps;s.lastRecordMicroseconds=lastRecordUs;s.lastDispatchMicroseconds=lastDispatchUs;s.lastPollMicroseconds=lastPollUs;if(graph){const auto g=graph->Stats();s.graphAllocations=g.allocations;s.graphPreparations=g.preparations;}return Status::Ok;}
#endif
    ~Impl(){
#ifdef NR_SCHEDULED_MIXED
        if(scheduled){if(!quarantined&&closed&&!scheduledDestroyed)static_cast<void>(ScheduledDestroy());return;}
#endif
        if(!quarantined&&!closed){static_cast<void>(session.BeginDrain());}
        if(!quarantined&&closed)static_cast<void>(Destroy());
    }
};

RuntimeDx12::RuntimeDx12(ID3D12Device* d,ID3D12CommandQueue* q):impl_(new Impl(d,q)){}
RuntimeDx12::~RuntimeDx12(){delete impl_;}
Status RuntimeDx12::Configure(const Config& c,const void* p,std::uint64_t n){return impl_?impl_->Configure(c,p,n):Status::Failed;}
Status RuntimeDx12::RecordInput(const Input& f,RecordResult& r){return impl_?impl_->RecordInput(f,r):Status::Failed;}
Status RuntimeDx12::NotifyInputSubmitted(const Token& t){return impl_?impl_->NotifyInput(t):Status::Failed;}
Status RuntimeDx12::Poll(Snapshot& s){return impl_?impl_->Poll(s):Status::Failed;}
Status RuntimeDx12::DispatchNext(){return impl_?impl_->Dispatch():Status::Failed;}
Status RuntimeDx12::RecordOutput(const Output& o,RecordResult& r){return impl_?impl_->RecordOutput(o,r):Status::Failed;}
Status RuntimeDx12::NotifyOutputSubmitted(const Token& t){return impl_?impl_->NotifyOutput(t):Status::Failed;}
Status RuntimeDx12::DropOutput(const Token& t){return impl_?impl_->Drop(t):Status::Failed;}
Status RuntimeDx12::AcknowledgeTerminal(const Token& t){return impl_?impl_->Ack(t):Status::Failed;}
Status RuntimeDx12::BeginDrain(){return impl_?impl_->BeginDrain():Status::Failed;}
Status RuntimeDx12::Shutdown(){return impl_?impl_->Shutdown():Status::Failed;}
Status RuntimeDx12::Quarantine(){return impl_?impl_->Quarantine():Status::Failed;}
Status RuntimeDx12::Destroy(){return impl_?impl_->Destroy():Status::Failed;}
#ifdef NR_SCHEDULED_MIXED
ScheduledRuntimeDx12::ScheduledRuntimeDx12(ID3D12Device* d,ID3D12CommandQueue* q):impl_(new RuntimeDx12::Impl(d,q,true)){}
ScheduledRuntimeDx12::~ScheduledRuntimeDx12(){
    if(!impl_)return;
    try {ScopedHipDevice device(impl_->hipDevice);delete impl_;}
    catch(...){impl_->quarantined=true;delete impl_;}
}
template<class T,class F>Status OnScheduledHipDevice(T* impl,F&& call) noexcept {
    if(!impl)return Status::Failed;
    try {ScopedHipDevice device(impl->hipDevice);return call();}
    catch(...){impl->quarantined=true;return Status::Quarantined;}
}
Status ScheduledRuntimeDx12::Configure(const Config& c,const void* p,std::uint64_t n){
    return OnScheduledHipDevice(impl_,[&]{return impl_->Configure(c,p,n);});}
Status ScheduledRuntimeDx12::RecordInput(const Input& f,RecordResult& r){return impl_?impl_->ScheduledRecordInput(f,r):Status::Failed;}
Status ScheduledRuntimeDx12::NotifyInputSubmitted(const Token& t){return impl_?impl_->ScheduledInputSubmitted(t):Status::Failed;}
Status ScheduledRuntimeDx12::Poll(Snapshot& s){
    const auto result=OnScheduledHipDevice(impl_,[&]{return impl_->ScheduledPoll(s);});
    if(result==Status::Quarantined)s.status=result;
    return result;}
Status ScheduledRuntimeDx12::Enqueue(const Token& t){
    return OnScheduledHipDevice(impl_,[&]{return impl_->ScheduledEnqueue(t);});}
Status ScheduledRuntimeDx12::RecordOutput(const Output& o,RecordResult& r){return impl_?impl_->ScheduledRecordOutput(o,r):Status::Failed;}
Status ScheduledRuntimeDx12::RecordOutputDeferred(const Output& o,RecordResult& r){return impl_?impl_->ScheduledRecordOutput(o,r,true):Status::Failed;}
Status ScheduledRuntimeDx12::ArmOutput(const Token& t){return impl_?impl_->ScheduledArmOutput(t):Status::Failed;}
Status ScheduledRuntimeDx12::NotifyOutputSubmitted(const Token& t){return impl_?impl_->ScheduledOutputSubmitted(t):Status::Failed;}
Status ScheduledRuntimeDx12::Drop(const Token& t){return impl_?impl_->ScheduledDrop(t):Status::Failed;}
Status ScheduledRuntimeDx12::AcknowledgeTerminal(const Token& t){return impl_?impl_->ScheduledAck(t):Status::Failed;}
Status ScheduledRuntimeDx12::BeginDrain(){return impl_?impl_->ScheduledDrain():Status::Failed;}
Status ScheduledRuntimeDx12::Shutdown(){
    return OnScheduledHipDevice(impl_,[&]{return impl_->ScheduledShutdown();});}
Status ScheduledRuntimeDx12::Quarantine(){if(!impl_)return Status::Failed;impl_->quarantined=true;return Status::Quarantined;}
Status ScheduledRuntimeDx12::Destroy(){
    return OnScheduledHipDevice(impl_,[&]{return impl_->ScheduledDestroy();});}
#endif
#ifdef NR_RUNTIME_V3_TEST_HOOKS
Status RuntimeDx12::ArmFailureForTest(){return impl_?impl_->Arm(0):Status::Failed;}
Status RuntimeDx12::ArmHeadNonfiniteForTest(){return impl_?impl_->Arm(1):Status::Failed;}
Status RuntimeDx12::ArmOutputNonfiniteForTest(){return impl_?impl_->Arm(2):Status::Failed;}
Status RuntimeDx12::ArmCleanupFailureForTest(){return impl_?impl_->Arm(3):Status::Failed;}
Status RuntimeDx12::ArmSignalFailureForTest(){return impl_?impl_->Arm(4):Status::Failed;}
Status RuntimeDx12::ArmEventFailureForTest(){return impl_?impl_->Arm(5):Status::Failed;}
Status RuntimeDx12::GetStateForTest(NrV3Test::State& s)const{return impl_?impl_->GetState(s):Status::Failed;}
#endif
} // namespace NrV3
