// Source-built native D3D12 NGX host. It exercises the actual OptiScaler DLL,
// not the standalone FSR2 fixture or the DX11-on-12 bridge.
#define NR_RUNTIME_DLL_PROBE_HELPERS_ONLY
#include "runtime_v2_dll_probe.cpp"
#include <nvsdk_ngx_params.h>
#include <dbghelp.h>
#include <chrono>
#include <filesystem>
#include <cmath>
#include "../include/nr_submission_runtime_contract.h"
#include "../include/nr_execution_profile.h"
#include "../include/nr_extent.h"
#include "../tests/submission/forwarding_list.h"
#include "../tests/common_color_reference.h"

namespace {
ID3D12Device* firstDeviceIdentity=nullptr;
bool referenceHdr=false,referenceApply=true;
float referenceTransfer=1;
ID3D12Resource* referenceExposure=nullptr;
float DecodeHalf(std::uint16_t bits){
    const unsigned exponent=(bits>>10)&31, fraction=bits&1023;
    const float value=std::ldexp(float(fraction+(exponent?1024:0)),exponent?int(exponent)-25:-24);
    return bits&0x8000?-value:value;
}
void UploadColor(ID3D12Device* device,ID3D12CommandQueue* queue,ID3D12Resource* target,const std::vector<Rgba16>& input){
    const auto format=target->GetDesc().Format;
    if(format==DXGI_FORMAT_R8G8B8A8_UNORM){
        std::vector<std::array<unsigned char,4>> data;data.reserve(input.size());
        for(const auto& p:input){std::array<unsigned char,4> v{};unsigned i=0;
            for(auto bits:{p.r,p.g,p.b,p.a})v[i++]=static_cast<unsigned char>(std::fmin(1.f,std::fmax(0.f,DecodeHalf(bits)))*255.f+.5f);
            data.push_back(v);}UploadTexture(device,queue,target,data);
    }else if(format==DXGI_FORMAT_R32G32B32A32_FLOAT){
        std::vector<std::array<float,4>> data;data.reserve(input.size());
        for(const auto& p:input)data.push_back({DecodeHalf(p.r),DecodeHalf(p.g),DecodeHalf(p.b),DecodeHalf(p.a)});
        UploadTexture(device,queue,target,data);
    }else if(format==DXGI_FORMAT_R11G11B10_FLOAT){
        std::vector<std::uint32_t> data;data.reserve(input.size());
        for(const auto& p:input){auto pack=[](std::uint16_t h,unsigned shift){
                if(h&0x8000)return 0u;return std::min(0x7bffu,unsigned(h)+(1u<<(shift-1)))>>shift;};
            data.push_back(pack(p.r,4)|(pack(p.g,4)<<11)|(pack(p.b,5)<<22));}
        UploadTexture(device,queue,target,data);
    }else UploadTexture(device,queue,target,input);
}
NrV2::Controls FrameControls(unsigned frame){
    const NrV2::Controls presets[]={{2,1,1,-1,0,1},{0,.25f,.5f,0,1,.75f},{1,1.5f,1.5f,.5f,0,.5f}};
    return presets[frame%3];
}
using InitFn=NVSDK_NGX_Result (*)(unsigned long long,const wchar_t*,ID3D12Device*,
                                   NVSDK_NGX_Version,const NVSDK_NGX_FeatureCommonInfo*);
using ParamsFn=NVSDK_NGX_Result (*)(NVSDK_NGX_Parameter**);
using DestroyFn=NVSDK_NGX_Result (*)(NVSDK_NGX_Parameter*);
using CreateFn=NVSDK_NGX_Result (*)(ID3D12GraphicsCommandList*,NVSDK_NGX_Feature,
                                     NVSDK_NGX_Parameter*,NVSDK_NGX_Handle**);
using EvaluateFn=NVSDK_NGX_Result (*)(ID3D12GraphicsCommandList*,const NVSDK_NGX_Handle*,
                                       NVSDK_NGX_Parameter*,void*);
using ReleaseFn=NVSDK_NGX_Result (*)(NVSDK_NGX_Handle*);
using ShutdownFn=NVSDK_NGX_Result (*)();
template<class T>T Entry(HMODULE module,const char* name){
    auto* address=GetProcAddress(module,name);Require(address!=nullptr,name);
    return reinterpret_cast<T>(address);
}
LONG WINAPI CrashDump(EXCEPTION_POINTERS* exception) {
    wchar_t path[32768]{};
    const DWORD length=GetEnvironmentVariableW(L"NR_NATIVE_CRASH_DUMP",path,
                                                DWORD(std::size(path)));
    if(length>0&&length<std::size(path)) {
        HANDLE file=CreateFileW(path,GENERIC_WRITE,FILE_SHARE_READ,nullptr,CREATE_ALWAYS,
                                FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file!=INVALID_HANDLE_VALUE) {
            MINIDUMP_EXCEPTION_INFORMATION info{GetCurrentThreadId(),exception,FALSE};
            MiniDumpWriteDump(GetCurrentProcess(),GetCurrentProcessId(),file,
                              MiniDumpWithThreadInfo,&info,nullptr,nullptr);
            CloseHandle(file);
        }
    }
    return EXCEPTION_EXECUTE_HANDLER;
}
void DebugClean(ID3D12Device* device){
    ComPtr<ID3D12InfoQueue> info;if(FAILED(device->QueryInterface(IID_PPV_ARGS(&info))))return;
    unsigned errors=0;
    for(UINT64 i=0;i<info->GetNumStoredMessagesAllowedByRetrievalFilter();++i){
        SIZE_T size=0;info->GetMessage(i,nullptr,&size);std::vector<unsigned char> data(size);
        auto* message=reinterpret_cast<D3D12_MESSAGE*>(data.data());info->GetMessage(i,message,&size);
        if(message->Severity<=D3D12_MESSAGE_SEVERITY_WARNING){
            std::fprintf(stderr,"D3D12 diagnostic severity=%u id=%u: %s\n",unsigned(message->Severity),unsigned(message->ID),message->pDescription);++errors;}
    }
    Require(errors==0,"D3D12 debug clean");info->ClearStoredMessages();
}
struct Reference {
    HMODULE module=nullptr;NrSubmissionApi::Api api{};void* runtime=nullptr;
    ID3D12Device* device;ID3D12CommandQueue* queue;unsigned width,height;UINT64 epoch=0;
    bool previousTemporal=true;
    struct Flight{Commands ingress,egress;CommonColorReference codec;NrV3::Token token{};bool live=false;};
    std::array<Flight,3> flights;
    Reference(ID3D12Device* d,ID3D12CommandQueue* q,const std::filesystem::path& directory,unsigned w,unsigned h):device(d),queue(q),width(w),height(h){
        module=LoadLibraryExW((directory/L"dlssnr_hip_scheduled_bridge.dll").c_str(),nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
        Require(module,"reference DLL");auto get=Entry<NrSubmissionApi::GetApi>(module,"DlssNrHipBackendGetScheduledApiV2");
        Require(get(NrExecution::AcceptedMixedId,2,&api,sizeof(api))==NrV3::Status::Ok,"reference API");
        Require(api.create(d,q,&runtime)==NrV3::Status::Ok,"reference create");
        std::ifstream in(directory/L"dlssnr_gfx1030_v1.nrwgt",std::ios::binary|std::ios::ate);Require(bool(in),"reference weights");
        std::vector<unsigned char> bytes(static_cast<size_t>(in.tellg()));in.seekg(0);in.read(reinterpret_cast<char*>(bytes.data()),bytes.size());Require(bool(in),"reference weight read");
        NrV3::Config config{};config.prefix={sizeof(config),NrV3::Version};
        config.sourceHash=ParseHash("E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E");
        config.packageHash=ParseHash("A7E6EE38172A81E12D613FA9A2F57E32AA1944908E56CD2A33E1F6C94369E3CB");
        config.graphVersion=1;config.mathContract=NrV2::MathContract;config.logicalWidth=w;config.logicalHeight=h;
        config.networkWidth=NrV2::AlignNetworkExtent(w);config.networkHeight=NrV2::AlignNetworkExtent(h);
        Require(api.configure(runtime,&config,sizeof(config),bytes.data(),bytes.size())==NrV3::Status::Ok,"reference configure");
    }
    NrV3::Snapshot Poll(){
        NrV3::Snapshot snapshot{};Require(api.poll(runtime,&snapshot,sizeof(snapshot))==NrV3::Status::Ok,"reference poll");
        for(auto& f:flights)if(f.live)for(const auto& s:snapshot.slots)
            if(s.token.key.submissionEpoch==f.token.key.submissionEpoch&&s.phase==NrV3::Phase::Terminal){
                NrV3::TokenRequest request{{sizeof(request),NrV3::Version},f.token};
                Require(api.acknowledgeTerminal(runtime,&request,sizeof(request))==NrV3::Status::Ok,"reference retire");f.live=false;}
        return snapshot;
    }
    void Apply(ID3D12Resource* color,ID3D12Resource* motion,unsigned mw,unsigned mh,bool reset,bool post,
               NrV2::Controls controls={2,1,1,-1,0,1},bool temporal=true){
        const auto snapshot=Poll();auto& f=flights[epoch%3];Require(!f.live,"reference slot");
        f.ingress=OpenCommands(device);f.egress=OpenCommands(device);
        f.codec.Open(device,width,height,color->GetDesc().Format);
        f.codec.exposure=referenceExposure;
        f.codec.Encode(f.ingress.list.Get(),color,post,referenceHdr);
        NrV3::Input input{};input.prefix={sizeof(input),NrV3::Version};input.key={0x1030,epoch+2,++epoch};
        input.commands=f.ingress.list.Get();input.queue=queue;input.generation=snapshot.generation;
        input.color={f.codec.proxy.Get(),{0,0,width,height,width,height},NrV2::Format::Rgba16Float,0};
        input.motion={motion,{0,0,mw,mh,mw,mh},NrV2::Format::Rg16Float,0};input.controls=controls;
        char temporalFlag[4]{};
        if((GetEnvironmentVariableA("DLSSNR_TEST_TEMPORAL_OFF",temporalFlag,sizeof(temporalFlag))==1&&temporalFlag[0]=='1')||!temporal)
            input.reserved=NrV3::DisableTemporalAccumulation;
        input.motionParameters={1,1,mw,mh,0,0,0,0,NrV2::JitterMode::AddPreviousMinusCurrent,0};
        input.resetReasons=(reset||temporal!=previousTemporal)?NrV2::Explicit:0;
        NrV3::RecordResult receipt{};Require(api.recordInput(runtime,&input,sizeof(input),&receipt,sizeof(receipt))==NrV3::Status::Ok&&receipt.recorded,"reference record input");
        f.token=receipt.token;f.live=true;
        NrV3::Output output{};output.prefix={sizeof(output),NrV3::Version};output.token=f.token;
        output.commands=f.egress.list.Get();output.queue=queue;
        output.target={f.codec.answer.Get(),{0,0,width,height,width,height},NrV2::Format::Rgba16Float,0};
        Require(api.recordOutputDeferred(runtime,&output,sizeof(output),&receipt,sizeof(receipt))==NrV3::Status::Ok&&receipt.recorded,"reference record output");
        f.codec.Resolve(f.egress.list.Get(),color,post,referenceHdr,referenceApply,referenceTransfer);
        Execute(queue,f.ingress.list.Get());NrV3::TokenRequest request{{sizeof(request),NrV3::Version},f.token};
        Require(api.notifyInputSubmitted(runtime,&request,sizeof(request))==NrV3::Status::Ok,"reference input submitted");
        Require(api.enqueue(runtime,&request,sizeof(request))==NrV3::Status::Ok,"reference enqueue");
        Require(api.armOutput(runtime,&request,sizeof(request))==NrV3::Status::Ok,"reference arm");
        Execute(queue,f.egress.list.Get());Require(api.notifyOutputSubmitted(runtime,&request,sizeof(request))==NrV3::Status::Ok,"reference output submitted");
        previousTemporal=temporal;
    }
    void Close(){Require(api.beginDrain(runtime)==NrV3::Status::Ok,"reference begin drain");
        const auto deadline=GetTickCount64()+10000;
        for(;;){Poll();const auto status=api.shutdown(runtime);if(status==NrV3::Status::Ok)break;
            Require(status==NrV3::Status::Busy&&GetTickCount64()<deadline,"reference shutdown");Sleep(1);}
        Require(api.destroy(&runtime)==NrV3::Status::Ok,"reference destroy");
        FreeLibrary(module);module=nullptr;}
};
Commands OpenCommands1(ID3D12Device* device){
    Commands result;
    Hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                      IID_PPV_ARGS(&result.allocator)),"Create1 allocator");
    ComPtr<ID3D12Device4> device4;
    Hr(device->QueryInterface(IID_PPV_ARGS(&device4)),"ID3D12Device4 for CreateCommandList1");
    ComPtr<ID3D12GraphicsCommandList1> extended;
    Hr(device4->CreateCommandList1(0,D3D12_COMMAND_LIST_TYPE_DIRECT,
        D3D12_COMMAND_LIST_FLAG_NONE,IID_PPV_ARGS(&extended)),"CreateCommandList1");
    Hr(extended.As(&result.list),"Create1 base alias");
    Hr(extended->Reset(result.allocator.Get(),nullptr),"Create1 Reset");
    return result;
}
}

int wmain(int argc,wchar_t** argv) try {
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);
    SetUnhandledExceptionFilter(CrashDump);
    Require(argc>=5&&argc<=8,"usage: optiscaler_dx12_evaluate_host OptiScaler.dll color.rgba.f16 width height [mode] [standard|write_immediate|batch|reuse|early_reset|scope|stream|paced_stream|recreate|queue_switch|resize|create1_alias|hdr|subrect_bypass|bundle_bypass|unknown_motion_state|device_reinit|controls|toggle|wrapper|wrapper_bypass|motion32_bypass] [frames]");
    const unsigned mode=argc>=6?std::stoul(argv[5]):0;
    const std::wstring scenario=argc>=7?argv[6]:L"standard";
    referenceHdr=scenario==L"hdr"||scenario==L"exposure";referenceApply=scenario!=L"apply_off";
    referenceTransfer=scenario==L"transfer_zero"?0.0f:1.0f;
    const unsigned frames=argc==8?std::stoul(argv[7]):4;
    // A same-size output lets the qualification host exercise 4K NR without
    // allocating an unrelated 8K upscaler target.
    wchar_t nativeOutput[4]{};
    const unsigned outputScale=GetEnvironmentVariableW(L"NR_HOST_NATIVE_OUTPUT",nativeOutput,4)==1&&
                               nativeOutput[0]==L'1'?1u:2u;
    Require(frames>=3&&frames<=64,"bounded diagnostic frame count");
    Require(scenario==L"rgba8"||scenario==L"rgba32"||scenario==L"r11"||scenario==L"standard"||scenario==L"write_immediate"||scenario==L"exposure"||scenario==L"apply_off"||scenario==L"transfer_zero"||scenario==L"batch"||scenario==L"reuse"||scenario==L"early_reset"||scenario==L"scope"||scenario==L"stream"||scenario==L"paced_stream"||scenario==L"recreate"||scenario==L"queue_switch"||scenario==L"resize"||scenario==L"drs"||scenario==L"create1_alias"||scenario==L"hdr"||scenario==L"subrect_bypass"||scenario==L"bundle_bypass"||scenario==L"unknown_motion_state"||scenario==L"device_reinit"||scenario==L"controls"||scenario==L"toggle"||scenario==L"temporal_toggle"||scenario==L"wrapper"||scenario==L"wrapper_bypass"||scenario==L"motion32_bypass","host scenario");
    Require(mode<=4,"host mode");
    Require(!(scenario==L"stream"&&frames>3&&mode>=3),"reference has three in-flight slots");
    Require(scenario!=L"recreate"||frames>=4,"recreate requires four frames");
    Require(scenario!=L"queue_switch"||frames>=4,"queue switch requires four frames");
    Require(scenario!=L"temporal_toggle"||frames>=6,"temporal toggle requires six frames");
    Require(scenario!=L"toggle"||frames>=5,"toggle requires at least five frames");
    const unsigned width=std::stoul(argv[3]),height=std::stoul(argv[4]);
    Require(width>=64&&height>=64&&width<=3840&&height<=2160,"bounded test dimensions");
    std::ifstream input(std::filesystem::path(argv[2]),std::ios::binary|std::ios::ate);
    Require(bool(input)&&input.tellg()==std::streampos(std::size_t(width)*height*8),"input RGBA16F size");
    std::vector<Rgba16> pixels(std::size_t(width)*height);
    input.seekg(0);input.read(reinterpret_cast<char*>(pixels.data()),pixels.size()*sizeof(Rgba16));
    Require(bool(input),"input read");

    ComPtr<ID3D12Debug> debug;
    wchar_t noDebug[8]{};
    const bool debugEnabled=GetEnvironmentVariableW(L"NR_NATIVE_NO_DEBUG",noDebug,8)==0;
    if(debugEnabled){Hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),"required D3D12 debug layer");debug->EnableDebugLayer();}
    ComPtr<IDXGIFactory6> factory;Hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"DXGI factory");
    ComPtr<ID3D12Device> device;ComPtr<ID3D12CommandQueue> queue;
    for(UINT index=0;;++index){
        ComPtr<IDXGIAdapter1> adapter;
        if(factory->EnumAdapterByGpuPreference(index,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
            IID_PPV_ARGS(&adapter))==DXGI_ERROR_NOT_FOUND)break;
        DXGI_ADAPTER_DESC1 desc{};adapter->GetDesc1(&desc);if(desc.VendorId!=0x1002)continue;
        ComPtr<ID3D12Device> candidate;
        if(FAILED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&candidate))))continue;
        D3D12_COMMAND_QUEUE_DESC qdesc{};qdesc.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> q;
        if(FAILED(candidate->CreateCommandQueue(&qdesc,IID_PPV_ARGS(&q))))continue;
        device=candidate;queue=q;break;
    }
    Require(device&&queue,"AMD native D3D12 device and queue");
    if(firstDeviceIdentity)std::printf("NativeDeviceReinit same_identity=%u\n",device.Get()==firstDeviceIdentity?1u:0u);
    const auto sourceFormat=scenario==L"rgba8"?DXGI_FORMAT_R8G8B8A8_UNORM:scenario==L"rgba32"?
        DXGI_FORMAT_R32G32B32A32_FLOAT:scenario==L"r11"?DXGI_FORMAT_R11G11B10_FLOAT:DXGI_FORMAT_R16G16B16A16_FLOAT;
    const auto color=CreateTexture(device.Get(),TextureDesc(width,height,sourceFormat),
                                   D3D12_RESOURCE_STATE_COPY_DEST);
    const auto motion=CreateTexture(device.Get(),TextureDesc(width,height,scenario==L"motion32_bypass"?DXGI_FORMAT_R32G32_FLOAT:DXGI_FORMAT_R16G16_FLOAT),
                                    D3D12_RESOURCE_STATE_COPY_DEST);
    const auto depth=CreateTexture(device.Get(),TextureDesc(width,height,DXGI_FORMAT_R32_FLOAT),
                                   D3D12_RESOURCE_STATE_COPY_DEST);
    ComPtr<ID3D12Resource> exposure;
    if(scenario==L"exposure"){
        exposure=CreateTexture(device.Get(),TextureDesc(1,1,DXGI_FORMAT_R32_FLOAT),D3D12_RESOURCE_STATE_COPY_DEST);
        UploadTexture(device.Get(),queue.Get(),exposure.Get(),std::vector<float>{.5f});
    }
    referenceExposure=exposure.Get();
    const auto output=CreateTexture(device.Get(),TextureDesc(width*outputScale,height*outputScale,
        DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    color->SetName(L"Host colour");
    motion->SetName(L"Host motion");
    depth->SetName(L"Host depth");
    output->SetName(L"Host upscale output");
    UploadColor(device.Get(),queue.Get(),color.Get(),pixels);
    if(scenario==L"motion32_bypass")UploadTexture(device.Get(),queue.Get(),motion.Get(),std::vector<std::array<float,2>>(pixels.size(),{0.f,0.f}));
    else UploadTexture(device.Get(),queue.Get(),motion.Get(),std::vector<Rg16>(pixels.size(),{0,0}));
    UploadTexture(device.Get(),queue.Get(),depth.Get(),std::vector<float>(pixels.size(),.5f));
    const auto scene=CreateTexture(device.Get(),TextureDesc(width,height,sourceFormat),D3D12_RESOURCE_STATE_COPY_DEST);
    scene->SetName(L"Host source scene");
    UploadColor(device.Get(),queue.Get(),scene.Get(),pixels);
    ComPtr<ID3D12Resource> immediateBuffer;
    if(scenario==L"write_immediate"){
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width=256;desc.Height=1;desc.DepthOrArraySize=1;desc.MipLevels=1;
        desc.SampleDesc.Count=1;desc.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
            D3D12_RESOURCE_STATE_COMMON,nullptr,IID_PPV_ARGS(&immediateBuffer)),"immediate buffer");
    }
    auto produceTo=[&](ID3D12GraphicsCommandList* list,ID3D12Resource* target){
        D3D12_RESOURCE_BARRIER barriers[]={Barrier(scene.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE),
            Barrier(target,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST)};
        list->ResourceBarrier(2,barriers);list->CopyResource(target,scene.Get());
        for(auto& b:barriers)std::swap(b.Transition.StateBefore,b.Transition.StateAfter);list->ResourceBarrier(2,barriers);
    };
    auto produce=[&](ID3D12GraphicsCommandList* list){produceTo(list,color.Get());};

    HMODULE module=LoadLibraryExW(argv[1],nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);
    Require(module,"load source-built OptiScaler");
    const auto init=Entry<InitFn>(module,"NVSDK_NGX_D3D12_Init_Ext");
    const auto allocate=Entry<ParamsFn>(module,"NVSDK_NGX_D3D12_AllocateParameters");
    const auto destroy=Entry<DestroyFn>(module,"NVSDK_NGX_D3D12_DestroyParameters");
    const auto create=Entry<CreateFn>(module,"NVSDK_NGX_D3D12_CreateFeature");
    const auto evaluate=Entry<EvaluateFn>(module,"NVSDK_NGX_D3D12_EvaluateFeature");
    const auto release=Entry<ReleaseFn>(module,"NVSDK_NGX_D3D12_ReleaseFeature");
    const auto shutdown=Entry<ShutdownFn>(module,"NVSDK_NGX_D3D12_Shutdown");
    Require(init(0x1337,L".",device.Get(),static_cast<NVSDK_NGX_Version>(0x15),nullptr)==
            NVSDK_NGX_Result_Success,"NGX D3D12 init");
    NVSDK_NGX_Parameter* params=nullptr;
    Require(allocate(&params)==NVSDK_NGX_Result_Success&&params,"NGX D3D12 parameters");
    params->Set(NVSDK_NGX_Parameter_Width,width);
    params->Set(NVSDK_NGX_Parameter_Height,height);
    params->Set(NVSDK_NGX_Parameter_OutWidth,width*outputScale);
    params->Set(NVSDK_NGX_Parameter_OutHeight,height*outputScale);
    params->Set(NVSDK_NGX_Parameter_PerfQualityValue,unsigned(NVSDK_NGX_PerfQuality_Value_MaxPerf));
    params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
        unsigned(NVSDK_NGX_DLSS_Feature_Flags_AutoExposure|NVSDK_NGX_DLSS_Feature_Flags_MVLowRes|
            (referenceHdr?NVSDK_NGX_DLSS_Feature_Flags_IsHDR:0)));
    auto creation=OpenCommands(device.Get());NVSDK_NGX_Handle* handle=nullptr;
    Require(create(creation.list.Get(),NVSDK_NGX_Feature_SuperSampling,params,&handle)==
            NVSDK_NGX_Result_Success&&handle,"NGX D3D12 create feature");
    ExecuteAndWait(device.Get(),queue.Get(),creation.list.Get());

    params->Set(NVSDK_NGX_Parameter_Color,color.Get());
    params->Set(NVSDK_NGX_Parameter_MotionVectors,motion.Get());
    params->Set(NVSDK_NGX_Parameter_Depth,depth.Get());
    params->Set(NVSDK_NGX_Parameter_Output,output.Get());
    if(exposure)params->Set(NVSDK_NGX_Parameter_ExposureTexture,exposure.Get());
    params->Set(NVSDK_NGX_Parameter_MV_Scale_X,1.f);
    params->Set(NVSDK_NGX_Parameter_MV_Scale_Y,1.f);
    params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X,0.f);
    params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y,0.f);
    params->Set(NVSDK_NGX_Parameter_FrameTimeDeltaInMsec,16.6667f);
    if(scenario==L"subrect_bypass"){
        params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,width-1);
        params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,height);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X,1u);
        params->Set(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X,1u);
    }
    using Counter=unsigned(*)();using Count64=UINT64(*)();
    auto ready=Entry<Counter>(module,"DlssNrNativeReady"),pending=Entry<Counter>(module,"DlssNrNativePending"),faults=Entry<Counter>(module,"DlssNrNativeFaults");
    auto applied=Entry<Count64>(module,"DlssNrNativeApplied");
    using ToggleFn=void (*)(unsigned);
    ToggleFn setEnabled=nullptr;
    ToggleFn setControls=nullptr;
    ToggleFn setTemporal=nullptr;
    if(scenario==L"toggle"&&(mode==1||mode==2))setEnabled=Entry<ToggleFn>(module,"DlssNrNativeTestSetEnabled");
    if(scenario==L"controls"&&(mode==1||mode==2))setControls=Entry<ToggleFn>(module,"DlssNrNativeTestSetControls");
    if(scenario==L"temporal_toggle"&&(mode==1||mode==2))setTemporal=Entry<ToggleFn>(module,"DlssNrNativeTestSetTemporal");
    if((mode==1||mode==2)&&scenario!=L"scope"&&scenario!=L"subrect_bypass"&&scenario!=L"bundle_bypass"&&scenario!=L"unknown_motion_state"&&scenario!=L"wrapper_bypass"&&scenario!=L"motion32_bypass"){
        params->Set(NVSDK_NGX_Parameter_Reset,1u);auto warmup=OpenCommands(device.Get());produce(warmup.list.Get());
        Require(evaluate(warmup.list.Get(),handle,params,nullptr)==NVSDK_NGX_Result_Success,"native warmup Evaluate");
        ExecuteAndWait(device.Get(),queue.Get(),warmup.list.Get());
        for(unsigned i=0;i<6000&&!ready()&&!faults();++i)Sleep(5);
        Require(ready()&&!faults(),"native warmup ready");
    }
    std::unique_ptr<Reference> reference;
    if(mode>=3)reference=std::make_unique<Reference>(device.Get(),queue.Get(),std::filesystem::path(argv[1]).parent_path(),
        mode==3?width:width*outputScale,mode==3?height:height*outputScale);
    const auto initiallyApplied=applied();
    Commands reused;
    ComPtr<ID3D12QueryHeap> scopeQueries;
    if(scenario==L"scope"){
        D3D12_QUERY_HEAP_DESC desc{};desc.Type=D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS;desc.Count=frames;
        Hr(device->CreateQueryHeap(&desc,IID_PPV_ARGS(&scopeQueries)),"scope query heap");
    }
    std::vector<Rgba16> all;
    all.reserve(std::size_t(width*outputScale)*(height*outputScale)*frames);
    std::vector<ComPtr<ID3D12Resource>> streamColors,streamOutputs;
    std::vector<Commands> streamCommands;
    std::array<Commands,3> pacedCommands;
    const bool streaming=scenario==L"stream"||scenario==L"paced_stream";
    std::uint64_t streamEvaluateUs=0,streamSubmitUs=0;
    if(streaming){
        streamColors.push_back(color);streamOutputs.push_back(output);
        for(unsigned i=1;i<frames;++i){
            auto nextColor=CreateTexture(device.Get(),TextureDesc(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT),
                                         D3D12_RESOURCE_STATE_COPY_DEST);
            UploadTexture(device.Get(),queue.Get(),nextColor.Get(),pixels);
            streamColors.push_back(std::move(nextColor));
            streamOutputs.push_back(CreateTexture(device.Get(),TextureDesc(width*outputScale,height*outputScale,
                DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        streamCommands.reserve(frames*2);
    }
    ComPtr<ID3D12Fence> pacingFence;
    if(scenario==L"paced_stream")Hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&pacingFence)),"host pacing fence");
    unsigned pacingWaits=0;
    const auto loopStart=std::chrono::steady_clock::now();
    for(unsigned frame=0;frame<frames;++frame){
        if(setControls)setControls(frame);
        const bool temporal=scenario!=L"temporal_toggle"||frame<2||frame>=4;
        if(setTemporal)setTemporal(temporal?1u:0u);
        const auto controls=FrameControls(scenario==L"controls"?frame:0);
        if(setEnabled&&(frame==1||frame==3)){
            setEnabled(frame==3?1u:0u);
            std::printf("NativeToggle frame=%u enabled=%u\n",frame,frame==3?1u:0u);
        }
        auto* frameColor=streaming?streamColors[frame].Get():color.Get();
        auto* frameOutput=streaming?streamOutputs[frame].Get():output.Get();
        params->Set(NVSDK_NGX_Parameter_Color,frameColor);
        params->Set(NVSDK_NGX_Parameter_Output,frameOutput);
        const bool resetFrame=frame==0||(scenario==L"toggle"&&frame==3)||
            ((scenario==L"recreate"||scenario==L"queue_switch")&&frame==frames/2);
        params->Set(NVSDK_NGX_Parameter_Reset,resetFrame?1u:0u);
        Commands commands;
        if(scenario==L"paced_stream"&&pacedCommands[frame%3].list){
            commands=std::move(pacedCommands[frame%3]);
            Hr(commands.allocator->Reset(),"paced allocator");
            Hr(commands.list->Reset(commands.allocator.Get(),nullptr),"paced list");
        }else if(scenario==L"reuse"&&reused.list){commands=std::move(reused);
            Hr(commands.allocator->Reset(),"reused allocator");Hr(commands.list->Reset(commands.allocator.Get(),nullptr),"reused list");}
        else commands=scenario==L"create1_alias"?OpenCommands1(device.Get()):OpenCommands(device.Get());
        Commands producer,tail;
        ComPtr<ID3D12CommandAllocator> resetAllocator;
        if(scenario==L"create1_alias"){
            ComPtr<ID3D12GraphicsCommandList1> alias;
            Hr(commands.list.As(&alias),"Create1 extended alias");
            alias->SetViewInstanceMask(0);
        }
        ComPtr<ID3D12CommandAllocator> bundleAllocator;
        ComPtr<ID3D12GraphicsCommandList> bundle;
        if(scenario==L"bundle_bypass"){
            Hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_BUNDLE,
                IID_PPV_ARGS(&bundleAllocator)),"bundle allocator");
            Hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_BUNDLE,bundleAllocator.Get(),
                nullptr,IID_PPV_ARGS(&bundle)),"empty bundle");
            Hr(bundle->Close(),"close empty bundle");
            commands.list->ExecuteBundle(bundle.Get());
        }
        produceTo(commands.list.Get(),frameColor);
        if(immediateBuffer){
            ComPtr<ID3D12GraphicsCommandList2> list2;
            Hr(commands.list.As(&list2),"WriteBufferImmediate list2");
            if(frame==0){
                D3D12_RESOURCE_BARRIER barrier{};
                barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrier.Transition={immediateBuffer.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_COPY_DEST};
                list2->ResourceBarrier(1,&barrier);
            }
            const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER write{
                immediateBuffer->GetGPUVirtualAddress(),frame+1};
            list2->WriteBufferImmediate(1,&write,nullptr);
        }
        if(mode==3){Execute(queue.Get(),commands.list.Get());
            producer=std::move(commands);
            reference->Apply(frameColor,motion.Get(),width,height,
                resetFrame,false,controls,temporal);commands=OpenCommands(device.Get());}
        if(scenario==L"batch"){
            producer=std::move(commands);Hr(producer.list->Close(),"batch producer close");
            commands=OpenCommands(device.Get());tail=OpenCommands(device.Get());
            D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;barrier.UAV.pResource=output.Get();
            tail.list->ResourceBarrier(1,&barrier);Hr(tail.list->Close(),"batch tail close");
        }
        if(scenario==L"scope")commands.list->BeginQuery(scopeQueries.Get(),D3D12_QUERY_TYPE_PIPELINE_STATISTICS,frame);
        ComPtr<ID3D12GraphicsCommandList> wrapper;
        if(scenario==L"wrapper_bypass"||scenario==L"wrapper")
            wrapper.Attach(new ForwardingList(commands.list.Get(),scenario==L"wrapper"));
        const auto evaluateStart=std::chrono::steady_clock::now();
        Require(evaluate(wrapper?wrapper.Get():commands.list.Get(),handle,params,nullptr)==NVSDK_NGX_Result_Success,
                "NGX D3D12 Evaluate");
        if(immediateBuffer){
            ComPtr<ID3D12GraphicsCommandList2> list2;
            Hr(commands.list.As(&list2),"WriteBufferImmediate suffix list2");
            const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER write{
                immediateBuffer->GetGPUVirtualAddress()+4,frame+1};
            const D3D12_WRITEBUFFERIMMEDIATE_MODE immediateMode=D3D12_WRITEBUFFERIMMEDIATE_MODE_DEFAULT;
            list2->WriteBufferImmediate(1,&write,&immediateMode);
        }
        if(streaming)streamEvaluateUs+=static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now()-evaluateStart).count());
        if(scenario==L"scope")commands.list->EndQuery(scopeQueries.Get(),D3D12_QUERY_TYPE_PIPELINE_STATISTICS,frame);
        if(scenario==L"batch"){
            Hr(commands.list->Close(),"batch marker close");
            ID3D12CommandList* batch[]={producer.list.Get(),commands.list.Get(),tail.list.Get()};
            queue->ExecuteCommandLists(3,batch);
            // The three list allocators are local to this iteration. Retire
            // them before destruction; delayed multi-frame ownership is
            // covered separately by stream and paced_stream.
            ComPtr<ID3D12Fence> batchFence;
            Hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&batchFence)),"batch fence");
            Hr(queue->Signal(batchFence.Get(),1),"batch signal");
            CpuWait(batchFence.Get(),1);
        }else if(scenario==L"early_reset"){
            Execute(queue.Get(),commands.list.Get());
            Hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&resetAllocator)),"early reset allocator");
            Hr(commands.list->Reset(resetAllocator.Get(),nullptr),"early reset original list");
            Hr(commands.list->Close(),"early reset close");
        }else if(streaming){
            const auto submitStart=std::chrono::steady_clock::now();
            Execute(queue.Get(),commands.list.Get());
            streamSubmitUs+=static_cast<std::uint64_t>(std::chrono::duration_cast<
                std::chrono::microseconds>(std::chrono::steady_clock::now()-submitStart).count());
        }
        else ExecuteAndWait(device.Get(),queue.Get(),commands.list.Get());
        if(mode==4)reference->Apply(frameOutput,motion.Get(),width,height,
            resetFrame,true,controls,temporal);
        if(streaming){
            if(producer.list)streamCommands.push_back(std::move(producer));
            if(pacingFence)pacedCommands[frame%3]=std::move(commands);
            else streamCommands.push_back(std::move(commands));
            if(pacingFence&&(frame+1)%3==0&&frame+1<frames){
                Hr(queue->Signal(pacingFence.Get(),frame+1),"host cohort signal");
                CpuWait(pacingFence.Get(),frame+1);++pacingWaits;
                streamCommands.clear();
            }
            continue;
        }
        const auto result=ReadTexture<Rgba16>(device.Get(),queue.Get(),output.Get());
        for(const auto& pixel:result){
            Require((pixel.r&0x7c00)!=0x7c00&&(pixel.g&0x7c00)!=0x7c00&&
                    (pixel.b&0x7c00)!=0x7c00,"finite output");
        }
        std::printf("NativeFrame frame=%u sha256=%s\n",frame,
            Sha256(result.data(),result.size()*sizeof(Rgba16)).c_str());
        all.insert(all.end(),result.begin(),result.end());
        DebugClean(device.Get());
        if(scenario==L"recreate"&&frame+1==frames/2){
            if(reference){reference->Close();reference=std::make_unique<Reference>(device.Get(),queue.Get(),
                std::filesystem::path(argv[1]).parent_path(),mode==3?width:width*outputScale,mode==3?height:height*outputScale);}
            Require(release(handle)==NVSDK_NGX_Result_Success,"release first feature");
            for(unsigned i=0;i<2000&&pending()&&!faults();++i)Sleep(5);
            Require(!pending()&&!faults(),"first native owner drained");
            auto nextCreation=OpenCommands(device.Get());handle=nullptr;
            Require(create(nextCreation.list.Get(),NVSDK_NGX_Feature_SuperSampling,params,&handle)==
                    NVSDK_NGX_Result_Success&&handle,"recreate feature");
            ExecuteAndWait(device.Get(),queue.Get(),nextCreation.list.Get());
            if(mode==1||mode==2){
                params->Set(NVSDK_NGX_Parameter_Reset,1u);
                auto nextWarmup=OpenCommands(device.Get());produce(nextWarmup.list.Get());
                Require(evaluate(nextWarmup.list.Get(),handle,params,nullptr)==
                        NVSDK_NGX_Result_Success,"recreated feature warmup");
                ExecuteAndWait(device.Get(),queue.Get(),nextWarmup.list.Get());
                for(unsigned i=0;i<6000&&!ready()&&!faults();++i)Sleep(5);
                Require(ready()&&!faults(),"recreated native owner ready");
            }
            std::printf("NativeRecreate drained=1 ready=%u after_frame=%u\n",ready(),frame);
        }
        if(scenario==L"queue_switch"&&frame+1==frames/2){
            if(reference)reference->Close();
            D3D12_COMMAND_QUEUE_DESC qdesc{};qdesc.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
            ComPtr<ID3D12CommandQueue> nextQueue;
            Hr(device->CreateCommandQueue(&qdesc,IID_PPV_ARGS(&nextQueue)),"replacement direct queue");
            queue=std::move(nextQueue);
            if(reference)reference=std::make_unique<Reference>(device.Get(),queue.Get(),
                std::filesystem::path(argv[1]).parent_path(),mode==3?width:width*outputScale,mode==3?height:height*outputScale);
            if(mode==1||mode==2){
                params->Set(NVSDK_NGX_Parameter_Reset,1u);
                auto nextWarmup=OpenCommands(device.Get());produce(nextWarmup.list.Get());
                Require(evaluate(nextWarmup.list.Get(),handle,params,nullptr)==
                        NVSDK_NGX_Result_Success,"replacement queue warmup");
                ExecuteAndWait(device.Get(),queue.Get(),nextWarmup.list.Get());
                for(unsigned i=0;i<6000&&!ready()&&!faults();++i)Sleep(5);
                Require(ready()&&!faults(),"replacement queue owner ready");
            }
            std::printf("NativeQueueSwitch ready=%u after_frame=%u\n",ready(),frame);
        }
        if(scenario==L"reuse")reused=std::move(commands);
    }
    if(streaming){
        ComPtr<ID3D12Fence> fence;
        Hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"stream completion fence");
        Hr(queue->Signal(fence.Get(),1),"stream final signal");
        CpuWait(fence.Get(),1);
        const auto loopUs=std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now()-loopStart).count();
        for(unsigned frame=0;frame<frames;++frame){
            const auto result=ReadTexture<Rgba16>(device.Get(),queue.Get(),streamOutputs[frame].Get());
            for(const auto& pixel:result)Require((pixel.r&0x7c00)!=0x7c00&&
                (pixel.g&0x7c00)!=0x7c00&&(pixel.b&0x7c00)!=0x7c00,"finite stream output");
            std::printf("NativeFrame frame=%u sha256=%s\n",frame,
                Sha256(result.data(),result.size()*sizeof(Rgba16)).c_str());
            all.insert(all.end(),result.begin(),result.end());
            DebugClean(device.Get());
        }
        std::printf("NativeStream queued=%u waits_during_submission=%u final_waits=1\n",frames,pacingWaits);
        std::printf("NativePacing cohort=3 host_waits=%u loop_us=%llu frames=%u\n",pacingWaits,
            static_cast<unsigned long long>(loopUs),frames);
        std::printf("NativeHostCpu evaluate_us_total=%llu submit_us_total=%llu frames=%u\n",
            static_cast<unsigned long long>(streamEvaluateUs),
            static_cast<unsigned long long>(streamSubmitUs),frames);
    }
    if(scenario==L"resize"||scenario==L"drs")for(unsigned phase=0;phase<(scenario==L"drs"?2u:1u);++phase){
        if(scenario!=L"drs")Require(release(handle)==NVSDK_NGX_Result_Success,"release feature before resize");
        if(scenario!=L"drs"){
            for(unsigned i=0;i<2000&&pending()&&!faults();++i)Sleep(5);
            Require(!pending()&&!faults(),"old owner drained before resize");
        }
        const unsigned nextWidth=phase?width:(width/2<64?64:width/2);
        const unsigned nextHeight=phase?height:(height/2<64?64:height/2);
        Require(phase||(nextWidth!=width&&nextHeight!=height),"resize must change both dimensions");
        const unsigned outWidth=scenario==L"drs"?width*outputScale:nextWidth*outputScale,
                       outHeight=scenario==L"drs"?height*outputScale:nextHeight*outputScale;
        std::vector<Rgba16> nextPixels(std::size_t(nextWidth)*nextHeight);
        for(unsigned y=0;y<nextHeight;++y)for(unsigned x=0;x<nextWidth;++x)
            nextPixels[std::size_t(y)*nextWidth+x]=pixels[std::size_t(y)*height/nextHeight*width+
                                                     std::size_t(x)*width/nextWidth];
        auto nextColor=CreateTexture(device.Get(),TextureDesc(nextWidth,nextHeight,DXGI_FORMAT_R16G16B16A16_FLOAT),D3D12_RESOURCE_STATE_COPY_DEST);
        auto nextScene=CreateTexture(device.Get(),TextureDesc(nextWidth,nextHeight,DXGI_FORMAT_R16G16B16A16_FLOAT),D3D12_RESOURCE_STATE_COPY_DEST);
        auto nextMotion=CreateTexture(device.Get(),TextureDesc(nextWidth,nextHeight,DXGI_FORMAT_R16G16_FLOAT),D3D12_RESOURCE_STATE_COPY_DEST);
        auto nextDepth=CreateTexture(device.Get(),TextureDesc(nextWidth,nextHeight,DXGI_FORMAT_R32_FLOAT),D3D12_RESOURCE_STATE_COPY_DEST);
        auto nextOutput=CreateTexture(device.Get(),TextureDesc(outWidth,outHeight,
            DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        UploadTexture(device.Get(),queue.Get(),nextColor.Get(),nextPixels);
        UploadTexture(device.Get(),queue.Get(),nextScene.Get(),nextPixels);
        UploadTexture(device.Get(),queue.Get(),nextMotion.Get(),std::vector<Rg16>(nextPixels.size(),{0,0}));
        UploadTexture(device.Get(),queue.Get(),nextDepth.Get(),std::vector<float>(nextPixels.size(),.5f));
        params->Set(NVSDK_NGX_Parameter_Width,nextWidth);
        params->Set(NVSDK_NGX_Parameter_Height,nextHeight);
        params->Set(NVSDK_NGX_Parameter_OutWidth,outWidth);
        params->Set(NVSDK_NGX_Parameter_OutHeight,outHeight);
        params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,nextWidth);
        params->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,nextHeight);
        params->Set(NVSDK_NGX_Parameter_Color,nextColor.Get());
        params->Set(NVSDK_NGX_Parameter_MotionVectors,nextMotion.Get());
        params->Set(NVSDK_NGX_Parameter_Depth,nextDepth.Get());
        params->Set(NVSDK_NGX_Parameter_Output,nextOutput.Get());
        if(reference&&(scenario!=L"drs"||mode==3)){reference->Close();reference=std::make_unique<Reference>(device.Get(),queue.Get(),
            std::filesystem::path(argv[1]).parent_path(),mode==3?nextWidth:outWidth,mode==3?nextHeight:outHeight);}
        if(scenario!=L"drs"){auto nextCreation=OpenCommands(device.Get());handle=nullptr;
        Require(create(nextCreation.list.Get(),NVSDK_NGX_Feature_SuperSampling,params,&handle)==
                NVSDK_NGX_Result_Success&&handle,"create resized feature");
        ExecuteAndWait(device.Get(),queue.Get(),nextCreation.list.Get());}
        auto produceResized=[&](ID3D12GraphicsCommandList* list){
            D3D12_RESOURCE_BARRIER barriers[]={Barrier(nextScene.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE),
                Barrier(nextColor.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST)};
            list->ResourceBarrier(2,barriers);list->CopyResource(nextColor.Get(),nextScene.Get());
            for(auto& b:barriers)std::swap(b.Transition.StateBefore,b.Transition.StateAfter);
            list->ResourceBarrier(2,barriers);
        };
        if(mode==1||(mode==2&&scenario!=L"drs")){
            params->Set(NVSDK_NGX_Parameter_Reset,1u);
            auto nextWarmup=OpenCommands(device.Get());produceResized(nextWarmup.list.Get());
            Require(evaluate(nextWarmup.list.Get(),handle,params,nullptr)==
                    NVSDK_NGX_Result_Success,"resized feature warmup");
            ExecuteAndWait(device.Get(),queue.Get(),nextWarmup.list.Get());
            for(unsigned i=0;i<6000&&!ready()&&!faults();++i)Sleep(5);
            Require(ready()&&!faults(),"resized native owner ready");
        }
        std::vector<Rgba16> resizedAll;resizedAll.reserve(std::size_t(outWidth)*outHeight*2);
        for(unsigned frame=0;frame<2;++frame){
            params->Set(NVSDK_NGX_Parameter_Reset,frame==0?1u:0u);
            auto commands=OpenCommands(device.Get());produceResized(commands.list.Get());
            Commands producer;
            if(mode==3){Execute(queue.Get(),commands.list.Get());producer=std::move(commands);
                reference->Apply(nextColor.Get(),nextMotion.Get(),nextWidth,nextHeight,frame==0,false);
                commands=OpenCommands(device.Get());}
            Require(evaluate(commands.list.Get(),handle,params,nullptr)==NVSDK_NGX_Result_Success,
                    "resized Evaluate");
            ExecuteAndWait(device.Get(),queue.Get(),commands.list.Get());
            if(mode==4)reference->Apply(nextOutput.Get(),nextMotion.Get(),nextWidth,nextHeight,frame==0,true);
            const auto result=ReadTexture<Rgba16>(device.Get(),queue.Get(),nextOutput.Get());
            for(const auto& pixel:result)Require((pixel.r&0x7c00)!=0x7c00&&
                (pixel.g&0x7c00)!=0x7c00&&(pixel.b&0x7c00)!=0x7c00,"finite resized output");
            resizedAll.insert(resizedAll.end(),result.begin(),result.end());
            DebugClean(device.Get());
        }
        std::printf("NativeResize phase=%u same_feature=%u source=%ux%u output=%ux%u frames=2 sha256=%s\n",
            phase,scenario==L"drs"?1u:0u,nextWidth,nextHeight,outWidth,outHeight,
            Sha256(resizedAll.data(),resizedAll.size()*sizeof(Rgba16)).c_str());
    }
    if(mode==1||mode==2){
        const auto count=applied()-initiallyApplied;
        const bool coverage=scenario==L"stream"&&frames>3
            ?count>=3&&count<=frames
            :count==((scenario==L"scope"||scenario==L"subrect_bypass"||scenario==L"bundle_bypass"||scenario==L"unknown_motion_state"||scenario==L"wrapper_bypass"||scenario==L"motion32_bypass")?0u:
                scenario==L"toggle"?frames-2:frames+(scenario==L"resize"?2u:scenario==L"drs"?4u:0u));
        Require(coverage&&!faults(),"native processing coverage");
    }
    if(reference)reference->Close();
    Require(release(handle)==NVSDK_NGX_Result_Success,"NGX D3D12 release feature");
    Require(destroy(params)==NVSDK_NGX_Result_Success,"NGX D3D12 destroy parameters");
    Require(shutdown()==NVSDK_NGX_Result_Success,"NGX D3D12 shutdown");
    if(mode==1||mode==2){for(unsigned i=0;i<2000&&pending();++i)Sleep(5);Require(!pending()&&!faults(),"native clean drain");}
    std::printf("NativeCoverage mode=%u applied=%llu debug_clean=1 debug_layer=%u\n",mode,static_cast<unsigned long long>(applied()-initiallyApplied),debugEnabled?1u:0u);
    std::printf("PASS OptiScaler native DX12 Evaluate source=%ux%u output=%ux%u frames=%u sha256=%s finite=1\n",
        width,height,width*outputScale,height*outputScale,frames,Sha256(all.data(),all.size()*sizeof(Rgba16)).c_str());
    if(scenario==L"device_reinit"){
        firstDeviceIdentity=device.Get();
        std::array<wchar_t*,8> second{};
        for(int i=0;i<argc;++i)second[i]=argv[i];
        second[6]=const_cast<wchar_t*>(L"standard");
        std::printf("NativeDeviceReinit first=%p beginning_second=1\n",static_cast<void*>(device.Get()));
        std::fflush(stdout);
        Require(wmain(argc,second.data())==0,"replacement device run");
        std::printf("NativeDeviceReinit completed=1\n");
        firstDeviceIdentity=nullptr;
    }
    // Match the DX11 host: OptiScaler owns process-wide Detours and cleanup
    // workers. Explicit FreeLibrary while D3D12 COM objects still exist can
    // detach code under those objects, so let process shutdown release it.
    std::fflush(stdout);
    return 0;
}catch(const std::exception& e){
    std::fprintf(stderr,"FAIL OptiScaler native DX12 Evaluate: %s\n",e.what());
    std::fflush(stderr);std::fflush(stdout);
    // A failed diagnostic may retain an undrained HIP owner. Do not run DLL
    // loader-lock teardown of that invalid test state; preserve a failing exit.
    TerminateProcess(GetCurrentProcess(),1);return 1;
}
