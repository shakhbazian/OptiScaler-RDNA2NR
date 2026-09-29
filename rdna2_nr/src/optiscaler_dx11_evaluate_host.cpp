// Calls the compiled OptiScaler DLL's public DX11 NGX entry points, so the
// actual FSR2FeatureDx11on12 / IFeature_Dx11wDx12::Evaluate path is exercised.
#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <nvsdk_ngx_params.h>
#include "../include/nr_extent.h"
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <string_view>
#include <chrono>
#include <stdexcept>
#include <cstdio>
#include <cstring>

using Microsoft::WRL::ComPtr;
namespace {
void Check(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
void Hr(HRESULT hr, const char* what) { if (FAILED(hr)) { std::fprintf(stderr, "%s: %08X\n", what, unsigned(hr)); throw std::runtime_error(what); } }
unsigned short Half(float value) {
    unsigned bits=0;std::memcpy(&bits,&value,sizeof(bits));
    const unsigned sign=(bits>>16)&0x8000;
    const int exponent=int((bits>>23)&0xff)-127+15;
    Check(exponent>0&&exponent<31,"half conversion range");
    return static_cast<unsigned short>(sign|(unsigned(exponent)<<10)|((bits>>13)&0x3ff));
}
using InitFn=NVSDK_NGX_Result (*)(unsigned long long,const wchar_t*,ID3D11Device*,NVSDK_NGX_Version,const NVSDK_NGX_FeatureCommonInfo*);
using ParamsFn=NVSDK_NGX_Result (*)(NVSDK_NGX_Parameter**);
using DestroyFn=NVSDK_NGX_Result (*)(NVSDK_NGX_Parameter*);
using CreateFn=NVSDK_NGX_Result (*)(ID3D11DeviceContext*,NVSDK_NGX_Feature,NVSDK_NGX_Parameter*,NVSDK_NGX_Handle**);
using EvaluateFn=NVSDK_NGX_Result (*)(ID3D11DeviceContext*,const NVSDK_NGX_Handle*,NVSDK_NGX_Parameter*,void*);
using ReleaseFn=NVSDK_NGX_Result (*)(NVSDK_NGX_Handle*);
using ShutdownFn=NVSDK_NGX_Result (*)();
using RetireProbeFn=unsigned (*)();
using AppliedProbeFn=UINT64 (*)();
template<class T> T Entry(HMODULE module,const char* name) { auto p=GetProcAddress(module,name); Check(p!=nullptr,name); return reinterpret_cast<T>(p); }
struct Texture { ComPtr<ID3D11Texture2D> resource; D3D11_TEXTURE2D_DESC desc{}; };
Texture MakeTexture(ID3D11Device* device,unsigned width,unsigned height,DXGI_FORMAT format,UINT bind) {
    Texture t; t.desc.Width=width; t.desc.Height=height; t.desc.MipLevels=1; t.desc.ArraySize=1;
    t.desc.Format=format; t.desc.SampleDesc.Count=1; t.desc.Usage=D3D11_USAGE_DEFAULT; t.desc.BindFlags=bind;
    Hr(device->CreateTexture2D(&t.desc,nullptr,&t.resource),"create DX11 texture"); return t;
}
std::vector<unsigned char> Readback(ID3D11Device* device,ID3D11DeviceContext* ctx,const Texture& source) {
    auto desc=source.desc; desc.Usage=D3D11_USAGE_STAGING;desc.BindFlags=0;desc.CPUAccessFlags=D3D11_CPU_ACCESS_READ;
    ComPtr<ID3D11Texture2D> staging;Hr(device->CreateTexture2D(&desc,nullptr,&staging),"create staging texture");
    ctx->CopyResource(staging.Get(),source.resource.Get()); D3D11_MAPPED_SUBRESOURCE mapped{};
    Hr(ctx->Map(staging.Get(),0,D3D11_MAP_READ,0,&mapped),"map staging texture");
    const unsigned rowBytes=desc.Width*8;std::vector<unsigned char> data(std::size_t(rowBytes)*desc.Height);
    for(unsigned y=0;y<desc.Height;++y) std::memcpy(data.data()+std::size_t(y)*rowBytes,
        static_cast<const unsigned char*>(mapped.pData)+std::size_t(y)*mapped.RowPitch,rowBytes);
    ctx->Unmap(staging.Get(),0);return data;
}
}
int wmain(int argc,wchar_t** argv) try {
    const bool debugEnabled=GetEnvironmentVariableW(L"NR_HOST_DEBUG",nullptr,0)>0;
    const bool hdr=GetEnvironmentVariableW(L"NR_HOST_HDR",nullptr,0)>0;
    if(debugEnabled){ComPtr<ID3D12Debug> debug;Hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),"DX12 debug");debug->EnableDebugLayer();}
    Check(argc>=6&&argc<=10,"usage: host OptiScaler.dll input.rgba.f16 width height output_dir [mode] [frames] [timed] [native]");
    const unsigned w=std::stoul(argv[3]),h=std::stoul(argv[4]);
    const std::wstring_view mode=argc>=7?std::wstring_view(argv[6]):L"static";
    const unsigned frameCount=argc==8?std::stoul(argv[7]):4;
    const bool timed=argc>=9&&std::wstring_view(argv[8])==L"timed";
    const bool native=argc==10&&std::wstring_view(argv[9])==L"native";
    Check(argc<9||timed,"unknown run option");
    Check(argc!=10||native,"unknown output option");
    const unsigned frames=argc>=9?std::stoul(argv[7]):frameCount;
    Check(frames>=1&&frames<=(timed?360u:32u),"bounded frame count");
    Check(mode==L"static"||mode==L"pan"||mode==L"pan_reset"||mode==L"occlusion"||mode==L"occlusion_reset"||mode==L"bad_motion"||mode==L"resize","unknown sequence mode");
    Check(w>=64&&h>=64&&NrV2::SupportedLogicalExtent(w,h)&&
        NrV2::SupportedGraphExtent(NrV2::AlignNetworkExtent(w),NrV2::AlignNetworkExtent(h)),
        "unsupported input extent");
    const unsigned outW=native?w:w*2,outH=native?h:h*2;
    Check(NrV2::SupportedLogicalExtent(outW,outH),"unsupported output extent");
    std::ifstream file(std::filesystem::path(argv[2]),std::ios::binary|std::ios::ate);
    Check(bool(file),"open input");Check(file.tellg()==std::streampos(std::size_t(w)*h*8),"RGBA16F input size");
    std::vector<unsigned char> input(std::size_t(w)*h*8);file.seekg(0);file.read(reinterpret_cast<char*>(input.data()),input.size());
    Check(bool(file),"read input");std::filesystem::create_directories(argv[5]);
    ComPtr<IDXGIFactory6> factory;Hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"DXGI factory");
    ComPtr<ID3D11Device> device;ComPtr<ID3D11DeviceContext> ctx;
    ComPtr<IDXGIAdapter3> memoryAdapter;
    for(unsigned i=0;;++i){ComPtr<IDXGIAdapter1> adapter;
        if(factory->EnumAdapterByGpuPreference(i,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&adapter))==DXGI_ERROR_NOT_FOUND)break;
        DXGI_ADAPTER_DESC1 ad{};adapter->GetDesc1(&ad);if(ad.VendorId!=0x1002)continue;
        D3D_FEATURE_LEVEL level{};
        if(SUCCEEDED(D3D11CreateDevice(adapter.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,nullptr,0,D3D11_SDK_VERSION,&device,&level,&ctx))){
            adapter.As(&memoryAdapter);break;
        }
    }
    Check(bool(device),"no AMD DX11 device");
    ComPtr<ID3D12Device> debugDevice;
    if(debugEnabled)Hr(D3D12CreateDevice(memoryAdapter.Get(),D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&debugDevice)),"debug bridge device");
    auto memorySample=[&](const char* phase,unsigned frame){
        if(!timed||!memoryAdapter)return;
        DXGI_QUERY_VIDEO_MEMORY_INFO info{};
        if(SUCCEEDED(memoryAdapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info)))
            std::printf("VideoMemory phase=%s frame=%u usage_bytes=%llu budget_bytes=%llu\n",phase,frame,
                static_cast<unsigned long long>(info.CurrentUsage),static_cast<unsigned long long>(info.Budget));
    };
    memorySample("before_feature",0);
    // Load only our source-built OptiScaler DLL, after creating the device.
    HMODULE dll=LoadLibraryExW(argv[1],nullptr,LOAD_WITH_ALTERED_SEARCH_PATH);Check(dll!=nullptr,"load OptiScaler DLL");
    auto init=Entry<InitFn>(dll,"NVSDK_NGX_D3D11_Init_Ext");
    auto allocate=Entry<ParamsFn>(dll,"NVSDK_NGX_D3D11_AllocateParameters");
    auto destroy=Entry<DestroyFn>(dll,"NVSDK_NGX_D3D11_DestroyParameters");
    auto create=Entry<CreateFn>(dll,"NVSDK_NGX_D3D11_CreateFeature");
    auto evaluate=Entry<EvaluateFn>(dll,"NVSDK_NGX_D3D11_EvaluateFeature");
    auto release=Entry<ReleaseFn>(dll,"NVSDK_NGX_D3D11_ReleaseFeature");
    auto shutdown=Entry<ShutdownFn>(dll,"NVSDK_NGX_D3D11_Shutdown");
    Check(init(0x1337,L".",device.Get(),static_cast<NVSDK_NGX_Version>(0x15),nullptr)==NVSDK_NGX_Result_Success,"NGX init");
    NVSDK_NGX_Parameter* params=nullptr;Check(allocate(&params)==NVSDK_NGX_Result_Success&&params,"allocate params");
    params->Set(NVSDK_NGX_Parameter_Width,w);params->Set(NVSDK_NGX_Parameter_Height,h);
    params->Set(NVSDK_NGX_Parameter_OutWidth,outW);params->Set(NVSDK_NGX_Parameter_OutHeight,outH);
    params->Set(NVSDK_NGX_Parameter_PerfQualityValue,unsigned(NVSDK_NGX_PerfQuality_Value_MaxPerf));
    params->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,unsigned(NVSDK_NGX_DLSS_Feature_Flags_AutoExposure|NVSDK_NGX_DLSS_Feature_Flags_MVLowRes|(hdr?NVSDK_NGX_DLSS_Feature_Flags_IsHDR:0)));
    NVSDK_NGX_Handle* handle=nullptr;
    Check(create(ctx.Get(),NVSDK_NGX_Feature_SuperSampling,params,&handle)==NVSDK_NGX_Result_Success&&handle,"create FSR2-on12 feature");
    memorySample("configured",0);
    auto color=MakeTexture(device.Get(),w,h,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D11_BIND_SHADER_RESOURCE);
    const bool badMotion=mode==L"bad_motion";
    auto motion=MakeTexture(device.Get(),w,h,badMotion?DXGI_FORMAT_R32G32_FLOAT:DXGI_FORMAT_R16G16_FLOAT,D3D11_BIND_SHADER_RESOURCE);
    auto depth=MakeTexture(device.Get(),w,h,DXGI_FORMAT_R32_FLOAT,D3D11_BIND_SHADER_RESOURCE);
    auto output=MakeTexture(device.Get(),outW,outH,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D11_BIND_UNORDERED_ACCESS);
    std::vector<unsigned char> movingColor=input;
    std::vector<unsigned short> mv(std::size_t(w)*h*2,0);
    const std::vector<float> wideMv(std::size_t(w)*h*2,0.0f);
    std::vector<float> z(std::size_t(w)*h,0.5f);
    const auto runStart=std::chrono::steady_clock::now();
    for(unsigned frame=0;frame<frames;++frame){
        if(mode==L"occlusion"||mode==L"occlusion_reset"){
            movingColor=input;
            std::fill(mv.begin(),mv.end(),static_cast<unsigned short>(0));
            std::fill(z.begin(),z.end(),0.8f);
            const unsigned step=w/80>4u?w/80:4u;
            const unsigned left=w/4+frame*step,right=left+w/4;
            const unsigned top=h/4,bottom=3*h/4;
            const unsigned short foregroundMotion=frame==0?0:Half(-float(step));
            for(unsigned y=top;y<bottom;++y)for(unsigned x=left;x<right&&x<w;++x){
                const unsigned localX=x-left,localY=y-top;
                // A moving cutout reveals the unmodified background. The
                // foreground has exact previous-frame motion and distinct depth.
                if(localX>w/12&&localX<w/6&&localY>h/8&&localY<h/3)continue;
                const bool border=localX<3||right-x<=3||localY<3||bottom-y<=3;
                const bool tile=((localX/8+localY/8)&1)!=0;
                const unsigned short rgba[4]={border?Half(1.0f):Half(tile?0.875f:0.625f),
                    border?Half(1.0f):Half(tile?0.125f:0.25f),
                    border?Half(1.0f):Half(tile?0.125f:0.25f),Half(1.0f)};
                const std::size_t pixel=std::size_t(y)*w+x;
                std::memcpy(movingColor.data()+pixel*8,rgba,8);
                z[pixel]=0.2f;mv[pixel*2]=foregroundMotion;
            }
        }else if(mode==L"pan"||mode==L"pan_reset"){
            const unsigned shift=frame*4;
            // Current x samples previous x-4. Negative four-pixel motion is
            // IEEE FP16 0xC400; the first/reset frame has no temporal input.
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
                const unsigned sourceX=x>=shift?x-shift:0;
                const std::size_t pixel=std::size_t(y)*w+x;
                std::memcpy(movingColor.data()+pixel*8,input.data()+(std::size_t(y)*w+sourceX)*8,8);
                z[pixel]=0.2f+0.6f*float(sourceX)/float(w-1);
                mv[pixel*2]=frame==0?0:0xC400;mv[pixel*2+1]=0;
            }
        }
        ctx->UpdateSubresource(color.resource.Get(),0,nullptr,movingColor.data(),w*8,0);
        ctx->UpdateSubresource(motion.resource.Get(),0,nullptr,badMotion?static_cast<const void*>(wideMv.data()):static_cast<const void*>(mv.data()),w*(badMotion?8:4),0);
        ctx->UpdateSubresource(depth.resource.Get(),0,nullptr,z.data(),w*4,0);
        params->Set(NVSDK_NGX_Parameter_Color,static_cast<ID3D11Resource*>(color.resource.Get()));
        params->Set(NVSDK_NGX_Parameter_MotionVectors,static_cast<ID3D11Resource*>(motion.resource.Get()));
        params->Set(NVSDK_NGX_Parameter_Depth,static_cast<ID3D11Resource*>(depth.resource.Get()));
        params->Set(NVSDK_NGX_Parameter_Output,static_cast<ID3D11Resource*>(output.resource.Get()));
        params->Set(NVSDK_NGX_Parameter_MV_Scale_X,1.0f);params->Set(NVSDK_NGX_Parameter_MV_Scale_Y,1.0f);
        params->Set(NVSDK_NGX_Parameter_Jitter_Offset_X,0.0f);params->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y,0.0f);
        params->Set(NVSDK_NGX_Parameter_Reset,unsigned(frame==0||
            ((mode==L"pan_reset"||mode==L"occlusion_reset")&&frame==2)));
        params->Set(NVSDK_NGX_Parameter_FrameTimeDeltaInMsec,16.6667f);
        const auto evalStart=std::chrono::steady_clock::now();
        Check(evaluate(ctx.Get(),handle,params,nullptr)==NVSDK_NGX_Result_Success,"NGX EvaluateFeature");
        const auto evalEnd=std::chrono::steady_clock::now();
        if(timed){
            const auto evalMs=std::chrono::duration<double,std::milli>(evalEnd-evalStart).count();
            std::printf("HostTiming frame=%u evaluate_ms=%.3f readback_ms=0 bytes=0\n",frame,evalMs);
            if(frame==0||frame+1==frames||frame%30==29)memorySample("frame",frame);
            continue;
        }
        auto pixels=Readback(device.Get(),ctx.Get(),output);
        const auto readEnd=std::chrono::steady_clock::now();
        for(std::size_t i=0;i<pixels.size();i+=2){unsigned half=unsigned(pixels[i])|(unsigned(pixels[i+1])<<8);
            Check((half&0x7c00)!=0x7c00,"nonfinite FP16 output");}
        const auto path=std::filesystem::path(argv[5])/(L"evaluate_"+std::to_wstring(frame)+L".rgba.f16");
        std::ofstream raw(path,std::ios::binary);Check(bool(raw),"open output");raw.write(reinterpret_cast<const char*>(pixels.data()),pixels.size());
        Check(bool(raw),"write output");
        const auto evalMs=std::chrono::duration<double,std::milli>(evalEnd-evalStart).count();
        const auto readMs=std::chrono::duration<double,std::milli>(readEnd-evalEnd).count();
        std::printf("HostTiming frame=%u evaluate_ms=%.3f readback_ms=%.3f bytes=%zu\n",
            frame,evalMs,readMs,pixels.size());
    }
    if(mode==L"resize"){
        Check(w>=64&&h>=64&&w%2==0&&h%2==0,"resize test requires even extent");
        Check(release(handle)==NVSDK_NGX_Result_Success,"release old resize feature");
        handle=nullptr;
        const unsigned nextW=w/2,nextH=h/2,nextOutW=nextW*2,nextOutH=nextH*2;
        params->Set(NVSDK_NGX_Parameter_Width,nextW);
        params->Set(NVSDK_NGX_Parameter_Height,nextH);
        params->Set(NVSDK_NGX_Parameter_OutWidth,nextOutW);
        params->Set(NVSDK_NGX_Parameter_OutHeight,nextOutH);
        Check(create(ctx.Get(),NVSDK_NGX_Feature_SuperSampling,params,&handle)==NVSDK_NGX_Result_Success&&handle,
            "create resized feature");
        auto nextColor=MakeTexture(device.Get(),nextW,nextH,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D11_BIND_SHADER_RESOURCE);
        auto nextMotion=MakeTexture(device.Get(),nextW,nextH,DXGI_FORMAT_R16G16_FLOAT,D3D11_BIND_SHADER_RESOURCE);
        auto nextDepth=MakeTexture(device.Get(),nextW,nextH,DXGI_FORMAT_R32_FLOAT,D3D11_BIND_SHADER_RESOURCE);
        auto nextOutput=MakeTexture(device.Get(),nextOutW,nextOutH,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D11_BIND_UNORDERED_ACCESS);
        std::vector<unsigned char> nextPixels(std::size_t(nextW)*nextH*8);
        for(unsigned y=0;y<nextH;++y)for(unsigned x=0;x<nextW;++x)
            std::memcpy(nextPixels.data()+(std::size_t(y)*nextW+x)*8,
                input.data()+(std::size_t(y*2)*w+x*2)*8,8);
        std::vector<unsigned short> nextMv(std::size_t(nextW)*nextH*2,0);
        std::vector<float> nextZ(std::size_t(nextW)*nextH,0.5f);
        for(unsigned frame=0;frame<3;++frame){
            ctx->UpdateSubresource(nextColor.resource.Get(),0,nullptr,nextPixels.data(),nextW*8,0);
            ctx->UpdateSubresource(nextMotion.resource.Get(),0,nullptr,nextMv.data(),nextW*4,0);
            ctx->UpdateSubresource(nextDepth.resource.Get(),0,nullptr,nextZ.data(),nextW*4,0);
            params->Set(NVSDK_NGX_Parameter_Color,static_cast<ID3D11Resource*>(nextColor.resource.Get()));
            params->Set(NVSDK_NGX_Parameter_MotionVectors,static_cast<ID3D11Resource*>(nextMotion.resource.Get()));
            params->Set(NVSDK_NGX_Parameter_Depth,static_cast<ID3D11Resource*>(nextDepth.resource.Get()));
            params->Set(NVSDK_NGX_Parameter_Output,static_cast<ID3D11Resource*>(nextOutput.resource.Get()));
            params->Set(NVSDK_NGX_Parameter_Reset,unsigned(frame==0));
            Check(evaluate(ctx.Get(),handle,params,nullptr)==NVSDK_NGX_Result_Success,"evaluate resized feature");
        }
        auto resized=Readback(device.Get(),ctx.Get(),nextOutput);
        for(std::size_t i=0;i<resized.size();i+=2){unsigned half=unsigned(resized[i])|(unsigned(resized[i+1])<<8);
            Check((half&0x7c00)!=0x7c00,"nonfinite resized output");}
        const auto path=std::filesystem::path(argv[5])/L"resize_last.rgba.f16";
        std::ofstream raw(path,std::ios::binary);Check(bool(raw),"open resized output");
        raw.write(reinterpret_cast<const char*>(resized.data()),resized.size());Check(bool(raw),"write resized output");
        std::printf("ResizeCycle old=%ux%u new=%ux%u frames=3 finite=PASS\n",w,h,nextW,nextH);
        Check(release(handle)==NVSDK_NGX_Result_Success,"release small resize feature");
        handle=nullptr;
        params->Set(NVSDK_NGX_Parameter_Width,w);
        params->Set(NVSDK_NGX_Parameter_Height,h);
        params->Set(NVSDK_NGX_Parameter_OutWidth,outW);
        params->Set(NVSDK_NGX_Parameter_OutHeight,outH);
        Check(create(ctx.Get(),NVSDK_NGX_Feature_SuperSampling,params,&handle)==NVSDK_NGX_Result_Success&&handle,
            "recreate original extent feature");
        for(unsigned frame=0;frame<3;++frame){
            ctx->UpdateSubresource(color.resource.Get(),0,nullptr,input.data(),w*8,0);
            ctx->UpdateSubresource(motion.resource.Get(),0,nullptr,mv.data(),w*4,0);
            ctx->UpdateSubresource(depth.resource.Get(),0,nullptr,z.data(),w*4,0);
            params->Set(NVSDK_NGX_Parameter_Color,static_cast<ID3D11Resource*>(color.resource.Get()));
            params->Set(NVSDK_NGX_Parameter_MotionVectors,static_cast<ID3D11Resource*>(motion.resource.Get()));
            params->Set(NVSDK_NGX_Parameter_Depth,static_cast<ID3D11Resource*>(depth.resource.Get()));
            params->Set(NVSDK_NGX_Parameter_Output,static_cast<ID3D11Resource*>(output.resource.Get()));
            params->Set(NVSDK_NGX_Parameter_Reset,unsigned(frame==0));
            Check(evaluate(ctx.Get(),handle,params,nullptr)==NVSDK_NGX_Result_Success,"evaluate restored extent feature");
        }
        auto restored=Readback(device.Get(),ctx.Get(),output);
        for(std::size_t i=0;i<restored.size();i+=2){unsigned half=unsigned(restored[i])|(unsigned(restored[i+1])<<8);
            Check((half&0x7c00)!=0x7c00,"nonfinite restored output");}
        const auto restoredPath=std::filesystem::path(argv[5])/L"resize_return.rgba.f16";
        std::ofstream restoredRaw(restoredPath,std::ios::binary);Check(bool(restoredRaw),"open restored output");
        restoredRaw.write(reinterpret_cast<const char*>(restored.data()),restored.size());
        Check(bool(restoredRaw),"write restored output");
        std::printf("ResizeReturn original=%ux%u frames=3 finite=PASS\n",w,h);
    }
    if(timed){
        const auto beforeReadback=std::chrono::steady_clock::now();
        auto pixels=Readback(device.Get(),ctx.Get(),output);
        const auto finished=std::chrono::steady_clock::now();
        for(std::size_t i=0;i<pixels.size();i+=2){unsigned half=unsigned(pixels[i])|(unsigned(pixels[i+1])<<8);
            Check((half&0x7c00)!=0x7c00,"nonfinite timed output");}
        const auto path=std::filesystem::path(argv[5])/L"evaluate_last.rgba.f16";
        std::ofstream raw(path,std::ios::binary);Check(bool(raw),"open timed output");
        raw.write(reinterpret_cast<const char*>(pixels.data()),pixels.size());Check(bool(raw),"write timed output");
        const double loopMs=std::chrono::duration<double,std::milli>(beforeReadback-runStart).count();
        const double readbackMs=std::chrono::duration<double,std::milli>(finished-beforeReadback).count();
        std::printf("TimedRun frames=%u evaluate_loop_ms=%.3f final_readback_ms=%.3f finite=PASS\n",
            frames,loopMs,readbackMs);
        const auto summaryPath=std::filesystem::path(argv[5])/L"host-summary.json";
        std::ofstream summary(summaryPath);Check(bool(summary),"open timed summary");
        summary<<"{\"frames\":"<<frames<<",\"evaluateLoopMs\":"<<loopMs
            <<",\"finalReadbackMs\":"<<readbackMs<<",\"finite\":true}\n";
        Check(bool(summary),"write timed summary");
        memorySample("after_readback",frames);
    }
    const auto releaseStart=std::chrono::steady_clock::now();
    Check(release(handle)==NVSDK_NGX_Result_Success,"release feature");
    const auto releaseEnd=std::chrono::steady_clock::now();
    Check(destroy(params)==NVSDK_NGX_Result_Success,"destroy params");
    Check(shutdown()==NVSDK_NGX_Result_Success,"shutdown");
    memorySample("after_shutdown",frames);
    const auto pending=reinterpret_cast<RetireProbeFn>(GetProcAddress(dll,"DlssNrNativePending"));
    const auto faulted=reinterpret_cast<RetireProbeFn>(GetProcAddress(dll,"DlssNrNativeFaults"));
    const auto applied=reinterpret_cast<AppliedProbeFn>(GetProcAddress(dll,"DlssNrNativeApplied"));
    Check(pending&&faulted&&applied,"retirement diagnostics unavailable");
    const unsigned initialPending=pending();
    const auto retireWaitStart=std::chrono::steady_clock::now();
    while(pending()!=0 && std::chrono::steady_clock::now()-retireWaitStart<std::chrono::seconds(15))Sleep(1);
    Check(pending()==0&&!faulted(),"scheduled retirement did not finish cleanly");
    if(timed)std::printf("TimedNr applied=%llu\n",static_cast<unsigned long long>(applied()));
    if(debugDevice){
        ComPtr<ID3D12InfoQueue> info;Hr(debugDevice.As(&info),"bridge debug info queue");unsigned errors=0;
        for(UINT64 i=0;i<info->GetNumStoredMessagesAllowedByRetrievalFilter();++i){SIZE_T size=0;
            info->GetMessage(i,nullptr,&size);std::vector<unsigned char> bytes(size);
            auto* m=reinterpret_cast<D3D12_MESSAGE*>(bytes.data());info->GetMessage(i,m,&size);
            if(m->Severity<=D3D12_MESSAGE_SEVERITY_WARNING){std::fprintf(stderr,"D3D12: %s\n",m->pDescription);++errors;}}
        Check(errors==0,"DX11 bridge D3D12 debug clean");std::puts("CommonFrontend bridge_debug_clean=1");
    }
    std::printf("Lifecycle release_ms=%.3f pending_after_shutdown=%u cleanup_wait_ms=%.3f faulted=0\n",
        std::chrono::duration<double,std::milli>(releaseEnd-releaseStart).count(),initialPending,
        std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-retireWaitStart).count());
    std::puts("PASS real OptiScaler DX11 Evaluate x4");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL Evaluate host: %s\n",e.what());return 1;}
