#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include "../include/nr_runtime_contract.h"
#include "../include/nr_runtime_v2_test.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;
#ifdef NR_RUNTIME_DLL_PROBE_HELPERS_ONLY
using Hash256=NrV2::Hash256;
using Frame=NrV2::Frame;
using Format=NrV2::Format;
using JitterMode=NrV2::JitterMode;
#else
using namespace NrV2;
#endif

namespace {
void Hr(HRESULT value, const char* operation) {
    if (FAILED(value)) throw std::runtime_error(std::string(operation) + " failed");
}
void Nt(NTSTATUS value, const char* operation) {
    if (value < 0) throw std::runtime_error(std::string(operation) + " failed");
}
void Require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Handle { HANDLE value=nullptr; ~Handle(){if(value)CloseHandle(value);} };
void CpuWait(ID3D12Fence* fence, std::uint64_t value) {
    if (fence->GetCompletedValue() >= value) return;
    Handle event; event.value = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    Require(event.value != nullptr, "CreateEvent failed");
    Hr(fence->SetEventOnCompletion(value, event.value), "SetEventOnCompletion");
    Require(WaitForSingleObject(event.value, 30000) == WAIT_OBJECT_0, "GPU wait timed out");
}
D3D12_RESOURCE_DESC TextureDesc(unsigned width,unsigned height,DXGI_FORMAT format,
                                D3D12_RESOURCE_FLAGS flags=D3D12_RESOURCE_FLAG_NONE){
    D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width=width;d.Height=height;d.DepthOrArraySize=1;d.MipLevels=1;d.Format=format;
    d.SampleDesc.Count=1;d.Layout=D3D12_TEXTURE_LAYOUT_UNKNOWN;d.Flags=flags;return d;
}
D3D12_RESOURCE_DESC BufferDesc(std::uint64_t bytes){D3D12_RESOURCE_DESC d{};
    d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=bytes;d.Height=1;
    d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;
    d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;return d;}
D3D12_RESOURCE_BARRIER Barrier(ID3D12Resource*r,D3D12_RESOURCE_STATES a,
                               D3D12_RESOURCE_STATES b){D3D12_RESOURCE_BARRIER x{};
    x.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;x.Transition.pResource=r;
    x.Transition.StateBefore=a;x.Transition.StateAfter=b;
    x.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;return x;}
ComPtr<ID3D12Resource> CreateTexture(ID3D12Device*device,const D3D12_RESOURCE_DESC&desc,
                                     D3D12_RESOURCE_STATES state){
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    ComPtr<ID3D12Resource> result;
    Hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,state,nullptr,
                                       IID_PPV_ARGS(&result)),"Create texture");return result;}
struct Commands{ComPtr<ID3D12CommandAllocator>allocator;ComPtr<ID3D12GraphicsCommandList>list;};
Commands OpenCommands(ID3D12Device*device){Commands r;
    Hr(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                      IID_PPV_ARGS(&r.allocator)),"Create allocator");
    Hr(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,r.allocator.Get(),nullptr,
                                 IID_PPV_ARGS(&r.list)),"Create list");return r;}
void Execute(ID3D12CommandQueue*queue,ID3D12GraphicsCommandList*list){
    Hr(list->Close(),"Close list");ID3D12CommandList*lists[]={list};
    queue->ExecuteCommandLists(1,lists);}
void ExecuteAndWait(ID3D12Device*device,ID3D12CommandQueue*queue,
                    ID3D12GraphicsCommandList*list){Execute(queue,list);ComPtr<ID3D12Fence>fence;
    Hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"Create wait fence");
    Hr(queue->Signal(fence.Get(),1),"Signal wait fence");CpuWait(fence.Get(),1);}

struct Rgba16{std::uint16_t r,g,b,a;};
struct Rg16{std::uint16_t r,g;};
static_assert(sizeof(Rgba16)==8&&sizeof(Rg16)==4);

template<class Pixel>void UploadTexture(ID3D12Device*device,ID3D12CommandQueue*queue,
                                        ID3D12Resource*texture,const std::vector<Pixel>&pixels){
    const auto desc=texture->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows=0;UINT64 rowBytes=0,total=0;
    device->GetCopyableFootprints(&desc,0,1,0,&footprint,&rows,&rowBytes,&total);
    Require(pixels.size()==desc.Width*desc.Height&&rowBytes==desc.Width*sizeof(Pixel),
            "upload footprint");D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_UPLOAD;
    const auto buffer=BufferDesc(total);ComPtr<ID3D12Resource>upload;
    Hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,
                                       D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,
                                       IID_PPV_ARGS(&upload)),"Create upload");
    unsigned char*mapped=nullptr;Hr(upload->Map(0,nullptr,reinterpret_cast<void**>(&mapped)),"Map upload");
    for(UINT y=0;y<rows;++y)std::memcpy(mapped+footprint.Offset+std::size_t(y)*footprint.Footprint.RowPitch,
                                        pixels.data()+std::size_t(y)*desc.Width,rowBytes);
    upload->Unmap(0,nullptr);auto commands=OpenCommands(device);
    D3D12_TEXTURE_COPY_LOCATION to{};to.pResource=texture;to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION from{};from.pResource=upload.Get();from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;from.PlacedFootprint=footprint;
    commands.list->CopyTextureRegion(&to,0,0,0,&from,nullptr);
    auto barrier=Barrier(texture,D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    commands.list->ResourceBarrier(1,&barrier);ExecuteAndWait(device,queue,commands.list.Get());}

template<class Pixel>std::vector<Pixel>ReadTexture(ID3D12Device*device,ID3D12CommandQueue*queue,
                                                   ID3D12Resource*texture){
    const auto desc=texture->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows=0;UINT64 rowBytes=0,total=0;device->GetCopyableFootprints(&desc,0,1,0,&footprint,&rows,&rowBytes,&total);
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_READBACK;const auto buffer=BufferDesc(total);
    ComPtr<ID3D12Resource>readback;Hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&buffer,
        D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&readback)),"Create readback");
    auto commands=OpenCommands(device);auto before=Barrier(texture,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
    commands.list->ResourceBarrier(1,&before);D3D12_TEXTURE_COPY_LOCATION to{};to.pResource=readback.Get();to.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;to.PlacedFootprint=footprint;
    D3D12_TEXTURE_COPY_LOCATION from{};from.pResource=texture;from.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    commands.list->CopyTextureRegion(&to,0,0,0,&from,nullptr);auto after=Barrier(texture,D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    commands.list->ResourceBarrier(1,&after);ExecuteAndWait(device,queue,commands.list.Get());
    const unsigned char*mapped=nullptr;D3D12_RANGE range{0,static_cast<SIZE_T>(total)};
    Hr(readback->Map(0,&range,reinterpret_cast<void**>(const_cast<unsigned char**>(&mapped))),"Map readback");
    std::vector<Pixel>result(std::size_t(desc.Width)*desc.Height);
    for(UINT y=0;y<rows;++y)std::memcpy(result.data()+std::size_t(y)*desc.Width,
        mapped+footprint.Offset+std::size_t(y)*footprint.Footprint.RowPitch,rowBytes);
    D3D12_RANGE written{0,0};readback->Unmap(0,&written);return result;}

Hash256 ParseHash(const char*text){Hash256 result{};auto digit=[](char c)->unsigned{
    if(c>='0'&&c<='9')return unsigned(c-'0');if(c>='A'&&c<='F')return unsigned(c-'A'+10);
    if(c>='a'&&c<='f')return unsigned(c-'a'+10);throw std::runtime_error("invalid hash");};
    Require(std::strlen(text)==64,"hash length");for(unsigned i=0;i<32;++i)
        result.bytes[i]=static_cast<std::uint8_t>((digit(text[i*2])<<4)|digit(text[i*2+1]));return result;}
std::string Sha256(const void*data,std::size_t size){BCRYPT_ALG_HANDLE algorithm=nullptr;
    BCRYPT_HASH_HANDLE hash=nullptr;DWORD objectBytes=0,resultBytes=0,hashBytes=0;
    Nt(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0),"Open SHA256");
    try{Nt(BCryptGetProperty(algorithm,BCRYPT_OBJECT_LENGTH,reinterpret_cast<PUCHAR>(&objectBytes),sizeof(objectBytes),&resultBytes,0),"SHA object size");
        Nt(BCryptGetProperty(algorithm,BCRYPT_HASH_LENGTH,reinterpret_cast<PUCHAR>(&hashBytes),sizeof(hashBytes),&resultBytes,0),"SHA length");
        std::vector<unsigned char>object(objectBytes),digest(hashBytes);
        Nt(BCryptCreateHash(algorithm,&hash,object.data(),objectBytes,nullptr,0,0),"Create SHA256");
        Nt(BCryptHashData(hash,const_cast<PUCHAR>(static_cast<const UCHAR*>(data)),static_cast<ULONG>(size),0),"Hash data");
        Nt(BCryptFinishHash(hash,digest.data(),hashBytes,0),"Finish SHA256");
        BCryptDestroyHash(hash);hash=nullptr;BCryptCloseAlgorithmProvider(algorithm,0);algorithm=nullptr;
        static const char hex[]="0123456789ABCDEF";std::string output;output.reserve(64);
        for(auto byte:digest){output.push_back(hex[byte>>4]);output.push_back(hex[byte&15]);}return output;
    }catch(...){if(hash)BCryptDestroyHash(hash);if(algorithm)BCryptCloseAlgorithmProvider(algorithm,0);throw;}}

std::vector<std::uint8_t>ReadFile(const char*path){std::ifstream file(path,std::ios::binary|std::ios::ate);
    Require(bool(file),"open package");const auto length=file.tellg();Require(length>0,"empty package");
    std::vector<std::uint8_t>result(static_cast<std::size_t>(length));file.seekg(0);
    Require(bool(file.read(reinterpret_cast<char*>(result.data()),length)),"read package");return result;}

Frame MakeFrame(std::uint64_t index,std::uint64_t epoch,ID3D12GraphicsCommandList*commands,
                ID3D12CommandQueue*queue,ID3D12Resource*color,ID3D12Resource*motion,
                ID3D12Resource*output,std::uint32_t reset=0){Frame frame{};
    frame.prefix={sizeof(Frame),NrV2::Version};frame.key={7,index,epoch};frame.commands=commands;frame.queue=queue;
    frame.color={color,{7,5,96,64,113,79},Format::Rgba16Float,0};
    frame.motion={motion,{5,3,96,64,109,75},Format::Rg16Float,0};
    frame.output={output,{0,0,96,64,96,64},Format::Rgba16Float,0};
    frame.controls={3,1,1,-1,-1,1};frame.motionParameters={1,-1,96,64,.25f,-.5f,-.25f,.5f,JitterMode::AddPreviousMinusCurrent,0};
    frame.resetReasons=reset;return frame;}
Frame MakeAsymmetricFrame(std::uint64_t index,std::uint64_t epoch,
                          ID3D12GraphicsCommandList*commands,ID3D12CommandQueue*queue,
                          ID3D12Resource*color,ID3D12Resource*motion,ID3D12Resource*mask,
                          ID3D12Resource*output,std::uint32_t reset=0){Frame frame{};
    frame.prefix={sizeof(Frame),NrV2::Version};frame.key={9,index,epoch};frame.commands=commands;frame.queue=queue;
    frame.color={color,{7,5,321,257,335,269},Format::Rgba16Float,0};
    frame.motion={motion,{5,3,321,257,331,267},Format::Rg16Float,0};
    frame.controlMask={mask,{9,7,173,139,187,151},Format::Rgba16Float,0};
    frame.output={output,{0,0,321,257,321,257},Format::Rgba16Float,0};
    frame.controls={3,.75f,.5f,1,1,.8f};
    frame.motionParameters={1,-1,321,257,0,0,0,0,JitterMode::AddPreviousMinusCurrent,0};
    frame.resetReasons=reset;return frame;}
} // namespace

#ifndef NR_RUNTIME_DLL_PROBE_HELPERS_ONLY
int main(int argc,char**argv){HMODULE library=nullptr;void*runtime=nullptr;Api api{};
try{
    const bool failureHooks=argc==4&&std::strcmp(argv[3],"--test-hooks")==0;
    const bool nonfiniteHooks=argc==4&&std::strcmp(argv[3],"--test-hooks-nonfinite")==0;
    const bool testHooks=failureHooks||nonfiniteHooks;
    Require(argc==3||testHooks,
            "usage: runtime_v2_dll_probe dll package [--test-hooks|--test-hooks-nonfinite]");
    library=LoadLibraryExA(argv[1],nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    Require(library!=nullptr,"LoadLibraryEx runtime v2");
    const auto getApi=reinterpret_cast<GetApi>(GetProcAddress(library,"DlssNrHipBackendGetApiV2"));
    Require(getApi!=nullptr,"GetProcAddress v2");
    const auto armFailure=reinterpret_cast<NrV2Test::ArmFailure>(
        GetProcAddress(library,"DlssNrHipBackendTestArmFailureV2"));
    const auto armNonfinite=reinterpret_cast<NrV2Test::ArmNonfinite>(
        GetProcAddress(library,"DlssNrHipBackendTestArmNonfiniteV2"));
    const auto getTestState=reinterpret_cast<NrV2Test::GetState>(
        GetProcAddress(library,"DlssNrHipBackendTestGetStateV2"));
    NrV2Test::State finalTestState{};
    Require(testHooks?(armFailure&&armNonfinite&&getTestState):
                      (!armFailure&&!armNonfinite&&!getTestState),
            "test-hook export isolation");
    Require(getApi(Version+1,&api,sizeof(api))==Status::Unsupported,"unknown API version");
    Require(getApi(Version,&api,sizeof(Prefix))==Status::InvalidSize,"truncated API table");
    Require(getApi(Version,&api,sizeof(api))==Status::Ok&&api.prefix.structSize==sizeof(Api)&&
            api.prefix.abiVersion==Version,"API negotiation");
    Require(api.create(nullptr,nullptr,nullptr)==Status::InvalidArgument,"null create output");

    ComPtr<ID3D12Debug>debug;if(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug))))debug->EnableDebugLayer();
    ComPtr<IDXGIFactory6>factory;Hr(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"Create factory");
    ComPtr<ID3D12Device>device;ComPtr<ID3D12CommandQueue>queue;
    for(UINT i=0;;++i){ComPtr<IDXGIAdapter1>adapter;
        if(factory->EnumAdapterByGpuPreference(i,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&adapter))==DXGI_ERROR_NOT_FOUND)break;
        ComPtr<ID3D12Device>candidate;if(FAILED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&candidate))))continue;
        D3D12_COMMAND_QUEUE_DESC desc{};desc.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;ComPtr<ID3D12CommandQueue>candidateQueue;
        if(FAILED(candidate->CreateCommandQueue(&desc,IID_PPV_ARGS(&candidateQueue))))continue;
        void*value=nullptr;if(api.create(candidate.Get(),candidateQueue.Get(),&value)==Status::Ok){device=candidate;queue=candidateQueue;runtime=value;break;}}
    Require(runtime&&device&&queue,"no D3D12 device accepted by v2 runtime");
    Config config{};config.prefix={sizeof(Config),Version};
    config.sourceHash=ParseHash("E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E");
    config.packageHash=ParseHash("A7E6EE38172A81E12D613FA9A2F57E32AA1944908E56CD2A33E1F6C94369E3CB");
    config.graphVersion=1;config.mathContract=MathContract;config.logicalWidth=96;config.logicalHeight=64;
    config.networkWidth=320;config.networkHeight=320;
    Require(api.configure(runtime,&config,sizeof(config)-1,nullptr,0)==Status::InvalidSize,"truncated configure");
    auto package=ReadFile(argv[2]);
    Require(api.configure(runtime,&config,sizeof(config),package.data(),16)==Status::Failed,"truncated package");
    Require(api.configure(runtime,&config,sizeof(config),package.data(),package.size())==Status::Ok,"configure package");
    if(testHooks){NrV2Test::State state{};
        Require(getTestState(runtime,&state,sizeof(state))==Status::Ok&&
                state.modelUploadOperations==1&&!state.historyValid&&state.commits==0&&
                state.graphPreparations==1&&state.graphEnqueues==0&&
                state.graphAllocations>0&&state.historySwaps==0&&state.completionChecks==0,
                "initial model upload state");}
    package.clear();package.shrink_to_fit();

    const auto colorDesc=TextureDesc(113,79,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto motionDesc=TextureDesc(109,75,DXGI_FORMAT_R16G16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto outputDesc=TextureDesc(96,64,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto color0=CreateTexture(device.Get(),colorDesc,D3D12_RESOURCE_STATE_COPY_DEST);
    auto color1=CreateTexture(device.Get(),colorDesc,D3D12_RESOURCE_STATE_COPY_DEST);
    auto motion0=CreateTexture(device.Get(),motionDesc,D3D12_RESOURCE_STATE_COPY_DEST);
    auto motion1=CreateTexture(device.Get(),motionDesc,D3D12_RESOURCE_STATE_COPY_DEST);
    std::array<ComPtr<ID3D12Resource>,4>outputs;
    for(auto&output:outputs)output=CreateTexture(device.Get(),outputDesc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    std::vector<Rgba16>colors0(113*79),colors1(113*79);std::vector<Rg16>motions0(109*75),motions1(109*75);
    for(std::size_t i=0;i<colors0.size();++i){colors0[i]={static_cast<std::uint16_t>(0x3000u+i%0x300u),static_cast<std::uint16_t>(0x3400u+i%0x300u),static_cast<std::uint16_t>(0x3800u+i%0x200u),0x3c00u};
        colors1[i]={static_cast<std::uint16_t>(0x3400u+i%0x280u),static_cast<std::uint16_t>(0x3800u+i%0x180u),static_cast<std::uint16_t>(0x3200u+i%0x300u),0x3c00u};}
    for(std::size_t i=0;i<motions0.size();++i){motions0[i]={0,0};motions1[i]={static_cast<std::uint16_t>(i&1?0x9400u:0x1400u),0};}
    UploadTexture(device.Get(),queue.Get(),color0.Get(),colors0);UploadTexture(device.Get(),queue.Get(),color1.Get(),colors1);
    UploadTexture(device.Get(),queue.Get(),motion0.Get(),motions0);UploadTexture(device.Get(),queue.Get(),motion1.Get(),motions1);

    auto commands0=OpenCommands(device.Get());auto frame0=MakeFrame(0,101,commands0.list.Get(),queue.Get(),color0.Get(),motion0.Get(),outputs[0].Get());
    Result result{};Require(api.evaluate(runtime,&frame0,sizeof(frame0),&result,sizeof(result)-1)==Status::InvalidSize,"truncated result");
    auto scaledColor=frame0;scaledColor.color.transform.extentWidth=95;
    Require(api.evaluate(runtime,&scaledColor,sizeof(scaledColor),&result,sizeof(result))==
            Status::Unsupported&&!result.frameAccepted,"scaled color status");
    auto invalid=frame0;invalid.output.resource=invalid.color.resource;
    Require(api.evaluate(runtime,&invalid,sizeof(invalid),&result,sizeof(result))==Status::InvalidResource&&!result.frameAccepted,"invalid alias mutation");
    Require(api.evaluate(runtime,&frame0,sizeof(frame0),&result,sizeof(result))==Status::Ok&&result.frameAccepted&&!result.outputReady,"frame0 evaluate");
    Require(api.shutdown(runtime)==Status::Busy,"unsubmitted shutdown");Require(api.notifySubmitted(runtime,999)==Status::StaleFrame,"wrong notify");
    Execute(queue.Get(),commands0.list.Get());Require(api.notifySubmitted(runtime,101)==Status::Ok,"frame0 notify");
    Require(api.notifySubmitted(runtime,101)==Status::StaleFrame,"duplicate notify");

    auto commands1=OpenCommands(device.Get());auto frame1=MakeFrame(1,102,commands1.list.Get(),queue.Get(),color1.Get(),motion1.Get(),outputs[1].Get());
    Require(api.evaluate(runtime,&frame1,sizeof(frame1),&result,sizeof(result))==Status::Ok&&result.outputReady&&result.outputDelayed&&
            result.output.streamId==7&&result.output.frameIndex==0&&result.output.submissionEpoch==101&&result.outputNoiseIndex==0,"frame1 delayed metadata");
    Execute(queue.Get(),commands1.list.Get());Require(api.notifySubmitted(runtime,102)==Status::Ok,"frame1 notify");
    const auto delayed0=ReadTexture<Rgba16>(device.Get(),queue.Get(),outputs[1].Get());
    const auto hash0=Sha256(delayed0.data(),delayed0.size()*sizeof(Rgba16));
    Require(hash0=="0FBFEFCF2E95ABF3F19455EB88BA48B67F23E1B092F79D39E489CB97BE396EBC","frame0 DLL hash");

    auto commands2=OpenCommands(device.Get());auto frame2=MakeFrame(2,103,commands2.list.Get(),queue.Get(),color0.Get(),motion0.Get(),outputs[2].Get(),Reset::Explicit);
    Require(api.evaluate(runtime,&frame2,sizeof(frame2),&result,sizeof(result))==Status::Ok&&!result.outputReady,"reset evaluate");
    Execute(queue.Get(),commands2.list.Get());Require(api.notifySubmitted(runtime,103)==Status::Ok,"reset notify");
    const auto resetPublic=ReadTexture<Rgba16>(device.Get(),queue.Get(),outputs[2].Get());std::size_t resetDiff=0;
    for(unsigned y=0;y<64;++y)for(unsigned x=0;x<96;++x){const auto&expected=colors0[std::size_t(y+5)*113+x+7];
        if(std::memcmp(&resetPublic[std::size_t(y)*96+x],&expected,sizeof(expected)))++resetDiff;}
    Require(resetDiff==0,"reset stale output");

    auto commands3=OpenCommands(device.Get());auto frame3=MakeFrame(3,104,commands3.list.Get(),queue.Get(),color1.Get(),motion1.Get(),outputs[3].Get());
    Require(api.evaluate(runtime,&frame3,sizeof(frame3),&result,sizeof(result))==Status::Ok&&result.outputReady&&
            result.output.frameIndex==2&&result.output.submissionEpoch==103&&result.outputNoiseIndex==0,"post-reset delayed metadata");
    Execute(queue.Get(),commands3.list.Get());Require(api.notifySubmitted(runtime,104)==Status::Ok,"frame3 notify");
    const auto delayedReset=ReadTexture<Rgba16>(device.Get(),queue.Get(),outputs[3].Get());
    const auto resetHash=Sha256(delayedReset.data(),delayedReset.size()*sizeof(Rgba16));Require(resetHash==hash0,"reset neural hash");

    package=ReadFile(argv[2]);
    Require(api.configure(runtime,&config,sizeof(config),package.data(),package.size())==Status::Ok,
            "identical configure must preserve state");
    if(testHooks){NrV2Test::State state{};
        Require(getTestState(runtime,&state,sizeof(state))==Status::Ok&&
                state.modelUploadOperations==1&&state.historyValid&&state.commits==4&&
                state.graphPreparations==1&&state.graphEnqueues==4&&
                state.historySwaps==4&&state.completionChecks==4,
                "identical configure state");}
    Require(api.evaluate(runtime,&frame3,sizeof(frame3),&result,sizeof(result))==Status::StaleFrame,
            "identical configure lost accepted-frame state");
    Config asymmetric=config;asymmetric.logicalWidth=321;asymmetric.logicalHeight=257;
    asymmetric.networkWidth=384;asymmetric.networkHeight=320;
    Require(api.configure(runtime,&asymmetric,sizeof(asymmetric),package.data(),package.size())==Status::Ok,
            "asymmetric configure");
    if(testHooks){NrV2Test::State state{};
        Require(getTestState(runtime,&state,sizeof(state))==Status::Ok&&
                state.modelUploadOperations==1&&!state.historyValid&&state.commits==4&&
                state.graphPreparations==2&&state.graphEnqueues==4&&
                state.historySwaps==4&&state.completionChecks==4,
                "geometry configure reuploaded model");}
    package.clear();package.shrink_to_fit();
    const auto asymmetricColorDesc=TextureDesc(335,269,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto asymmetricMotionDesc=TextureDesc(331,267,DXGI_FORMAT_R16G16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto asymmetricMaskDesc=TextureDesc(187,151,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    const auto asymmetricOutputDesc=TextureDesc(321,257,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto asymmetricColor=CreateTexture(device.Get(),asymmetricColorDesc,D3D12_RESOURCE_STATE_COPY_DEST);
    auto asymmetricMotion=CreateTexture(device.Get(),asymmetricMotionDesc,D3D12_RESOURCE_STATE_COPY_DEST);
    auto asymmetricMask=CreateTexture(device.Get(),asymmetricMaskDesc,D3D12_RESOURCE_STATE_COPY_DEST);
    std::array<ComPtr<ID3D12Resource>,4>asymmetricOutputs;
    for(auto&output:asymmetricOutputs)output=CreateTexture(device.Get(),asymmetricOutputDesc,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    std::vector<Rgba16>asymmetricColors(335*269),asymmetricMasks(187*151);std::vector<Rg16>asymmetricMotions(331*267);
    for(std::size_t i=0;i<asymmetricColors.size();++i)asymmetricColors[i]={
        static_cast<std::uint16_t>(0x3100u+i%0x280u),
        static_cast<std::uint16_t>(0x3500u+i%0x280u),
        static_cast<std::uint16_t>(0x3900u+i%0x180u),0x3c00u};
    for(auto&value:asymmetricMotions)value={0,0};
    for(unsigned y=0;y<151;++y)for(unsigned x=0;x<187;++x){
        auto&value=asymmetricMasks[std::size_t(y)*187+x];
        value={static_cast<std::uint16_t>((x+y)%3?0x3800u:0x3c00u),
               static_cast<std::uint16_t>(x&1?0x3400u:0x3a00u),
               static_cast<std::uint16_t>(y&1?0x3800u:0x3c00u),0x3c00u};}
    UploadTexture(device.Get(),queue.Get(),asymmetricColor.Get(),asymmetricColors);
    UploadTexture(device.Get(),queue.Get(),asymmetricMotion.Get(),asymmetricMotions);
    UploadTexture(device.Get(),queue.Get(),asymmetricMask.Get(),asymmetricMasks);
    for(unsigned frameIndex=0;frameIndex<4;++frameIndex){
        auto commands=OpenCommands(device.Get());const bool reset=frameIndex==2;
        auto frame=MakeAsymmetricFrame(frameIndex,201+frameIndex,commands.list.Get(),queue.Get(),
            asymmetricColor.Get(),asymmetricMotion.Get(),asymmetricMask.Get(),
            asymmetricOutputs[frameIndex].Get(),
            reset?Reset::Explicit:0);
        if(frameIndex==0){auto invalidMask=frame;
            invalidMask.controlMask={asymmetricMotion.Get(),{5,3,173,139,331,267},Format::Rgba16Float,0};
            Require(api.evaluate(runtime,&invalidMask,sizeof(invalidMask),&result,sizeof(result))==
                    Status::InvalidResource&&!result.frameAccepted,"invalid control mask mutation");
            invalidMask=frame;
            invalidMask.controlMask={asymmetricColor.Get(),{7,5,173,139,335,269},Format::Rgba16Float,0};
            Require(api.evaluate(runtime,&invalidMask,sizeof(invalidMask),&result,sizeof(result))==
                    Status::InvalidResource&&!result.frameAccepted,"aliased control mask mutation");}
        Require(api.evaluate(runtime,&frame,sizeof(frame),&result,sizeof(result))==Status::Ok,
                "asymmetric evaluate");
        if(frameIndex==0||frameIndex==2)Require(!result.outputReady,"asymmetric warmup/reset output");
        else Require(result.outputReady&&result.output.streamId==9&&
                     result.output.frameIndex==frameIndex-1&&
                     result.output.submissionEpoch==200+frameIndex,
                     "asymmetric delayed metadata");
        Execute(queue.Get(),commands.list.Get());
        Require(api.notifySubmitted(runtime,201+frameIndex)==Status::Ok,"asymmetric notify");
    }
    const auto asymmetricFirst=ReadTexture<Rgba16>(device.Get(),queue.Get(),asymmetricOutputs[1].Get());
    const auto asymmetricReset=ReadTexture<Rgba16>(device.Get(),queue.Get(),asymmetricOutputs[3].Get());
    const auto asymmetricHash=Sha256(asymmetricFirst.data(),asymmetricFirst.size()*sizeof(Rgba16));
    const auto asymmetricResetHash=Sha256(asymmetricReset.data(),asymmetricReset.size()*sizeof(Rgba16));
    Require(asymmetricHash==asymmetricResetHash,"asymmetric reset neural hash");
    if(testHooks){NrV2Test::State before{};
        Require(getTestState(runtime,&before,sizeof(before))==Status::Ok&&
                before.modelUploadOperations==1&&before.historyValid&&before.commits==8&&
                !before.faulted&&!before.recorded&&before.graphPreparations==2&&
                before.graphEnqueues==8&&before.historySwaps==8&&
                before.completionChecks==8&&before.lastEnqueueMicroseconds>0,
                "pre-failure state");
        auto faultOutput=CreateTexture(device.Get(),asymmetricOutputDesc,
                                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto faultCommands=OpenCommands(device.Get());
        auto faultFrame=MakeAsymmetricFrame(4,205,faultCommands.list.Get(),queue.Get(),
            asymmetricColor.Get(),asymmetricMotion.Get(),asymmetricMask.Get(),faultOutput.Get());
        Require(api.evaluate(runtime,&faultFrame,sizeof(faultFrame),&result,sizeof(result))==
                Status::Ok&&result.frameAccepted&&result.outputReady,"fault frame evaluate");
        Execute(queue.Get(),faultCommands.list.Get());
        Require((nonfiniteHooks?armNonfinite(runtime):armFailure(runtime))==Status::Ok,
                "arm injected failure");
        NrV2Test::State armed{};
        Require(getTestState(runtime,&armed,sizeof(armed))==Status::Ok&&armed.recorded&&
                armed.historyValid&&armed.commits==8,"armed failure state");
        Require(api.notifySubmitted(runtime,205)==Status::Failed,"injected notify failure");
        NrV2Test::State failed{};
        Require(getTestState(runtime,&failed,sizeof(failed))==Status::Ok&&failed.faulted&&
                !failed.recorded&&!failed.ready&&!failed.historyValid&&failed.commits==8&&
                failed.modelUploadOperations==1&&failed.graphEnqueues==9&&
                failed.graphPreparations==2&&failed.historySwaps==8&&
                failed.completionChecks==(nonfiniteHooks?9u:8u)&&
                failed.finiteFailures==(nonfiniteHooks?1u:0u),"post-failure state");
        finalTestState=failed;
        auto rejectedCommands=OpenCommands(device.Get());
        auto rejected=MakeAsymmetricFrame(5,206,rejectedCommands.list.Get(),queue.Get(),
            asymmetricColor.Get(),asymmetricMotion.Get(),asymmetricMask.Get(),faultOutput.Get());
        Require(api.evaluate(runtime,&rejected,sizeof(rejected),&result,sizeof(result))==
                Status::Failed&&!result.frameAccepted,"faulted runtime accepted frame");
    }
    Require(api.shutdown(runtime)==Status::Ok,"shutdown");api.destroy(runtime);runtime=nullptr;
    ComPtr<ID3D12InfoQueue>info;if(SUCCEEDED(device.As(&info)))Require(info->GetNumStoredMessagesAllowedByRetrievalFilter()==0,"D3D12 debug messages");
    FreeLibrary(library);library=nullptr;
    if(testHooks)std::printf("PASS %s uploads=1 commits_before=8 commits_after=8 "
        "history_invalidated=PASS faulted=PASS shutdown=PASS graph_allocations=%llu "
        "graph_preparations=%llu graph_enqueues=%llu history_swaps=%llu completion_checks=%llu "
        "finite_failures=%llu enqueue_us=%llu completion_wait_us=%llu\n",
        nonfiniteHooks?"v2_nonfinite":"v2_fault",
        static_cast<unsigned long long>(finalTestState.graphAllocations),
        static_cast<unsigned long long>(finalTestState.graphPreparations),
        static_cast<unsigned long long>(finalTestState.graphEnqueues),
        static_cast<unsigned long long>(finalTestState.historySwaps),
        static_cast<unsigned long long>(finalTestState.completionChecks),
        static_cast<unsigned long long>(finalTestState.finiteFailures),
        static_cast<unsigned long long>(finalTestState.lastEnqueueMicroseconds),
        static_cast<unsigned long long>(finalTestState.lastCompletionWaitMicroseconds));
    std::printf("PASS v2_dll api=2 frames=8 delayed=7:0:101 reset_hash=%s asymmetric=321x257/384x320 asymmetric_reset_hash=%s mask=PASS negotiation=PASS retirement=PASS debug=PASS\n",
                hash0.c_str(),asymmetricHash.c_str());return 0;
}catch(const std::exception&e){if(runtime&&api.shutdown&&api.destroy){if(api.shutdown(runtime)==Status::Ok)api.destroy(runtime);}if(library)FreeLibrary(library);
    std::fprintf(stderr,"FAIL v2_dll: %s\n",e.what());return 1;}}
#endif
