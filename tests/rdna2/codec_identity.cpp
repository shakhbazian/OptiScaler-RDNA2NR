// Exercise the committed encode/resize/resolve shader without a neural network.
#define NR_RUNTIME_DLL_PROBE_HELPERS_ONLY
#include "../../rdna2_nr/src/runtime_v2_dll_probe.cpp"
#include "../../rdna2_nr/tests/common_color_reference.h"
#include <d3d12sdklayers.h>
#include <filesystem>
#ifdef NR_HOST_AGILITY_SDK_VERSION
extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion=NR_HOST_AGILITY_SDK_VERSION;
__declspec(dllexport) extern const char* D3D12SDKPath=".\\AgilitySDK\\";
}
#endif
int wmain(int argc,wchar_t** argv)try{
    Require(argc==8||argc==9,"input width height output hdr scale clamp-model [profile]");
    const unsigned width=std::stoul(argv[2]),height=std::stoul(argv[3]);
    const bool hdr=std::stoul(argv[5])!=0,clampModel=std::stoul(argv[7])!=0;
    const float scale=std::stof(argv[6]);
    const std::wstring profile=argc==9?argv[8]:L"default";
    std::vector<Rgba16> pixels(std::size_t(width)*height);
    std::ifstream input(std::filesystem::path(argv[1]),std::ios::binary);
    Require(bool(input.read(reinterpret_cast<char*>(pixels.data()),pixels.size()*sizeof(Rgba16))),"input read");
    ComPtr<ID3D12Debug> debug;Hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),"debug");debug->EnableDebugLayer();
    ComPtr<ID3D12Device> device;Hr(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)),"device");
    ComPtr<ID3D12CommandQueue> queue;D3D12_COMMAND_QUEUE_DESC q{};Hr(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)),"queue");
    const auto desc=TextureDesc(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto color=CreateTexture(device.Get(),desc,D3D12_RESOURCE_STATE_COPY_DEST);
    UploadTexture(device.Get(),queue.Get(),color.Get(),pixels);
    CommonColorReference codec;codec.Open(device.Get(),width,height,desc.Format,
        unsigned(std::lround(width*scale)),unsigned(std::lround(height*scale)));
    auto encode=OpenCommands(device.Get());codec.Encode(encode.list.Get(),color.Get(),false,hdr);
    auto* proxy=codec.ModelInput(encode.list.Get());ExecuteAndWait(device.Get(),queue.Get(),encode.list.Get());
    auto values=ReadTexture<Rgba16>(device.Get(),queue.Get(),proxy,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if(clampModel)for(auto& pixel:values){
        for(auto* channel:{&pixel.r,&pixel.g,&pixel.b})*channel=(*channel&0x8000)?0:std::min<std::uint16_t>(*channel,0x3c00);
    }
    if(profile==L"ulp"||profile==L"skin-mask")for(auto& pixel:values){
        if(pixel.r>0&&pixel.r<0x3bff)++pixel.r;
    }
    auto answer=CreateTexture(device.Get(),codec.answer->GetDesc(),D3D12_RESOURCE_STATE_COPY_DEST);
    UploadTexture(device.Get(),queue.Get(),answer.Get(),values);
    auto resolve=OpenCommands(device.Get());
    auto a=Barrier(answer.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
    auto b=Barrier(codec.answer.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_DEST);
    resolve.list->ResourceBarrier(1,&a);resolve.list->ResourceBarrier(1,&b);
    resolve.list->CopyResource(codec.answer.Get(),answer.Get());std::swap(b.Transition.StateBefore,b.Transition.StateAfter);
    auto settings=codec.Settings(hdr,true,1);
    if(profile==L"skin-half"||profile==L"skin-zero"||profile==L"skin-mask"){
        settings.SkinProtection=1;
        settings.SkinDetail=settings.SkinColour=settings.EnvironmentDetail=settings.EnvironmentColour=
            profile==L"skin-zero"?0.f:.5f;
        settings.ShowSkinMask=profile==L"skin-mask"?1u:0u;
    }else if(profile==L"wipe")settings.CompareMode=2;
    else if(profile==L"lighting")settings.Transfer=3;
    else if(profile==L"replace-native"){
        settings.ReversibleMode=2;settings.ReplaceDetailStrength=1;settings.ModelWorkScale=1;
    }else Require(profile==L"default"||profile==L"ulp","unknown profile");
    resolve.list->ResourceBarrier(1,&b);codec.Resolve(resolve.list.Get(),color.Get(),false,hdr,true,1,&settings);
    ExecuteAndWait(device.Get(),queue.Get(),resolve.list.Get());
    const auto result=ReadTexture<Rgba16>(device.Get(),queue.Get(),color.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if(profile==L"default"||profile==L"skin-half"||profile==L"skin-zero"||profile==L"replace-native"||profile==L"lighting")
        Require(std::memcmp(pixels.data(),result.data(),pixels.size()*sizeof(Rgba16))==0,"neutral RGBA changed");
    else if(profile==L"ulp")
        Require(std::memcmp(pixels.data(),result.data(),pixels.size()*sizeof(Rgba16))!=0,"one-ULP edit disappeared");
    else if(profile==L"skin-mask")for(std::size_t i=0;i<result.size();++i){
        const auto& p=result[i];
        Require(p.r==p.g&&p.g==p.b&&p.r<=0x3c00&&p.a==pixels[i].a,"mask has headroom or changed alpha");
    }else if(profile==L"wipe")for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x){
        const auto& p=result[std::size_t(y)*width+x];
        if(std::abs((float(x)+.5f)/float(width)-.5f)<1.f/float(width))
            Require(p.r==0x3c00&&p.g==0x3c00&&p.b==0x3c00,"wipe divider missing");
        else Require(std::memcmp(&p,&pixels[std::size_t(y)*width+x],sizeof(p))==0,"neutral wipe changed RGBA");
    }
    std::ofstream output(std::filesystem::path(argv[4]),std::ios::binary);
    output.write(reinterpret_cast<const char*>(result.data()),result.size()*sizeof(Rgba16));Require(bool(output),"output write");
    ComPtr<ID3D12InfoQueue> info;Hr(device.As(&info),"info queue");
    for(UINT64 i=0;i<info->GetNumStoredMessages();++i){
        SIZE_T size=0;info->GetMessage(i,nullptr,&size);std::vector<unsigned char> data(size);
        auto* m=reinterpret_cast<D3D12_MESSAGE*>(data.data());Hr(info->GetMessage(i,m,&size),"debug read");
        if(m->Severity<=D3D12_MESSAGE_SEVERITY_WARNING){std::fprintf(stderr,"D3D12 %u %s\n",unsigned(m->ID),m->pDescription);throw std::runtime_error("debug clean");}
    }
    std::printf("PASS codec_identity hdr=%u scale=%.2f clamp=%u debug_clean=1\n",hdr?1u:0u,scale,clampModel?1u:0u);return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL codec_identity %s\n",e.what());return 1;}
