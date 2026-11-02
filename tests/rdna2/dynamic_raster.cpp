// Independent depth readbacks make omitted/reordered raster state observable.
#define NR_RUNTIME_DLL_PROBE_HELPERS_ONLY
#include "../../rdna2_nr/src/runtime_v2_dll_probe.cpp"
#include <dlssnr/native/NativeRecording.h>
#include <d3dcompiler.h>
#include <d3d12sdklayers.h>
#include <cmath>
#ifdef NR_HOST_AGILITY_SDK_VERSION
extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion=NR_HOST_AGILITY_SDK_VERSION;
__declspec(dllexport) extern const char* D3D12SDKPath=".\\AgilitySDK\\";
}
#endif
namespace {
constexpr unsigned Size=32;
struct Bias {int value;float clamp,slope;};
constexpr Bias Zero{0,0,0},Positive{524288,.03125f,.5f},Negative{-262144,0,-.5f};
struct Fixture {
    ComPtr<ID3D12Device> device;ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12RootSignature> root;ComPtr<ID3D12DescriptorHeap> dsv;
    ComPtr<ID3D12Resource> indices;
    ComPtr<ID3DBlob> vertex;
    Fixture(){
        ComPtr<ID3D12Debug> debug;Hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),"debug interface");debug->EnableDebugLayer();
        Hr(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&device)),"device");
        D3D12_COMMAND_QUEUE_DESC q{};Hr(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)),"queue");
        D3D12_FEATURE_DATA_D3D12_OPTIONS16 depth{};D3D12_FEATURE_DATA_D3D12_OPTIONS15 strip{};
        Hr(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16,&depth,sizeof(depth)),"depth query");
        Hr(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS15,&strip,sizeof(strip)),"strip query");
        Require(depth.DynamicDepthBiasSupported&&strip.DynamicIndexBufferStripCutSupported,"dynamic raster support");
        D3D12_ROOT_SIGNATURE_DESC signature{};signature.Flags=D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
        ComPtr<ID3DBlob> blob,error;Hr(D3D12SerializeRootSignature(&signature,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error),"serialize root");
        Hr(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)),"root");
        constexpr char shader[]=R"(
float4 VS(uint id:SV_VertexID):SV_Position {
    if(id==65535)return float4(0,0,.5,1);
    float2 p=float2(float(id&1)*2-1,float(id>>1)*2-1);
    return float4(p,.5+p.x*.03125,1);
})";
        Hr(D3DCompile(shader,sizeof(shader),nullptr,nullptr,nullptr,"VS","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&vertex,&error),"VS");
        D3D12_DESCRIPTOR_HEAP_DESC heap{};heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_DSV;heap.NumDescriptors=1;
        Hr(device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&dsv)),"DSV heap");
        const auto desc=BufferDesc(10);D3D12_HEAP_PROPERTIES upload{};upload.Type=D3D12_HEAP_TYPE_UPLOAD;
        Hr(device->CreateCommittedResource(&upload,D3D12_HEAP_FLAG_NONE,&desc,D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,IID_PPV_ARGS(&indices)),"index buffer");
        const unsigned short data[]={0,1,65535,2,3};void* mapped=nullptr;
        Hr(indices->Map(0,nullptr,&mapped),"index map");std::memcpy(mapped,data,sizeof(data));indices->Unmap(0,nullptr);
    }
    ComPtr<ID3D12PipelineState> Pipeline(Bias bias,bool cut,unsigned dynamic,bool variant=false){
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};desc.pRootSignature=root.Get();
        desc.VS={vertex->GetBufferPointer(),vertex->GetBufferSize()};
        desc.SampleMask=variant?(UINT_MAX&~2u):UINT_MAX;desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;desc.RasterizerState.DepthClipEnable=TRUE;
        desc.RasterizerState.DepthBias=bias.value;desc.RasterizerState.DepthBiasClamp=bias.clamp;
        desc.RasterizerState.SlopeScaledDepthBias=bias.slope;
        desc.DepthStencilState.DepthEnable=TRUE;desc.DepthStencilState.DepthWriteMask=D3D12_DEPTH_WRITE_MASK_ALL;
        desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
        desc.DepthStencilState.FrontFace={D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_COMPARISON_FUNC_ALWAYS};
        desc.DepthStencilState.BackFace=desc.DepthStencilState.FrontFace;
        desc.DSVFormat=DXGI_FORMAT_D32_FLOAT;desc.SampleDesc.Count=1;desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.IBStripCutValue=cut?D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF:D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED;
        desc.Flags=static_cast<D3D12_PIPELINE_STATE_FLAGS>(((dynamic&1)?D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS:0)|
            ((dynamic&2)?D3D12_PIPELINE_STATE_FLAG_DYNAMIC_INDEX_BUFFER_STRIP_CUT:0));
        ComPtr<ID3D12PipelineState> result;Hr(device->CreateGraphicsPipelineState(&desc,IID_PPV_ARGS(&result)),"depth PSO");return result;
    }
    // mode: 0 static oracle, 1 override, 2 same-PSO reset, 3 other-PSO reset,
    // 4 ClearState, 5 suffix override, 6 same-PSO reset after the split,
    // 7 materialize at Draw then rebind, 8 different PSO with same raster defaults.
    std::vector<float> Draw(unsigned dynamic,unsigned mode,bool mirrored,Bias expected,bool expectedCut){
        auto pipeline=Pipeline(mode?Zero:expected,mode?false:expectedCut,mode?dynamic:0);
        auto other=Pipeline(mode==8?Zero:Negative,false,dynamic,mode==8);
        auto desc=TextureDesc(Size,Size,DXGI_FORMAT_D32_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
        D3D12_CLEAR_VALUE clear{};clear.Format=DXGI_FORMAT_D32_FLOAT;clear.DepthStencil.Depth=1;
        ComPtr<ID3D12Resource> target;Hr(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
            D3D12_RESOURCE_STATE_DEPTH_WRITE,&clear,IID_PPV_ARGS(&target)),"depth target");
        device->CreateDepthStencilView(target.Get(),nullptr,dsv->GetCPUDescriptorHandleForHeapStart());
        auto original=OpenCommands(device.Get());ComPtr<ID3D12GraphicsCommandList9> list;
        Hr(original.list.As(&list),"list9");
        std::unique_ptr<DlssNr::Native::Recording> recording;
        if(mirrored)recording=std::make_unique<DlssNr::Native::Recording>(device.Get(),1);
        auto emit=[&](auto call){if(recording)call(recording.get());else call(list.Get());};
        const auto handle=dsv->GetCPUDescriptorHandleForHeapStart();
        const D3D12_VIEWPORT viewport{0,0,float(Size),float(Size),0,1};const D3D12_RECT rect{0,0,LONG(Size),LONG(Size)};
        const D3D12_INDEX_BUFFER_VIEW view{indices->GetGPUVirtualAddress(),10,DXGI_FORMAT_R16_UINT};
        auto bindings=[&]{
            emit([&](auto* p){p->SetGraphicsRootSignature(root.Get());p->RSSetViewports(1,&viewport);p->RSSetScissorRects(1,&rect);
                p->OMSetRenderTargets(0,nullptr,FALSE,&handle);p->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
                p->IASetIndexBuffer(&view);});
        };
        emit([&](auto* p){p->SetPipelineState(pipeline.Get());p->ClearDepthStencilView(handle,D3D12_CLEAR_FLAG_DEPTH,1,0,0,nullptr);});
        bindings();
        auto overrideState=[&]{emit([&](auto* p){
            if(dynamic&1)p->RSSetDepthBias(float(Positive.value),Positive.clamp,Positive.slope);
            if(dynamic&2)p->IASetIndexBufferStripCutValue(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF);
        });};
        if(mode&&mode!=5)overrideState();
        if(mode==2)emit([&](auto* p){p->SetPipelineState(pipeline.Get());});
        if(mode==3||mode==8)emit([&](auto* p){p->SetPipelineState(other.Get());});
        if(mode==4){emit([&](auto* p){p->ClearState(pipeline.Get());});bindings();}
        if(mode==7)emit([&](auto* p){p->DrawIndexedInstanced(5,1,0,0,0);
            p->ClearDepthStencilView(handle,D3D12_CLEAR_FLAG_DEPTH,1,0,0,nullptr);p->SetPipelineState(pipeline.Get());});
        if(recording)Require(recording->Cut(),"raster state accepted at split");
        if(mode==5)overrideState();
        if(mode==6)emit([&](auto* p){p->SetPipelineState(pipeline.Get());});
        emit([](auto* p){p->DrawIndexedInstanced(5,1,0,0,0);});
        if(recording){
            recording->Close(true);Require(recording->Ready(),"raster recording ready");
            ID3D12CommandList* batch[]={recording->prefix->list.Get(),recording->suffix->list.Get()};
            queue->ExecuteCommandLists(2,batch);
            ComPtr<ID3D12Fence> fence;Hr(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"fence");
            Hr(queue->Signal(fence.Get(),1),"signal");CpuWait(fence.Get(),1);
        }else ExecuteAndWait(device.Get(),queue.Get(),original.list.Get());
        return ReadTexture<float>(device.Get(),queue.Get(),target.Get(),D3D12_RESOURCE_STATE_DEPTH_WRITE);
    }
    void Clean(){
        ComPtr<ID3D12InfoQueue> info;Hr(device.As(&info),"info queue");
        for(UINT64 i=0;i<info->GetNumStoredMessages();++i){
            SIZE_T size=0;info->GetMessage(i,nullptr,&size);std::vector<unsigned char> bytes(size);
            auto* message=reinterpret_cast<D3D12_MESSAGE*>(bytes.data());Hr(info->GetMessage(i,message,&size),"debug read");
            if(message->Severity<=D3D12_MESSAGE_SEVERITY_WARNING){std::fprintf(stderr,"D3D12 id=%u %s\n",unsigned(message->ID),message->pDescription);
                throw std::runtime_error("dynamic raster debug validation");}
        }
    }
};
}
int main()try{
    Fixture fixture;
    const auto zero=fixture.Draw(0,0,false,Zero,false);
    const auto biased=fixture.Draw(0,0,false,Positive,false);
    const auto cut=fixture.Draw(0,0,false,Zero,true);
    Require(zero!=biased&&zero!=cut,"static oracles must distinguish state");
    unsigned specDifferences=0;
    for(unsigned dynamic:{1u,2u,3u})for(unsigned mode=1;mode<=8;++mode){
        const bool overrides=mode==1||mode==5;
        const Bias expected=mode==3?Negative:overrides&&(dynamic&1)?Positive:Zero;
        const bool expectedCut=overrides&&(dynamic&2);
        const auto reference=fixture.Draw(0,0,false,expected,expectedCut);
        const auto direct=fixture.Draw(dynamic,mode,false,Zero,false);
        const auto replay=fixture.Draw(dynamic,mode,true,Zero,false);
        if(direct!=replay){
            for(unsigned i=0;i<direct.size();++i)if(direct[i]!=replay[i]){
                std::fprintf(stderr,"RASTER_DIFF mask=%u mode=%u pixel=%u direct=%.9g replay=%.9g\n",dynamic,mode,i,direct[i],replay[i]);break;
            }
        }
        Require(direct==replay,"split replay lost dynamic raster state");
        if(overrides)Require(reference==direct,"single dynamic override differs from static PSO oracle");
        else if(reference!=direct)++specDifferences;
        fixture.Clean();std::printf("PASS dynamic_raster mask=%u mode=%u direct_split_exact=1 static_default_match=%u\n",dynamic,mode,reference==direct?1u:0u);
    }
    std::printf("PASS dynamic_raster cases=24 static_override_oracle=1 spec_default_differences=%u pso_rebind=1 clear_state=1 suffix=1 debug_clean=1\n",specDifferences);return 0;
}catch(const std::exception& error){std::fprintf(stderr,"FAIL dynamic_raster %s\n",error.what());return 1;}
