#pragma once
#include <d3dcompiler.h>

// A real raster producer, rather than a texture copy. Each frame changes in the
// pixel shader, so stale/replayed inputs cannot pass merely by looking finite.
struct GraphicsProducer {
    ComPtr<ID3D12RootSignature> root,interleavedComputeRoot;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12DescriptorHeap> resources,rtv,decoy;
    void (*restore)(ID3D12GraphicsCommandList*,ID3D12DescriptorHeap*)=nullptr;
    ID3D12Device* device=nullptr;
    ID3D12Resource* source=nullptr;
    unsigned dynamicRaster=0;
    void Open(ID3D12Device* d,ID3D12Resource* scene,unsigned dynamic=0){
        device=d;source=scene;
        dynamicRaster=dynamic;
        if(dynamic&1){D3D12_FEATURE_DATA_D3D12_OPTIONS16 options{};
            Hr(d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16,&options,sizeof(options)),"dynamic depth bias query");
            Require(options.DynamicDepthBiasSupported,"dynamic depth bias support");}
        if(dynamic&2){D3D12_FEATURE_DATA_D3D12_OPTIONS15 options{};
            Hr(d->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS15,&options,sizeof(options)),"dynamic strip cut query");
            Require(options.DynamicIndexBufferStripCutSupported,"dynamic strip cut support");}
        D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,0,0,0};
        D3D12_ROOT_PARAMETER parameters[3]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[0].DescriptorTable={1,&range};
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[1].Constants={0,0,1};
        D3D12_DESCRIPTOR_RANGE unusedRange{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,7,0,0};
        parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[2].DescriptorTable={1,&unusedRange}; // deliberately unused by the shader
        D3D12_ROOT_SIGNATURE_DESC signature{3,parameters,0,nullptr,D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT};
        ComPtr<ID3DBlob> blob,error;
        Hr(D3D12SerializeRootSignature(&signature,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error),"raster root serialize");
        Hr(d->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)),"raster root");
        // Slot 0 deliberately has a different kind in each domain. Sharing a
        // saved table between compute/graphics would restore the wrong API call.
        std::swap(parameters[0],parameters[1]);
        parameters[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[2].Constants={7,0,1};
        Hr(D3D12SerializeRootSignature(&signature,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error),"interleaved root serialize");
        Hr(d->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&interleavedComputeRoot)),"interleaved root");
        constexpr char shader[]=R"(
Texture2D<float4> source : register(t0);
cbuffer Frame : register(b0) { uint frame; }
float4 VS(uint id:SV_VertexID):SV_Position {
    float2 uv=float2((id<<1)&2,id&2);return float4(uv*float2(2,-2)+float2(-1,1),0,1);
}
float4 PS(float4 position:SV_Position):SV_Target {
    float4 c=source.Load(int3(int2(position.xy),0));
    c.r=frac(c.r+frame*0.03125);c.a=1;return c;
})";
        ComPtr<ID3DBlob> vs,ps;
        Hr(D3DCompile(shader,sizeof(shader),nullptr,nullptr,nullptr,"VS","vs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&vs,&error),"raster VS");
        Hr(D3DCompile(shader,sizeof(shader),nullptr,nullptr,nullptr,"PS","ps_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&ps,&error),"raster PS");
        D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
        desc.pRootSignature=root.Get();
        desc.VS={vs->GetBufferPointer(),vs->GetBufferSize()};desc.PS={ps->GetBufferPointer(),ps->GetBufferSize()};
        auto& blend=desc.BlendState.RenderTarget[0];
        blend.SrcBlend=blend.SrcBlendAlpha=D3D12_BLEND_ONE;
        blend.DestBlend=blend.DestBlendAlpha=D3D12_BLEND_ZERO;
        blend.BlendOp=blend.BlendOpAlpha=D3D12_BLEND_OP_ADD;blend.LogicOp=D3D12_LOGIC_OP_NOOP;
        blend.RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
        desc.SampleMask=UINT_MAX;desc.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;
        desc.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;desc.RasterizerState.DepthClipEnable=TRUE;
        desc.DepthStencilState.DepthFunc=D3D12_COMPARISON_FUNC_ALWAYS;
        desc.DepthStencilState.FrontFace={D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_STENCIL_OP_KEEP,D3D12_COMPARISON_FUNC_ALWAYS};
        desc.DepthStencilState.BackFace=desc.DepthStencilState.FrontFace;
        desc.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        desc.NumRenderTargets=1;desc.RTVFormats[0]=DXGI_FORMAT_R16G16B16A16_FLOAT;desc.SampleDesc.Count=1;
        desc.Flags=static_cast<D3D12_PIPELINE_STATE_FLAGS>(((dynamic&1)?D3D12_PIPELINE_STATE_FLAG_DYNAMIC_DEPTH_BIAS:0)|
            ((dynamic&2)?D3D12_PIPELINE_STATE_FLAG_DYNAMIC_INDEX_BUFFER_STRIP_CUT:0));
        Hr(d->CreateGraphicsPipelineState(&desc,IID_PPV_ARGS(&pipeline)),"raster PSO");
        D3D12_DESCRIPTOR_HEAP_DESC heap{};heap.NumDescriptors=1;
        heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Hr(d->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&resources)),"raster SRV heap");
        Hr(d->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&decoy)),"raster decoy heap");
        heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV;heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_NONE;
        Hr(d->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&rtv)),"raster RTV heap");
        D3D12_SHADER_RESOURCE_VIEW_DESC view{};view.Format=scene->GetDesc().Format;
        view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;view.Texture2D.MipLevels=1;
        view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d->CreateShaderResourceView(scene,&view,resources->GetCPUDescriptorHandleForHeapStart());
        view.Shader4ComponentMapping=D3D12_ENCODE_SHADER_4_COMPONENT_MAPPING(
            D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,
            D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_0,D3D12_SHADER_COMPONENT_MAPPING_FORCE_VALUE_1);
        d->CreateShaderResourceView(scene,&view,decoy->GetCPUDescriptorHandleForHeapStart());
    }
    void Draw(ID3D12GraphicsCommandList* list,ID3D12Resource* target,unsigned w,unsigned h,unsigned frame){
        auto barrier=Barrier(target,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_RENDER_TARGET);
        auto sourceBarrier=Barrier(source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1,&sourceBarrier);
        list->ResourceBarrier(1,&barrier);
        device->CreateRenderTargetView(target,nullptr,rtv->GetCPUDescriptorHandleForHeapStart());
        const auto handle=rtv->GetCPUDescriptorHandleForHeapStart();
        list->SetPipelineState(pipeline.Get());
        if(dynamicRaster){ComPtr<ID3D12GraphicsCommandList9> extended;Hr(list->QueryInterface(IID_PPV_ARGS(&extended)),"dynamic raster list9");
            if(dynamicRaster&1)extended->RSSetDepthBias(2.f,.01f,.5f);
            if(dynamicRaster&2)extended->IASetIndexBufferStripCutValue(D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED);}
        if(restore){
            list->SetGraphicsRootSignature(interleavedComputeRoot.Get());
            list->SetGraphicsRoot32BitConstant(2,123,0);
        }
        // A new graphics signature invalidates the unused slot 2. Restoring
        // its old constant after this switch would be an invalid table setter.
        list->SetGraphicsRootSignature(root.Get());
        ID3D12DescriptorHeap* heaps[]={resources.Get()};list->SetDescriptorHeaps(1,heaps);
        list->SetGraphicsRootDescriptorTable(0,resources->GetGPUDescriptorHandleForHeapStart());
        list->SetGraphicsRoot32BitConstant(1,frame,0);
        const D3D12_VIEWPORT viewport{0,0,float(w),float(h),0,1};
        const D3D12_RECT rect{0,0,LONG(w),LONG(h)};
        list->RSSetViewports(1,&viewport);list->RSSetScissorRects(1,&rect);
        list->OMSetRenderTargets(1,&handle,FALSE,nullptr);
        list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        if(restore){
            list->SetComputeRootSignature(interleavedComputeRoot.Get());
            list->SetComputeRoot32BitConstant(0,frame+123,0);
            list->SetGraphicsRootSignature(root.Get()); // same-signature rebind must retain graphics arguments
            restore(list,decoy.Get());
            // Null unbinds the signature too: G -> null -> G must discard a
            // handle from an old heap, even though the final pointer is G again.
            ID3D12DescriptorHeap* oldHeap=decoy.Get();list->SetDescriptorHeaps(1,&oldHeap);
            list->SetGraphicsRootDescriptorTable(2,decoy->GetGPUDescriptorHandleForHeapStart());
            list->SetGraphicsRootSignature(nullptr);
            list->SetGraphicsRootSignature(root.Get());
            list->SetDescriptorHeaps(1,heaps);
            list->SetGraphicsRootDescriptorTable(0,resources->GetGPUDescriptorHandleForHeapStart());
            list->SetGraphicsRoot32BitConstant(1,frame,0);
            restore(list,decoy.Get());
        }
        list->DrawInstanced(3,1,0,0);
        std::swap(barrier.Transition.StateBefore,barrier.Transition.StateAfter);list->ResourceBarrier(1,&barrier);
        std::swap(sourceBarrier.Transition.StateBefore,sourceBarrier.Transition.StateAfter);list->ResourceBarrier(1,&sourceBarrier);
    }
};
