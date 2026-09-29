#pragma once
// Independent orchestration oracle: the committed OptiScaler codec shader,
// explicit per-flight resources, and no native recorder or production RunPass.
#pragma warning(push)
#pragma warning(disable:4324) // Intentional D3D12 256-byte constant-buffer alignment.
#include "../../OptiScaler/shaders/dlssnr/DlssNr_Common.h"
#pragma warning(pop)
#include "../../OptiScaler/shaders/dlssnr/precompile/DlssNr_Shader.h"

struct CommonColorReference {
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12DescriptorHeap> heap;
    std::array<ComPtr<ID3D12Resource>,2> constants;
    ComPtr<ID3D12Resource> proxy,original,answer,composed;
    ID3D12Device* device=nullptr;unsigned width=0,height=0,stride=0;
    ID3D12Resource* exposure=nullptr;
    void Open(ID3D12Device* d,unsigned w,unsigned h,DXGI_FORMAT format=DXGI_FORMAT_R16G16B16A16_FLOAT){
        if(device)return;device=d;width=w;height=h;
        D3D12_DESCRIPTOR_RANGE ranges[3]{};
        ranges[0]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,5,0,0,0};
        ranges[1]={D3D12_DESCRIPTOR_RANGE_TYPE_UAV,2,0,0,5};
        ranges[2]={D3D12_DESCRIPTOR_RANGE_TYPE_CBV,1,0,0,7};
        D3D12_ROOT_PARAMETER parameter{};parameter.ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable={3,ranges};parameter.ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
        D3D12_STATIC_SAMPLER_DESC sampler{};sampler.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        sampler.AddressU=sampler.AddressV=sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
        sampler.MaxLOD=D3D12_FLOAT32_MAX;sampler.ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
        D3D12_ROOT_SIGNATURE_DESC desc{};desc.NumParameters=1;desc.pParameters=&parameter;
        desc.NumStaticSamplers=1;desc.pStaticSamplers=&sampler;
        ComPtr<ID3DBlob> blob,error;
        Hr(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error),"reference codec root serialize");
        Hr(d->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)),"reference codec root");
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};pso.pRootSignature=root.Get();pso.CS={DlssNr_cso,sizeof(DlssNr_cso)};
        Hr(d->CreateComputePipelineState(&pso,IID_PPV_ARGS(&pipeline)),"reference codec pipeline");
        D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        hd.NumDescriptors=16;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Hr(d->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&heap)),"reference codec heap");
        stride=d->GetDescriptorHandleIncrementSize(hd.Type);
        D3D12_HEAP_PROPERTIES upload{};upload.Type=D3D12_HEAP_TYPE_UPLOAD;
        const auto cb=BufferDesc(sizeof(DlssNrConstants));
        for(auto& buffer:constants)Hr(d->CreateCommittedResource(&upload,D3D12_HEAP_FLAG_NONE,&cb,
            D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&buffer)),"reference codec constants");
        auto td=TextureDesc(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        proxy=CreateTexture(d,td,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        answer=CreateTexture(d,td,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        td.Format=format;
        original=CreateTexture(d,td,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        composed=CreateTexture(d,td,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    void Dispatch(ID3D12GraphicsCommandList* list,unsigned slot,const DlssNrConstants& value,
                  ID3D12Resource* source,ID3D12Resource* model,ID3D12Resource* base,
                  ID3D12Resource* target,ID3D12Resource* keep,ID3D12Resource* exposureInput=nullptr){
        auto cpu=heap->GetCPUDescriptorHandleForHeapStart();cpu.ptr+=SIZE_T(slot*8)*stride;
        // The current shader reads its exposure input through t3 (gMotion).
        // The older fork used t4; keep this independent binding aligned with
        // the audited shader contract rather than with the old pilot layout.
        ID3D12Resource* srvs[]={source,model?model:source,base?base:source,
                                exposureInput?exposureInput:source,source};
        for(unsigned i=0;i<5;++i){D3D12_SHADER_RESOURCE_VIEW_DESC srv{};srv.Format=srvs[i]->GetDesc().Format;
            srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels=1;device->CreateShaderResourceView(srvs[i],&srv,cpu);cpu.ptr+=stride;}
        for(auto* resource:{target,keep?keep:target}){D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
            uav.Format=resource->GetDesc().Format;uav.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
            device->CreateUnorderedAccessView(resource,nullptr,&uav,cpu);cpu.ptr+=stride;}
        void* mapped=nullptr;Hr(constants[slot]->Map(0,nullptr,&mapped),"reference codec map");
        std::memcpy(mapped,&value,sizeof(value));constants[slot]->Unmap(0,nullptr);
        D3D12_CONSTANT_BUFFER_VIEW_DESC cbv{constants[slot]->GetGPUVirtualAddress(),sizeof(DlssNrConstants)};
        device->CreateConstantBufferView(&cbv,cpu);
        auto gpu=heap->GetGPUDescriptorHandleForHeapStart();gpu.ptr+=UINT64(slot*8)*stride;
        ID3D12DescriptorHeap* heaps[]={heap.Get()};list->SetDescriptorHeaps(1,heaps);
        list->SetComputeRootSignature(root.Get());list->SetPipelineState(pipeline.Get());
        list->SetComputeRootDescriptorTable(0,gpu);list->Dispatch((width+7)/8,(height+7)/8,1);
    }
    DlssNrConstants Settings(bool hdr,bool apply,float strength){
        DlssNrConstants c{};c.WhitePoint=1;c.Width=width;c.Height=height;
        c.TransferStrength=strength;c.ColourStrength=1;c.MaxRatio=2;c.Passthrough=hdr?0u:1u;
        c.Transfer=1;c.DebugScale=1;c.CompareSplit=.5f;c.CompareZoom=1;c.ApplyModel=apply?1u:0u;
        c.ExposureMode=exposure?1u:0u;c.PreExposure=1.f;c.ExposureTrim=1.f;
        c.SkinDetail=c.SkinColour=c.EnvironmentDetail=c.EnvironmentColour=1;return c;
    }
    void Encode(ID3D12GraphicsCommandList* list,ID3D12Resource* color,bool post,bool hdr){
        if(post){auto b=Barrier(color,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);list->ResourceBarrier(1,&b);}
        auto p=Barrier(proxy.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto o=Barrier(original.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        list->ResourceBarrier(1,&p);list->ResourceBarrier(1,&o);
        auto c=Settings(hdr,true,1);c.Mode=DlssNrMode_Encode;
        Dispatch(list,0,c,color,nullptr,nullptr,proxy.Get(),original.Get(),exposure);
        std::swap(p.Transition.StateBefore,p.Transition.StateAfter);std::swap(o.Transition.StateBefore,o.Transition.StateAfter);
        list->ResourceBarrier(1,&p);list->ResourceBarrier(1,&o);
    }
    void Resolve(ID3D12GraphicsCommandList* list,ID3D12Resource* color,bool post,bool hdr,bool apply=true,float strength=1){
        auto ready=Barrier(answer.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->ResourceBarrier(1,&ready);
        auto c=Settings(hdr,apply,strength);c.Mode=DlssNrMode_Resolve;
        Dispatch(list,1,c,proxy.Get(),answer.Get(),original.Get(),composed.Get(),nullptr,exposure);
        auto a=Barrier(composed.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto b=Barrier(color,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_DEST);
        list->ResourceBarrier(1,&a);list->ResourceBarrier(1,&b);list->CopyResource(color,composed.Get());
        std::swap(a.Transition.StateBefore,a.Transition.StateAfter);
        b.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.StateAfter=post?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        list->ResourceBarrier(1,&a);list->ResourceBarrier(1,&b);
        std::swap(ready.Transition.StateBefore,ready.Transition.StateAfter);list->ResourceBarrier(1,&ready);
    }
};
