#pragma once
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <stdexcept>

namespace DlssNr::Native {
// Each instance belongs to one retired color workspace, so its descriptors
// cannot be overwritten by another in-flight frame. This is an exact gather,
// matching prepare_features' integer center sampling, not motion filtering.
class MotionReducer {
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    Microsoft::WRL::ComPtr<ID3D12RootSignature> root;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptors;
    Microsoft::WRL::ComPtr<ID3D12Resource> output;
    unsigned width=0,height=0;
    static void Check(HRESULT hr){if(FAILED(hr))throw std::runtime_error("native motion gather creation");}
public:
    void Open(ID3D12Device* d){
        if(device)return;
        constexpr char shader[]=R"(
Texture2D<float2> source : register(t0);
RWTexture2D<float2> target : register(u0);
cbuffer Shape : register(b0) { uint width; uint height; uint sourceWidth; uint sourceHeight; };
[numthreads(8,8,1)] void main(uint3 id:SV_DispatchThreadID) {
    if(id.x>=width||id.y>=height)return;
    uint2 p=uint2((2*id.x+1)*sourceWidth/(2*width),(2*id.y+1)*sourceHeight/(2*height));
    target[id.xy]=source.Load(int3(p,0));
})";
        Microsoft::WRL::ComPtr<ID3DBlob> code,error,signature;
        Check(D3DCompile(shader,sizeof(shader)-1,nullptr,nullptr,nullptr,"main","cs_5_0",
                         D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&error));
        D3D12_DESCRIPTOR_RANGE ranges[2]={{D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,0,0,0},
                                        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV,1,0,0,1}};
        D3D12_ROOT_PARAMETER parameters[2]{};
        parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[0].DescriptorTable={2,ranges};
        parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[1].Constants={0,0,4};
        D3D12_ROOT_SIGNATURE_DESC desc{};desc.NumParameters=2;desc.pParameters=parameters;
        Check(D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&signature,&error));
        Check(d->CreateRootSignature(0,signature->GetBufferPointer(),signature->GetBufferSize(),IID_PPV_ARGS(&root)));
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};pso.pRootSignature=root.Get();
        pso.CS={code->GetBufferPointer(),code->GetBufferSize()};
        Check(d->CreateComputePipelineState(&pso,IID_PPV_ARGS(&pipeline)));
        D3D12_DESCRIPTOR_HEAP_DESC heap{};heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap.NumDescriptors=2;heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        Check(d->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&descriptors)));device=d;
    }
    ID3D12Resource* Record(ID3D12GraphicsCommandList* list,ID3D12Resource* source,unsigned w,unsigned h){
        if(!output||width!=w||height!=h){
            output.Reset();width=w;height=h;
            D3D12_RESOURCE_DESC desc{};desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width=w;desc.Height=h;desc.DepthOrArraySize=1;desc.MipLevels=1;
            desc.Format=DXGI_FORMAT_R16G16_FLOAT;desc.SampleDesc.Count=1;desc.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
            Check(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&output)));
            output->SetName(L"NR exact motion gather");
        }
        auto cpu=descriptors->GetCPUDescriptorHandleForHeapStart();
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};srv.Format=DXGI_FORMAT_R16G16_FLOAT;
        srv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;srv.Texture2D.MipLevels=1;
        srv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        device->CreateShaderResourceView(source,&srv,cpu);
        cpu.ptr+=device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};uav.Format=DXGI_FORMAT_R16G16_FLOAT;uav.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
        device->CreateUnorderedAccessView(output.Get(),nullptr,&uav,cpu);
        ID3D12DescriptorHeap* heaps[]={descriptors.Get()};list->SetDescriptorHeaps(1,heaps);
        list->SetComputeRootSignature(root.Get());list->SetPipelineState(pipeline.Get());
        list->SetComputeRootDescriptorTable(0,descriptors->GetGPUDescriptorHandleForHeapStart());
        const auto sourceDesc=source->GetDesc();const unsigned shape[]={w,h,unsigned(sourceDesc.Width),sourceDesc.Height};
        list->SetComputeRoot32BitConstants(1,4,shape,0);list->Dispatch((w+7)/8,(h+7)/8,1);
        return output.Get();
    }
};
}
