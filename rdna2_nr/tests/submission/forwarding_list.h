#pragma once
// Source-built COM wrapper with its own identity and forwarding endpoints.
// This models an opaque wrapper; it is not a substitute for a particular
// game's Streamline/ReShade implementation or a native-interface contract.
class ForwardingList final:public ID3D12GraphicsCommandList {
    std::atomic<ULONG> refs{1};
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> original;
    bool exposePrivate;
public:
    explicit ForwardingList(ID3D12GraphicsCommandList* p,bool expose=true):original(p),exposePrivate(expose){}
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid,void** out)override{
        if(!out)return E_POINTER;*out=nullptr;
        if(iid!=__uuidof(IUnknown)&&iid!=__uuidof(ID3D12Object)&&iid!=__uuidof(ID3D12DeviceChild)&&
           iid!=__uuidof(ID3D12CommandList)&&iid!=__uuidof(ID3D12GraphicsCommandList))return E_NOINTERFACE;
        *out=static_cast<ID3D12GraphicsCommandList*>(this);AddRef();return S_OK;
    }
    ULONG STDMETHODCALLTYPE AddRef()override{return ++refs;}
    ULONG STDMETHODCALLTYPE Release()override{const auto n=--refs;if(!n)delete this;return n;}
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID g,UINT* n,void* p)override{
        return exposePrivate?original->GetPrivateData(g,n,p):DXGI_ERROR_NOT_FOUND;
    }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID g,UINT n,const void* p)override{return original->SetPrivateData(g,n,p);}
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID g,const IUnknown* p)override{return original->SetPrivateDataInterface(g,p);}
    HRESULT STDMETHODCALLTYPE SetName(LPCWSTR n)override{return original->SetName(n);}
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID iid,void** out)override{return original->GetDevice(iid,out);}
    D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE GetType()override{return original->GetType();}
    HRESULT STDMETHODCALLTYPE Close()override{return original->Close();}
    HRESULT STDMETHODCALLTYPE Reset(ID3D12CommandAllocator* a,ID3D12PipelineState* p)override{return original->Reset(a,p);}
    template<class F>void Record(F&& fn){fn(original.Get());}
#include "native_shadow_forwarders.inl"
};
