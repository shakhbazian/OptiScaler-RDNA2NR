#pragma once
#include <Windows.h>
#include <d3d12.h>
#include <array>
#include <cstring>
#include <cwchar>

namespace DlssNr::Native {
// Agility can replace a command list's vtable at Reset with direct driver
// entry points. Keep the already observed runtime route for native AMD lists:
// shared driver functions cannot distinguish two different COM method slots.
// Never copy/truncate a vtable or change a wrapper/driver's implementation.
struct RuntimeDispatch {
    ID3D12GraphicsCommandList* owner=nullptr;
    void** table=nullptr;
    unsigned slots=0;
    std::array<void*,86> prefix{};
    bool workGraphsUnsupported=false;

    static HMODULE Module(const void* address)noexcept{
        HMODULE value=nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           static_cast<LPCWSTR>(address),&value);
        return value;
    }
    static bool Named(HMODULE module,const wchar_t* name)noexcept{
        wchar_t path[1024]{};
        if(!module||!GetModuleFileNameW(module,path,1024))return false;
        const auto slash=wcsrchr(path,L'\\');return _wcsicmp(slash?slash+1:path,name)==0;
    }
    static unsigned PublicSlots(ID3D12GraphicsCommandList* list)noexcept{
        // A different tear-off is a different object header. This narrow guard
        // only handles the native interface whose public versions share self.
        const GUID ids[]={__uuidof(ID3D12GraphicsCommandList),__uuidof(ID3D12GraphicsCommandList1),
            __uuidof(ID3D12GraphicsCommandList2),__uuidof(ID3D12GraphicsCommandList3),
            __uuidof(ID3D12GraphicsCommandList4),__uuidof(ID3D12GraphicsCommandList5),
            __uuidof(ID3D12GraphicsCommandList6),__uuidof(ID3D12GraphicsCommandList7),
            __uuidof(ID3D12GraphicsCommandList8),__uuidof(ID3D12GraphicsCommandList9),__uuidof(ID3D12GraphicsCommandList10)};
        constexpr unsigned sizes[]={60,66,67,68,77,79,80,81,82,84,86};
        unsigned count=0;
        for(unsigned i=0;i<std::size(ids);++i){
            void* pointer=nullptr;
            if(FAILED(list->QueryInterface(ids[i],&pointer))||!pointer)continue;
            const bool same=pointer==list;static_cast<IUnknown*>(pointer)->Release();
            if(!same)return 0;count=sizes[i];
        }
        return count;
    }
    static RuntimeDispatch Capture(ID3D12GraphicsCommandList* list)noexcept{
        RuntimeDispatch value;
        auto** candidate=*reinterpret_cast<void***>(list);
        MEMORY_BASIC_INFORMATION region{};
        if(!VirtualQuery(candidate,&region,sizeof(region))||region.Type!=MEM_IMAGE)return value;
        const auto module=Module(candidate);
        if(!Named(module,L"D3D12Core.dll")||Module(candidate[10])!=module)return value;
        const auto count=PublicSlots(list);
        if(count&&*reinterpret_cast<void***>(list)==candidate){
            SIZE_T read=0;
            if(!ReadProcessMemory(GetCurrentProcess(),candidate,value.prefix.data(),count*sizeof(void*),&read)||
               read!=count*sizeof(void*))return {};
            value.owner=list;value.table=candidate;value.slots=count;
            ID3D12Device* device=nullptr;
            if(SUCCEEDED(list->GetDevice(IID_PPV_ARGS(&device)))&&device){
                D3D12_FEATURE_DATA_D3D12_OPTIONS21 options{};
                value.workGraphsUnsupported=SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS21,&options,sizeof(options)))&&
                    options.WorkGraphsTier==D3D12_WORK_GRAPHS_TIER_NOT_SUPPORTED;
                device->Release();
            }
        }return value;
    }
    static bool BypassSlot(unsigned slot)noexcept{
        // Public APIs eligible for Runtime Bypass, from the DirectX spec.
        return (slot>=12&&slot<=14)||(slot>=20&&slot<=25)||(slot>=29&&slot<=46)||
               slot==62||slot==63||slot==65||(slot>=72&&slot<=77)||slot==79||(slot>=81&&slot<=85);
    }
    bool Unchanged(ID3D12GraphicsCommandList* list)const noexcept{
        return list!=owner||!table||(*reinterpret_cast<void***>(list)==table&&Intact());
    }
    bool Intact()const noexcept{
        std::array<void*,86> current{};SIZE_T read=0;
        return ReadProcessMemory(GetCurrentProcess(),table,current.data(),slots*sizeof(void*),&read)&&
            read==slots*sizeof(void*)&&std::memcmp(current.data(),prefix.data(),read)==0;
    }
    bool Restore(ID3D12GraphicsCommandList* list,bool& restored,unsigned& rejectedSlot)const noexcept{
        restored=false;
        rejectedSlot=~0u;
        if(list!=owner||!table)return true;
        if(!Intact())return false;
        auto** current=*reinterpret_cast<void***>(list);
        if(current==table)return true;
        // Read only the negotiated public prefix. The retained original table
        // itself remains whole, including any private/future tail and RTTI.
        std::array<void*,86> snapshot{};SIZE_T read=0;
        if(!ReadProcessMemory(GetCurrentProcess(),current,snapshot.data(),slots*sizeof(void*),&read)||
           read!=slots*sizeof(void*))return false;
        bool driverChanged=false;
        for(unsigned i=0;i<slots;++i){
            if(snapshot[i]==table[i])continue;
            const auto module=Module(snapshot[i]);
            const bool driver=Named(module,L"amdxc64.dll");
            // On this runtime DispatchGraph switches to a Core-owned fallback
            // instead of a driver thunk. Other changed slots must be AMD APIs.
            const bool graphFallback=i==85&&workGraphsUnsupported&&module==Module(table);
            if(!BypassSlot(i)||(!driver&&!graphFallback)){rejectedSlot=i;return false;}
            driverChanged=driverChanged||driver;
        }
        if(!driverChanged)return false; // An unknown wrapper is not a bypass profile.
        // Reset has returned and the recording thread owns this list. Do not
        // overwrite a newer runtime choice if it changed during inspection.
        if(InterlockedCompareExchangePointer(reinterpret_cast<void* volatile*>(list),table,current)!=current)return false;
        restored=true;return true;
    }
};
}
