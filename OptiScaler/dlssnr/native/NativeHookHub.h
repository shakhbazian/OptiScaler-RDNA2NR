#pragma once

#include <Windows.h>
#include <TlHelp32.h>
#include <detours/detours.h>
#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <type_traits>
#include <vector>

// One physical detour per endpoint. Existing OptiScaler hooks are ordered
// listeners; their original pointers become typed continuation thunks. A direct
// call of a continuation (OptiScaler state restoration) still reaches the
// observer, without re-entering legacy state trackers.
namespace DlssNr::NativeHooks {
enum class Origin { Game, Restore, Private };
inline thread_local Origin origin = Origin::Game;
struct OriginScope {
    Origin previous;
    explicit OriginScope(Origin value) noexcept : previous(origin) { origin = value; }
    ~OriginScope() { origin = previous; }
};
using SameObject = bool (*)(const void*, const void*) noexcept;
inline std::atomic<SameObject> sameObject{nullptr};
inline bool Same(const void* a, const void* b) noexcept {
    if (a == b) return true;
    const auto fn = sameObject.load(std::memory_order_acquire);
    return fn && fn(a, b);
}

struct Setup {
    std::mutex mutex;
    std::map<void*, unsigned> signatures;
};
inline Setup& State() { static auto* value = new Setup; return *value; }
inline thread_local bool transaction = false;
inline thread_local std::vector<HANDLE> enlisted;
inline thread_local std::vector<std::function<void(bool)>> changes;
inline thread_local LONG setupError = NO_ERROR;

inline LONG Begin() {
    if (transaction) return ERROR_INVALID_OPERATION;
    State().mutex.lock();
    const auto result = DetourTransactionBegin();
    if (result != NO_ERROR) { State().mutex.unlock(); return result; }
    transaction = true; setupError = NO_ERROR;
    DetourUpdateThread(GetCurrentThread());
    return NO_ERROR;
}
inline LONG Commit() {
    if (!transaction) return ERROR_INVALID_OPERATION;
    // Collect handles before suspending any thread: do not allocate from the
    // application CRT heap while a suspended thread could own its lock.
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) setupError = GetLastError();
    else {
        THREADENTRY32 item{}; item.dwSize = sizeof(item);
        if (Thread32First(snapshot, &item)) do {
            if (item.th32OwnerProcessID != GetCurrentProcessId() ||
                item.th32ThreadID == GetCurrentThreadId()) continue;
            HANDLE thread = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
                                       THREAD_SET_CONTEXT | THREAD_QUERY_INFORMATION,
                                       FALSE, item.th32ThreadID);
            if (!thread) {
                // A thread may have ended between enumeration and OpenThread.
                if (GetLastError() != ERROR_INVALID_PARAMETER) setupError = GetLastError();
                continue;
            }
            enlisted.push_back(thread);
        } while (Thread32Next(snapshot, &item));
        CloseHandle(snapshot);
    }
    for (HANDLE thread : enlisted) {
        const auto error = DetourUpdateThread(thread);
        if (error != NO_ERROR) setupError = error;
    }
    LONG result = setupError;
    if (result == NO_ERROR) {
        // Publish listeners while enlisted threads are still suspended. Detours
        // resumes them inside Commit, so publication afterwards leaves a gap.
        for (auto& change : changes) change(true);
        result = DetourTransactionCommit();
        if (result != NO_ERROR) for (auto& change : changes) change(false);
    } else {
        DetourTransactionAbort();
        for (auto& change : changes) change(false);
    }
    changes.clear();
    for (HANDLE thread : enlisted) CloseHandle(thread);
    enlisted.clear(); transaction = false;
    State().mutex.unlock();
    return result;
}
inline void Abort() noexcept {
    if(!transaction)return;
    DetourTransactionAbort();
    for(auto& change:changes)change(false);
    changes.clear();for(HANDLE thread:enlisted)CloseHandle(thread);
    enlisted.clear();transaction=false;State().mutex.unlock();
}
inline LONG Signature(void* address, unsigned tag) {
    const auto [it, inserted] = State().signatures.emplace(address, tag);
    if (!inserted && it->second != tag) return ERROR_INVALID_FUNCTION;
    return NO_ERROR;
}
inline bool Pin() {
    HMODULE module = nullptr;
    return GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_PIN,
                             reinterpret_cast<LPCWSTR>(&Pin), &module) != 0;
}

template<unsigned Tag, class Fn> struct Method;
template<class R,class Self,class... A> struct ObserverSignature {using Type=void(*)(Self,A...,R);};
template<class Self,class... A> struct ObserverSignature<void,Self,A...> {using Type=void(*)(Self,A...);};
// The common implementation keeps HRESULT/void observation typed without ever
// packing call arguments into an untyped buffer.
template<unsigned Tag, class R, class Self, class... A>
struct MethodBody {
    using Fn = R(STDMETHODCALLTYPE*)(Self, A...);
    using Observer = typename ObserverSignature<R,Self,A...>::Type;
    static constexpr unsigned MaxEndpoints = 4, MaxListeners = 4;
    struct Endpoint {
        void* address = nullptr;
        Fn original = nullptr;
        std::atomic<bool> installed{false};
        std::array<std::atomic<Fn>,MaxListeners> listeners{};
        std::array<Fn*,MaxListeners> slots{};
    };
    static inline std::array<Endpoint,MaxEndpoints> endpoints{};
    static inline std::atomic<Observer> observer{nullptr};
    struct Invocation { unsigned endpoint; Self self; Invocation* previous; };
    static inline thread_local Invocation* active = nullptr;

    static R Original(unsigned e, Self self, A... args) {
        return endpoints[e].original(self, args...);
    }
    static R Next(unsigned e, int last, Self self, A... args) {
        if (origin == Origin::Game)
            for (int i=last;i>=0;--i)
                if (const auto fn=endpoints[e].listeners[i].load(std::memory_order_acquire))
                    return fn(self,args...);
        return Original(e,self,args...);
    }
    template<class Call> static R Observed(unsigned e, Self self, Call&& call, A... args) {
        Invocation invocation{e,self,active};
        bool nested = false;
        for (auto* p=active;p;p=p->previous)
            if (Same(self,p->self)) { nested=true;break; }
        struct Guard {
            Invocation* previous;
            explicit Guard(Invocation* value):previous(active){active=value;}
            ~Guard(){active=previous;}
        } guard(&invocation);
        const auto observe = (origin != Origin::Private && !nested)
            ? observer.load(std::memory_order_acquire) : nullptr;
        if constexpr (std::is_void_v<R>) {
            call(); if(observe)observe(self,args...);
        } else {
            const auto result=call(); if(observe)observe(self,args...,result);return result;
        }
    }
    template<unsigned E> static R STDMETHODCALLTYPE Entry(Self self,A... args) {
        return Observed(E,self,[&] { return Next(E,int(MaxListeners)-1,self,args...); },args...);
    }
    template<unsigned E,unsigned L> static R STDMETHODCALLTYPE Continue(Self self,A... args) {
        if(active && active->endpoint==E && Same(self,active->self))
            return Next(E,int(L)-1,self,args...);
        // Original-bypass calls restore state. Observe them exactly once, with
        // no callbacks to the legacy tracker whose mutex may guard the snapshot.
        OriginScope scope(origin==Origin::Private?Origin::Private:Origin::Restore);
        return Observed(E,self,[&] { return Original(E,self,args...); },args...);
    }
    template<unsigned E> static Fn Continuation(unsigned l) {
        switch(l) {
        case 0:return Continue<E,0>;case 1:return Continue<E,1>;
        case 2:return Continue<E,2>;case 3:return Continue<E,3>;
        default:return nullptr;
        }
    }
    static Fn Continuation(unsigned e,unsigned l) {
        switch(e) {
        case 0:return Continuation<0>(l);case 1:return Continuation<1>(l);
        case 2:return Continuation<2>(l);case 3:return Continuation<3>(l);
        default:return nullptr;
        }
    }
    static Fn EntryPoint(unsigned e) {
        switch(e) {
        case 0:return Entry<0>;case 1:return Entry<1>;
        case 2:return Entry<2>;case 3:return Entry<3>;
        default:return nullptr;
        }
    }
    static LONG Ensure(void* address,unsigned& index) {
        if(!transaction||!address)return ERROR_INVALID_PARAMETER;
        const auto signature=Signature(address,Tag);
        if(signature!=NO_ERROR){setupError=signature;return signature;}
        for(index=0;index<MaxEndpoints;++index)
            if(endpoints[index].address==address) return NO_ERROR;
        for(index=0;index<MaxEndpoints;++index)if(!endpoints[index].address)break;
        if(index==MaxEndpoints){setupError=ERROR_TOO_MANY_NAMES;return setupError;}
        if(!Pin()){setupError=GetLastError();return setupError;}
        auto& e=endpoints[index];e.address=address;e.original=reinterpret_cast<Fn>(address);
        const auto result=DetourAttach(reinterpret_cast<void**>(&e.original),
                                      reinterpret_cast<void*>(EntryPoint(index)));
        if(result!=NO_ERROR)setupError=result;
        changes.emplace_back([index](bool ok){
            auto& endpoint=endpoints[index];endpoint.installed.store(ok,std::memory_order_release);
            if(!ok){State().signatures.erase(endpoint.address);endpoint.address=nullptr;endpoint.original=nullptr;}
        });
        return result;
    }
    static LONG Ensure(void* address) {unsigned index=0;return Ensure(address,index);}
    static LONG Attach(Fn* original,Fn listener) {
        if(!original||!listener){setupError=ERROR_INVALID_PARAMETER;return setupError;}
        unsigned index=0;const auto result=Ensure(reinterpret_cast<void*>(*original),index);
        if(result!=NO_ERROR)return result;
        auto& e=endpoints[index];unsigned l=0;
        for(;l<MaxListeners;++l)if(!e.slots[l]||e.slots[l]==original)break;
        if(l==MaxListeners){setupError=ERROR_TOO_MANY_NAMES;return setupError;}
        const auto previous=*original;e.slots[l]=original;*original=Continuation(index,l);
        changes.emplace_back([index,l,original,previous,listener](bool ok){
            if(ok)endpoints[index].listeners[l].store(listener,std::memory_order_release);
            else{*original=previous;endpoints[index].slots[l]=nullptr;}
        });
        return NO_ERROR;
    }
    static LONG Detach(Fn* original,Fn) {
        if(!transaction)return ERROR_INVALID_OPERATION;
        for(unsigned e=0;e<MaxEndpoints;++e)for(unsigned l=0;l<MaxListeners;++l)
            if(endpoints[e].slots[l]==original){
                const auto listener=endpoints[e].listeners[l].load();
                const auto continuation=*original;
                changes.emplace_back([e,l,original,listener,continuation](bool ok){if(ok){
                    endpoints[e].listeners[l].store(nullptr,std::memory_order_release);
                    *original=endpoints[e].original;endpoints[e].slots[l]=nullptr;
                }else{*original=continuation;endpoints[e].slots[l]=original;
                    endpoints[e].listeners[l].store(listener,std::memory_order_release);}});return NO_ERROR;
            }
        return ERROR_INVALID_HANDLE;
    }
    static bool Covered(void* address) noexcept {
        for(auto& e:endpoints)if(e.installed.load(std::memory_order_acquire)&&e.address==address)return true;
        return false;
    }
};

template<unsigned Tag,class R,class Self,class... A>
struct Method<Tag,R(STDMETHODCALLTYPE*)(Self,A...)> : MethodBody<Tag,R,Self,A...> {};
template<unsigned Tag,class Fn> LONG Attach(Fn* original,Fn listener) {
    return Method<Tag,Fn>::Attach(original,listener);
}
template<unsigned Tag,class Fn> LONG Detach(Fn* original,Fn listener) {
    return Method<Tag,Fn>::Detach(original,listener);
}
} // namespace DlssNr::NativeHooks
