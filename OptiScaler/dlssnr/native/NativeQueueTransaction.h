#pragma once

#include <d3d12.h>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

// The existing ResTrack queue hook is the only owner of ExecuteCommandLists.
// A native NR adapter may claim a whole batch here; otherwise the original
// call is made exactly once. Queue Signal/Wait participate in the same lane.
namespace DlssNr::NativeQueue {
using Execute = void(STDMETHODCALLTYPE*)(ID3D12CommandQueue*,UINT,ID3D12CommandList* const*);
using FenceOp = HRESULT(STDMETHODCALLTYPE*)(ID3D12CommandQueue*,ID3D12Fence*,UINT64);
enum class Result { Unclaimed, Submitted, Quarantined };
using Submitter = Result(*)(ID3D12CommandQueue*,UINT,ID3D12CommandList* const*,Execute) noexcept;
struct Prepared {
    virtual ~Prepared()=default;
    virtual Result Submit(ID3D12CommandQueue*,UINT,ID3D12CommandList* const*,Execute) noexcept=0;
};
using Prepare = std::shared_ptr<Prepared>(*)(ID3D12CommandQueue*,UINT,ID3D12CommandList* const*) noexcept;
inline std::atomic<Prepare> prepare{nullptr};

inline std::atomic<Submitter> submitter{nullptr};
inline std::mutex registryMutex;
inline std::unordered_map<ID3D12CommandQueue*,std::shared_ptr<std::mutex>> lanes;
inline thread_local ID3D12CommandQueue* internalQueue=nullptr;

inline std::shared_ptr<std::mutex> Lane(ID3D12CommandQueue* queue) {
    std::lock_guard lock(registryMutex);
    auto& stored=lanes[queue];
    if(!stored)stored=std::make_shared<std::mutex>();
    auto strong=stored;
    if(lanes.size()>128)
        for(auto it=lanes.begin();it!=lanes.end();)
            it=it->second.use_count()==1?lanes.erase(it):++it;
    return strong;
}
struct InternalPermit {
    ID3D12CommandQueue* previous;
    explicit InternalPermit(ID3D12CommandQueue* queue):previous(internalQueue){internalQueue=queue;}
    ~InternalPermit(){internalQueue=previous;}
    InternalPermit(const InternalPermit&)=delete;
    InternalPermit& operator=(const InternalPermit&)=delete;
};

// Register only after the adapter is fully ready; unregister before its code is
// unloaded and after its in-flight callbacks have finished. The queue hooks can
// remain installed while the handler is null, preserving the existing path.
inline void SetSubmitter(Submitter fn) noexcept {submitter.store(fn,std::memory_order_release);}
inline void SetPrepare(Prepare fn) noexcept {prepare.store(fn,std::memory_order_release);}
inline bool Active() noexcept {return submitter.load(std::memory_order_acquire)||prepare.load(std::memory_order_acquire);}

inline bool ExecuteBatch(ID3D12CommandQueue* queue,UINT count,
                         ID3D12CommandList* const* lists,Execute original) {
    const auto fn=submitter.load(std::memory_order_acquire);
    const auto pin=prepare.load(std::memory_order_acquire);
    if((!fn&&!pin)||internalQueue==queue){original(queue,count,lists);return true;}
    // Pin immutable generations before taking the queue lane. No recorder or
    // Config lock is acquired from the submission callback under that lane.
    auto prepared=pin?pin(queue,count,lists):std::shared_ptr<Prepared>{};
    auto lane=Lane(queue);std::lock_guard lock(*lane);InternalPermit permit(queue);
    const auto result=prepared?prepared->Submit(queue,count,lists,original):
        fn?fn(queue,count,lists,original):Result::Unclaimed;
    switch(result) {
    case Result::Unclaimed: original(queue,count,lists);return true;
    case Result::Submitted: return true;
    case Result::Quarantined: return false;
    }
    return false;
}
inline HRESULT Signal(ID3D12CommandQueue* queue,ID3D12Fence* fence,UINT64 value,FenceOp original) {
    if(!Active()||internalQueue==queue)
        return original(queue,fence,value);
    auto lane=Lane(queue);std::lock_guard lock(*lane);InternalPermit permit(queue);
    return original(queue,fence,value);
}
inline HRESULT Wait(ID3D12CommandQueue* queue,ID3D12Fence* fence,UINT64 value,FenceOp original) {
    if(!Active()||internalQueue==queue)
        return original(queue,fence,value);
    auto lane=Lane(queue);std::lock_guard lock(*lane);InternalPermit permit(queue);
    return original(queue,fence,value);
}
template<class Call> inline void Mutation(ID3D12CommandQueue* queue,Call&& call) {
    if(!Active()||internalQueue==queue){call();return;}
    auto lane=Lane(queue);std::lock_guard lock(*lane);InternalPermit permit(queue);call();
}
} // namespace DlssNr::NativeQueue
