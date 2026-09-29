#pragma once
#include "NativeRecording.h"
#include "NativeQueueTransaction.h"
#include "../../../rdna2_nr/include/nr_submission_runtime_contract.h"
#include <string>

namespace DlssNr::Native {
struct Intent {
    ComPtr<ID3D12Resource> color,motion,output;
    // Optional common-color commands flank HIP inside the same queue transaction.
    // The game's original list remains a complete raw fallback.
    std::shared_ptr<Recording> frontend;
    std::shared_ptr<void> frontendLease;
    NrV2::Controls controls{};
    NrV2::Motion motionParameters{};
    std::uint64_t sourceFrame=0,feature=0;
    unsigned width=0,height=0,motionWidth=0,motionHeight=0;
    D3D12_RESOURCE_STATES colorState=D3D12_RESOURCE_STATE_COMMON,motionState=D3D12_RESOURCE_STATE_COMMON;
    D3D12_RESOURCE_STATES outputState=D3D12_RESOURCE_STATE_COMMON;
    bool reset=false,before=true,temporalAccumulation=true,warmOnly=false;
    std::shared_ptr<std::atomic<bool>> featureLive;
};
struct NativeBatch final : NativeQueue::Prepared {
    std::shared_ptr<Recording> recording;
    std::shared_ptr<const Intent> intent;
    unsigned markerIndex=0;
    bool repeated=false;
    NativeQueue::Result Submit(ID3D12CommandQueue*,UINT,ID3D12CommandList* const*,NativeQueue::Execute) noexcept override;
};
void SetAssetPaths(std::wstring dll,std::wstring package);
bool WarmSession(ID3D12Device*,ID3D12CommandQueue*,unsigned,unsigned,std::uint64_t,bool) noexcept;
std::uint64_t AppliedFrames() noexcept;
std::uint64_t BypassedFrames() noexcept;
bool SessionReady() noexcept;
bool SessionReadyFor(ID3D12Device*, ID3D12CommandQueue*, unsigned, unsigned,
                     std::uint64_t feature, bool before) noexcept;
void StopSession(std::uint64_t feature=0) noexcept;
void HistoryDiscontinuity() noexcept;
}
