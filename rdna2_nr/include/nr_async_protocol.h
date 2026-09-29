#pragma once

#include "nr_runtime_contract.h"
#include <array>
#include <limits>

namespace NrAsync {
using NrV2::Status;
using NrV2::FrameKey;
constexpr std::uint64_t Removed = std::numeric_limits<std::uint64_t>::max();
enum class Phase { Free, Recorded, Queued, Running, Ready, CopyRecorded, Retiring, Cancelled };
enum class Completion { Pending, Success, Failure };

// CPU-only ownership protocol. The adapter supplies observed GPU facts; this
// class cannot query devices, issue signals, wait, or release GPU resources.
// All methods are serialized by one owner. One neural inference at a time;
// multiple acknowledged inputs and completed/leased outputs may coexist.
template<std::size_t Capacity = 3> class Protocol {
public:
    static_assert(Capacity > 0);
    struct Slot {
        Phase phase = Phase::Free;
        FrameKey key{};
        std::uint64_t generation = 0, inputFence = 0, retireFence = 0;
        std::uint32_t noise = 0;
        bool reset = false, motion = false, usesHistory = false, hipOutstanding = false;
    };
    struct Decision {
        std::uint64_t ticket = 0;
        FrameKey key{};
        std::uint32_t noise = 0;
        bool useHistory = false;
    };

    Status Record(FrameKey key, bool reset, bool motion, std::uint64_t& ticket) {
        ticket = 0;
        if (faulted_ || draining_) return Status::Failed;
        if (!key.streamId || !key.submissionEpoch || key.submissionEpoch == Removed)
            return Status::InvalidArgument;
        if (key.submissionEpoch <= lastAccepted_.submissionEpoch ||
            (lastAccepted_.streamId == key.streamId && key.frameIndex <= lastAccepted_.frameIndex && !reset))
            return Status::StaleFrame;
        for (const auto& slot : slots_)
            if (slot.phase == Phase::Recorded) return Status::NotSubmitted;
        for (auto& slot : slots_) if (slot.phase == Phase::Free) {
            slot = {}; slot.phase = Phase::Recorded; slot.key = key;
            slot.reset = reset; slot.motion = motion; slot.generation = generation_;
            lastAccepted_ = key; ticket = key.submissionEpoch;
            return Status::Ok;
        }
        return Status::Busy;
    }
    // Call only after ExecuteCommandLists AND successful queue Signal. On signal
    // failure preserve the unretired slot; submission may already be executing.
    Status Submitted(std::uint64_t ticket, std::uint64_t fence) {
        auto* slot = Find(ticket);
        if (!slot || slot->phase != Phase::Recorded) return Status::StaleFrame;
        if (!NewFence(fence)) return Status::InvalidArgument;
        slot->inputFence = fence; lastD3dSignal_ = fence;
        slot->phase = (faulted_ || draining_) ? Phase::Cancelled : Phase::Queued;
        return Status::Ok;
    }
    Status Begin(std::uint64_t completedInputFence, Decision& decision) {
        decision = {};
        if (completedInputFence == Removed) { DeviceLost(); return Status::Failed; }
        if (faulted_ || draining_) return Status::Failed;
        Slot* oldest = nullptr;
        for (auto& slot : slots_) {
            if (slot.phase == Phase::Running) return Status::Busy;
            if (slot.phase == Phase::Queued &&
                (!oldest || slot.key.submissionEpoch < oldest->key.submissionEpoch)) oldest = &slot;
        }
        if (!oldest || completedInputFence < oldest->inputFence) return Status::Busy;
        const bool consecutive = historyValid_ && committed_.streamId == oldest->key.streamId &&
            committed_.frameIndex != Removed && oldest->key.frameIndex == committed_.frameIndex + 1;
        if (!consecutive || oldest->reset || !oldest->motion) Invalidate();
        oldest->noise = noise_; oldest->usesHistory = historyValid_;
        oldest->phase = Phase::Running; oldest->hipOutstanding = true;
        decision = {oldest->key.submissionEpoch, oldest->key, noise_, historyValid_};
        return Status::Ok;
    }
    // Success requires a recorded HIP completion event, checked numerical
    // status, and observation of the separate HIP-written egress fence.
    // The egress value is the ticket (strictly increasing, never UINT64_MAX).
    Status Complete(std::uint64_t ticket, Completion event, std::uint64_t completedEgress) {
        if (completedEgress == Removed) { DeviceLost(); return Status::Failed; }
        if (faulted_ || draining_) return Status::Failed;
        auto* slot = Find(ticket);
        if (!slot || slot->phase != Phase::Running) return Status::StaleFrame;
        if (event == Completion::Failure) { Fault(); return Status::Failed; }
        if (event == Completion::Pending || completedEgress < ticket) return Status::Busy;
        slot->hipOutstanding = false; slot->phase = Phase::Ready;
        committed_ = slot->key; historyValid_ = true; ++noise_; ++commits_;
        return Status::Ok;
    }
    // Reserve the ready output BEFORE recording its copy into a host list.
    // Host must preserve the corresponding full key/generation resolve carrier.
    Status Lease(std::uint64_t ticket) {
        if (faulted_ || draining_) return Status::Failed;
        auto* slot = Find(ticket);
        if (!slot || slot->phase != Phase::Ready) return Status::StaleFrame;
        slot->phase = Phase::CopyRecorded;
        return Status::Ok;
    }
    // Also allowed after Fault/Drain: an already recorded copy must still retire.
    Status CopySubmitted(std::uint64_t ticket, std::uint64_t fence) {
        auto* slot = Find(ticket);
        if (!slot || slot->phase != Phase::CopyRecorded) return Status::StaleFrame;
        if (!NewFence(fence)) return Status::InvalidArgument;
        slot->retireFence = fence; lastD3dSignal_ = fence; slot->phase = Phase::Retiring;
        return Status::Ok;
    }
    Status Drop(std::uint64_t ticket) {
        auto* slot = Find(ticket);
        if (!slot || slot->phase != Phase::Ready) return Status::StaleFrame;
        *slot = {}; // No D3D output reference exists. History lives elsewhere.
        return Status::Ok;
    }
    // Nonblocking. hipIdleProven means successful hipStreamQuery on the owning
    // stream after all enqueue attempts, NOT timeout/error/device removal.
    Status Retire(std::uint64_t completedD3d, bool hipIdleProven = false) {
        if (completedD3d == Removed) { DeviceLost(); return Status::Failed; }
        if (quarantined_) return Status::Failed;
        for (auto& slot : slots_) {
            if (slot.phase == Phase::Retiring && completedD3d >= slot.retireFence) slot = {};
            else if (slot.phase == Phase::Cancelled && completedD3d >= slot.inputFence &&
                     (!slot.hipOutstanding || hipIdleProven)) slot = {};
        }
        return Empty() ? Status::Ok : Status::Busy;
    }
    Status Reconfigure() {
        if (faulted_ || draining_) return Status::Failed;
        if (!Empty()) return Status::Busy;
        if (generation_ == Removed - 1) return Status::Failed;
        ++generation_; Invalidate(); return Status::Ok;
    }
    void Fault() { faulted_ = true; Cancel(); }
    void Drain() { draining_ = true; Cancel(); }
    void DeviceLost() { quarantined_ = true; Fault(); }
    // No release permission can be inferred from a driver error. A terminal
    // quarantine remains owned until process exit; never unload its code module.
    bool CanDestroy() const { return Empty() && !quarantined_; }
    bool HistoryValid() const { return historyValid_; }
    bool Faulted() const { return faulted_; }
    bool Quarantined() const { return quarantined_; }
    std::uint64_t Commits() const { return commits_; }
    std::uint64_t Generation() const { return generation_; }
    const std::array<Slot, Capacity>& Slots() const { return slots_; }
    const Slot* Lookup(std::uint64_t ticket) const {
        for (const auto& slot : slots_)
            if (slot.phase != Phase::Free && slot.key.submissionEpoch == ticket) return &slot;
        return nullptr;
    }
private:
    Slot* Find(std::uint64_t ticket) { return const_cast<Slot*>(Lookup(ticket)); }
    bool NewFence(std::uint64_t fence) const { return fence > lastD3dSignal_ && fence < Removed; }
    bool Empty() const {
        for (const auto& slot : slots_) if (slot.phase != Phase::Free) return false;
        return true;
    }
    void Invalidate() { historyValid_ = false; noise_ = 0; }
    void Cancel() {
        Invalidate();
        for (auto& slot : slots_) {
            if (slot.phase == Phase::Queued || slot.phase == Phase::Running) slot.phase = Phase::Cancelled;
            else if (slot.phase == Phase::Ready) slot = {};
        }
    }
    std::array<Slot, Capacity> slots_{};
    FrameKey lastAccepted_{}, committed_{};
    std::uint64_t lastD3dSignal_ = 0, generation_ = 1, commits_ = 0;
    std::uint32_t noise_ = 0;
    bool historyValid_ = false, faulted_ = false, draining_ = false, quarantined_ = false;
};
} // namespace NrAsync
