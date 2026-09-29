#pragma once
#include "nr_runtime_v3_validation.h"
#include "nr_async_protocol.h"

namespace NrV3 {
// CPU executable specification, NOT a GPU adapter. All GPU observations are
// trusted inputs supplied by the future adapter; no public API accepts fences.
// The inner proven protocol owns GPU lifetime; this ledger owns host receipts.
class Session {
public:
    using Decision = NrAsync::Protocol<Capacity>::Decision;
    Status Configure() {
        if (state_ == Lifecycle::Quarantined) return Status::Quarantined;
        if (state_ != Lifecycle::Created && state_ != Lifecycle::Live) return Status::Closed;
        if (Occupied()) return Status::Busy;
        if (state_ == Lifecycle::Live) {
            const auto s = Convert(protocol_.Reconfigure());
            if (s != Status::Ok) return s;
        }
        configured_ = true; state_ = Lifecycle::Live; return Status::Ok;
    }
    Status Record(const Token& token, bool reset, bool motion) {
        const auto live = Live(); if (live != Status::Ok) return live;
        if (!Valid(token)) return Status::InvalidArgument;
        if (token.generation != protocol_.Generation()) return Status::StaleFrame;
        if (token.key.submissionEpoch <= lastAccepted_) return Status::StaleFrame;
        Slot* free = nullptr;
        for (auto& slot : slots_) if (slot.phase == Phase::Free) { free = &slot; break; }
        if (!free) return Status::Busy;
        if (!CanReserveSignal()) return Status::Failed;
        std::uint64_t ticket = 0;
        const auto s = Convert(protocol_.Record(token.key, reset, motion, ticket));
        if (s != Status::Ok) return s;
        *free = {}; free->token = token; free->phase = Phase::Recorded;
        lastAccepted_ = ticket; return Status::Ok;
    }
    // Adapter validates the exact token/phase BEFORE queue Signal, reserves a
    // fresh serial, then invokes this only if Signal succeeded after Execute.
    Status CheckInputAck(const Token& token) const { return CheckPhase(token, Phase::Recorded); }
    Status InputSubmitted(const Token& token, std::uint64_t serial) {
        auto s = CheckInputAck(token); if (s != Status::Ok) return s;
        if (!ValidSignal(serial)) return Status::InvalidArgument;
        s = Convert(protocol_.Submitted(token.key.submissionEpoch, serial));
        if (s == Status::Ok) lastD3dSerial_ = serial;
        Sync(); return s;
    }
    Status Dispatch(std::uint64_t observedInput, Decision& decision) {
        decision = {};
        const auto live = Live(); if (live != Status::Ok) return live;
        if (observedInput == ReservedSerial) return Quarantine();
        const auto s = Convert(protocol_.Begin(observedInput, decision));
        Sync(); return s;
    }
    Status Complete(const Token& token, NrAsync::Completion completion, std::uint64_t egress) {
        auto s = CheckPhase(token, Phase::Running); if (s != Status::Ok) return s;
        if (state_ != Lifecycle::Live) return Status::Closed;
        if (egress == ReservedSerial) return Quarantine();
        // Failure means a known failed candidate. Query/driver errors instead
        // call Quarantine; neither is evidence of stream retirement.
        if (completion == NrAsync::Completion::Failure) { Fault(&token); return Status::Failed; }
        s = Convert(protocol_.Complete(token.key.submissionEpoch, completion, egress));
        Sync(); return s;
    }
    Status Lease(const Token& token) {
        const auto live = Live(); if (live != Status::Ok) return live;
        auto s = CheckPhase(token, Phase::Ready); if (s != Status::Ok) return s;
        if (!CanReserveSignal()) return Status::Failed;
        s = Convert(protocol_.Lease(token.key.submissionEpoch)); Sync(); return s;
    }
    Status CheckOutputAck(const Token& token) const { return CheckPhase(token, Phase::CopyRecorded); }
    Status OutputSubmitted(const Token& token, std::uint64_t serial) {
        auto s = CheckOutputAck(token); if (s != Status::Ok) return s;
        if (!ValidSignal(serial)) return Status::InvalidArgument;
        s = Convert(protocol_.CopySubmitted(token.key.submissionEpoch, serial));
        if (s == Status::Ok) lastD3dSerial_ = serial;
        Sync(); return s;
    }
    Status Drop(const Token& token) {
        const auto live = Live(); if (live != Status::Ok) return live;
        auto s = CheckPhase(token, Phase::Ready); if (s != Status::Ok) return s;
        s = Convert(protocol_.Drop(token.key.submissionEpoch));
        if (s == Status::Ok) { auto* slot = Find(token); slot->outcome = Outcome::Dropped; Sync(); }
        return s;
    }
    Status AcknowledgeTerminal(const Token& token) {
        auto s = CheckPhase(token, Phase::Terminal); if (s != Status::Ok) return s;
        *Find(token) = {}; return Status::Ok;
    }
    Status Retire(std::uint64_t observedD3d, bool streamIdleProven) {
        if (state_ == Lifecycle::Quarantined) return Status::Quarantined;
        if (state_ == Lifecycle::Closed) return Status::Closed;
        if (observedD3d == ReservedSerial) return Quarantine();
        const auto s = Convert(protocol_.Retire(observedD3d, streamIdleProven));
        Sync(); return s;
    }
    Status BeginDrain() {
        if (state_ == Lifecycle::Quarantined) return Status::Quarantined;
        if (state_ == Lifecycle::Closed) return Status::Ok;
        if (state_ != Lifecycle::Faulted) state_ = Lifecycle::Draining;
        MarkCancelled(nullptr); protocol_.Drain(); Sync(); return Status::Ok;
    }
    void Fault(const Token* failed) {
        if (state_ == Lifecycle::Quarantined || state_ == Lifecycle::Closed) return;
        state_ = Lifecycle::Faulted; MarkCancelled(failed); protocol_.Fault(); Sync();
    }
    Status Quarantine() {
        if (state_ == Lifecycle::Closed) return Status::Closed;
        state_ = Lifecycle::Quarantined; protocol_.DeviceLost();
        for (auto& slot : slots_) if (slot.phase != Phase::Free && slot.phase != Phase::Terminal) {
            slot.phase = Phase::Quarantined; slot.outcome = Outcome::Quarantined;
        }
        return Status::Quarantined;
    }
    // Adapter first polls driver facts and verifies its retained objects. This
    // method alone proves only CPU protocol closure, not safe GPU destruction.
    Status Close() {
        if (state_ == Lifecycle::Quarantined) return Status::Quarantined;
        if (state_ == Lifecycle::Closed) return Status::Ok;
        if (state_ != Lifecycle::Draining && state_ != Lifecycle::Faulted) return Status::Busy;
        if (Occupied() || !protocol_.CanDestroy()) return Status::Busy;
        state_ = Lifecycle::Closed; return Status::Ok;
    }
    // Cleanup can fail AFTER Close. The adapter then retains the owner and its
    // remaining fields; normal destruction must not run on that partial object.
    void CleanupFailed() { state_ = Lifecycle::Quarantined; protocol_.DeviceLost(); }
    Snapshot View() const {
        Snapshot result{}; result.prefix = {sizeof(result), Version};
        result.status = state_ == Lifecycle::Quarantined ? Status::Quarantined :
            state_ == Lifecycle::Faulted ? Status::Failed : Status::Ok;
        result.lifecycle = state_; result.generation = configured_ ? protocol_.Generation() : 0;
        result.commits = protocol_.Commits(); result.occupied = Occupied(); result.capacity = Capacity;
        result.lastAcceptedEpoch = lastAccepted_;
        for (std::uint32_t i = 0; i < Capacity; ++i) result.slots[i] = slots_[i];
        return result;
    }
    bool HistoryValid() const { return protocol_.HistoryValid(); }
private:
    Status Live() const {
        if (state_ == Lifecycle::Quarantined) return Status::Quarantined;
        if (state_ == Lifecycle::Created) return Status::NotConfigured;
        if (state_ == Lifecycle::Faulted) return Status::Failed;
        return state_ == Lifecycle::Live ? Status::Ok : Status::Closed;
    }
    Slot* Find(const Token& token) {
        for (auto& slot : slots_) if (slot.phase != Phase::Free && Same(slot.token, token)) return &slot;
        return nullptr;
    }
    Status CheckPhase(const Token& token, Phase phase) const {
        if (state_ == Lifecycle::Quarantined) return Status::Quarantined;
        if (!Valid(token)) return Status::InvalidArgument;
        for (const auto& slot : slots_) if (slot.phase != Phase::Free && Same(slot.token, token))
            return slot.phase == phase ? Status::Ok : Status::StaleFrame;
        return Status::StaleFrame;
    }
    std::uint32_t Occupied() const {
        std::uint32_t count = 0;
        for (const auto& slot : slots_) if (slot.phase != Phase::Free) ++count;
        return count;
    }
    std::uint64_t PendingSignals() const {
        std::uint64_t count = 0;
        for (const auto& slot : slots_)
            if (slot.phase == Phase::Recorded || slot.phase == Phase::CopyRecorded) ++count;
        return count;
    }
    bool CanReserveSignal() const { return PendingSignals() < ReservedSerial - 1 - lastD3dSerial_; }
    bool ValidSignal(std::uint64_t value) const {
        // The current ack consumes one reservation; preserve room for all
        // other already-recorded uses even if this observation skips serials.
        const auto pending = PendingSignals();
        return pending && value > lastD3dSerial_ && value < ReservedSerial &&
            pending - 1 <= ReservedSerial - 1 - value;
    }
    void MarkCancelled(const Token* failed) {
        for (auto& slot : slots_) {
            if (slot.phase == Phase::Free || slot.phase == Phase::Terminal ||
                slot.phase == Phase::CopyRecorded || slot.phase == Phase::Retiring) continue;
            // Keep a previously attributed failure across repeated drain calls.
            if (slot.outcome != Outcome::Failed)
                slot.outcome = failed && Same(slot.token, *failed) ? Outcome::Failed : Outcome::Cancelled;
        }
    }
    void Sync() {
        for (auto& slot : slots_) {
            if (slot.phase == Phase::Free || slot.phase == Phase::Terminal) continue;
            const auto* inner = protocol_.Lookup(slot.token.key.submissionEpoch);
            if (!inner) {
                if (slot.phase == Phase::Retiring) slot.outcome = Outcome::OutputRetired;
                slot.phase = Phase::Terminal; continue;
            }
            // NrAsync has the same first seven states; Cancelled is cancelling
            // GPU use here, not a terminal receipt until actual retirement.
            slot.phase = inner->phase == NrAsync::Phase::Cancelled ? Phase::Cancelling :
                static_cast<Phase>(inner->phase);
            slot.inputSerial = inner->inputFence; slot.outputSerial = inner->retireFence;
            slot.noiseIndex = inner->noise; slot.usesHistory = inner->usesHistory ? 1u : 0u;
        }
    }
    NrAsync::Protocol<Capacity> protocol_;
    std::array<Slot, Capacity> slots_{};
    Lifecycle state_ = Lifecycle::Created;
    std::uint64_t lastAccepted_ = 0, lastD3dSerial_ = 0;
    bool configured_ = false;
};
} // namespace NrV3
