#pragma once
#include "nr_runtime_v3_contract.h"
#ifdef NR_RUNTIME_V3_TEST_HOOKS
#include "nr_runtime_v3_test.h"
#endif

namespace NrV3 {
#ifdef NR_SCHEDULED_MIXED
class ScheduledRuntimeDx12;
#endif
class RuntimeDx12 {
public:
    RuntimeDx12(ID3D12Device* device, ID3D12CommandQueue* queue);
    ~RuntimeDx12();
    RuntimeDx12(const RuntimeDx12&) = delete;
    RuntimeDx12& operator=(const RuntimeDx12&) = delete;

    Status Configure(const Config&, const void* package, std::uint64_t packageBytes);
    Status RecordInput(const Input&, RecordResult&);
    Status NotifyInputSubmitted(const Token&);
    Status Poll(Snapshot&);
    Status DispatchNext();
    Status RecordOutput(const Output&, RecordResult&);
    Status NotifyOutputSubmitted(const Token&);
    Status DropOutput(const Token&);
    Status AcknowledgeTerminal(const Token&);
    Status BeginDrain();
    Status Shutdown();
    Status Quarantine();
    Status Destroy();
#ifdef NR_RUNTIME_V3_TEST_HOOKS
    Status ArmFailureForTest();
    Status ArmHeadNonfiniteForTest();
    Status ArmOutputNonfiniteForTest();
    Status ArmCleanupFailureForTest();
    Status ArmSignalFailureForTest();
    Status ArmEventFailureForTest();
    Status GetStateForTest(NrV3Test::State&) const;
#endif

private:
#ifdef NR_SCHEDULED_MIXED
    friend class ScheduledRuntimeDx12;
#endif
    struct Impl;
    Impl* impl_ = nullptr;
};
#ifdef NR_SCHEDULED_MIXED
class ScheduledRuntimeDx12 {
public:
    ScheduledRuntimeDx12(ID3D12Device*,ID3D12CommandQueue*);
    ~ScheduledRuntimeDx12();
    ScheduledRuntimeDx12(const ScheduledRuntimeDx12&)=delete;
    ScheduledRuntimeDx12& operator=(const ScheduledRuntimeDx12&)=delete;
    Status Configure(const Config&,const void*,std::uint64_t);
    Status RecordInput(const Input&,RecordResult&);
    Status NotifyInputSubmitted(const Token&);
    Status Poll(Snapshot&);
    Status Enqueue(const Token&);
    Status RecordOutput(const Output&,RecordResult&);
    Status RecordOutputDeferred(const Output&,RecordResult&);
    Status ArmOutput(const Token&);
    Status NotifyOutputSubmitted(const Token&);
    Status Drop(const Token&);
    Status AcknowledgeTerminal(const Token&);
    Status BeginDrain();
    Status Shutdown();
    Status Quarantine();
    Status Destroy();
private:
    RuntimeDx12::Impl* impl_=nullptr;
};
#endif
} // namespace NrV3
