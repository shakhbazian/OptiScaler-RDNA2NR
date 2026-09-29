#pragma once

// Test-only ABI. These exports exist only in dlssnr_hip_runtime_v2_fault_test.dll.
#include "nr_runtime_contract.h"

#include <cstdint>

namespace NrV2Test {
constexpr std::uint32_t Version = 1;

#pragma pack(push, 8)
struct State {
    std::uint32_t structSize, version;
    std::uint32_t configured, recorded, ready, faulted, historyValid, reserved;
    std::uint64_t commits, modelUploadOperations;
    std::uint64_t graphAllocations, graphReuses, graphEnqueues, graphPreparations;
    std::uint64_t historySwaps, completionChecks;
    std::uint64_t lastEnqueueMicroseconds, lastCompletionWaitMicroseconds;
    std::uint64_t finiteFailures;
};
#pragma pack(pop)

static_assert(sizeof(State) == 120);
using ArmFailure = NrV2::Status (__cdecl*)(void*);
using ArmNonfinite = NrV2::Status (__cdecl*)(void*);
using GetState = NrV2::Status (__cdecl*)(void*, State*, std::uint32_t);
} // namespace NrV2Test
