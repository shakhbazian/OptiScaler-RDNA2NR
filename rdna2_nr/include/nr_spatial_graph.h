#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>

namespace NrV2 {
class GpuModel;

struct SpatialGraphStats {
    std::size_t allocations = 0;
    std::size_t reuses = 0;
    std::size_t reservedBytes = 0;
    std::size_t peakInUseBytes = 0;
    std::size_t frames = 0;
    std::size_t preparations = 0;
    std::size_t int8ReservedBytes = 0;
    std::size_t int8Invocations = 0;
    bool experimentalGlobalInt8 = false;
};

// Diagnostic-only profiling. Events has 72 entries, blockWallMs has 71.
// Isolation deliberately inserts waits and changes scheduling; it is not a
// measurement of normal asynchronous throughput.
struct SpatialGraphProfile {
    const hipEvent_t* events = nullptr;
    bool isolateBlocks = false;
    // Paired compact-student benchmarks can execute the frozen teacher through
    // the same runtime and resident model without rebuilding or reloading.
    bool disableCompactC32 = false;
    bool disableCompactBranched = false;
    bool disableLinkedW128 = false;
    double* blockWallMs = nullptr;
    // Diagnostic observer. It runs on the owning stream immediately after an
    // activation becomes available; implementations must enqueue their work
    // and return without retaining the device pointer.
    void (*activationObserver)(void* context, std::uint32_t block,
                               const char* role, const void* deviceValues,
                               std::size_t rows, std::size_t columns,
                               hipStream_t stream) = nullptr;
    void* activationObserverContext = nullptr;
    // Queued diagnostic boundaries only. Caller owns timing events and reads
    // them after completion; nested pack ranges are inclusive in parent stages.
    void (*stageEvent)(void* context, const char* stage, bool begin,
                       hipStream_t stream) = nullptr;
    void* stageEventContext = nullptr;
};

// Persistent, serialized executor for the 71-block FP16 graph. Input and output
// are device pointers to NHWC half arrays of HxWx16 and HxWx4 elements.
class SpatialGraphRuntime {
  public:
    SpatialGraphRuntime(const GpuModel& model, hipStream_t owningStream);
    ~SpatialGraphRuntime();
    SpatialGraphRuntime(const SpatialGraphRuntime&) = delete;
    SpatialGraphRuntime& operator=(const SpatialGraphRuntime&) = delete;

    // Enqueue is serialized and nonblocking. Every allocation/reuse and kernel
    // is ordered on owningStream. Caller retains input/output until completion.
    void Enqueue(const void* preparedDevice, std::uint32_t height,
                 std::uint32_t width, void* outputDevice,
                 const SpatialGraphProfile* profile = nullptr);
    // Optional diagnostics: 72 caller-owned timing events at the 71 block
    // boundaries. Retain them until completion; no readback/wait is inserted.
    // Compatibility wrapper: Enqueue plus one owning-stream synchronization.
    void Execute(const void* preparedDevice, std::uint32_t height,
                 std::uint32_t width, void* outputDevice);
    // Warm the arena for this geometry before accepting frames.
    void Prepare(const void* preparedDevice, std::uint32_t height,
                 std::uint32_t width, void* outputDevice);
    // Call only after the owning stream is known idle. The normal FP16 path is
    // a no-op; the opt-in W8A8 experiment reports nonfinite data or guard damage.
    void ValidateExperimental() const;
    SpatialGraphStats Stats() const noexcept;
    // Returns false without deleting ownership bookkeeping when HIP cleanup is
    // ambiguous. Call only after the owning stream has been proved idle.
    bool Release() noexcept;

  private:
    struct Impl;
    Impl* impl_ = nullptr;
};
} // namespace NrV2
