#pragma once

// ABI v2 contract, Windows x64 / pack 8. Implemented by dlssnr_hip_runtime_v2.dll.
#include <cstddef>
#include <cstdint>
#include <type_traits>

struct ID3D12Device;
struct ID3D12CommandQueue;
struct ID3D12GraphicsCommandList;
struct ID3D12Resource;

namespace NrV2 {
constexpr std::uint32_t Version = 2;
constexpr std::uint32_t MathContract = 1; // gfx1030 DOT2 + Cephes FP32/device sqrt

enum class Status : std::uint32_t {
    Ok, InvalidSize, Unsupported, InvalidArgument, InvalidResource,
    NotConfigured, Busy, NotSubmitted, StaleFrame, Failed
};
enum class Format : std::uint32_t { None, Rgba16Float, Rg16Float };
enum class JitterMode : std::uint32_t { AlreadyIncluded, AddPreviousMinusCurrent };
enum Reset : std::uint32_t { NoReset = 0, Explicit = 1, SceneCut = 2 };

#pragma pack(push, 8)
struct Prefix { std::uint32_t structSize, abiVersion; };
struct Hash256 { std::uint8_t bytes[32]; };
struct FrameKey {
    std::uint64_t streamId, frameIndex, submissionEpoch;
};
struct Transform {
    std::uint32_t baseX, baseY, extentWidth, extentHeight, backingWidth, backingHeight;
};
struct Texture {
    ID3D12Resource* resource;
    Transform transform;
    Format format;
    std::uint32_t reserved;
};
struct Controls {
    float styleIndex, tone, structure, skinStrength, autoStrength, intensity;
};
struct Motion {
    float scaleX, scaleY;
    std::uint32_t effectiveWidth, effectiveHeight;
    float previousJitterX, previousJitterY, currentJitterX, currentJitterY;
    JitterMode jitterMode;
    std::uint32_t reserved;
};
struct Config {
    Prefix prefix;
    Hash256 sourceHash, packageHash;
    std::uint32_t graphVersion, mathContract;
    std::uint32_t logicalWidth, logicalHeight, networkWidth, networkHeight;
    std::uint32_t reserved[2];
};
struct Frame {
    Prefix prefix;
    FrameKey key;
    ID3D12GraphicsCommandList* commands;
    ID3D12CommandQueue* queue;
    Texture color, motion, controlMask, output;
    Controls controls;
    Motion motionParameters;
    std::uint32_t resetReasons, reserved;
};
struct Result {
    Prefix prefix;
    Status status;
    std::uint32_t frameAccepted, outputReady, outputDelayed;
    FrameKey accepted, output;
    std::uint32_t outputNoiseIndex, reserved;
};
struct Api {
    Prefix prefix;
    Status (__cdecl* create)(ID3D12Device*, ID3D12CommandQueue*, void**);
    Status (__cdecl* configure)(void*, const Config*, std::uint32_t configBytes,
                               const void* package, std::uint64_t packageBytes);
    Status (__cdecl* evaluate)(void*, const Frame*, std::uint32_t frameBytes,
                              Result*, std::uint32_t resultBytes);
    Status (__cdecl* notifySubmitted)(void*, std::uint64_t submissionEpoch);
    Status (__cdecl* shutdown)(void*); // Retires submitted work; Busy if recorded work is unsubmitted.
    void (__cdecl* destroy)(void*); // Only after successful shutdown.
};
// Exported as DlssNrHipBackendGetApiV2(requestedVersion, table, tableBytes).
using GetApi = Status (__cdecl*)(std::uint32_t, Api*, std::uint32_t);
#pragma pack(pop)

static_assert(sizeof(void*) == 8, "Windows x64 contract only");
static_assert(sizeof(Prefix) == 8 && sizeof(Hash256) == 32);
static_assert(sizeof(FrameKey) == 24 && sizeof(Transform) == 24);
static_assert(sizeof(Texture) == 40 && sizeof(Controls) == 24 && sizeof(Motion) == 40);
static_assert(sizeof(Config) == 104 && sizeof(Frame) == 280 && sizeof(Result) == 80);
static_assert(sizeof(Api) == 56);
static_assert(offsetof(Frame, color) == 48 && offsetof(Frame, controls) == 208);
static_assert(std::is_standard_layout<Frame>::value && std::is_trivially_copyable<Frame>::value);
static_assert(std::is_standard_layout<Config>::value && std::is_trivially_copyable<Config>::value);
} // namespace NrV2
