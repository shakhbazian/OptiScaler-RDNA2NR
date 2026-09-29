#pragma once
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#ifdef NR_PINNED_MIXED
#if !defined(NR_DIRECT_PUBLICATION) || NR_DIRECT_PUBLICATION != 1 || !defined(NR_ATTENTION_DOT2) || NR_ATTENTION_DOT2 != 0
#error The accepted pinned profile requires direct publication and the unpaired attention contract.
#endif
#endif
#if defined(NR_SCHEDULED_MIXED) && !defined(NR_PINNED_MIXED)
#error The scheduled mixed runtime must use the accepted pinned execution profile.
#endif

// Build-local selection: no process environment mutation, no change to research
// probes unless NR_PINNED_MIXED is explicitly defined for that binary.
namespace NrExecution {
constexpr std::uint32_t AcceptedMixedId = 0x20260924u;
constexpr const char* AcceptedMixedName = "mixed-v5-group64-c32-fp16-split4-20260924";
struct Setting { const char* name; const char* value; };
inline constexpr Setting AcceptedMixed[] = {
    {"RDNA2_GLOBAL_INT8", "64"}, {"RDNA2_GROUPED_INT8_SCOPE", "mixed-v5"},
    {"RDNA2_C32_INT8", "0"}, {"RDNA2_GLOBAL_GROUP128", "0"},
    {"RDNA2_BRANCH_RESIDENT", "1"}, {"RDNA2_BRANCH_PROJECT_FUSED", "1"},
    {"RDNA2_INT8_WAVE_PACK", "1"}, {"RDNA2_INT8_TILE64", "1"},
    {"RDNA2_INT8_ROW32", "1"}, {"RDNA2_INT8_HW_LAYOUT", "1"},
    {"RDNA2_QKV_HW_LAYOUT", "1"}, {"RDNA2_QKV_NORM_FUSED", "1"},
    {"RDNA2_BRANCH_FP16_PACKED", "1"}, {"RDNA2_C32_ATTENTION_WAVE", "1"},
    {"RDNA2_C32_FP16_PACKED", "1"}, {"RDNA2_C32_FUSED", "1"},
    {"RDNA2_C32_ATTENTION_FUSED", "1"}, {"RDNA2_WINDOW_PARALLEL", "1"},
    {"RDNA2_ATTENTION_LDS_PAD", "1"}, {"RDNA2_INT8_PREFETCH", "0"},
    {"RDNA2_SPLIT_RESIDENT", "4"}, {"RDNA2_GLOBAL_INT8_ALL", "0"},
    {"RDNA2_BUFFER_GUARDS", "0"}, {"RDNA2_INT8_VALIDATE_GUARDS", "0"},
    {"RDNA2_SPATIAL_TRACE", nullptr}, {"RDNA2_INT8_CAPTURE", nullptr},
    {"RDNA2_LINKED_W128_BUNDLE", nullptr}, {"RDNA2_LINKED_W128_SHA256", nullptr},
    {"RDNA2_COMPACT_C32_BUNDLE", nullptr}, {"RDNA2_COMPACT_C32_BLOCK2", nullptr},
    {"RDNA2_COMPACT_C32_SHA256", nullptr}, {"RDNA2_COMPACT_BRANCHED_BUNDLE", nullptr},
    {"RDNA2_COMPACT_BRANCHED_SHA256", nullptr}
};

inline int ReadSetting(std::size_t* needed, char* output, std::size_t capacity, const char* name) {
#ifndef NR_PINNED_MIXED
    return getenv_s(needed, output, capacity, name);
#else
    if (!needed || !name || (!output && capacity)) return EINVAL;
    *needed = 0;
    if (output && capacity) output[0] = 0;
    for (const auto& setting : AcceptedMixed) {
        if (std::strcmp(name, setting.name)) continue;
        if (!setting.value) return 0;
        *needed = std::strlen(setting.value) + 1;
        if (!output && !capacity) return 0;
        if (capacity < *needed) return ERANGE;
        std::memcpy(output, setting.value, *needed);
        return 0;
    }
    // A new graph setting must be consciously added to the pinned profile.
    return EINVAL;
#endif
}
} // namespace NrExecution
