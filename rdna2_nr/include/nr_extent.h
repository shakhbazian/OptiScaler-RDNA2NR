#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace NrV2 {

// D3D12 texture dimensions and the graph's signed 32-bit element indexing,
// rather than a fixed 1440p/4K resolution list, bound diagnostic extents.
// Residency and the shared-memory attention row are checked at runtime.
inline constexpr std::uint32_t MaximumLogicalWidth = 16384;
inline constexpr std::uint32_t MaximumLogicalHeight = 16384;
inline constexpr std::uint32_t MaximumNetworkWidth = 16384;
inline constexpr std::uint32_t MaximumNetworkHeight = 16384;
inline constexpr std::size_t MaximumLogicalPixels =
    (std::numeric_limits<std::int32_t>::max)() / 64;
inline constexpr std::size_t MaximumNetworkPixels =
    (std::numeric_limits<std::int32_t>::max)() / 64;

inline constexpr std::uint32_t AlignNetworkExtent(std::uint32_t value) {
    const auto clamped = value < 320 ? 320 : value;
    return (clamped + 63) / 64 * 64;
}

inline constexpr bool SupportedLogicalExtent(std::uint32_t width, std::uint32_t height) {
    return width && height && width <= MaximumLogicalWidth &&
        height <= MaximumLogicalHeight && std::size_t(width) * height <= MaximumLogicalPixels;
}

inline constexpr bool SupportedGraphExtent(std::uint32_t width, std::uint32_t height) {
    return width >= 128 && height >= 128 && !(width % 64) && !(height % 64) &&
        width <= MaximumNetworkWidth && height <= MaximumNetworkHeight &&
        std::size_t(width) * height <= MaximumNetworkPixels;
}

static_assert(SupportedLogicalExtent(3840, 2160));
static_assert(SupportedGraphExtent(3840, 2176));
static_assert(SupportedLogicalExtent(4096, 2304));
static_assert(SupportedLogicalExtent(7680, 4320));
static_assert(SupportedGraphExtent(7680, 4352));

} // namespace NrV2
