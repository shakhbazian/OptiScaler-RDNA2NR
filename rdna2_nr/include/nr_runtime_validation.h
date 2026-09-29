#pragma once
#include "nr_runtime_contract.h"
#include "nr_extent.h"
#include <cmath>
#include <cstring>

namespace NrV2 {
// Explicit accessible lengths prevent reads past a truncated ABI prefix.
template<class T> inline Status Decode(const void* data, std::size_t bytes, T& result) {
    if (!data || bytes < sizeof(Prefix)) return Status::InvalidSize;
    Prefix prefix{};
    std::memcpy(&prefix, data, sizeof(prefix));
    if (prefix.abiVersion != Version) return Status::Unsupported;
    if (prefix.structSize != sizeof(T) || bytes < sizeof(T)) return Status::InvalidSize;
    std::memcpy(&result, data, sizeof(T));
    return Status::Ok;
}
inline bool Nonzero(const Hash256& hash) {
    for (auto byte : hash.bytes) if (byte) return true;
    return false;
}
inline bool Same(const Hash256& a, const Hash256& b) {
    return std::memcmp(a.bytes, b.bytes, 32) == 0;
}
inline Status Validate(const Config& c) {
    if (c.prefix.structSize != sizeof(c)) return Status::InvalidSize;
    if (c.prefix.abiVersion != Version || c.graphVersion != 1 || c.mathContract != MathContract)
        return Status::Unsupported;
    if (!Nonzero(c.sourceHash) || !Nonzero(c.packageHash) || c.reserved[0] || c.reserved[1])
        return Status::InvalidArgument;
    if (!SupportedLogicalExtent(c.logicalWidth, c.logicalHeight))
        return Status::Unsupported;
    if (c.networkWidth != AlignNetworkExtent(c.logicalWidth) ||
        c.networkHeight != AlignNetworkExtent(c.logicalHeight) ||
        !SupportedGraphExtent(c.networkWidth, c.networkHeight))
        return Status::InvalidArgument;
    return Status::Ok;
}
inline bool Empty(const Texture& t) {
    const auto& v = t.transform;
    return !t.resource && t.format == Format::None && !t.reserved && !v.baseX && !v.baseY &&
           !v.extentWidth && !v.extentHeight && !v.backingWidth && !v.backingHeight;
}
inline bool ValidTexture(const Texture& t, Format wanted) {
    const auto& v = t.transform;
    return t.resource && t.format == wanted && !t.reserved && v.extentWidth && v.extentHeight &&
           v.backingWidth && v.backingHeight && v.backingWidth <= 16384 && v.backingHeight <= 16384 &&
           std::uint64_t(v.baseX) + v.extentWidth <= v.backingWidth &&
           std::uint64_t(v.baseY) + v.extentHeight <= v.backingHeight;
}
// Metadata checks only. The adapter must additionally query real D3D12 descriptions,
// device/queue identity, resource states and allocation overlap before recording work.
inline Status Validate(const Frame& f, const Config& c) {
    auto status = Validate(c);
    if (status != Status::Ok) return status;
    if (f.prefix.structSize != sizeof(f)) return Status::InvalidSize;
    if (f.prefix.abiVersion != Version) return Status::Unsupported;
    if (!f.commands || !f.queue || !f.key.streamId || !f.key.submissionEpoch || f.reserved ||
        (f.resetReasons & ~(Explicit | SceneCut))) return Status::InvalidArgument;
    if (!ValidTexture(f.color, Format::Rgba16Float) || !ValidTexture(f.output, Format::Rgba16Float) ||
        (!Empty(f.motion) && !ValidTexture(f.motion, Format::Rg16Float)) ||
        (!Empty(f.controlMask) && !ValidTexture(f.controlMask, Format::Rgba16Float)))
        return Status::InvalidResource;
    if (f.output.resource == f.color.resource || f.output.resource == f.motion.resource ||
        f.output.resource == f.controlMask.resource) return Status::InvalidResource;
    const auto& out = f.output.transform;
    if (out.baseX || out.baseY || out.extentWidth != c.logicalWidth || out.extentHeight != c.logicalHeight ||
        out.backingWidth != c.logicalWidth || out.backingHeight != c.logicalHeight) return Status::Unsupported;
    const auto& controls = f.controls;
    const float values[] = {controls.styleIndex, controls.tone, controls.structure,
        controls.skinStrength, controls.autoStrength, controls.intensity};
    for (float value : values) if (!std::isfinite(value)) return Status::InvalidArgument;
    if (std::fabs(controls.styleIndex / 128.f) > 65504.f || std::fabs(controls.tone) > 65504.f ||
        std::fabs(controls.structure) > 65504.f || std::fabs(controls.skinStrength) > 65504.f ||
        std::fabs(controls.autoStrength) > 65504.f) return Status::InvalidArgument;
    const auto& m = f.motionParameters;
    const float mv[] = {m.scaleX, m.scaleY, m.previousJitterX, m.previousJitterY,
                        m.currentJitterX, m.currentJitterY};
    for (float value : mv) if (!std::isfinite(value)) return Status::InvalidArgument;
    if (m.reserved || (m.jitterMode != JitterMode::AlreadyIncluded &&
        m.jitterMode != JitterMode::AddPreviousMinusCurrent)) return Status::Unsupported;
    if (m.jitterMode == JitterMode::AlreadyIncluded && (m.previousJitterX || m.previousJitterY ||
        m.currentJitterX || m.currentJitterY)) return Status::InvalidArgument;
    if (!Empty(f.motion) && (!m.effectiveWidth || !m.effectiveHeight ||
        m.effectiveWidth > 16384 || m.effectiveHeight > 16384)) return Status::InvalidArgument;
    if (m.jitterMode == JitterMode::AddPreviousMinusCurrent &&
        (!std::isfinite(m.previousJitterX - m.currentJitterX) ||
         !std::isfinite(m.previousJitterY - m.currentJitterY))) return Status::InvalidArgument;
    return Status::Ok;
}
} // namespace NrV2
