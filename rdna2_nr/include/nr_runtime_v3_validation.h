#pragma once
#include "nr_runtime_v3_contract.h"
#include "nr_runtime_validation.h"
#include <limits>

namespace NrV3 {
constexpr std::uint64_t ReservedSerial = std::numeric_limits<std::uint64_t>::max();
inline Status Convert(NrV2::Status value) { return static_cast<Status>(value); }
template<class T> inline Status Decode(const void* data, std::size_t bytes, T& result) {
    if (!data || bytes < sizeof(Prefix)) return Status::InvalidSize;
    Prefix prefix{};
    std::memcpy(&prefix, data, sizeof(prefix));
    if (prefix.abiVersion != Version) return Status::Unsupported;
    if (prefix.structSize != sizeof(T) || bytes < sizeof(T)) return Status::InvalidSize;
    std::memcpy(&result, data, sizeof(T));
    return Status::Ok;
}
template<class T> inline Status Header(const T& value) {
    if (value.prefix.abiVersion != Version) return Status::Unsupported;
    return value.prefix.structSize == sizeof(T) ? Status::Ok : Status::InvalidSize;
}
inline bool Same(const Token& a, const Token& b) {
    return a.generation == b.generation && a.key.streamId == b.key.streamId &&
        a.key.frameIndex == b.key.frameIndex && a.key.submissionEpoch == b.key.submissionEpoch;
}
inline bool Valid(const Token& t) {
    return t.key.streamId && t.key.submissionEpoch && t.key.submissionEpoch != ReservedSerial &&
        t.generation && t.generation != ReservedSerial;
}
inline Status Validate(const TokenRequest& r) {
    const auto h = Header(r);
    return h != Status::Ok ? h : Valid(r.token) ? Status::Ok : Status::InvalidArgument;
}
inline NrV2::Config AsV2(const Config& c) {
    return {{sizeof(NrV2::Config), NrV2::Version}, c.sourceHash, c.packageHash,
        c.graphVersion, c.mathContract, c.logicalWidth, c.logicalHeight,
        c.networkWidth, c.networkHeight, {c.reserved[0], c.reserved[1]}};
}
inline Status Validate(const Config& c) {
    const auto h = Header(c);
    return h != Status::Ok ? h : Convert(NrV2::Validate(AsV2(c)));
}
inline Status Validate(const Input& f, const Config& c, bool scheduledPolicy = false) {
    auto s = Validate(c); if (s != Status::Ok) return s;
    s = Header(f); if (s != Status::Ok) return s;
    if (!Valid({f.key, f.generation})) return Status::InvalidArgument;
    if (!f.commands || !f.queue ||
        (scheduledPolicy ? (f.reserved & ~DisableTemporalAccumulation) : f.reserved) ||
        (f.resetReasons & ~(NrV2::Explicit | NrV2::SceneCut)))
        return Status::InvalidArgument;
    if (!NrV2::ValidTexture(f.color, NrV2::Format::Rgba16Float) ||
        (!NrV2::Empty(f.motion) && !NrV2::ValidTexture(f.motion, NrV2::Format::Rg16Float)) ||
        (!NrV2::Empty(f.controlMask) && !NrV2::ValidTexture(f.controlMask, NrV2::Format::Rgba16Float)))
        return Status::InvalidResource;
    if (f.color.resource == f.motion.resource || f.color.resource == f.controlMask.resource ||
        (f.motion.resource && f.motion.resource == f.controlMask.resource)) return Status::InvalidResource;
    // Matches the existing texture adapter; scaled motion/mask remain supported.
    if (f.color.transform.extentWidth != c.logicalWidth || f.color.transform.extentHeight != c.logicalHeight)
        return Status::Unsupported;
    const auto& v = f.controls;
    const float controls[] = {v.styleIndex, v.tone, v.structure, v.skinStrength, v.autoStrength, v.intensity};
    for (float x : controls) if (!std::isfinite(x)) return Status::InvalidArgument;
    if (std::fabs(v.styleIndex / 128.f) > 65504.f || std::fabs(v.tone) > 65504.f ||
        std::fabs(v.structure) > 65504.f || std::fabs(v.skinStrength) > 65504.f ||
        std::fabs(v.autoStrength) > 65504.f) return Status::InvalidArgument;
    const auto& m = f.motionParameters;
    const float motion[] = {m.scaleX,m.scaleY,m.previousJitterX,m.previousJitterY,m.currentJitterX,m.currentJitterY};
    for (float x : motion) if (!std::isfinite(x)) return Status::InvalidArgument;
    if (m.reserved || (m.jitterMode != NrV2::JitterMode::AlreadyIncluded &&
        m.jitterMode != NrV2::JitterMode::AddPreviousMinusCurrent)) return Status::Unsupported;
    if (m.jitterMode == NrV2::JitterMode::AlreadyIncluded &&
        (m.previousJitterX || m.previousJitterY || m.currentJitterX || m.currentJitterY))
        return Status::InvalidArgument;
    if (!NrV2::Empty(f.motion) && (!m.effectiveWidth || !m.effectiveHeight ||
        m.effectiveWidth > 16384 || m.effectiveHeight > 16384)) return Status::InvalidArgument;
    if (m.jitterMode == NrV2::JitterMode::AddPreviousMinusCurrent &&
        (!std::isfinite(m.previousJitterX - m.currentJitterX) ||
         !std::isfinite(m.previousJitterY - m.currentJitterY))) return Status::InvalidArgument;
    return Status::Ok;
}
inline Status Validate(const Output& o, const Config& c) {
    auto s = Validate(c); if (s != Status::Ok) return s;
    s = Header(o); if (s != Status::Ok) return s;
    if (!Valid(o.token) || !o.commands || !o.queue || o.reserved[0] || o.reserved[1])
        return Status::InvalidArgument;
    if (!NrV2::ValidTexture(o.target, NrV2::Format::Rgba16Float)) return Status::InvalidResource;
    const auto& t = o.target.transform;
    if (t.baseX || t.baseY || t.extentWidth != c.logicalWidth || t.extentHeight != c.logicalHeight ||
        t.backingWidth != c.logicalWidth || t.backingHeight != c.logicalHeight) return Status::Unsupported;
    return Status::Ok;
}
} // namespace NrV3
