#pragma once
#include <cstdint>
#include <cstddef>

// Plain device layout shared by the HIP producer and its host-side oracle.
// A single HIP stream is the writer. The next frame reads it only after the
// preceding resolve kernel; no CPU observation is needed between frames.
namespace NrScheduled {
struct Frame {
    std::uint64_t streamId, frameIndex;
    std::uint32_t reset, hasMotion, disableAccumulation;
};
struct State {
    std::uint64_t committedStream, committedFrame;
    std::uint32_t valid, noise, commits, lastInvalid;
};
struct Decision {
    std::uint32_t useHistory, noise;
};
static_assert(sizeof(Frame)==32 && sizeof(State)==32 && sizeof(Decision)==8);
static_assert(offsetof(State,valid)==16 && offsetof(State,lastInvalid)==28);
} // namespace NrScheduled
