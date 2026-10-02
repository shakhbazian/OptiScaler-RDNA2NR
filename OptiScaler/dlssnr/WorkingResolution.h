#pragma once

#include <algorithm>
#include <cmath>

namespace DlssNr
{
struct WorkingResolution
{
    unsigned width;
    unsigned height;
    float scale;
};

// Admission, warmup and the frontend must agree on the rounded model size.
// Otherwise a scaled pass continually alternates between two HIP sessions.
inline WorkingResolution ModelResolution(unsigned width, unsigned height, float scale)
{
    scale = std::isfinite(scale) ? std::clamp(scale, 0.25f, 2.0f) : 1.0f;
    return { std::max(1u, static_cast<unsigned>(width * scale + 0.5f)),
             std::max(1u, static_cast<unsigned>(height * scale + 0.5f)), scale };
}
} // namespace DlssNr
