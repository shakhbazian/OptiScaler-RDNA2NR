#pragma once
#include <shaders/dlssnr/DlssNr_Guides.h>

namespace DlssNr
{
struct ModelMotionReference
{
    GuideExtent frame{}, work{}, guide{}, render{}, output{};
    bool lowResolution = false;
    bool matched = false;
    bool renderMotionScale = true;
};
struct ModelMotionFactors
{
    GuideExtent samples, reference;
    float x = 1, y = 1;
};

// NGX measures its input in pixels of the supplied motion grid. The game can
// instead express displacement in render pixels or output pixels. Resampling
// the guide changes the grid, not the vector values, so convert units once.
inline ModelMotionFactors MotionFactors(const ModelMotionReference& source)
{
    ModelMotionFactors result;
    result.samples = !source.renderMotionScale || source.matched ? source.work : source.guide;
    result.reference = !source.renderMotionScale ? source.frame : source.lowResolution ? source.render : source.output;
    // Missing metadata uses the original effective guide region, before matching.
    if (!result.reference.width) result.reference.width = source.guide.width;
    if (!result.reference.height) result.reference.height = source.guide.height;
    result.x = result.reference.width ? float(result.samples.width) / result.reference.width : 1.f;
    result.y = result.reference.height ? float(result.samples.height) / result.reference.height : 1.f;
    return result;
}
} // namespace DlssNr
