#include <dlssnr/ModelGuideContract.h>
#include <cassert>
#include <cmath>
#include <cstdio>

int main()
{
    unsigned count = 0;
    for (auto frame : {DlssNr::GuideExtent{160, 90}, DlssNr::GuideExtent{1505, 847}})
        for (unsigned percent : {100u, 75u, 50u})
            for (bool low : {false, true})
                for (bool matched : {false, true})
                    for (bool metadata : {false, true})
                    {
                        DlssNr::GuideExtent output{2560, 1440};
                        const auto ref = low ? frame : output;
                        DlssNr::GuideExtent work{frame.width*percent/100, frame.height*percent/100};
                        DlssNr::ModelMotionReference c{frame, work, ref,
                            metadata ? frame : DlssNr::GuideExtent{},
                            metadata ? output : DlssNr::GuideExtent{}, low, matched, true};
                        auto result = DlssNr::MotionFactors(c);
                        for (auto motion : {6.f, -6.f, .5f})
                        {
                            assert(std::abs(motion*result.x/result.samples.width - motion/ref.width) < 1e-7f);
                            assert(std::abs(motion*result.y/result.samples.height - motion/ref.height) < 1e-7f);
                        }
                        // Matching resamples values; it does not introduce another
                        // model-scale multiplier into the game's displacement.
                        c.renderMotionScale = false;
                        auto legacy = DlssNr::MotionFactors(c);
                        assert(legacy.x == float(work.width)/frame.width);
                        assert(legacy.y == float(work.height)/frame.height);
                        ++count;
                    }
    auto regions = DlssNr::ResolveGuideRegions({32,16},{40,20},{11,5},{22,10},true,3,2,7,4);
    assert(regions.depth.x==3 && regions.depth.width==11);
    assert(regions.motion.x==7 && regions.motion.y==4 && regions.motion.width==11 && regions.motion.height==5);
    std::printf("PASS motion units: %u grid/scale/missing-metadata cases and independent guide subrects\n",count);
}
