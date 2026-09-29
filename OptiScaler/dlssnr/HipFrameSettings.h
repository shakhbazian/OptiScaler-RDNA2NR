#pragma once
#include "PassProfiles.h"
#include "../../rdna2_nr/include/nr_runtime_contract.h"

namespace DlssNr::HipFrameSettings {
inline NrV2::Controls Controls(const Config& config, unsigned pass = 0) {
    const auto tuning=Profiles::PassSettings(config,pass);
    return {static_cast<float>(tuning.style),tuning.localTone,tuning.localStructure,
            tuning.skinStructure,tuning.autoMask?1.f:0.f,tuning.intensity};
}
inline bool TemporalAccumulation(const Config& config) {
    return config.DlssNrTemporalAccumulation.value_or_default();
}
}
