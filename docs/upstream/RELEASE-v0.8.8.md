> Inherited parent-fork reference. This document describes NVIDIA paths or historical work; see the [RDNA2NR documentation](../README.md) for current AMD support.

# v0.8.8 prerelease - optional lighting and colour reconstruction

This release adds the Display Filter structural composition approach as optional OptiScaler NR enlargement modes. Below 100% model resolution, resizing an RGB residual can introduce a bright rim between dark hair and light skin even when the raw model answer has no rim. The new path resizes relative lighting gain and chromaticity changes separately, reconstructs the model answer on the full-resolution proxy, then applies the existing final composition controls once.

In the NR menu, set model resolution below 100% and choose:

- **Lighting + colour**: bilinear reconstruction, `Transfer=3` under `[DlssNr]`. Available on DX12 and native Vulkan processing paths.
- **Lighting + colour + DLSS**: the same fields enlarged by private DLSS SR, `Transfer=4`. Requires post-upscale DX12 processing, including the existing DX12 bridges. It is not available before the game upscaler or in native Vulkan processing.

The default remains Matched residual (`Transfer=1`). Classic, Matched residual and Matched residual + DLSS retain their existing algorithms. Native/supersampled model processing bypasses the new reconstruction. Raw model inspection remains available. Switching private DLSS carrier types retires the old feature through the existing GPU lifetime tracking, preventing mixed temporal history.

The new private-DLSS mode includes the standalone path's spatial carrier range guard and inverse-HDR white-limit fallback. Fully black source pixels remain black; relative lighting cannot carry newly generated light at true zero. This targets resize-induced halos, not artifacts already present in the NR output. See [the enlargement documentation](NR-DLSS-ENLARGEMENT.md).

## Validation

- 300 shipped-shader WARP fixtures: 25%, 65%, 99%, 100% and 101% resolution, SDR and all five HDR modes, brightening/darkening and colour edits. Maximum difference from the native-resolution synthetic reference was 0.0000018 in the tested output units.
- Synthetic private-carrier encode/reconstruction and overshoot guards, model bypass, alpha preservation and true-black handling.
- 864 consumed shader outputs for the existing modes matched the v0.8.7 shader bit-for-bit.
- Full automated NR prerelease suite, including GPU lifetime, shutdown, proxy, pipeline, queue-readiness and Vulkan shader checks.
- Standard and optional RTX40 MFG Release x64 builds and package checksum verification.

No games were launched. The fixtures do not evaluate NVIDIA NR or private DLSS themselves, and do not establish real-game image quality or performance. Private-DLSS temporal behaviour and gameplay remain for user verification. The RTX40 MFG implementation is unchanged; no RTX40 hardware validation is claimed. Existing v0.8.7 queue-safety limitations remain.

Use `OptiScaler-NR-v0.8.8.zip` for the standard build or `OptiScaler-NR-v0.8.8-rtx40-mfg.zip` for the optional RTX40 MFG build. Extract the complete package. NVIDIA model/FG runtime DLLs are not bundled. No game installations are modified by this release task.
