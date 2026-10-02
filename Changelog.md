# Changelog

## r2-0.8.91

- Faster HIP matrix execution with calibrated static INT8, folded scales, longer INT32 accumulation, fused handoffs and attention workspace reuse. The network and controls are retained; 32-channel QKV and attention remain FP16.
- NR model resolution now supports 25–200% before or after upscaling while preserving the output frame size.
- New defaults: Standard, 75% NR resolution, pre-upscale placement, FSR 3 when available and XeFG when its libraries are present. Existing INI files are preserved.
- Fix NR being skipped when the shared D3D12 lifetime tracker is active.
- Reuses the existing converted model cache. See the [release notes](docs/release-notes/r2-0.8.91.md).

## r1-0.8.91

First public release of OptiScaler-RDNA2NR, based on wilsjo2's OptiScaler-DLSSNR v0.8.91.

- AMD HIP execution of the original NR network on `gfx1030`, using mixed INT8/FP16 arithmetic.
- One-pass NR before or after upscaling on D3D12, and through OptiScaler's D3D11 w/Dx12 upscaler bridge.
- Backend Auto selection for the rendering adapter, with manual HIP/NVIDIA selection.
- Local model conversion and GUI installation from a user-supplied DLSS 5 Neural Rendering 310.8.0.0 DLL.
- Queue, resolution and feature-lifetime handling, with raw-frame fallback for unqualified submissions.
- User documentation and source-check automation.

NR application in Cyberpunk 2077 is confirmed on RX 6900 XT. Performance remains a limitation; D3D11 has standalone coverage and HIP Vulkan support is not implemented. See [compatibility](docs/COMPATIBILITY.md) and the [release notes](docs/release-notes/r1-0.8.91.md).
