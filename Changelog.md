# Changelog

## r3-0.8.91_fix-1

- Enable the frame-generation Active switch by default for the FSR 3.0 FG input → XeFG output route.
- Apply the enabled default to existing automatic configurations during installation; preserve explicitly saved on/off choices.

See the [hotfix notes](docs/release-notes/r3-0.8.91_fix-1.md).

## r3-0.8.91

- Fix critical frame-generation instability in Cyberpunk 2077: native FSR FG input no longer repeatedly enables and disables the output backend. FSR FG input to XeFG output is now the default route.
- Add optional FSR 4.0.2c INT8 DLL selection to the GUI installer. The runtime is copied into the game folder and retained on updates, without an external library path.
- Prefer FSR 4 when the loaded runtime provides it, with FSR 3 as the fallback; saved provider choices are respected.
- Enable NR by default with Standard, 75% model resolution and pre-upscale placement.
- Installation clears diagnostic views and console logging.
- Allow confirmed updates of manually replaced installation files, keeping backups and rollback support.
- Consolidate installation into one launcher and clarify HUD handling in the FG menu and user guides.

See the [release notes](docs/release-notes/r3-0.8.91.md).

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
