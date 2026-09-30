> Inherited parent-fork reference. This document describes NVIDIA paths or historical work; see the [RDNA2NR documentation](../README.md) for current AMD support.

# v0.8.5 prerelease

The NR branch is rebased onto OptiScaler master `93fbf1b2` (20 September 2026), including its current descriptor caching, resource tracking and XeFG synchronization changes.

## Fixes

- NR can initialize with the legacy menu when the presentation counter stays at zero. Actual GPU completion now releases the initialization gate.
- D3D12 finished-picture composition acquires swapchain buffers before NR locks. Disabled NR does not query the swapchain, avoiding the OptiFG lock inversion.
- Non-native finished-picture composition waits only for its selected, compatible, submitted producer. Native Streamline handoffs retain the no-wait rule.
- After-upscale and deferred NR support multi-mip, single-layer output textures. Copies address mip zero; input and output ownership remain separate. Texture arrays remain unsupported.
- A failed forced RR initialization falls back to the detected upscaler rather than unconditionally selecting FSR 2.1.2. No create-time guess about G-buffer pointers is used.
- Zero or non-finite game motion-vector scales use the NGX default. Explicit frame-hold motion remains unaffected.

## Optional controls

- **Restore sharpness** adds bounded luminance detail in Replace modes below 100% model resolution. It defaults to zero, preserves hue and avoids crushing dark edges to black.
- **White point source** offers manual, game exposure or automatic HDR metering. Automatic exposure runs on the GPU in D3D12 and Vulkan, with highlight protection, a separate trim (default 5), and up to eight logarithmic trim anchors. Enter anchors as `1:5 100:2`; game and automatic sources retain separate curves. Missing/invalid exposure falls back to manual paper white. Direct finished-picture processing retains its own white-point override.
- **History confidence threshold** optionally makes RR residual accumulation respond faster to disagreement. Zero preserves the existing blend.
- **Profiles** save/load named settings beside the DLL in `OptiScalerProfiles`. Startup-only settings still require a game restart. Ctrl/Alt requirements for the overlay shortcut are available through `[Menu] ShortcutKeyRequireCtrl` and `ShortcutKeyRequireAlt`.
- GPU diagnostics report rolling mean and p99 NR/model timings. These intervals can include other GPU work.

## Packages

Choose **OptiScaler-NR-v0.8.5.zip** for the standard build. The separate **OptiScaler-NR-v0.8.5-rtx40-mfg.zip** adds OTA provider discovery, Streamline ceiling handling, presented-frame telemetry, optional Ada PTX correction and optional software pacing. The unlock, PTX selection and pacing patch remain opt-in; hardware-specific gameplay validation on RTX 40 is still needed.

NVIDIA NR/FG runtime binaries are not included. Retain your existing runtime installation. Back up the previous DLL and INI before testing.

## Validation and limits

Release x64 builds and the targeted prerelease suite cover GPU lifetime/replay/concurrency, proxy readiness, MFG patch helpers, capture, native Streamline routing, multi-mip copies, cross-queue ordering, residual composition, sharpness and exposure. WARP exercises production D3D shader bytes; an RTX 5090 runs the production Vulkan shader. Shader headers are checked against their compiled binaries, and both archives include SHA256 manifests.

This is a prerelease: no new full game-session validation or RTX 40 hardware validation is claimed. Half-rate NR and evaluation cadence are excluded. Resource scanning, temporary mask/distortion probes and research-download tooling are excluded. Smooth Motion interoperability remains outside this change.

Based on ideas and contributions in PRs [#54](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/54), [#70](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/70), [#72](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/72), [#77](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/77), [#79](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/79), [#80](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/80), [#81](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/81) and [#82](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/82). See [implementation review](PR-REWRITE-REVIEW-v0.8.5.md) for correctness and scope decisions.
