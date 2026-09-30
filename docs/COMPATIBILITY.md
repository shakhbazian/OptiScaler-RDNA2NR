# Compatibility and performance

[Documentation](README.md) · [Troubleshooting](TROUBLESHOOTING.md)

## Hardware

The HIP companion contains code for `gfx1030`. RX 6900 XT with 16 GB VRAM is the tested GPU. This is an architecture-specific build, not support for every card sold as RDNA2: `gfx1031` and `gfx1032`, RDNA1, Vega and later architectures are not included in this companion. Other `gfx1030` cards still need hardware qualification.

Auto selection matches HIP to the D3D12 rendering adapter. Having a supported Radeon installed alongside an unsupported rendering GPU is not sufficient. Multi-GPU execution and splitting the network between cards are not implemented.

Ordinary NVIDIA NGX compatibility remains in the frontend. It needs a separately supplied runtime and a driver/GPU combination that accepts it. This project does not remove NVIDIA's original runtime restrictions or claim NVIDIA hardware qualification from the AMD tests.

## Rendering paths

| Path | HIP status | Evidence / limitation |
| --- | --- | --- |
| D3D12 before or after SR | Implemented | Standalone regressions and user-confirmed operation in Cyberpunk 2077 |
| D3D11 through a w/Dx12 upscaler | Implemented | Standalone regressions; game use not yet qualified |
| Native D3D11-only processing | Unavailable | Select the existing D3D12 upscaler bridge instead |
| Vulkan NR | Unavailable on HIP | Parent-fork NVIDIA Vulkan work does not provide a HIP path |
| Multipass, separate-edit, finished-picture and frame-hold paths | Unavailable on HIP | The shared menu limits choices according to backend capabilities |

Standalone tests cover resolution and queue changes, feature recreation, forwarding command-list wrappers, selected resource formats, failure handling and pre/post placement. This confirms specific contracts, rather than compatibility with every game's wrappers or HDR implementation.

## Inputs and formats

NR is attached to a supported upscaler evaluation, rather than a generic desktop capture. A game needs a DLSS, FSR or XeSS input path that OptiScaler can intercept. Driver-level Radeon Super Resolution and frame generation alone do not expose that evaluation to OptiScaler.

The frontend passes colour, depth, motion vectors and frame metadata when available. Colour encoding and output composition stay in OptiScaler; the HIP backend consumes the common model representation. Standalone coverage includes FP16, FP32, RGBA8 and R11G11B10 colour cases and HDR/exposure handling. A game-specific state or format that cannot be qualified retains the raw upscaler path.

## Resolution and memory

There is no product cutoff at 1440p. Diagnostic hosts have exercised frames up to 3840×2160. This establishes the ability to process those dimensions, not acceptable 4K performance or VRAM headroom while a game is running.

Resolution changes can require a new resource/session preparation. During that transition, frames may temporarily use the raw path. Larger dimensions consume more scratch/history storage as well as more computation. Available memory must cover the game, NR, the upscaler and frame generation together.

## Performance

This is a developer preview. No playable frame-rate target is guaranteed. The original network remains computationally expensive even with the mixed INT8/FP16 executor.

Pre-upscale NR usually costs less because it processes the internal render size. Post-upscale NR processes the upscaler output. A game's lower render resolution or a different SR preset can reduce the workload; HIP's own working-size setting remains 100%.

Compare total frame time with NR disabled and enabled after preparation has completed, in the same scene with the same settings. Model-only timers do not include the entire integration, and a background game workload competes for the same GPU. Account for real rendered frames separately from generated frames.

Cyberpunk 2077 currently confirms that NR is applied in a game. It is not a broad game compatibility matrix or a performance certification.
