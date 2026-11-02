# Architecture

[Documentation](README.md) · [Building](BUILDING.md)

## Frontend and backend

OptiScaler supplies the game hooks, upscaler routing, NR placement, colour processing, menu and settings. A backend selection layer chooses ordinary NVIDIA NGX or the AMD HIP companion for the rendering adapter. The companion is a separate DLL because HIP compilation and GPU execution have different dependencies from the frontend.

On AMD, `nvngx_dlssnr.dll` is an offline source of model data. The game does not use it as an NVIDIA inference runtime. On NVIDIA, the retained NGX path uses a user-supplied runtime accepted by the driver.

## Frame path

1. The game's supported upscaler evaluation supplies colour and available depth, motion and frame metadata.
2. OptiScaler records the common NR colour preparation and composition work around the selected pre/post-upscale point.
3. At the intercepted D3D12 queue submission, the transport orders the input transfer, HIP inference and output publication relative to the game's command lists.
4. The game resumes through its ordinary upscaler/rendering path with the edited image. An unqualified or failed admission keeps the raw path.

Frames remain on the GPU. D3D12 and HIP exchange resources and use completion fences to order the transfers and inference.

The transport uses the actual submitted queue and rendering adapter. Swapchain or frame-generation presentation queues need not be the same as the queue executing the upscaler. Readiness during command recording does not require those queues to match; submission validates the actual execution queue before scheduling GPU work.

For D3D11, OptiScaler's existing w/Dx12 upscaler bridge reaches the same D3D12 NR integration. There is no native HIP Vulkan transport in this release.

## Network and arithmetic

The converter restores NVIDIA's network tensors into a verified FP16 package. Model A is the original network; B is NVIDIA's optimized network from NR 310.9.16.0. The runtime preserves each model's topology and control inputs.

Both models combine INT8 matrix operations with FP16 attention and other sensitive stages. A uses calibrated scales for selected projections and feed-forward layers, with folded scaling and INT32 accumulation. B uses dynamic activation scales in large matrices and calibrated static INT8 in its final 32-channel feed-forward block. Weights are prepared for these execution paths at startup; the stored model packages remain FP16.

Custom HIP kernels, fused operations and shared workspaces reduce intermediate transfers and repeated packing. Each model has its own calibration and layout metadata.

Style, Intensity, Local structure, Local tone, Skin structure and Auto skin mask are inputs to the network. Transfer and Colour strength act later, when composing its edit with the frame. At reduced NR resolutions, enlargement and optional halo filtering also happen during this composition stage.

## Lifetime and failure handling

Model state, scratch buffers and temporal history belong to a device/session. Model switches, resolution changes, feature recreation and queue changes are handled through preparation and replacement rather than reusing incompatible state. Resources stay alive until their submitted GPU work retires.

The frontend only schedules qualified frames. Unsupported formats, states, command-list wrappers or configuration combinations can bypass NR while the upscaler continues. A fatal backend error is reported in the menu; after fixing its cause, restart the game to recreate the session.

## Source map

Paths below are relative to the repository root.

| Area | Location |
| --- | --- |
| Backend selection and capabilities | `OptiScaler/dlssnr/NrBackendSelection.cpp` and related headers |
| Game-facing NR pipeline and controls | `OptiScaler/dlssnr/` and `OptiScaler/shaders/dlssnr/` |
| HIP model, transport and weight validation | `rdna2_nr/src/` and `rdna2_nr/include/` |
| Offline model conversion | `tools/model_converter/` |
| Installer and release assembly | `Install-RDNA2NR.ps1`, `package_release.ps1` |
| Standalone integration regressions | `tests/rdna2/` and `rdna2_nr/tests/` |

The [parent-fork notes](upstream/README.md) describe inherited NVIDIA features in more detail. They are not the HIP capability specification.
