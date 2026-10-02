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

This path exchanges GPU resources and synchronizes D3D12/HIP work. It is not a CPU screenshot-upload loop. Transfers and synchronization still have a cost; retaining GPU residency does not imply zero-copy inference or zero overhead.

The transport uses the actual submitted queue and rendering adapter. Swapchain or frame-generation presentation queues need not be the same as the queue executing the upscaler. Readiness during command recording does not require those queues to match; submission validates the actual execution queue before scheduling GPU work.

For D3D11, OptiScaler's existing w/Dx12 upscaler bridge reaches the same D3D12 NR integration. There is no native HIP Vulkan transport in this release.

## Network and arithmetic

The converter restores the original network's tensors into a checked FP16 package. The runtime retains the network topology and control inputs. It does not use a separately trained compact replacement.

The executor uses INT8 for selected matrix projections and feed-forward layers, including the 32-channel feed-forward stages. Activation scales are calibrated constants; selected projections fold them into the weight scales at load time. Integer dot products accumulate in INT32 before conversion, reducing repeated scale operations. Sensitive 32-channel QKV and attention remain FP16. Other stages use their supported FP16/float arithmetic. Fused kernels and shared workspaces reduce intermediate transfers and repeated packing. Custom HIP kernels provide the execution path; rocBLAS is not a runtime dependency. The conversion package itself is FP16, so offline conversion and runtime quantization are distinct steps.

The Style, Intensity, Local structure, Local tone, Skin structure and Auto skin mask controls reach the original network. Transfer and Colour strength are later composition controls. Keeping the network and controls does not claim numerical identity with the original NVIDIA runtime.

## Lifetime and failure handling

Model state, scratch buffers and temporal history belong to a device/session. Resolution changes, feature recreation and queue changes are handled through preparation and replacement rather than reusing incompatible state. Resources stay alive until their submitted GPU work retires.

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
