# OptiScaler-RDNA2NR

DLSS Neural Rendering on Radeon, integrated into OptiScaler.

This fork adds a HIP backend for RDNA2 to [wilsjo2's OptiScaler-DLSSNR](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass). It keeps OptiScaler's DLSS/FSR/XeSS routing, frame-generation features and ordinary NVIDIA NGX compatibility. NR can run before or after upscaling, with controls for lighting, detail and colour.

**Developer preview:** NR is working in Cyberpunk 2077 on an RX 6900 XT. Performance is still a major limitation; broad game compatibility and playable frame rates are not guaranteed. DirectX 11 support has been checked in standalone tests, rather than games.

The original network is retained, with selected matrix operations in INT8 and the remaining paths in FP16. This is not a trained replacement network. **You supply the original model DLL; the installer prepares its weights locally. No NVIDIA model or driver runtime is distributed.**

## Requirements

For the AMD backend:

| Requirement | What to install or provide |
| --- | --- |
| Windows x64 with DirectX 12 | Tested on Windows 11. |
| A `gfx1030` Radeon GPU | Tested on RX 6900 XT, 16 GB. Other `gfx1030` cards are not yet qualified; this build does not cover every RDNA2 GPU. |
| AMD driver with the HIP 6 runtime | Install a [compatible AMD graphics driver](https://www.amd.com/en/support/download/drivers.html). Adrenalin 26.8.1 is the tested version, not an established minimum. The runtime must provide `amdhip64_6.dll`. |
| Microsoft Visual C++ runtime, x64 | Install the [Visual C++ v14 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) ([x64 download](https://aka.ms/vc14/vc_redist.x64.exe)). |
| Your own compatible `nvngx_dlssnr.dll` | One source version is supported, identified by its [SHA-256 hash](docs/MODEL.md#supported-source). |
| A game with a supported upscaler input | OptiScaler must intercept the game's DLSS, FSR or XeSS calls. Use single-player games. |

**No separate Python, HIP SDK, rocBLAS or Visual Studio installation is needed to use a packaged build.** The converter includes portable Python and NumPy. Allow space for the extracted release, about 278 MiB for the converted model, and backups of any files replaced in the game folder.

## Quick start

1. Close the game. Extract the **complete release archive** into a separate folder.
2. Run **`Install-RDNA2NR.cmd`**. Select the folder containing the game's actual executable and your original `nvngx_dlssnr.dll`. Keep `dxgi.dll` unless the game's installation guidance calls for another proxy. Click **Install / update**.
3. Start the game and enable a supported in-game upscaler. **FSR is a valid input on AMD; you do not need to enable NVIDIA DLSS.**
4. Press **Insert**, open **DLSS Neural Rendering**, and enable NR. Select **Auto** for the backend. For a less expensive starting point, enable **before upscaling**; keep one pass, 100% working size and **Apply model** enabled.
5. Wait for model preparation. **AMD HIP (gfx1030)** identifies the backend; **RDNA2 NR active** with a growing frame count confirms that frames are being processed. Toggle NR to compare the image.

The installer converts and verifies the model before replacing game files. It keeps an existing `OptiScaler.ini`, backs up replaced files, and reuses a verified model cache on subsequent installations. A fresh configuration leaves NR disabled until you enable it.

For DirectX 11, choose an upscaler marked **w/Dx12**; for example, `Dx11Upscaler=fsr22_12` in the `[Upscalers]` section. See [installation](docs/INSTALLATION.md) for updates, removal and command-line use, or [troubleshooting](docs/TROUBLESHOOTING.md) if NR is inactive.

## Supported NR paths

| Path | AMD HIP backend |
| --- | --- |
| DirectX 12 | One pass before or after upscaling |
| DirectX 11 | Through OptiScaler's DirectX 12 upscaler bridge |
| Vulkan | Not implemented for this backend |
| Multipass, finished-picture, separate-edit or frame-hold modes | Not supported by the HIP backend |

The NVIDIA NGX route remains available on supported NVIDIA hardware with a user-supplied runtime. AMD and NVIDIA do not have the same set of NR features. NR backend selection is independent of the selected upscaler and frame generator.

NR shares GPU time and memory with the game, upscaling and frame generation. Running it before upscaling normally processes fewer pixels. There is no fixed 1440p cutoff, but larger frames can be slow or exceed available VRAM. See [compatibility and performance](docs/COMPATIBILITY.md).

## Documentation

- [Install, update and uninstall](docs/INSTALLATION.md)
- [NR settings and comparisons](docs/SETTINGS.md)
- [Model preparation and storage](docs/MODEL.md)
- [Compatibility and performance](docs/COMPATIBILITY.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [How the integration works](docs/ARCHITECTURE.md)

## Build from source

Install Git, PowerShell 7, Visual Studio's C++ desktop tools and Windows SDK, the **v145 platform toolset with MSVC 14.44 binaries**, and **AMD HIP SDK 6.4**. The scripts default to Visual Studio 18 Community and `C:\Program Files\AMD\ROCm\6.4`; see [build instructions](docs/BUILDING.md) for CMake, Python, paths and packaging requirements.

From a checkout with its submodules initialized:

```powershell
git submodule update --init --recursive
.\tools\Build-HipBackend.ps1
.\tools\Build-OptiScaler.ps1
```

The outputs are `build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll` and `x64/Release/OptiScaler.dll`. A source build still needs locally prepared model weights to run NR.

## Origins and credits

This is an unofficial fork of [wilsjo2's OptiScaler-DLSSNR](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass), built on [OptiScaler](https://github.com/optiscaler/OptiScaler) and earlier NR integration work by [Dagherbou](https://github.com/Dagherbou/OptiScaler_DLSSNR). OptiScaler began with [CyberFSR2](https://github.com/PotatoOfDoom/CyberFSR2); parts of the NR colour processing derive from [RenoDX](https://github.com/clshortfuse/renodx). See [credits](docs/CREDITS.md), [third-party notices](Licenses), and the [GPL-3.0 license](LICENSE). Model ownership is separate from the code license.

For OptiScaler's general features and game-specific advice, use the [upstream wiki](https://github.com/optiscaler/OptiScaler/wiki) and [community Discord](https://discord.gg/wEyd9w4hG5). Support upstream development through [cdozdil](https://github.com/sponsors/cdozdil?frequency=one-time) or [nitec](https://buymeacoffee.com/nitec).
