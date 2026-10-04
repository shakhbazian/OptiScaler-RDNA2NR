# OptiScaler-RDNA2NR

DLSS Neural Rendering on Radeon, integrated into OptiScaler.

This fork adds a HIP backend for RDNA2 to [wilsjo2's OptiScaler-DLSSNR](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass). It keeps OptiScaler's DLSS/FSR/XeSS routing, frame-generation features and ordinary NVIDIA NGX compatibility. NR can run before or after upscaling, with controls for lighting, detail and colour.

**Research project:** NR is working in Cyberpunk 2077 on an RX 6900 XT. A user test reached approximately 45 FPS at 1440p with XeFG frame generation, FSR 3 in Performance mode and NR at 75% before upscaling. This is a reported game result, rather than a controlled benchmark. DirectX 11 support has been checked in standalone tests.

The original network is retained, with selected matrix operations in INT8 and the remaining paths in FP16. You supply the original model DLL; the installer prepares its weights locally.

## Requirements

For the AMD backend:

| Requirement | What to install or provide |
| --- | --- |
| Windows x64 | DirectX 12 support. |
| A `gfx1030` Radeon GPU | RX 6900 XT; see [hardware compatibility](docs/COMPATIBILITY.md#hardware). |
| AMD Software: Adrenalin Edition | Install the [Radeon RX 6000 driver](https://www.amd.com/en/support/download/drivers.html) with the HIP 6 runtime. |
| Microsoft Visual C++ runtime, x64 | Install the [Visual C++ v14 Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) ([x64 download](https://aka.ms/vc14/vc_redist.x64.exe)). |
| DLSS 5 Neural Rendering 310.8.0.0 | Obtain `nvngx_dlssnr.dll` from a game that includes DLSS 5 Neural Rendering. Use the original build matching the [supported SHA-256](docs/MODEL.md#supported-source). |
| A game with a supported upscaler input | OptiScaler must intercept the game's DLSS, FSR or XeSS calls. Use single-player games. |

Allow space for the extracted release, about 278 MiB for the converted model, and backups of any files replaced in the game folder.

## Quick start

1. Download the installer ZIP from [Releases](https://github.com/shakhbazian/OptiScaler-RDNA2NR/releases). Close the game and extract the complete archive into a separate folder.
2. Run `Install-RDNA2NR.cmd`. Select the folder containing the game's actual executable and your original `nvngx_dlssnr.dll`. Keep `dxgi.dll` unless the game's installation guidance calls for another proxy. Click **Install / update**.
3. Start the game and enable a supported in-game upscaler. FSR is a valid input on AMD.
4. Press **Insert** and open **DLSS Neural Rendering**. NR is enabled in a fresh configuration, with **Auto** backend, **Standard**, **Model resolution** at 75%, **before upscaling**, one pass and **Apply model** enabled.
5. Wait for model preparation. **AMD HIP (gfx1030)** identifies the backend; **RDNA2 NR active** with a growing frame count confirms that frames are being processed. Toggle NR to compare the image.

To lower the network's working resolution while keeping the frame size, adjust **Model resolution**. At 50%, a 1080p NR input runs at 540p. See [settings](docs/SETTINGS.md#model-resolution) for placement and composition choices.

The installer converts and verifies the model before replacing game files. It keeps an existing `OptiScaler.ini`, backs up replaced files, and reuses a verified model cache on subsequent installations. A fresh configuration selects FSR 4 when the loaded runtime offers it, otherwise FSR 3, and FSR 3.0 FG → XeFG for frame generation. To use this FG route, enable the game's native FSR frame generation and **Active** in OptiScaler's FG menu. Games with a different FG input need the matching source; **OptiFG (Upscaler)** is available for games without native FG. Save settings and restart after changing the FG source or output.

For DirectX 11, choose an upscaler marked **w/Dx12**; for example, `Dx11Upscaler=ffx_12` in the `[Upscalers]` section. See [installation](docs/INSTALLATION.md) for updates, removal and command-line use, or [troubleshooting](docs/TROUBLESHOOTING.md) if NR is inactive.

**FSR 4 INT8 on RDNA2:** an optional FSR 4.0.2c INT8 runtime can be used with
the HIP NR backend. Select its DLL in the installer; it is copied into the game
folder and retained on updates. See [setup and verification](docs/INSTALLATION.md#optional-fsr-4-int8-on-rdna2).

## Supported NR paths

| Path | AMD HIP backend |
| --- | --- |
| DirectX 12 | One pass before or after upscaling |
| DirectX 11 | Through OptiScaler's DirectX 12 upscaler bridge |
| Vulkan | Not implemented for this backend |
| Multipass, finished-picture, separate-edit or frame-hold modes | Not supported by the HIP backend |

The NVIDIA NGX route remains available on supported NVIDIA hardware with a user-supplied runtime. AMD and NVIDIA do not have the same set of NR features. NR backend selection is independent of the selected upscaler and frame generator.

NR processes the image at the selected stage and model scale. Running it before upscaling normally processes fewer pixels. Higher resolutions increase processing time and VRAM usage; large frames can exhaust the memory available alongside the game, upscaling and frame generation. See [compatibility and performance](docs/COMPATIBILITY.md).

## Documentation

- [Install, update and uninstall](docs/INSTALLATION.md)
- [NR settings and comparisons](docs/SETTINGS.md)
- [Model preparation and storage](docs/MODEL.md)
- [Compatibility and performance](docs/COMPATIBILITY.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [How the integration works](docs/ARCHITECTURE.md)

## Build from source

Install Git, PowerShell 7, Visual Studio's C++ desktop tools and Windows SDK, the v145 platform toolset with MSVC 14.44 binaries, and AMD HIP SDK 6.4. The scripts default to Visual Studio 18 Community and `C:\Program Files\AMD\ROCm\6.4`; see [build instructions](docs/BUILDING.md) for CMake, Python, paths and packaging requirements.

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
