# OptiScaler-RDNA2NR

DLSS 5 Neural Rendering on Radeon, integrated into OptiScaler.

This fork adds an AMD HIP backend to [wilsjo2's OptiScaler-DLSSNR](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass), retaining OptiScaler's upscaler routing, frame generation and NVIDIA NGX support. NR can run before or after upscaling, with adjustable model resolution, lighting, detail and colour.

The project uses NVIDIA's NR networks with mixed INT8/FP16 execution on AMD. You supply the original model DLL; the installer prepares the models locally. This is a research project, tested on an RX 6900 XT in Cyberpunk 2077 and The Witcher 3 (DirectX 12).

[Download r4-0.8.10](https://github.com/shakhbazian/OptiScaler-RDNA2NR/releases/tag/r4-0.8.10) · [Changelog](Changelog.md) · [Documentation](docs/README.md)

## Requirements

For the AMD backend:

| Requirement | Details |
| --- | --- |
| Windows x64 | DirectX 12 support |
| A `gfx1030` Radeon | RX 6900 XT is tested; see [hardware compatibility](docs/COMPATIBILITY.md#hardware) |
| AMD graphics driver | [AMD Software: Adrenalin Edition for Radeon RX 6000](https://www.amd.com/en/support/download/drivers.html), including the HIP 6 runtime |
| Visual C++ runtime, x64 | [Microsoft Visual C++ v14 Redistributable](https://aka.ms/vc14/vc_redist.x64.exe) |
| A supported game input | DLSS, FSR or XeSS calls that OptiScaler can intercept; use single-player games |

### NR model DLL

Provide an original `nvngx_dlssnr.dll` from a game that includes DLSS 5 Neural Rendering.

| DLL version | Models included |
| --- | --- |
| 310.9.16.0 (recommended) | Models A and B |
| 310.8.0.0 | Model A only |

A is the original NR model; B is NVIDIA's newer, optimized model. On AMD, **Auto** selects B when available. The installer checks the source DLL and prepares its models automatically. See [model preparation](docs/MODEL.md) for supported file hashes and storage details.

Allow space for the extracted release, about 494 MiB for both prepared models, and backups of replaced game files.

## Quick start

1. Close the game. Download the installer ZIP from [Releases](https://github.com/shakhbazian/OptiScaler-RDNA2NR/releases/latest) and extract it completely.
2. Run `Install-RDNA2NR.cmd`. Select the folder containing the game's actual executable and your original NR DLL. Keep the default `dxgi.dll` proxy unless the game's setup requires another name. Optionally select the [FSR 4.0.2c INT8 DLL](docs/INSTALLATION.md#optional-fsr-4-int8-on-rdna2), then click **Install / update**.
3. Start the game and enable a supported in-game upscaler. For the default frame-generation route, also enable the game's native FSR frame generation.
4. Press **Insert**, open **DLSS Neural Rendering**, and wait for model preparation. An increasing **RDNA2 NR active** frame count confirms processing. Use the NR switch to compare the result; choose A or B under **NR model**.

A fresh installation enables NR with **Auto** model selection, **Standard** style, **75% Model resolution**, and placement **before upscaling**. It selects FSR 4 when available, otherwise FSR 3, and uses FSR 3.0 FG input → XeFG output with **Active** enabled. Updating preserves saved settings.

For DirectX 11, select an upscaler marked **Dx12**, which uses OptiScaler's DirectX 12 bridge. See [installation](docs/INSTALLATION.md) for game setup, updates and removal, or [troubleshooting](docs/TROUBLESHOOTING.md) if NR is inactive.

## Adjusting NR

**Model resolution** controls the network's working size while preserving the output frame size. At 50%, a 1920×1080 NR input is processed at 960×540. At reduced sizes, **Enlargement** controls how its edit is transferred back; r4 adds optional Edge-aware and halo filters. See [settings](docs/SETTINGS.md).

Higher NR resolutions increase processing time and VRAM use. Large frames can exhaust the memory available alongside the game, upscaling and frame generation. [Compatibility](docs/COMPATIBILITY.md) lists supported paths and game coverage; [NVIDIA setup](docs/NVIDIA.md) covers the NGX route.

## Documentation

- [Install, update and uninstall](docs/INSTALLATION.md)
- [NR settings](docs/SETTINGS.md)
- [Model preparation and storage](docs/MODEL.md)
- [Compatibility and performance](docs/COMPATIBILITY.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Architecture](docs/ARCHITECTURE.md)

## Build from source

Install Git, PowerShell 7, Visual Studio's C++ desktop tools and Windows SDK, the v145 platform toolset with MSVC 14.44 binaries, and AMD HIP SDK 6.4. The scripts default to Visual Studio 18 Community and `C:\Program Files\AMD\ROCm\6.4`. See [build instructions](docs/BUILDING.md) for dependencies, paths and release packaging.

```powershell
git submodule update --init --recursive
.\tools\Build-HipBackend.ps1
.\tools\Build-OptiScaler.ps1
```

Outputs: `build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll` and `x64/Release/OptiScaler.dll`. Prepare model weights with the [converter](docs/MODEL.md#standalone-conversion).

## Origins and credits

This is an unofficial fork of [wilsjo2's OptiScaler-DLSSNR](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass), built on [OptiScaler](https://github.com/optiscaler/OptiScaler) and earlier NR work by [Dagherbou](https://github.com/Dagherbou/OptiScaler_DLSSNR). OptiScaler began with [CyberFSR2](https://github.com/PotatoOfDoom/CyberFSR2); parts of NR colour processing derive from [RenoDX](https://github.com/clshortfuse/renodx). See [credits](docs/CREDITS.md), [third-party notices](Licenses), and the [GPL-3.0 license](LICENSE). Model ownership is separate from the code license.

For general OptiScaler features and game-specific advice, use the [upstream wiki](https://github.com/optiscaler/OptiScaler/wiki) and [community Discord](https://discord.gg/wEyd9w4hG5). Support upstream development through [cdozdil](https://github.com/sponsors/cdozdil?frequency=one-time) or [nitec](https://buymeacoffee.com/nitec).
