# Installation

[Documentation](README.md) · [Troubleshooting](TROUBLESHOOTING.md)

## Before installing

For the AMD backend, you need:

- Windows x64 with DirectX 12.
- A Radeon with the `gfx1030` architecture, such as RX 6900 XT. See [hardware compatibility](COMPATIBILITY.md#hardware).
- [AMD Software: Adrenalin Edition](https://www.amd.com/en/support/download/drivers.html) for Radeon RX 6000, with the HIP 6 runtime.
- The [Microsoft Visual C++ v14 Redistributable, x64](https://aka.ms/vc14/vc_redist.x64.exe). See Microsoft's [runtime requirements](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).
- Original `nvngx_dlssnr.dll` from **DLSS 5 Neural Rendering 310.8.0.0**. Obtain it from a game's installation that includes DLSS 5 Neural Rendering and check the [supported SHA-256](MODEL.md#supported-source).
- A game whose DLSS, FSR or XeSS upscaler calls OptiScaler can intercept.

The HIP runtime is included in the AMD graphics driver; see AMD's [deployment guidance](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/conceptual/deployment-guidelines.html). This build uses the HIP 6 runtime (`amdhip64_6.dll`).

Allow disk space for the extracted release, a converted model of about 278 MiB, and backups of replaced game files. VRAM usage depends on the game and NR resolution.

## GUI installation

1. Close the game and launcher. Extract the complete release into a separate folder.
2. Run `Install-RDNA2NR.cmd`. This opens the Windows PowerShell installer GUI.
3. Set **Game executable folder** to the folder containing the executable that actually renders the game. For Cyberpunk 2077 this is normally `bin\x64`.
4. Select your original `nvngx_dlssnr.dll`. It is read as data and converted locally, rather than loaded as executable code.
5. Select **Proxy DLL name**. The default is `dxgi.dll`; a game or another loader may require a different name. Follow [OptiScaler's game-specific guidance](https://github.com/optiscaler/OptiScaler/wiki) and check existing mods before replacing a proxy.
6. Click **Install / update** and wait for completion. Model conversion can briefly make the window unresponsive. Installation verifies the model and installed files; it does not start the game or prove GPU compatibility.

Do not use the inherited `setup_windows.bat` for this AMD installation path. It does not prepare the HIP model.

## First launch

Enable an in-game upscaler, open OptiScaler with **Insert**, and enter **DLSS Neural Rendering**. Enable NR, choose **Auto**, and start with NR before upscaling, one pass, 100% working size and **Apply model** enabled. A fresh configuration leaves NR disabled.

Wait for **AMD HIP (gfx1030)** and an increasing **RDNA2 NR active** frame count. Toggle the master NR switch to compare the image and performance. The model-preparation and queue-preparation stages can take time on first use or after a resolution change.

For DirectX 11, select an upscaler marked **w/Dx12**. For example:

```ini
[Upscalers]
Dx11Upscaler=fsr22_12
```

This uses OptiScaler's existing DirectX 12 upscaler bridge. It has standalone test coverage; game compatibility still needs testing.

## Update and uninstall

Close the game, extract the new release separately, and run its installer against the same game folder and proxy name. The installer reuses a verified model cache and preserves an existing `OptiScaler.ini`; it does not reset old or unsupported settings. Review [NR settings](SETTINGS.md) when upgrading from another fork.

The game folder contains `.optiscaler-rdna2nr-install.json` and `.optiscaler-rdna2nr-backup`. Keep both: they record ownership and retain replaced files. To change the proxy name, uninstall the existing installation first.

Choose **Uninstall** in the installer to restore owned files and backups. A subsequently edited INI is preserved. Removal does not delete the shared model cache. If an installed binary was changed by another mod or a manual update, the installer refuses to overwrite it blindly; see [installation conflicts](TROUBLESHOOTING.md#installation-conflicts).

## Command-line use

Run these commands from the extracted release folder. Replace the example paths with your own.

```powershell
.\Install-RDNA2NR.cmd -Mode Install -GameDirectory "C:\Games\Example\bin\x64" -SourceDll "C:\Models\nvngx_dlssnr.dll"
.\Install-RDNA2NR.cmd -Mode Verify -GameDirectory "C:\Games\Example\bin\x64"
.\Install-RDNA2NR.cmd -Mode Uninstall -GameDirectory "C:\Games\Example\bin\x64"
```

The launcher uses Windows PowerShell 5.1. PowerShell 7 is not required for installation. The installer does not install the AMD driver or Microsoft redistributable for you.

## NVIDIA route

The frontend retains ordinary NVIDIA NGX compatibility. The AMD installer above always requires the supported source DLL and prepares HIP weights; it is not an automatic NVIDIA runtime installer.

For NVIDIA, follow this build's [manual runtime setup](NVIDIA.md) with a user-supplied runtime accepted by the installed NVIDIA driver. The original runtime's GPU restrictions still apply. Historical hybrid/NVFP4 experiments and RTX 40 MFG unlocks are not part of the standard RDNA2NR build. See [compatibility](COMPATIBILITY.md) before treating a parent-fork feature as available here.
