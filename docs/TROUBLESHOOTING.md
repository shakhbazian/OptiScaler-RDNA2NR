# Troubleshooting

[Documentation](README.md) · [Installation](INSTALLATION.md)

## Menu or upscaler missing

Check that installation targets the actual rendering executable's folder and that the game loads the selected proxy DLL. Replacing `OptiScaler.dll` beside an old `dxgi.dll` does not update the proxy the game loads. Check competing loaders, the process filter and the game's supported OptiScaler input path.

Press **Insert**, or **Alt+Insert** on an alternate keyboard layout. For general game setup and overlay conflicts, consult the [OptiScaler wiki](https://github.com/optiscaler/OptiScaler/wiki).

## NR status messages

| Status | What to check |
| --- | --- |
| Waiting for the upscaler to run | Enable a supported in-game upscaler and enter a rendered scene. Having the menu open is not proof that an upscaler evaluation is occurring. |
| Preparing model and queue | Wait for preparation. A change of resolution, device or submission queue can trigger it again. Persistent preparation needs a log report. |
| Waiting for a qualified submission | Ensure the game is actually submitting the intercepted upscaler work. Unsupported wrappers/states may bypass NR. |
| Ready; waiting for the next frame | Preparation completed; wait for a new rendered frame. |
| Active, increasing applied-frame count | NR is processing frames. Check Apply model and composition strength if the edit is invisible. |
| Skipped the last submission | The frame used the raw path. Repeated skips need the exact log reason, API, format and active settings. |
| Fault | Correct the reported model, dependency or device error, then restart the game. |

The AMD backend should be identified as **AMD HIP (gfx1030)**. Auto uses the actual rendering adapter. Do not force NVIDIA on AMD to work around a preparation error.

## Active NR, unchanged picture

Keep **Apply model** enabled and Transfer/Colour strength above zero. Leave debug/comparison views off when judging the ordinary output. Confirm that the *applied* frame count increases, rather than only a preparation or background counter.

Use a scene where the full model has a visible effect, keep settings consistent, and compare with the master NR switch. Some scenes respond subtly. If the applied counter grows but the expected edit is absent, include a paired screenshot and log in a report.

## Missing runtime or model

- **`amdhip64_6.dll`:** install a compatible AMD graphics driver. The current companion uses the HIP 6 runtime ABI. A driver providing only another major runtime is not sufficient. Do not download isolated DLLs from file-mirror sites.
- **`VCRUNTIME140`, `MSVCP140` or similar:** install the [Visual C++ v14 Redistributable, x64](https://aka.ms/vc14/vc_redist.x64.exe).
- **HIP companion missing:** extract the complete release and reinstall with `Install-RDNA2NR.cmd`. The companion must accompany the frontend.
- **Source DLL rejected:** compare its SHA-256 with the [supported source](MODEL.md#supported-source). Renaming a different DLL does not make it compatible.
- **Weight package missing or invalid:** verify the model cache or the absolute `[DlssNr] ModelPath`. Move aside a corrupt cached file and rerun conversion with a verified source. Restart the game afterward.

The current HIP executor does not depend on rocBLAS. A rocBLAS error points to another component, an older package or a different tool; report which executable produced it instead of installing arbitrary libraries.

## Installation conflicts

The installer records file hashes and ownership. If an installed binary has changed since installation, update/uninstall refuses to overwrite it. Close the game, retain the manifest and backup folder, and identify the changed file before deciding which mod owns it. Do not delete the backups as a first troubleshooting step.

An edited INI is intentionally preserved. Unsupported modes left in an old configuration can prevent HIP admission: restore one pass, 100% working size, ordinary pre/post placement and no separate-edit, finished-picture or frame-hold mode.

For script/path errors, use a newly extracted complete release. Supply actual paths when using the command line; do not run the launcher from inside a ZIP archive. Copy the full error text into a report.

## Slow frames, trails or memory pressure

Measure after preparation, using the same scene and rendering settings. Turn off NR entirely for the baseline; disabling Apply model only hides its output. Try pre-upscale placement to reduce the model's pixel count.

For trails or flicker, retain a short moving comparison and report the temporal-accumulation setting, motion-vector path and camera-cut behaviour. A still screenshot cannot characterize a temporal defect. Large resolutions can exhaust shared VRAM; lower the rendered/model input size and distinguish an allocation failure from sustained inference cost.

## Report a problem

Enable file logging temporarily:

```ini
[Log]
LogToFile=true
LogLevel=2
```

Reproduce once, locate the newly written OptiScaler log beside the game executable, then restore your usual logging setting. Include:

- Release/build identifier, Windows version, GPU and AMD driver version.
- Game, rendering API, loaded proxy, input/output upscalers and frame-generation selection.
- Render/output resolutions, before/after placement, temporal setting and exact NR status.
- The relevant log and a short reproduction sequence; screenshots/video if the defect is visual.

Do not attach the original NVIDIA DLL or converted model weights to public reports.
