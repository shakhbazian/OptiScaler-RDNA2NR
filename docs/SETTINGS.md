# NR settings

[Documentation](README.md) · [Compatibility](COMPATIBILITY.md)

Open OptiScaler with **Insert** and expand **DLSS Neural Rendering**. These settings describe the AMD HIP path; the NVIDIA path has additional capabilities.

## Starting configuration

Enable NR, select **Auto**, generate before upscaling, and leave one pass, 75% working size and Standard style and **Apply model** enabled. Start with default model controls, then change one at a time. Save the configuration through OptiScaler if you want it retained.

Equivalent basic settings in `OptiScaler.ini`:

```ini
[DlssNr]
Enabled=true
Backend=4
HipModel=auto
RunBeforeSR=true
Passes=1
WorkingScale=0.75
Preset=0
Style=0
ApplyModel=true
```

A fresh configuration enables NR with Standard, 75% model resolution and pre-upscale placement. For D3D12 and the D3D11/D3D12 bridge, the loaded runtime selects FSR 4 when available, otherwise FSR 3. An explicit provider selection is retained when saving settings. The existing FSR fallback applies if initialization fails. Frame generation defaults to **FSR 3.0 FG** input with **XeFG** output and **Active** enabled. Enable the game's native FSR frame generation to use this route. A saved explicit **Active** off setting is preserved on updates; enable it once if desired. Other games may need a different input; **OptiFG (Upscaler)** works without native FG. Save settings and restart after changing the FG source or output. Existing INI files are preserved by the installer.

For FSR 4 INT8, use the [optional runtime setup](INSTALLATION.md#optional-fsr-4-int8-on-rdna2).
Choose the FSR provider version under **FFX Upscaler**. **Watermark** can
temporarily identify the implementation running on the rendered image.

### HUD handling

XeFG can handle HUD without a separate HUDless input, though results vary by game.
FSR3-FG may need **HUDFix**, under **Frame Generation (OptiFG)**. Cyberpunk's
compatibility profile disables HUDFix because of known issues; the menu explains
when it is unavailable. Keep that restriction in place when using XeFG.

## Backend and placement

| Control | Meaning |
| --- | --- |
| Auto (`Backend=4`) | Choose HIP on the supported AMD adapter, or ordinary NGX on NVIDIA. Selection uses the rendering adapter. |
| NVIDIA (`Backend=0`) | Force the NVIDIA route. This does not make it usable on AMD. |
| AMD HIP (`Backend=3`) | Force the HIP route. This build requires `gfx1030` on the rendering device. |
| Before upscaling | Process the game's upscaler input, then let SR reconstruct the edited image. |
| After upscaling | Process the upscaler output. Usually more expensive because it contains more pixels. |
| Enable NR | Master switch. Turning it off stops NR processing. |
| Apply model | Show the processed result. Turning it off keeps computation running for comparisons. |

Backend choice is independent of FSR, XeSS or DLSS selection and of frame generation. Forcing an incompatible backend reports it unavailable; it does not quietly choose another GPU.

HIP supports one pass before or after upscaling. See [compatibility](COMPATIBILITY.md#rendering-paths) for other processing modes.

## NR model

The HIP **NR model** selector is independent of the backend and style preset:

| Choice | Behaviour |
| --- | --- |
| Auto (`HipModel=auto`) | Prefer B when its prepared package is available; otherwise use A. |
| Model B (`HipModel=b`) | The newer, faster 31-block network from 310.9.16.0. |
| Model A (`HipModel=a`) | The original 71-block network, available in both supported DLL builds. |

Use NR DLL 310.9.16.0 (recommended) to prepare models A and B, or 310.8.0.0 for A only.
Unavailable choices are disabled. A model switch drains the previous session,
prepares the new weights and resets history. NR can briefly show the original
image while preparation finishes. The Style, Tone, Structure and scale settings
are retained; A and B can produce different edits with the same settings.

## Model resolution

**Model resolution** sets the width and height used by NR independently of the
image at its selected stage. The range is 25–200%; changes apply when the slider
is released. At 50%, a 1920×1080 NR input is processed at 960×540, while the final
image keeps its original dimensions. Before-SR percentages refer to the render
input; after-SR percentages refer to the upscaler output.

Below 100%, OptiScaler filters the model input down and transfers the resulting
edit back onto the full-size image. **Enlargement** offers Classic, Matched
residual, Lighting + colour and Edge-aware on HIP. See [enlargement modes](#enlarging-the-nr-edit)
for their differences. Test faces and moving edges when choosing a scale.

Above 100%, NR runs on an enlarged input, then its result is reduced to frame
size with the selected **Downscaler (NR)**. This increases work and memory use.
Changing model dimensions prepares a new HIP session and clears its temporal
history; the unprocessed upscaler output may appear during preparation. Network
padding and fixed per-frame work mean execution cost is not exactly the pixel
ratio. WorkingScale=0.5 in the INI corresponds to 50%.

## Model controls

Style, tone and structure control the original network. Intensity adjusts the amount of its effect.

| Control | Range / default | Use |
| --- | --- | --- |
| Style | Standard, Natural, Cinematic / Standard | Select the model's rendering style. |
| Intensity | 0–2 / 1 | Adjust the overall model effect. |
| Local structure | 0–2 / 1 | Adjust local detail and structure. |
| Local tone | 0–2 / 1 | Adjust local lighting and tone. |
| Skin structure | -1–2 / -1 | Adjust skin structure; -1 follows Local structure. |
| Auto skin mask | On | Use the model's skin selection. |
| Temporal accumulation | On | Use available motion/history across real input frames. |

On HIP, these controls edit the single active model pass. Slider edits apply when
the slider is released. Skin controls depend on the scene and the model's skin
selection; they do not apply a uniform filter to the whole image.

Control effects depend on the scene. A large setting does not guarantee a better result, and model controls are not general performance knobs.

**Transfer strength** and **Colour strength** blend the model's edit into the output. Reducing their values can soften the visible change, but does not remove the cost of inference. Additional skin-protection composition controls are colour-based filters; they are separate from the model's Auto skin mask.

### Enlarging the NR edit

At model resolutions below 100%, **Enlargement** controls how the smaller
model result is transferred onto the full-sized image:

| Mode | Behaviour |
| --- | --- |
| Classic | Resize the model answer for composition. |
| Matched residual (default) | Transfer the difference between the model input and answer. |
| Lighting + colour | Transfer relative lighting and colour separately. |
| Edge-aware | Use brightness edges to reduce leakage of the edit between surfaces. |

All modes preserve the output frame size and use the same network inference.

The **Halo suppression (experimental)** section filters dark rims against
brighter or darker backgrounds. Both strength sliders start at zero (off).
Enabling either adds a GPU pass and can lighten real shadows; compare before saving.

On NVIDIA, **NR model** selects Automatic, Original or Optimized. Optimized
uses model B from 310.9.16.0; Original uses A. HIP uses its separate A/B selector.

### Additional DX12 FFX runtimes

Place an alternative FFX runtime in `custom/<name>/ffx` beside the OptiScaler
installation, then restart the game. It appears as `<name> (FFX)` in the
upscaler list. Use `amd_fidelityfx_upscaler_dx12.dll` as the filename when the
folder contains several DLLs. This is separate from the installer-managed FSR
4.0.2c runtime. Each named runtime keeps its own provider index.

For a driver provider stored outside the detected AMD driver store, set
`[Libraries] Fsr4DriverPath` to its DLL or directory. The usual automatic search
remains the default.

## Compare the result

Use the normal view and the master NR toggle for an ordinary before/after comparison. Comparison views and debug input/output/difference views help inspect the edit, but an amplified difference is not a quality judgment by itself.

Evaluate moving scenes as well as still frames. Watch faces, fine detail, camera cuts and moving edges for flicker or trails. Keep placement, render resolution, style and intensity the same when comparing backends or builds.

Use total game frame time to judge performance. Hiding the edit with **Apply model** leaves the NR workload running; disable NR for the actual baseline.
