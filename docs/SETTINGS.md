# NR settings

[Documentation](README.md) В· [Compatibility](COMPATIBILITY.md)

Open OptiScaler with **Insert** and expand **DLSS Neural Rendering**. These settings describe the AMD HIP path; the NVIDIA path has additional capabilities.

## Starting configuration

Enable NR, select **Auto**, generate before upscaling, and leave one pass, 75% working size and Standard style and **Apply model** enabled. Start with default model controls, then change one at a time. Save the configuration through OptiScaler if you want it retained.

Equivalent basic settings in `OptiScaler.ini`:

```ini
[DlssNr]
Enabled=true
Backend=4
RunBeforeSR=true
Passes=1
WorkingScale=0.75
Preset=0
Style=0
ApplyModel=true
```

A fresh configuration selects Standard, 75% model resolution and pre-upscale placement. NR remains disabled until enabled. FSR 3 is preferred for D3D12 and the D3D11/D3D12 bridge; the existing FSR fallback applies if it cannot initialize. XeFG is the default FG output when its libraries are available, with Upscaler as the source. Enable FG separately in the menu. Existing INI files are preserved by the installer.

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

Multipass, spatial compression, finished-picture processing, separate edit upscaling and frame holding are unavailable on HIP. NVIDIA precision choices are not HIP quality selectors. The HIP arithmetic split is fixed by this build.

## Model resolution

**Model resolution** sets the width and height used by NR independently of the
image at its selected stage. The range is 25–200%; changes apply when the slider
is released. At 50%, a 1920×1080 NR input is processed at 960×540, while the final
image keeps its original dimensions. Before-SR percentages refer to the render
input; after-SR percentages refer to the upscaler output.

Below 100%, OptiScaler filters the model input down and transfers the resulting
edit back onto the full-size image. **Enlargement** selects Classic, Matched
residual or Lighting + colour on HIP. The latter two transfer the change rather
than simply stretching the entire processed picture. Test faces and moving
edges when choosing a scale and composition mode.

Above 100%, NR runs on an enlarged input, then its result is reduced to frame
size with the selected **Downscaler (NR)**. This increases work and memory use.
Changing model dimensions prepares a new HIP session and clears its temporal
history; the unprocessed upscaler output may appear during preparation. Network
padding and fixed per-frame work mean execution cost is not exactly the pixel
ratio. WorkingScale=0.5 in the INI corresponds to 50%.

## Model controls

These values are inputs to the original network, rather than post-processing substitutes for it.

| Control | Range / default | Use |
| --- | --- | --- |
| Style | Standard, Natural, Cinematic / Standard | Select the model's rendering style. |
| Intensity | 0вЂ“2 / 1 | Adjust the overall model effect. |
| Local structure | 0вЂ“2 / 1 | Adjust local detail and structure. |
| Local tone | 0вЂ“2 / 1 | Adjust local lighting and tone. |
| Skin structure | в€’1вЂ“2 / в€’1 | Adjust skin structure; в€’1 follows Local structure. |
| Auto skin mask | On | Use the model's skin selection. |
| Temporal accumulation | On | Use available motion/history across real input frames. |

Control effects depend on the scene. A large setting does not guarantee a better result, and model controls are not general performance knobs.

**Transfer strength** and **Colour strength** blend the model's edit into the output. Reducing their values can soften the visible change, but does not remove the cost of inference. Additional skin-protection composition controls are colour-based filters; they are separate from the model's Auto skin mask.

## Compare the result

Use the normal view and the master NR toggle for an ordinary before/after comparison. Comparison views and debug input/output/difference views help inspect the edit, but an amplified difference is not a quality judgment by itself.

Evaluate moving scenes as well as still frames. Watch faces, fine detail, camera cuts and moving edges for flicker or trails. Keep placement, render resolution, style and intensity the same when comparing backends or builds.

Use total game frame time to judge performance. Hiding the edit with **Apply model** leaves the NR workload running; disable NR for the actual baseline.
