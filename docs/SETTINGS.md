# NR settings

[Documentation](README.md) · [Compatibility](COMPATIBILITY.md)

Open OptiScaler with **Insert** and expand **DLSS Neural Rendering**. These settings describe the AMD HIP path; the NVIDIA path has additional capabilities.

## Starting configuration

Enable NR, select **Auto**, generate before upscaling, and leave one pass, 100% working size and **Apply model** enabled. Start with default model controls, then change one at a time. Save the configuration through OptiScaler if you want it retained.

Equivalent basic settings in `OptiScaler.ini`:

```ini
[DlssNr]
Enabled=true
Backend=4
RunBeforeSR=true
Passes=1
WorkingScale=1.0
ApplyModel=true
```

The release's fresh configuration has NR disabled and post-upscale placement. The recommendation above enables the less expensive pre-upscale path explicitly. Existing INI files are preserved by the installer.

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

Multipass, reduced model working size, spatial compression, finished-picture processing, separate edit upscaling and frame holding are unavailable on HIP. NVIDIA precision choices such as FP8 or a hybrid model are not HIP quality selectors. The HIP arithmetic split is fixed by this build.

## Model controls

These values are inputs to the original network, rather than post-processing substitutes for it.

| Control | Range / default | Use |
| --- | --- | --- |
| Style | Standard, Natural, Cinematic / Standard | Select the model's rendering style. |
| Intensity | 0–2 / 1 | Adjust the overall model effect. |
| Local structure | 0–2 / 1 | Adjust local detail and structure. |
| Local tone | 0–2 / 1 | Adjust local lighting and tone. |
| Skin structure | −1–2 / −1 | Adjust skin structure; −1 follows Local structure. |
| Auto skin mask | On | Use the model's skin selection. |
| Temporal accumulation | On | Use available motion/history across real input frames. |

Control effects depend on the scene. A large setting does not guarantee a better result, and model controls are not general performance knobs.

**Transfer strength** and **Colour strength** blend the model's edit into the output. Reducing their values can soften the visible change, but does not remove the cost of inference. Additional skin-protection composition controls are colour-based filters; they are separate from the model's Auto skin mask.

## Compare the result

Use the normal view and the master NR toggle for an ordinary before/after comparison. Comparison views and debug input/output/difference views help inspect the edit, but an amplified difference is not a quality judgment by itself.

Evaluate moving scenes as well as still frames. Watch faces, fine detail, camera cuts and moving edges for flicker or trails. Keep placement, render resolution, style and intensity the same when comparing backends or builds.

Use total game frame time to judge performance. Hiding the edit with **Apply model** leaves the NR workload running; disable NR for the actual baseline.
