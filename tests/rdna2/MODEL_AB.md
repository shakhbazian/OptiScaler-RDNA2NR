# Comparing NVIDIA NR model A and model B

These tools exercise the production HIP kernels independently of a game. Model A
retains the released graph and arithmetic. Model B uses the 31-block graph from
310.9.16.0 with its own weights, window phases, skip edges and temporal coefficient.
The default B executable is the FP16 reference. B-specific mixed builds and
calibration are described in [MODEL_B_MIXED.md](MODEL_B_MIXED.md).

## Convert and run

Use the normal project Python environment (NumPy and Pillow), HIP 6.4 and the
Windows C++ toolchain. Keep packages, fixtures and outputs on a spacious local disk.
The converter reads the supplied DLL as data.

```powershell
python tools/model_converter/write_runtime_package.py --source nvngx_dlssnr.dll --model original --output D:/TEMP/nr-ab/model-a.nrwgt
python tools/model_converter/write_runtime_package.py --source nvngx_dlssnr.dll --model optimized --output D:/TEMP/nr-ab/model-b.nrwgt
./tests/rdna2/Build-ModelAB.ps1
./tests/rdna2/Build-ModelAB.ps1 -Mixed
python tests/rdna2/prepare_model_ab.py portrait.png D:/TEMP/nr-ab/portrait --frames 8
./tests/rdna2/Run-ModelAB.ps1 -CorpusRoot D:/TEMP/nr-ab/portrait -PackageRoot D:/TEMP/nr-ab -Models a,b,release
```

The runner expects both packages from the recognized **310.9.16.0** DLL. A package
from **310.8.0.0** remains supported by the converter and runtime; its source and
package hashes are different, so use the executable directly with that identity.
Unknown source DLLs, modified packages and B with the A mixed profile are rejected.

The fixture generator uses identical RGB pixels and controls for both models,
with model-specific padding. `--height` resizes the input; omitting it retains the
source size. `--standard` selects Style 0 / Tone 1 / Structure 1; the default is
Style 1 / Tone 1.5 / Structure 1.5. `--controls` takes eight numbers: style, tone,
structure, skin, automatic, intensity, jitter X and jitter Y.

For a sequence, pass a folder of ordered PNGs and `--sequence --frames 60
--temporal`. The diagnostic fixture has **zero motion vectors**, so recorded video
does not validate game reprojection. `-SaveAllFrames -ImagesOnly` saves the RGBA
result of each frame without bulky intermediate tensors.

## Validation

- `test_model_profiles.py` checks source/model compatibility and malformed PE
  input. Its optional package directory verifies all three pinned package hashes
  and identical A tensor tables/payloads from old and new DLLs.
- `model_spec.cpp` checks skip ownership, phases, common and extreme dimensions.
- `model_package.cpp` checks valid A/B loading and damaged identity, header,
  payload, tensor type, rank, shape and role. Link with `nr_weight_package.cpp`
  and `bcrypt`.
- `model_ab_controls.py` prepares/checks a settings sweep, history and reset.
- `Build-ModelAB.ps1 -BaselineSource <saved original spatial_graph.cpp>` builds
  the old executor with the same probe and compiler flags. Compare the raw
  feature, head, RGBA and history tensors, not screenshots.

Use `stream_ms` from normal runs for the GPU pipeline cost. Per-block and stage
profiling change scheduling and are diagnostics. Timing excludes disk I/O,
host-to-device input copies, and D3D interop. The first two frame samples are
excluded from the comparison page's median.

`render_model_ab.py` and `model_ab_gallery.html` produce the local review page for
the named portrait/street/dialogue corpus. They retain full-resolution PNGs,
signed edit maps and per-model video. The public NGX comparison is a separate
check with exactly the published input and controls; FP16 here does not imply
bit-identical output to NVIDIA's runtime.

Graph metadata provenance is recorded in `Licenses/DLSSNR-AMD_LICENSE.txt`.
No original DLL or model weights are part of these sources.
