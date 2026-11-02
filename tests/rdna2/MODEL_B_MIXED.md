# Model B mixed execution

This is the standalone HIP comparison for the 31-block model B from NVIDIA
310.9.16.0. It uses the same prepared package as the FP16 reference. INT8 weights
are prepared once when the graph warms up, then retained by that graph instance.
The architecture, channels, window phases, skip edges and controls stay unchanged.

## Execution profiles

Build with `Build-ModelAB.ps1 -Mixed`, plus the options below. Use a separate
`-OutputRoot` for each executable; keep the earlier A mixed and B FP16 binaries.

| Profile | Options | Arithmetic |
| --- | --- | --- |
| Dynamic | `-BMode 0` | Group64 W8A8 in large matrices; dynamic activation scales; C32 FP16 |
| Static large | `-BMode 2` | B-calibrated large activation scales, folded scales and long INT32; C32 FP16 |
| Full static | `-BMode 1` | Static large, plus static W8A8 in all six C32 FFNs |
| Balanced | `-BMode 3 -BC32Mask 32` | Dynamic large matrices, static W8A8 in the final C32 FFN |

`BC32Mask` bits select B blocks 0, 1, 2, 28, 29, 30. With mode 1, mask 32
tests only the last C32 and mask 33 tests the first and last. These are diagnostic
alternatives, not changes to the package or saved game settings. B mixed must use
group64 and the model-aware mixed policy. Builds without `BMode` retain the guard
against accidentally running B through the A mixed profile.

The port reuses DOT4 INT8 kernels, resident branch/split FFNs, packed FP16 branch
projections, fused epilogues, global FFN handoffs, fused normalization/attention,
the optimized C32 wave kernel and the persistent arena. Layer selection now uses
the model's block families. A keeps its original selection and calibration.

Static B constants are independent, read-only device arrays. No upload replaces
A's process-global constants. Calibration indexes select B data explicitly.
Long accumulation retains the bound `K * 127 * 127 <= INT32_MAX`.

## Calibration and validation

`model_b_mixed.py prepare` selects 32 scenes from the existing calibration split,
with Standard and strong controls at 960×540. `calibrate` runs B FP16 through the
diagnostic activation observer. The observer reads activations synchronously and
samples up to 2048 evenly spaced rows per tensor/frame, accumulating logarithmic
per-channel histograms. Its timings are not performance measurements.

`fit_model_b_quantization.py` generates `nr_model_b_quantization.h` from that CSV:
per-channel q99.99, a 1.15 range margin, power-of-two channel balancing for C32,
and group64 ranges for large projections. The generated header records the input
statistics hash. Neither A's calibration nor its weights are used to fit B.

`validate_model_b_mixed.py` uses all 18 scenes from the separate holdout split
with two presets, plus the retained full-resolution portrait with a control sweep.
The visual fixtures add native-size portraits, a 720p street and a 60-frame
720p dialogue. Recorded video has zero motion vectors; this does not qualify game
reprojection. Corpus paths are local research inputs and are not distributed.

`check_model_b_mixed.py` checks finite outputs, exact zero-intensity bypass and
history reset, control-response direction, RGB error, and preservation of the NR
edit. Edit projection is `dot(candidate-source, reference-source) /
dot(reference-source, reference-source)`: it measures the component along the
reference edit, not a percentage of perceptual quality.

`benchmark_model_b_mixed.py` uses two forward/reverse cycles, 24 frames per run,
discarding the first four. It reports 80 uninstrumented samples per profile and
logical resolution, median, p95 and individual run medians. Each run's first
output must match its retained quality result byte for byte. The GPU stream
interval includes preparation, graph and composition, excluding disk I/O,
host input upload and graphics-API interop.

`render_model_b_mixed.py --video` builds lossless PNGs, signed NR edits/differences,
and synchronized comparison videos in a new `comparison/mixed` directory. The
previous A/B page remains intact. Full static C32 is an aggressive comparison
candidate: inspect temporal edit retention as well as still-frame RGB error.

The tests preserve the release A and FP16 B outputs. This block does not select
a new game default; the visual comparison is the acceptance point.
