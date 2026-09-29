# Section B: HIP migration qualification (2026-09-29)

Base: wilsjo2 OptiScaler v0.8.91, commit `f45ccf3a`; product branch
`codex/rdna2nr`. The accepted mixed-v5 / C32-FP16 graph and its local
user-supplied model package were retained. This report concerns the new
OptiScaler integration, not a claim of game compatibility or target FPS.

## Result

- The source-built frontend uses the existing D3D12 and D3D11-on-12 upscaler
  paths. Backend Auto selects HIP only for the observed `gfx1030` adapter LUID;
  a mismatched forced backend does not fall through to HIP. The NVIDIA NGX
  compatibility route remains available on a matching device.
- The production companion carries the accepted mixed-v5 / FP16 C32 executor.
  The common OptiScaler color pass, one pre- or post-upscale NR pass, and ordered
  copy/queue submission are integrated without a game-specific bridge.
- A reusable output target previously made a queued frame fail `record output`
  with `InvalidResource`. The transport now permits reuse only for a deferred
  output whose previous writer is ordered on the same queue and whose target is
  distinct from the earlier slot's inputs. Other in-flight aliases still fail.
  The old pilot applied 2/3 frames in a quick Full HD sequence; this build
  applied 3/3. On one accepted Full HD frame, old and new outputs were SHA-256
  identical (`AD2F95364CA0EDDC3E985880F1AD44CB5F082F52696CF796B364228BB5717484`).
- The default model path is under the user's LocalAppData; `ModelPath` can
  override it. Missing and wrong-sized packages and a missing HIP companion
  have distinct status messages. The backend uses the same failure code for
  full-size package validation and VRAM allocation, so the menu names both
  possibilities and the runtime log carries the more specific cause.
  Unsupported HIP modes are visibly
  disabled instead of being stored as usable presets. The menu's auto-sized
  window is constrained to the ImGui viewport so fullscreen scaling can scroll
  oversized content. This last layout guard is build-checked; an actual game
  menu screenshot remains an integration check.
- FreeType 2.13.3 is built from pinned source with `/MD`; the final frontend
  build contains neither C4744 nor LNK4098. The allowlist packager excludes
  the model, NVIDIA runtime and AMD driver DLLs, and checks production exports.

A second checkout in `D:\TEMP` was cloned from the local branch, with all nine
submodules checked against their pinned commits. Both HIP and OptiScaler built
there from source before any model was copied into that checkout. The clean
frontend log contained zero C4744/LNK4098 occurrences and the checkout had
zero `.nrwgt` files. A provisional archive from the primary checkout contains
29 checksum-verified files (30 ZIP entries including `SHA256SUMS.txt`) and no
model or driver runtime. This is a build/package receipt, not the public
installer or final user-facing release.

## Reproducible checks

Run the source-built hosts from `tests/rdna2`. Their logs and exact hashes
remain under ignored `build/tests/rdna2`; the co-load receipt is
`build/tests/rdna2/coload/receipt.json`. The private package path is supplied
at invocation and does not enter Git or the release archive.

- `tests/dlssnr_proxy/run.ps1`: upstream NGX mock regression passed.
- D3D12 pre/post exact parity with the independent color codec passed for
  standard, queue change, resize/DRS/recreation, COM wrapper, early reset,
  HDR, exposure, RGBA8, RGBA32F, R11G11B10, apply-off, zero transfer, batch,
  reuse, and six-frame stream/paced-stream. Subrect, bundle, unknown motion
  state, wrapper bypass, and unsupported motion format yielded exact raw
  fallback. Debug layer and lifecycle checks were clean.
- D3D11-on-12 pre/post passed static, pan/reset, occlusion/reset, and resize.
  Auto selected HIP on AMD; forced NVIDIA on AMD stayed unavailable and left
  the raw result unchanged. Debug layer and lifetime checks were clean.
- Functional D3D12 pre/post ladder passed at 960x540, 1280x720 and 1920x1080
  with 2x FSR output, and at 2560x1440 and 3840x2160 with same-size output.
  A 2560x1440 input with 5120x2880 FSR output failed cleanly at model warmup:
  insufficient VRAM for the spatial graph. This is a memory budget boundary,
  not a hard-coded 1440p resolution cap.
- Old and migrated HIP companions produced identical pre/post hashes on the
  four-frame small fixture. The accepted single Full HD visual frame was also
  identical. A three-frame pilot/candidate comparison intentionally differs
  because the pilot rejected an in-flight output; compare only accepted frames.
- The real 1920x1080 screenshot fixture produced an NR-on image with a visible
  facial/detail response. Raw, NR, face crop and amplified difference are under
  ignored `build/tests/rdna2/visual-comparison`. The A/B is a single-frame
  visual check, not temporal game-quality qualification.

## Uninstrumented paired co-load

The source-built D3D11-on-12 host ran 60 moving 960x540 frames into a
1920x1080 FSR output. A separate source-built D3D12 compute process occupied
the same RX 6900 XT in the loaded runs. NR applied 60/60 frames in each run;
the final output hash was identical with and without load. Results are loop
wall-time per frame, including host/upscaler work, not isolated kernel time:

| Run | NR | Idle/load | ms/frame |
| --- | --- | --- | ---: |
| A | on | idle | 55.08 |
| B | on | loaded | 158.85 |
| C | on | loaded | 160.74 |
| D | on | idle | 51.86 |
| Control | off | idle | 10.98 |
| Control | off | loaded | 21.35 |

The startup/model warmup (about seven seconds on this fixture) is outside the
timed frame loop. The first five frame calls were excluded from the separately
recorded median Evaluate time; the whole-loop values above remain the primary
comparable measurement. These numbers do not imply 1080p60. They establish
the migration's current performance cost and the effect of shared-GPU load.

## Next boundary

Section C is the local converter/installer and transactional update/uninstall.
It must accept a user's own original DLL, cache a validated converted model,
and handle permission, disk, Unicode path, and rollback cases. Section D is
product documentation, UI/game qualification, and publication review. Do not
interpret this Section B host matrix as having completed either section.
