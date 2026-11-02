# Native D3D12 regression tests

The runtime records a bounded command stream, splits it at NR, restores state
on the suffix, and retires private work against GPU fences. These tests exercise
those behaviours without an injected game or timed test modes.

Build the ordinary frontend with `tools/Build-OptiScaler.ps1` and the hosts with
`tests/rdna2/Build-Hosts.ps1`. For graphics-listener and fault-injection tests,
build a separate frontend with `-NativeTestHooks` in another output/object
directory. Release packaging rejects its test exports.

## GPU checks

`Run-Smoke.ps1` runs NR off, before/after SR, and the independent colour oracle.
NR must modify the frame and match the oracle exactly. Rejected inputs preserve
the clean frame. Every D3D12 warning/error fails the debug-layer check.

```powershell
tests/rdna2/Run-Smoke.ps1 -FrontendPath $frontend -CompanionPath $companion `
    -ModelPath $model -Scenario reuse -WorkingScale .75 -SkipBuild
```

Repeat with `standard`, `wrapper`, and these graphics fixtures:

- `draw_reuse`: different frames from the same reset list and allocator.
- `draw_reuse_postcopy`: a listener appends a copy after Draw; private replay
  must not invoke the listener again or reorder its copy.
- `draw_reuse_restore`: graphics-state restoration with signature changes,
  null unbind and independent compute/graphics tables.

The graphics fixtures require the test frontend. The two listener fixtures use
fresh lists for NR-off, because the AMD runtime can bypass legacy hooks on a
reused list when NR is disabled. NR-enabled lists are reused throughout.

For FSR 4, add `-Upscaler ffx -RuntimeDirectory $fsrRuntime`. Its two host paths
can produce different hashes even on an unchanged frontend. Use `-FunctionalOnly`
and require exact paired before/after hashes for cleanup regressions; record
this separately from independent oracle parity. Use a new output directory
for each run. `NR_NATIVE_NO_DEBUG=1` selects a separate retail
qualification; it does not constitute a debug-layer pass.

For a game-specific Agility SDK, build a separate host directory:

```powershell
tests/rdna2/Build-Hosts.ps1 -OutputRoot build/tests/agility619 `
    -AgilitySdkDirectory 'C:/path/to/game/D3D12_0' -AgilitySdkVersion 619
```

Pass `-HostDirectory build/tests/agility619 -SkipBuild` to `Run-Smoke.ps1`.
The builder stages Core and SDKLayers locally; it changes no system SDK selection.
Set `NR_HOST_DYNAMIC_RASTER=1`, `2` or `3` for depth bias, strip-cut or both.
Requested unsupported capabilities fail rather than being skipped.

`dynamic_raster.exe` checks 24 direct/split pixel cases, including redundant PSO
binds, ClearState and suffix overrides. Static-default differences are reported
separately: the tested AMD route can retain overrides at redundant binds. The
recorder preserves actual PSO/raster setter order.

## Other checks

- `Build-HookTests.ps1 -Run`: listener order, shared endpoints, concurrency and
  runtime-dispatch guards.
- `handoff_fallback.py`: pre-submission failure preserves the current clean
  frame; post-submission failure quarantines the owner safely.
- [CODEC.md](CODEC.md): exact neutral colour, masks, overlays and headroom.
- `python tools/generate_native_hook_methods.py --check`: generated declarations
  and callbacks agree with qualified handlers in `NativeRecording.h`.

Standalone success must be followed by live visual confirmation for a game fix.
