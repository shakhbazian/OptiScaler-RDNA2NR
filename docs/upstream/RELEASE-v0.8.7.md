> Inherited parent-fork reference. This document describes NVIDIA paths or historical work; see the [RDNA2NR documentation](../README.md) for current AMD support.

# v0.8.7 prerelease — finished-picture queue safety

This release removes the unsafe cross-queue wait identified in the follow-up review of [PR #70](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/70). It also includes all [v0.8.6 shutdown fixes](RELEASE-v0.8.6.md).

Finished Picture NR now skips unfinished cross-queue input on every presentation path. Same-queue submission order and completed cross-queue input remain eligible, and composition selects the newest compatible eligible slot. A submitted producer signal can itself depend on presentation; adding a presentation wait for that signal can create a GPU dependency cycle even when the wait API returns success. The v0.8.5 rewrite removed waits on unused slots but retained this unsafe assumption through v0.8.6.

This can cause NR to be skipped when another queue has not finished producing its input, including the RDR2 configuration that motivated PR #70. Guaranteeing NR on every frame in that configuration requires an integration point with proven producer/presentation ordering. Pre-SR-only processing is unaffected by this change.

## Validation

- WARP queue-readiness test: blocked producer/presentation dependencies, same-queue ordering, completed input and removed-device fences.
- Automated NR prerelease suite and standard/optional RTX40 MFG Release x64 builds.
- Both release archives checked against their SHA-256 manifests, source commit and build variant.

The review's bounded WARP counterexample reproduced the old wait policy stalling while returning success. It was a synchronization test, not an affected-game or NVIDIA driver reproduction. The reported shutdown hang and Event 153 still need a retest on the affected game and driver. No RTX40 hardware validation is claimed; the optional MFG implementation is unchanged.

Use `OptiScaler-NR-v0.8.7.zip` for the standard build or `OptiScaler-NR-v0.8.7-rtx40-mfg.zip` for the optional RTX40 MFG build. Extract the complete package. NVIDIA model/FG runtime DLLs are not bundled.
