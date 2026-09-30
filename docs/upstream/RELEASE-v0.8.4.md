> Inherited parent-fork reference. This document describes NVIDIA paths or historical work; see the [RDNA2NR documentation](../README.md) for current AMD support.

# v0.8.4 pre-release

Fixes NR memory retention when switching pre-SR/finished-picture placement or changing model resolution. Old, inactive command recordings no longer hold unrelated replacement models in VRAM. Cancellation preserves GPU-completion checks for work still in flight.

The Model passes control is now a slider. Enable **Unlock up to 10 passes** to extend its range from two to ten. Changes apply when the slider is released; turning the unlock off clamps the count back to two.

This release retains v0.8.3's NR runtime loading/recovery fixes and Starfield recording/reset synchronization.

Choose the standard ZIP unless you need the optional RTX 40 MFG unlock. NVIDIA NR and FG runtime binaries are not included; retain your existing runtime installation.

Validation includes GPU lifetime/replay/concurrency regressions, the NGX routing tests, and standard/RTX 40 MFG Release x64 builds. The standalone memory reproduction previously retained 3,663 MiB after twelve model replacements; the corrected generation tracking returned to roughly 51 MiB after each cleanup. Full game-session verification remains with testers.

Known compatibility issue: Smooth Motion triggered the reported Silent Hill 2/f startup crashes. Disable Smooth Motion for those games; this release does not implement Smooth Motion interoperability changes.
