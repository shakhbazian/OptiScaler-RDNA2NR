# RDNA2 NR port: build and qualification notes

This branch starts from the wilsjo2 OptiScaler v0.8.91 source. The D3D12 and
D3D11-on-12 upscaler entry points remain upstream's. On an actual `gfx1030`
device, backend Auto selects the HIP companion by the D3D12 adapter LUID; on
NVIDIA it keeps the ordinary NGX route. A forced backend that does not match
the adapter is unavailable rather than silently using another device.

The HIP companion implements the accepted mixed-v5 graph with FP16 C32. The
common OptiScaler color pass runs on privately recorded D3D12 lists. At the
game queue marker, the transport inserts input copy, HIP work, and the
ordered output copy, then resumes the game's lists. A failed or unqualified
frame takes the raw upscaler path. Old command allocators, private textures,
and model sessions remain owned until their GPU use retires.

`[DlssNr] ModelPath` names a locally converted `.nrwgt` file. If empty, the
runtime looks in `%LOCALAPPDATA%\OptiScaler-RDNA2NR\models\1\E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E\mixed-v5-gfx1030.nrwgt`.
No model, NVIDIA runtime, or AMD driver DLL is shipped by the packager. HIP
comes from the installed AMD driver. A missing or incompatible package is
reported in the NR menu and requires a game restart after correction.

Build from this checkout with `tools/Build-HipBackend.ps1`, then
`tools/Build-OptiScaler.ps1`; or use `package_release.ps1`, which builds both
and assembles an allowlisted archive with SHA-256 checksums. The toolchain is
VS v145 / MSVC 14.44 and ROCm HIP 6.4 for gfx1030. The frontend build fetches
official FreeType 2.13.3 source at the pinned revision, builds it with `/MD`,
and links that library instead of upstream's `/MT` archive. A clean build
therefore has neither C4744 nor LNK4098. FreeType's FTL is included with the
package.

The source-built regression hosts live in `tests/rdna2`. `Build-Hosts.ps1`
compiles them; `Run-Smoke.ps1 -ModelPath <private package>` checks D3D12
pre/post results against an independent color codec and GPU executor.
`Run-Dx11.ps1` checks the D3D11-on-12 route, and `Run-Failures.ps1` checks
distinct asset failures. Test logs and binaries remain in ignored `build/`.
The HIP source closure and hashes are recorded in `rdna2_nr/import-manifest.json`.
The measured migration matrix and its limits are in
`docs/RDNA2NR_SECTION_B_RESULTS.md`.

The first HIP profile qualifies one ordinary pass before or after upscaling.
Finished-picture, separate edit upscaling, multiple passes and frame hold
remain available to the original NVIDIA route; HIP presents their inactive
reason and does not save one of them as a working HIP preset. This is a
capability boundary, not a change to the model's response to style and
strength controls.
