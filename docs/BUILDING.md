# Build from source

[Documentation](README.md) · [Architecture](ARCHITECTURE.md)

These instructions build the Windows frontend and `gfx1030` HIP companion. To use a ready-made release, follow [installation](INSTALLATION.md); the developer tools below are unnecessary for that path.

## Developer requirements

| Tool | Purpose / tested selection |
| --- | --- |
| Git for Windows | Checkout, submodules and the pinned FreeType source fetch |
| PowerShell 7 | Build scripts; `Build-OptiScaler.ps1` uses modern .NET process APIs |
| Visual Studio C++ desktop tools | Visual Studio 18 Community, **v145 platform toolset** and **MSVC 14.44** compiler binaries |
| Windows SDK | Desktop headers, libraries and tools; the project selects the installed Windows 10/11 SDK (`10.0`) |
| CMake | FreeType build; standalone CMake or Visual Studio's bundled CMake |
| AMD HIP SDK 6.4 | HIP compiler and `gfx1030` code-object tools |
| Python + NumPy | Offline model conversion; release staging specifically requires a full CPython 3.12 x64 installation |

Select both the required platform toolset and compiler components in Visual Studio Installer. The scripts deliberately pair v145's build infrastructure with MSVC 14.44; do not assume the default compiler in a newer VS installation is equivalent. The build command overrides the upstream project's v143 setting.

Use the tested HIP SDK 6.4 rather than assuming a newer SDK has identical RDNA2 support. The [AMD HIP SDK page](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) provides SDK information. Actual GPU tests also need the [runtime requirements](INSTALLATION.md#before-installing).

## Paths and checkout

The default installation paths are:

```text
C:\Program Files\Microsoft Visual Studio\18\Community
C:\Program Files\AMD\ROCm\6.4
```

If yours differ, adjust the defaults in `tools/Enter-Toolchain.ps1`, `tools/Build-FreeType.ps1` and `tools/Build-HipBackend.ps1` consistently. The HIP and FreeType scripts accept `-VsRoot`, and the HIP script also accepts `-HipRoot`; the frontend wrapper currently uses the shared defaults. Merely opening another VS developer shell does not override the script's toolchain selection.

Open PowerShell 7 in the repository root and initialize submodules:

```powershell
git submodule update --init --recursive
```

GitHub's source ZIP alone is insufficient if submodule contents are missing. The first frontend build fetches pinned FreeType 2.13.3 from its official source mirror. Its NMake build uses an ASCII temporary directory to avoid response-file problems with Unicode checkout paths. FreeType links with the shared CRT (`/MD`), matching the frontend.

## Compile

```powershell
.\tools\Build-HipBackend.ps1
.\tools\Build-OptiScaler.ps1
```

| Output | File |
| --- | --- |
| HIP companion | `build/hip-gfx1030/dlssnr_hip_scheduled_bridge.dll` |
| OptiScaler frontend | `x64/Release/OptiScaler.dll` |

The HIP script verifies the scheduled API exports and architecture bundle. The frontend wrapper builds FreeType as needed and disables upstream's legacy pre/post-build file operations. Compilation and release assembly are separate steps; copying arbitrary files from the output directory is not the release process.

Reports are written beneath ignored `build/` directories. Source builds do not produce or include model weights automatically. Prepare a private package from your own supported DLL using the [converter](MODEL.md#standalone-conversion).

## Assemble an installable release

Prepare a full CPython 3.12 x64 installation with NumPy. It must contain `python.exe`, `python312.dll`, `DLLs`, `Lib`, the Python license, and NumPy's library/distribution metadata. An executable path, a virtual environment alone or the minimal embeddable Python ZIP is not the required input.

Release versions use `r<release number>-<parent version>`, for example `r1-0.8.91`.
The parent version identifies the wilsjo2 base. The `r` number increases for every
release and continues when the parent version changes. `VERSION.txt` supplies the
default package label; release notes and the changelog use the same version.

Use that installation's interpreter to install NumPy, then package:

```powershell
& "C:\Tools\Python312\python.exe" -m pip install numpy
.\package_release.ps1 -PortablePythonHome "C:\Tools\Python312"
```

The script builds both DLLs and assembles an allowlisted directory and ZIP in `release/`. `-SkipBuild` reuses existing binaries; use it only after confirming they are the intended production build. `-NoZip` creates the staged directory without an archive. Use `-Version "local-1"` for a separate local package rather than overwriting an archive used for testing. The installer records the package version in its installation manifest.

Packaging audits DLL exports and default settings, includes runtime licenses and portable Python/NumPy, and emits file checksums. It excludes NVIDIA model DLLs, converted weights, AMD driver libraries and diagnostic test exports. Runtime preparation remains the user's local installation step.

## Standalone checks

Before publishing source changes, run `python tools/check_release_source.py`. It checks public documentation links and rejects tracked model/driver assets. GitHub runs the same check; GPU builds and qualification remain separate.

These tests execute the source-built hosts and HIP backend on the GPU, outside a game. They need the compatible AMD runtime and a private model package. Start with:

```powershell
.\tests\rdna2\Run-Smoke.ps1 -ModelPath "D:\Models\NR\mixed-v5-gfx1030.nrwgt"
.\tests\rdna2\Run-Dx11.ps1 -ModelPath "D:\Models\NR\mixed-v5-gfx1030.nrwgt"
.\tests\rdna2\Run-Failures.ps1 -ModelPath "D:\Models\NR\mixed-v5-gfx1030.nrwgt"
```

The smoke test builds the hosts and binaries, produces deterministic input, and checks pre/post NR against independent colour-codec and executor results. The other two commands reuse its generated input. Smoke scenarios also cover queue switching, resize, forwarding wrappers and selected formats; consult `Run-Smoke.ps1` for its accepted `-Scenario` values. Host correctness does not establish game compatibility or agreement with NVIDIA hardware.

The optional `Build-OptiScaler.ps1 -NativeTestHooks` build adds source-test hooks for queue diagnostics. It is not a release frontend. Rebuild without that switch before packaging; the packager rejects diagnostic exports.

Use `Run-Smoke.ps1 -WorkingScale 0.75` for reduced-resolution codec parity, `-Scenario notifier` for destruction-notification compatibility, or `-WorkingScale 2 -FunctionalOnly` for supersampling admission. `Run-Dx11.ps1` also accepts `-WorkingScale`. Both accept explicit `-FrontendPath` and `-CompanionPath` for isolated builds.

To keep large build artifacts outside the checkout, both build scripts accept `-OutputRoot`. The frontend also accepts `-ObjectRoot` with `-OutputRoot` to reuse a compiler cache. Package these binaries with `package_release.ps1 -SkipBuild -FrontendPath <DLL> -HipBackendPath <DLL> -OutputRoot <directory> -PortablePythonHome <Python directory>`.

The standard frontend build explicitly disables FG transition tracing and native test exports. The packager rejects trace binaries, diagnostic exports, enabled default watermarks and machine-specific FSR paths. See the [tool inventory](../tools/README.md) for build helpers and source-only tests.

For performance work, compare paired, uninstrumented full-frame runs. Treat stage profiling as a diagnostic aid, because instrumentation can alter the measured workload.
