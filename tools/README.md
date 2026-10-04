# Build and verification tools

The release has one installation entry point: `Install-RDNA2NR.cmd`. Its PowerShell
implementation performs installation, verification and removal. Scripts in this
directory belong to the source checkout, except the model converter and portable
Python copied into releases.

| Tool | Purpose |
| --- | --- |
| `Build-HipBackend.ps1` | Build the HIP executor. |
| `Build-OptiScaler.ps1` | Build the production frontend. Test exports require an explicit switch. |
| `Enter-Toolchain.ps1`, `Build-FreeType.ps1` | Shared build dependencies; called by the build scripts. |
| `Stage-PortablePython.ps1` | Prepare the Python and NumPy runtime needed for model conversion. |
| `check_release_source.py` | Check public documentation, version labels and excluded private assets. |
| `Test-NrPresentQueue.ps1` | Optional GPU queue regression test; requires a prepared model and input. |
| `model_converter/` | Parse a supported user-provided DLL and prepare runtime weights. |

`package_release.ps1` at the repository root assembles and checks an allowlisted
release. `tests/` contains regression hosts and developer checks; they are not
shipped in the installer archive. Shader build helpers and dependency fetchers
are inherited build tools, not additional installers.

See [building](../docs/BUILDING.md) for prerequisites and commands. Local research
captures, diagnostic packages and one-off experiments do not belong in a public
release or its build instructions.

CPU checks used by CI are `tests/rdna2/frame_generation_contracts.cpp` (C++20)
and `tests/rdna2/Installer.Tests.ps1` (Windows PowerShell 5.1). The installer tests
use tiny data fixtures, including model and DLL placeholders; they do not load
vendor binaries or require the GPU. GPU hosts remain separate, opt-in checks.
The Windows contracts also check that disabling a FidelityFX watermark removes
its environment variable from both CRT readers and the Win32 process environment.
