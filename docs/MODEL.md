# Model preparation

[Documentation](README.md) · [Installation](INSTALLATION.md)

## Supported source

The converter accepts one original NVIDIA-signed `nvngx_dlssnr.dll`, identified by its SHA-256:

```text
E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E
```

A matching filename or version string is not sufficient. Modified DLLs and other model releases are rejected. Check your file with:

```powershell
Get-FileHash -LiteralPath "C:\Models\nvngx_dlssnr.dll" -Algorithm SHA256
```

You must obtain and supply the source yourself under its applicable terms. This project distributes its integration and converter code; it does not distribute NVIDIA's model, a converted weight package or a license to redistribute either. The code's GPL license does not change the model's ownership.

## What conversion does

The installer reads the DLL as a binary data container. It extracts the network tensors, decodes their FP8 E4M3 representation into FP16, restores the layouts and scalar controls required by the executor, and writes a checksummed `.nrwgt` weight package. The original DLL is not executed or copied into the game folder by the AMD installer.

This keeps the original network. There is no student-network training, pruning or downloadable replacement model in this release. The stored package contains FP16 tensors; the HIP executor prepares selected weights for INT8 execution when loading it. Sensitive 32-channel stages remain FP16. See [architecture](ARCHITECTURE.md#network-and-arithmetic) for the execution split.

Conversion and package validation happen before game files are changed. A verified cache is reused, so installing into another game does not require converting the same source again. GPU preparation is separate and takes place when the game starts NR.

## Storage

The default per-user cache is:

```text
%LOCALAPPDATA%\OptiScaler-RDNA2NR\models\1\E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E\mixed-v5-gfx1030.nrwgt
```

The filename is a package identifier, rather than a user-selectable quality preset. The supported package is 291,595,458 bytes, about 278 MiB. Its SHA-256 is:

```text
A7E6EE38172A81E12D613FA9A2F57E32AA1944908E56CD2A33E1F6C94369E3CB
```

Leave `[DlssNr] ModelPath` empty to use this cache. To keep the package elsewhere, set an absolute path in the game's `OptiScaler.ini`:

```ini
[DlssNr]
ModelPath=D:\Models\NR\mixed-v5-gfx1030.nrwgt
```

Keep the package accessible to the Windows account running the game. A corrupt cache is refused rather than silently overwritten. Back up or move the affected file before regenerating it from a verified source. Restart the game after correcting a missing or invalid model.

Uninstalling a game integration keeps the shared cache, because another installed game may use it. You can remove it manually when no installation needs it.

## Standalone conversion

The GUI installer is the usual path. Developers can run the bundled converter directly from an extracted release:

```powershell
.\tools\python\python.exe -I .\tools\model_converter\write_runtime_package.py --source "C:\Models\nvngx_dlssnr.dll" --output "D:\Models\NR\mixed-v5-gfx1030.nrwgt"
```

In a source checkout, use your own Python with NumPy instead of `tools\python\python.exe`. Conversion runs on the CPU; it does not require an NVIDIA GPU or execute the original NVIDIA runtime.
