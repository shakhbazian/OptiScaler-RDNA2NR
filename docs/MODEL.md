# Model preparation

[Documentation](README.md) · [Installation](INSTALLATION.md)

## Supported source

Supply an original `nvngx_dlssnr.dll` from a game that includes DLSS 5 Neural
Rendering. Two versions are supported:

| Source version | Prepared models | SHA-256 |
| --- | --- | --- |
| 310.9.16.0 (recommended) | Models A and B | `2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F` |
| 310.8.0.0 | Model A only | `E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E` |

A is the original NR network; B is NVIDIA's newer, optimized network. The
installer validates the file by SHA-256. To check it manually:

```powershell
Get-FileHash -LiteralPath "C:\Models\nvngx_dlssnr.dll" -Algorithm SHA256
```

You supply the source under its applicable terms. This project distributes the
integration and converter code, not NVIDIA's DLL or converted weights. The code
license does not change model ownership or grant redistribution rights.

## What conversion does

The installer reads the DLL as data, extracts its tensors, decodes FP8 E4M3 into
FP16, restores layouts and controls, then writes checked `.nrwgt` packages. It
prepares both networks from 310.9.16.0. A's tensor payload is the same in both supported DLLs.

The packages store FP16 tensors. At startup, HIP prepares selected weights for
mixed INT8/FP16 execution. See [architecture](ARCHITECTURE.md#network-and-arithmetic)
for the execution approach.

Conversion and validation finish before game files are changed. A verified
package is reused on subsequent installations. GPU preparation happens when NR
starts or the selected model changes.

## Storage and selection

Packages are cached below `%LOCALAPPDATA%\OptiScaler-RDNA2NR\models`, separated
by graph and source hash. A occupies about 278 MiB, B about 216 MiB; together
about 494 MiB. Uninstalling one game retains this shared cache.

Prepared files are named `nr-model-a-gfx1030.nrwgt` and
`nr-model-b-gfx1030.nrwgt`. An update verifies and reuses files stored under the
older names. Explicit custom paths continue to work.

The installer records the prepared A path in `[DlssNr] ModelPath` and B path in
`ModelBPath`. Empty paths use the default per-user cache. To store them elsewhere:

```ini
[DlssNr]
HipModel=auto
ModelPath=D:\Models\NR\nr-model-a-gfx1030.nrwgt
ModelBPath=D:\Models\NR\nr-model-b-gfx1030.nrwgt
```

Auto selects B when available, otherwise A. The menu also allows an explicit
A or B choice. A model change preserves controls and scale, drains the previous
session, and resets history. A corrupt or mismatched package is reported as an
error. Move the damaged cache file aside before regenerating it, then restart
the game. Keep packages accessible to the account running the game.

## Standalone conversion

The GUI prepares every model available in your source. Developers can choose
one with `--model original` (A) or `--model optimized` (B):

```powershell
.\tools\python\python.exe .\tools\model_converter\write_runtime_package.py --source "C:\Models\nvngx_dlssnr.dll" --model optimized --output "D:\Models\NR\nr-model-b-gfx1030.nrwgt"
```

In a source checkout, use Python with NumPy. B requires the 310.9.16.0 source.
The catalog in `tools/model_converter/model_catalog.json` pins package sizes and
hashes; the converter, installer and runtime validate the selected identity.
