# NVIDIA compatibility

[Documentation](README.md) · [Compatibility](COMPATIBILITY.md)

The frontend retains the ordinary NVIDIA NGX route. This is separate from the AMD HIP executor: it uses the installed NVIDIA driver's NGX core and a user-supplied `nvngx_dlssnr.dll`. A supported model runtime, driver and GPU combination is still required. AMD testing does not qualify that combination.

## Manual setup

1. Close the game and back up an existing OptiScaler installation and configuration.
2. Place this release's `OptiScaler.dll`, `OptiScaler.ini` and complete `OptiScaler/` backend folder beside the actual rendering executable. Preserve an existing INI if needed, then review its NR settings.
3. Use the inherited `setup_windows.bat` to select a proxy filename, or rename the frontend manually according to the game's OptiScaler setup. Handle existing loaders before replacing their files.
4. Place your compatible `nvngx_dlssnr.dll` beside the game executable/proxy. This route loads the NVIDIA runtime; it does not use the HIP converter or `.nrwgt` cache.
5. Enable a supported in-game upscaler, open the overlay, choose Auto or NVIDIA, and enable NR. Start with one pass and ordinary pre/post-upscale placement.

The AMD GUI installer does not perform this setup: it requires the supported original source DLL and creates HIP weights. Do not mix its ownership manifest with an independently managed manual installation.

The integrated NGX path does not require the obsolete `nvngx.dll_dlssnr.dll` forwarder. General OptiScaler loaders and a game's genuine NGX/DLSS files serve other purposes; do not remove them indiscriminately.

## Scope

The standard build excludes experimental NVIDIA hybrid/NVFP4 and RTX 40 MFG-unlock implementations. It does not unlock the original runtime for unsupported NVIDIA cards. Menu options and capabilities differ between NGX and HIP.

The [parent-fork references](upstream/README.md) retain provenance and historical experiments. Parent releases can have different helper DLLs and setup requirements; copying their installation instructions wholesale is not a supported update procedure for this build.
