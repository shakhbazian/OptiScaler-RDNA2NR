# NVIDIA compatibility

[Documentation](README.md) · [Compatibility](COMPATIBILITY.md)

The NVIDIA route uses the installed driver's NGX core and a user-supplied `nvngx_dlssnr.dll` compatible with the GPU and driver.

## Manual setup

1. Close the game and back up an existing OptiScaler installation and configuration.
2. Place this release's `OptiScaler.dll`, `OptiScaler.ini` and complete `OptiScaler/` backend folder beside the actual rendering executable. Preserve an existing INI if needed, then review its NR settings.
3. Rename the frontend to a proxy filename such as `dxgi.dll`, according to the game's OptiScaler setup. Handle existing loaders before replacing their files.
4. Place your compatible `nvngx_dlssnr.dll` beside the game executable/proxy. This route loads the NVIDIA runtime; it does not use the HIP converter or `.nrwgt` cache.
5. Enable a supported in-game upscaler, open the overlay, choose Auto or NVIDIA, and enable NR. Start with one pass and ordinary pre/post-upscale placement.

The AMD GUI installer does not perform this setup: it requires the supported original source DLL and creates HIP weights. Do not mix its ownership manifest with an independently managed manual installation.

The integrated NGX path does not require the obsolete `nvngx.dll_dlssnr.dll` forwarder. General OptiScaler loaders and a game's genuine NGX/DLSS files serve other purposes; do not remove them indiscriminately.

## Scope

NR DLL 310.9.16.0 is recommended and contains models A and B; 310.8.0.0 contains
only A. In the NVIDIA menu, **Original** selects A and **Optimized** selects B.
**Automatic** follows the runtime's default choice.

The standard build excludes experimental NVIDIA hybrid/NVFP4 and RTX 40 MFG-unlock implementations. It does not unlock the original runtime for unsupported NVIDIA cards. Menu options and capabilities differ between NGX and HIP.

The [parent-fork references](upstream/README.md) retain provenance and historical experiments. Parent releases can have different helper DLLs and setup requirements; copying their installation instructions wholesale is not a supported update procedure for this build.
