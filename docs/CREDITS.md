# Credits

OptiScaler-RDNA2NR adds its AMD HIP executor, model converter and installer to [wilsjo2's OptiScaler-DLSSNR](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass). The parent fork builds on [Dagherbou/OptiScaler_DLSSNR](https://github.com/Dagherbou/OptiScaler_DLSSNR) and [OptiScaler](https://github.com/optiscaler/OptiScaler). OptiScaler began with [PotatoOfDoom's CyberFSR2](https://github.com/PotatoOfDoom/CyberFSR2).

Colour processing derives from [clshortfuse's RenoDX](https://github.com/clshortfuse/renodx); see [attribution/licence](../Licenses/RenoDX_ATTRIBUTION.txt).

Inherited integrations include hhkbble's multipass/composition, [y4my4my4m's Vulkan work](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/blob/main/docs/NR-VULKAN.md) and [cmh1448's motion metadata](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/blob/main/docs/NR-MOTION-METADATA.md). These credits describe provenance, rather than HIP feature support; see the [compatibility guide](COMPATIBILITY.md).

The standard RDNA2NR build excludes the experimental NVIDIA hybrid/NVFP4 and RTX 40 MFG-unlock paths. Attribution for inherited source is retained below.

## RTX 40 MFG unlock

The parent fork's RTX 40 multi frame generation unlock was adapted from [y4my4my4m's fork](https://github.com/y4my4my4m/OptiScaler_DLSSNR_Multipass_MFG) (GPL-3.0). Its provider discovery, Streamline plugin frame-ceiling patch, software frame pacing option and PTX temporal fix derive from [KleberMotta's fork](https://github.com/KleberMotta/OptiScaler-DLSS5-MFG-RTX40) (MIT), a port of the MFG Unlock ReShade addon by [Dreamt](https://github.com/ImDreamt/MFGAdaUnlock-RenoDx) and [mavismmg](https://github.com/mavismmg/MFGAdaUnlock-RenoDx). The technique originates from [dashdogy's RTX40MFG-Unlock](https://github.com/dashdogy/RTX40MFG-Unlock). See the parent's [MFG notes](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/blob/main/docs/RTX40-MFG.md) and retained [license](../Licenses/MFGUnlock_LICENSE.txt).

## Fork contributors

- [Yuri Grib / @BeliyG3](https://github.com/BeliyG3) for the MIT-licensed peripheral spatial mapping adapted from [Optimizer FPS for DLSS5](https://github.com/BeliyG3/optimizer-fps-dlss5/tree/64902dd6a02460e5f6b778504ec2a4005faf4d9c). See the parent's [spatial-compression notes](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/blob/main/docs/NR-SPATIAL-COMPRESSION.md) for integration details.

- [@LorisPicariello](https://github.com/LorisPicariello) for investigating and testing RDR2's Finished Picture NR / OptiFG interaction, and for [PR #70](https://github.com/wilsjo2/OptiScaler-DLSSNR-PreSR-Multipass/pull/70), which informed the rewritten finished-picture ordering and subsequent queue-safety work.

## OptiScaler contributors

- @PotatoOfDoom for CyberFSR2.
- @Artur for DLSS Enabler and help with the NVNGX API.
- @LukeFZ and @Nukem for their mods and shared knowledge.
- @FakeMichau for support, testing and features.
- @QM for testing and access to games.
- @TheRazerMD for testing and support.
- @Cryio, @krispy, @krisshietala, @Lordubuntu, @scz and @Veeqo for the earlier compatibility matrix.
- The DLSS2FSR community for its support.

This project uses [FreeType](https://gitlab.freedesktop.org/freetype/freetype), licensed under the [FTL](https://gitlab.freedesktop.org/freetype/freetype/-/blob/master/docs/FTL.TXT). Other notices are in [Licenses](../Licenses).

The local model converter's packaged runtime uses [CPython](https://www.python.org/) and [NumPy](https://numpy.org/). Their licenses and distribution metadata accompany the portable runtime in `tools/python/`. NVIDIA's model and AMD's driver runtime are user-supplied and are not distributed by this project.

## Upstream sponsorship

Upstream credits [SignPath.io](https://signpath.io/) for Windows code signing and the [SignPath Foundation](https://signpath.org/) for its certificate.

Support upstream: [cdozdil](https://github.com/sponsors/cdozdil?frequency=one-time) and [nitec](https://buymeacoffee.com/nitec).
