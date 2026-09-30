# Neural Rendering implementation

OptiScaler-RDNA2NR retains the common NR frontend and adds an AMD HIP backend. Auto selects a backend for the rendering adapter: HIP on supported `gfx1030` devices, ordinary NGX on NVIDIA. The HIP executor lives in `rdna2_nr/`; NVIDIA uses a separately supplied runtime through the installed driver's NGX core.

See the [architecture](../../docs/ARCHITECTURE.md), [build instructions](../../docs/BUILDING.md) and [compatibility matrix](../../docs/COMPATIBILITY.md). The [parent-fork implementation notes](../../docs/upstream/IMPLEMENTATION.md) preserve the inherited NVIDIA source map and design references. They describe more capabilities than the HIP backend implements.
