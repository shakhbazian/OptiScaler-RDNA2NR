#pragma once

#include <d3d12.h>
#include <cstdint>

namespace DlssNr {

enum class NrBackendSelection { NvidiaNgx, AmdHip, Unsupported };

// Auto is tied to the D3D12 device actually used by the feature. The bundled
// HIP binary currently targets gfx1030; another AMD adapter must not silently
// receive a model compiled for different hardware.
NrBackendSelection SelectNrBackend(ID3D12Device* device, std::uint32_t configured) noexcept;
const char* NrBackendName(NrBackendSelection backend) noexcept;

} // namespace DlssNr
