#include "pch.h"
#include "NrBackendSelection.h"

#include <device_info/device_info.hpp>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <mutex>
#include <unordered_map>
#include <cstring>

namespace DlssNr {
namespace {

NrBackendSelection Detect(ID3D12Device* device) noexcept
{
    if (!device)
        return NrBackendSelection::Unsupported;
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) ||
        FAILED(factory->EnumAdapterByLuid(device->GetAdapterLuid(), IID_PPV_ARGS(&adapter))))
        return NrBackendSelection::Unsupported;

    DXGI_ADAPTER_DESC1 desc {};
    if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE))
        return NrBackendSelection::Unsupported;
    if (desc.VendorId == 0x10DE)
        return NrBackendSelection::NvidiaNgx;
    if (desc.VendorId != device_info::kAmdVendorId)
        return NrBackendSelection::Unsupported;

    const auto info = device_info::GetCardInfo({ desc.VendorId, desc.DeviceId, desc.Revision });
    if (info && info->gfx_target && std::strcmp(info->gfx_target, "gfx1030") == 0)
        return NrBackendSelection::AmdHip;
    return NrBackendSelection::Unsupported;
}

} // namespace

NrBackendSelection SelectNrBackend(ID3D12Device* device, std::uint32_t configured) noexcept
{
    if ((configured != 0 && configured != 3 && configured != 4) || !device)
        return NrBackendSelection::Unsupported;

    const LUID luid = device->GetAdapterLuid();
    const auto key = (std::uint64_t(static_cast<std::uint32_t>(luid.HighPart)) << 32) | luid.LowPart;
    static std::mutex mutex;
    static std::unordered_map<std::uint64_t, NrBackendSelection> cache;
    std::lock_guard lock(mutex);
    const auto found = cache.find(key);
    const auto detected = found != cache.end() ? found->second : cache.emplace(key, Detect(device)).first->second;
    if (configured == 0 && detected != NrBackendSelection::NvidiaNgx)
        return NrBackendSelection::Unsupported;
    if (configured == 3 && detected != NrBackendSelection::AmdHip)
        return NrBackendSelection::Unsupported;
    return detected;
}

const char* NrBackendName(NrBackendSelection backend) noexcept
{
    switch (backend)
    {
    case NrBackendSelection::NvidiaNgx: return "NVIDIA NGX";
    case NrBackendSelection::AmdHip: return "AMD HIP (gfx1030)";
    default: return "Unavailable for this GPU";
    }
}

} // namespace DlssNr
