#pragma once
#include <d3d12.h>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <dlssnr/backend/INrBackend.h>
struct NVSDK_NGX_Parameter;
namespace DlssNr::Native {
struct RuntimeStatus {
    bool ready=false,pending=false,fault=false;
    unsigned lastOutcome=0; // 0 no submission, 1 applied, 2 raw fallback.
    std::uint64_t applied=0,bypassed=0;
    std::string reason;
};
RuntimeStatus ReadRuntimeStatus() noexcept;
struct EvaluationScope {
    void* previous=nullptr;
    void* context=nullptr;
    EvaluationScope(std::uint64_t feature,bool hdr);
    ~EvaluationScope();
};
bool Initialize(ID3D12Device* device) noexcept;
bool PrepareOwned(std::uint64_t,ID3D12Device*,ID3D12CommandQueue*,unsigned,unsigned,bool) noexcept;
std::uint64_t AppliedFrames() noexcept;
std::uint64_t BypassedFrames() noexcept;
bool SessionReady() noexcept;
bool FrontendSessionReady(NVSDK_NGX_Parameter*, bool beforeUpscale, ID3D12Device*,
                          ID3D12CommandQueue*) noexcept;
const char* Mark(ID3D12GraphicsCommandList* list,NVSDK_NGX_Parameter* parameters,bool beforeUpscale,
                 bool warmOnly=false,bool interop=false) noexcept;
// Records the shared codec privately. Only a fully recorded model+resolve is
// attached to the game's marker; raw submission never executes codec commands.
const char* RecordFrontend(ID3D12GraphicsCommandList*,NVSDK_NGX_Parameter*,bool,
                          std::shared_ptr<void>,const std::function<void(ID3D12GraphicsCommandList*)>&,
                          bool interop=false) noexcept;
bool FrontendActive() noexcept;
D3D12_RESOURCE_STATES FrontendColorState() noexcept;
void CompleteFrontend() noexcept;
std::unique_ptr<INrBackend> MakeModelBackend();
void ReleaseFeature(std::uint64_t feature) noexcept;
void Shutdown() noexcept;
}
