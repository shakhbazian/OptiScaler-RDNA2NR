#pragma once

#include <d3d12.h>
#include <dxgiformat.h>
#include <cstdint>

namespace DlssNr
{
enum class NrBackendResult
{
    Success,
    NotInitialized,
    InvalidArgument,
    InvalidResource,
    Unsupported
};

struct NrBackendEvaluation
{
    NrBackendResult result = NrBackendResult::NotInitialized;
    std::uint64_t outputEpoch = 0;
    bool outputDelayed = false;
};

struct NrBackendSize
{
    unsigned int width = 0;
    unsigned int height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
};

// The caller owns resource lifetime and command-list submission. On entry, input is in
// NON_PIXEL_SHADER_RESOURCE and output is in UNORDERED_ACCESS. Evaluate must return both resources
// to those states so the existing encode/resolve pipeline does not depend on the selected backend.
struct NrBackendFrame
{
    ID3D12GraphicsCommandList* commands = nullptr;
    // Cross-API backends use the queue and monotonically changing submission epoch to process the
    // preceding frame. They must not assume commands recorded in this list have already executed.
    ID3D12CommandQueue* queue = nullptr;
    std::uint64_t submissionEpoch = 0;
    ID3D12Resource* input = nullptr;
    ID3D12Resource* depth = nullptr;
    ID3D12Resource* motion = nullptr;
    ID3D12Resource* output = nullptr;
    NrBackendSize size{};
    unsigned int guideWidth = 0;
    unsigned int guideHeight = 0;
    unsigned int motionWidth = 0;
    unsigned int motionHeight = 0;
    unsigned int depthBaseX = 0;
    unsigned int depthBaseY = 0;
    unsigned int motionBaseX = 0;
    unsigned int motionBaseY = 0;
    float motionScaleX = 1.0f;
    float motionScaleY = 1.0f;
    bool depthInverted = false;
    bool reset = false;
};

class INrBackend
{
  public:
    virtual ~INrBackend() = default;
    virtual const char* Name() const noexcept = 0;
    virtual bool Initialize(ID3D12Device* device) = 0;
    virtual bool Resize(const NrBackendSize& size) = 0;
    virtual NrBackendEvaluation Evaluate(const NrBackendFrame& frame) = 0;
    virtual void Shutdown() noexcept = 0;
};
} // namespace DlssNr
