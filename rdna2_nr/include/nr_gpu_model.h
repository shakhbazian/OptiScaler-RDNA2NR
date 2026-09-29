#pragma once

#include "nr_weight_package.h"

#include <cstddef>

namespace NrV2 {
class GpuModel {
  public:
    explicit GpuModel(const WeightPackage& package);
    ~GpuModel();
    GpuModel(const GpuModel&) = delete;
    GpuModel& operator=(const GpuModel&) = delete;

    const void* TensorData(std::uint32_t block, const char* role,
                           std::size_t elementCount) const;
    std::size_t ByteLength() const noexcept { return byteLength_; }
    std::size_t TensorCount() const noexcept { return package_.TensorCount(); }
    std::size_t UploadOperations() const noexcept { return device_ ? 1 : 0; }
    const Sha256Digest& PackageHash() const noexcept { return package_.PackageHash(); }
    const Sha256Digest& SourceHash() const noexcept { return package_.SourceHash(); }
    // Checked teardown for production owners. On failure the pointer remains
    // owned and must be quarantined; the destructor is only a best-effort fallback.
    bool Release() noexcept;

  private:
    const WeightPackage& package_;
    unsigned char* device_ = nullptr;
    std::size_t byteLength_ = 0;
};
} // namespace NrV2
