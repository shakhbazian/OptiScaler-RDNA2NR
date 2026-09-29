#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace NrV2 {
using Sha256Digest = std::array<std::uint8_t, 32>;

Sha256Digest Sha256(const void* data, std::size_t size);
Sha256Digest ParseSha256(const std::string& text);
std::string FormatSha256(const Sha256Digest& digest);

struct WeightTensor {
    std::uint32_t block = 0;
    std::string role;
    std::vector<std::uint32_t> shape;
    std::uint64_t byteOffset = 0;
    std::uint64_t byteLength = 0;
    Sha256Digest sha256{};
};

class WeightPackage {
  public:
    static WeightPackage LoadFile(const wchar_t* path,
                                  const Sha256Digest& expectedPackage,
                                  const Sha256Digest& expectedSource);
    static WeightPackage LoadFile(const char* path,
                                  const Sha256Digest& expectedPackage,
                                  const Sha256Digest& expectedSource);
    static WeightPackage LoadMemory(std::vector<std::uint8_t> bytes,
                                    const Sha256Digest& expectedPackage,
                                    const Sha256Digest& expectedSource);

    const WeightTensor& Tensor(std::uint32_t block, const std::string& role) const;
    const std::uint16_t* HalfData(const WeightTensor& tensor) const;
    const std::uint8_t* Data() const noexcept { return bytes_.data(); }
    std::size_t TensorCount() const noexcept { return tensors_.size(); }
    std::size_t ByteLength() const noexcept { return bytes_.size(); }
    const Sha256Digest& PackageHash() const noexcept { return packageHash_; }
    const Sha256Digest& SourceHash() const noexcept { return sourceHash_; }

  private:
    std::vector<std::uint8_t> bytes_;
    std::vector<WeightTensor> tensors_;
    std::unordered_map<std::string, std::size_t> index_;
    Sha256Digest packageHash_{};
    Sha256Digest sourceHash_{};
};
} // namespace NrV2
