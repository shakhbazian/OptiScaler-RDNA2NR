#include "../include/nr_gpu_model.h"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace {
void Check(hipError_t result, const char* operation) {
    if (result != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(result));
}
} // namespace

namespace NrV2 {
GpuModel::GpuModel(const WeightPackage& package)
    : package_(package), byteLength_(package.ByteLength()) {
    Check(hipMalloc(&device_, byteLength_), "hipMalloc model");
    try {
        Check(hipMemcpy(device_, package.Data(), byteLength_, hipMemcpyHostToDevice),
              "hipMemcpy model");
    } catch (...) {
        static_cast<void>(hipFree(device_));
        device_ = nullptr;
        throw;
    }
}

GpuModel::~GpuModel() {
    if (device_) static_cast<void>(hipFree(device_));
}

bool GpuModel::Release() noexcept {
    if (!device_) return true;
    if (hipFree(device_) != hipSuccess) return false;
    device_ = nullptr;
    byteLength_ = 0;
    return true;
}

const void* GpuModel::TensorData(std::uint32_t block, const char* role,
                                 std::size_t elementCount) const {
    if (!device_) throw std::runtime_error("GPU model is not resident");
    const auto& tensor = package_.Tensor(block, role);
    if (tensor.byteLength != elementCount * 2)
        throw std::runtime_error("GPU model tensor shape mismatch");
    return device_ + tensor.byteOffset;
}
} // namespace NrV2
