#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include "../include/nr_weight_package.h"

#include <windows.h>
#include <bcrypt.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace {
using NrV2::Sha256Digest;

template<class T> T Read(const std::uint8_t* data) {
    T result{};
    std::memcpy(&result, data, sizeof(result));
    return result;
}

std::string Key(std::uint32_t block, const std::string& role) {
    return std::to_string(block) + ":" + role;
}

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool IsZero(const std::uint8_t* begin, const std::uint8_t* end) {
    return std::all_of(begin, end, [](std::uint8_t value) { return value == 0; });
}
} // namespace

namespace NrV2 {
Sha256Digest Sha256(const void* data, std::size_t size) {
    Require(size <= std::numeric_limits<ULONG>::max(), "SHA-256 input exceeds Windows CNG limit");
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD objectBytes = 0, hashBytes = 0, returned = 0;
    std::vector<std::uint8_t> object;
    Sha256Digest result{};
    auto check = [](NTSTATUS status, const char* operation) {
        if (status < 0) throw std::runtime_error(operation);
    };
    try {
        check(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0),
              "BCryptOpenAlgorithmProvider failed");
        check(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                                reinterpret_cast<PUCHAR>(&objectBytes), sizeof(objectBytes),
                                &returned, 0), "BCryptGetProperty object failed");
        check(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                                reinterpret_cast<PUCHAR>(&hashBytes), sizeof(hashBytes),
                                &returned, 0), "BCryptGetProperty hash failed");
        Require(hashBytes == result.size(), "Unexpected SHA-256 size");
        object.resize(objectBytes);
        check(BCryptCreateHash(algorithm, &hash, object.data(), objectBytes, nullptr, 0, 0),
              "BCryptCreateHash failed");
        check(BCryptHashData(hash, static_cast<PUCHAR>(const_cast<void*>(data)),
                             static_cast<ULONG>(size), 0), "BCryptHashData failed");
        check(BCryptFinishHash(hash, result.data(), static_cast<ULONG>(result.size()), 0),
              "BCryptFinishHash failed");
        BCryptDestroyHash(hash);
        BCryptCloseAlgorithmProvider(algorithm, 0);
        return result;
    } catch (...) {
        if (hash) BCryptDestroyHash(hash);
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
        throw;
    }
}

Sha256Digest ParseSha256(const std::string& text) {
    Require(text.size() == 64, "SHA-256 text must contain 64 hex digits");
    Sha256Digest result{};
    auto digit = [](char value) -> unsigned {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        throw std::runtime_error("Invalid SHA-256 hex digit");
    };
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = static_cast<std::uint8_t>((digit(text[index * 2]) << 4) |
                                                   digit(text[index * 2 + 1]));
    return result;
}

std::string FormatSha256(const Sha256Digest& digest) {
    constexpr char digits[] = "0123456789ABCDEF";
    std::string result(digest.size() * 2, '0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
        result[index * 2] = digits[digest[index] >> 4];
        result[index * 2 + 1] = digits[digest[index] & 15];
    }
    return result;
}

WeightPackage WeightPackage::LoadFile(const wchar_t* path,
                                      const Sha256Digest& expectedPackage,
                                      const Sha256Digest& expectedSource) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    Require(bool(stream), "Unable to open weight package");
    const auto length = stream.tellg();
    Require(length >= 0 && length <= std::streamoff(512ull * 1024 * 1024),
            "Weight package size is invalid");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    stream.seekg(0);
    Require(bool(stream.read(reinterpret_cast<char*>(bytes.data()), length)),
            "Unable to read weight package");
    return LoadMemory(std::move(bytes), expectedPackage, expectedSource);
}

WeightPackage WeightPackage::LoadFile(const char* path,
                                      const Sha256Digest& expectedPackage,
                                      const Sha256Digest& expectedSource) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    Require(bool(stream), "Unable to open weight package");
    const auto length = stream.tellg();
    Require(length >= 0 && length <= std::streamoff(512ull * 1024 * 1024),
            "Weight package size is invalid");
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
    stream.seekg(0);
    Require(bool(stream.read(reinterpret_cast<char*>(bytes.data()), length)),
            "Unable to read weight package");
    return LoadMemory(std::move(bytes), expectedPackage, expectedSource);
}

WeightPackage WeightPackage::LoadMemory(std::vector<std::uint8_t> bytes,
                                        const Sha256Digest& expectedPackage,
                                        const Sha256Digest& expectedSource) {
    constexpr std::size_t headerSize = 128, entrySize = 128, alignment = 64;
    Require(bytes.size() >= headerSize, "Truncated weight package header");
    const auto actualPackage = Sha256(bytes.data(), bytes.size());
    Require(actualPackage == expectedPackage, "Weight package SHA-256 mismatch");
    const auto* header = bytes.data();
    Require(std::memcmp(header, "NRWGT001", 8) == 0, "Unsupported weight package magic");
    Require(Read<std::uint32_t>(header + 8) == 1 && Read<std::uint32_t>(header + 12) == 1,
            "Unsupported graph or math contract");
    Require(Read<std::uint64_t>(header + 16) == bytes.size(), "Weight package length mismatch");
    const auto count = Read<std::uint32_t>(header + 24);
    Require(count == 641 && Read<std::uint32_t>(header + 28) == entrySize,
            "Unexpected tensor manifest size");
    Sha256Digest source{}, tableExpected{}, payloadExpected{};
    std::memcpy(source.data(), header + 32, 32);
    std::memcpy(tableExpected.data(), header + 64, 32);
    std::memcpy(payloadExpected.data(), header + 96, 32);
    Require(source == expectedSource, "Weight package source SHA-256 mismatch");
    const std::size_t tableBytes = std::size_t(count) * entrySize;
    Require(tableBytes / entrySize == count && headerSize + tableBytes <= bytes.size(),
            "Tensor table exceeds package");
    const std::size_t payloadStart = (headerSize + tableBytes + alignment - 1) /
                                     alignment * alignment;
    Require(payloadStart <= bytes.size(), "Payload offset exceeds package");
    Require(Sha256(bytes.data() + headerSize, tableBytes) == tableExpected,
            "Tensor table SHA-256 mismatch");
    Require(Sha256(bytes.data() + payloadStart, bytes.size() - payloadStart) == payloadExpected,
            "Weight payload SHA-256 mismatch");
    Require(IsZero(bytes.data() + headerSize + tableBytes, bytes.data() + payloadStart),
            "Nonzero table alignment padding");

    WeightPackage package;
    package.packageHash_ = actualPackage;
    package.sourceHash_ = source;
    package.bytes_ = std::move(bytes);
    std::uint64_t previousEnd = payloadStart;
    std::uint32_t previousBlock = 0;
    std::string previousRole;
    bool havePrevious = false;
    for (std::uint32_t index = 0; index < count; ++index) {
        const auto* entry = package.bytes_.data() + headerSize + std::size_t(index) * entrySize;
        WeightTensor tensor;
        tensor.block = Read<std::uint32_t>(entry);
        const auto dtype = Read<std::uint32_t>(entry + 4);
        const auto rank = Read<std::uint32_t>(entry + 8);
        Require(dtype == 1 && rank >= 1 && rank <= 6 && Read<std::uint32_t>(entry + 12) == 0,
                "Invalid tensor type, rank, or reserved field");
        Require(tensor.block <= 70, "Tensor block is outside the graph");
        std::uint64_t elements = 1;
        for (std::uint32_t dimension = 0; dimension < 6; ++dimension) {
            const auto extent = Read<std::uint32_t>(entry + 16 + dimension * 4);
            if (dimension < rank) {
                Require(extent != 0 && elements <= std::numeric_limits<std::uint64_t>::max() / extent,
                        "Invalid tensor shape");
                elements *= extent;
                tensor.shape.push_back(extent);
            } else {
                Require(extent == 0, "Unused tensor dimension is nonzero");
            }
        }
        const auto* roleBytes = entry + 40;
        const auto* terminator = static_cast<const std::uint8_t*>(std::memchr(roleBytes, 0, 32));
        Require(terminator && terminator != roleBytes && IsZero(terminator, roleBytes + 32),
                "Invalid tensor role termination");
        for (const auto* cursor = roleBytes; cursor != terminator; ++cursor)
            Require((*cursor >= 'a' && *cursor <= 'z') || (*cursor >= '0' && *cursor <= '9') ||
                    *cursor == '_', "Invalid tensor role character");
        tensor.role.assign(reinterpret_cast<const char*>(roleBytes),
                           reinterpret_cast<const char*>(terminator));
        tensor.byteOffset = Read<std::uint64_t>(entry + 72);
        tensor.byteLength = Read<std::uint64_t>(entry + 80);
        std::memcpy(tensor.sha256.data(), entry + 88, 32);
        Require(IsZero(entry + 120, entry + 128), "Tensor entry reserved bytes are nonzero");
        Require(elements <= std::numeric_limits<std::uint64_t>::max() / 2 &&
                tensor.byteLength == elements * 2, "Tensor byte length does not match shape");
        Require(tensor.byteOffset % alignment == 0 && tensor.byteOffset >= previousEnd &&
                tensor.byteOffset <= package.bytes_.size() &&
                tensor.byteLength <= package.bytes_.size() - tensor.byteOffset,
                "Tensor range is invalid or overlapping");
        Require(IsZero(package.bytes_.data() + previousEnd,
                       package.bytes_.data() + tensor.byteOffset),
                "Nonzero tensor alignment padding");
        const auto key = Key(tensor.block, tensor.role);
        Require(!havePrevious || previousBlock < tensor.block ||
                (previousBlock == tensor.block && previousRole < tensor.role),
                "Tensor table is not canonically sorted");
        Require(package.index_.emplace(key, package.tensors_.size()).second,
                "Duplicate tensor key");
        Require(Sha256(package.bytes_.data() + tensor.byteOffset,
                       static_cast<std::size_t>(tensor.byteLength)) == tensor.sha256,
                "Tensor SHA-256 mismatch");
        const auto* halves = package.bytes_.data() + tensor.byteOffset;
        for (std::uint64_t element = 0; element < elements; ++element)
            Require((Read<std::uint16_t>(halves + element * 2) & 0x7c00u) != 0x7c00u,
                    "Nonfinite FP16 tensor value");
        previousEnd = tensor.byteOffset + tensor.byteLength;
        previousBlock = tensor.block;
        previousRole = tensor.role;
        havePrevious = true;
        package.tensors_.push_back(std::move(tensor));
    }
    Require(previousEnd == package.bytes_.size(), "Trailing bytes after final tensor");
    return package;
}

const WeightTensor& WeightPackage::Tensor(std::uint32_t block, const std::string& role) const {
    const auto found = index_.find(Key(block, role));
    if (found == index_.end()) throw std::runtime_error("Missing weight tensor");
    return tensors_[found->second];
}

const std::uint16_t* WeightPackage::HalfData(const WeightTensor& tensor) const {
    Require(tensor.byteOffset <= bytes_.size() && tensor.byteLength <= bytes_.size() - tensor.byteOffset,
            "Stale tensor view");
    return reinterpret_cast<const std::uint16_t*>(bytes_.data() + tensor.byteOffset);
}
} // namespace NrV2
