#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

namespace rdna2_nr {
// OCP/ONNX E4M3FN: bias 7, signed zero, finite exponent 15,
// NaN only at 0x7f/0xff. Not E4M3FNUZ (bias 8).
// All finite values expand exactly to binary16; preserve NaN sign,
// canonicalize its payload to a quiet NaN.
__host__ __device__ inline std::uint16_t e4m3fn_to_half_bits(std::uint8_t input) {
    const unsigned sign = (input & 0x80u) << 8;
    unsigned exponent = (input >> 3) & 15u;
    unsigned mantissa = input & 7u;
    if (exponent == 15 && mantissa == 7) return sign | 0x7e00u;
    if (exponent != 0) return sign | ((exponent + 8u) << 10) | (mantissa << 7);
    if (mantissa == 0) return sign;
    // FP8 subnormals m*2^-9 are normal in FP16.
    unsigned half_exponent = 9;
    while (mantissa < 8) { mantissa <<= 1; --half_exponent; }
    return sign | (half_exponent << 10) | ((mantissa - 8) << 7);
}
}
