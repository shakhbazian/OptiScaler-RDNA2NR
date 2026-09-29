#pragma once
#include <hip/hip_fp16.h>
#include "e4m3fn.h"

namespace rdna2_nr {
// Equivalent encode/decode publication without reconstructing FP8 exponents.
// Normals retain the FP16 exponent and round away seven fraction bits. Tiny
// values round on the exact 2^-9 lattice; preserve signed zero and NaN sign.
__device__ inline unsigned short publish_e4m3_half_bits(unsigned short bits) {
    const unsigned sign=bits&0x8000u,magnitude=bits&0x7fffu;
    if(magnitude>0x7c00u)return sign|0x7e00u;
    if(magnitude>=0x5f00u)return sign|0x5f00u;
    if(magnitude<0x2400u){
        const int rounded=__float2int_rn(__half2float(__ushort_as_half(magnitude))*512.f);
        return sign|__half_as_ushort(__float2half(float(rounded)*(1.f/512.f)));
    }
    return sign|((magnitude+63u+((magnitude>>7)&1u))&~127u);
}

// FP16 -> E4M3FN, nearest-even, finite saturation, signed canonical NaNs.
// Integer rounding for normals; exact power-of-two scaling for subnormals.
__device__ inline unsigned char half_bits_to_e4m3fn(unsigned short bits) {
    unsigned sign=(bits>>8)&128, magnitude=bits&0x7fff;
    if (magnitude>0x7c00) return sign|127;
    if (magnitude>=0x5f00) return sign|126;
    unsigned code;
    if (magnitude<0x2400) {
        code=__float2int_rn(__half2float(__ushort_as_half(magnitude))*512.0f);
    } else {
        unsigned retained=magnitude>>7, remainder=magnitude&127;
        code=retained-64+(remainder>64 || (remainder==64 && (retained&1)));
    }
    return sign|code;
}

// Attributed candidate: pinned MLX-DLSS quadratic gate, not vendor parity.
// Keep each half FMA/multiply rounding explicit; no GELU substitution.
__device__ inline __half ffn_gate(__half value) {
    __half c=__float2half(fminf(4.0f,fmaxf(-4.0f,__half2float(value))));
    __half linear=__hfma(__float2half(fabsf(__half2float(c))),
                         __float2half(-0.055908203125f),__float2half(0.447265625f));
    __half gate=__hfma(c,linear,__float2half(0.89453125f));
    return __hmul(value,gate);
}
}
