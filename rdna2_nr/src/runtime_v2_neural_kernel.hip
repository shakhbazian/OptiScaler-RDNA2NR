#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "../include/nr_gpu_scheduled_state.h"

#include <cstddef>
#include <cstdint>

namespace {
struct Axis { float center[3], weight[3]; };

__device__ std::uint32_t Mix(std::uint32_t value) {
    const std::uint32_t shift = (value >> 28) + 4;
    return (value ^ (value >> shift)) * 0x108ef2d9u;
}
__device__ float NoiseLog(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    float exponent = float(int((bits >> 23) & 255) - 126);
    float mantissa = __uint_as_float((bits & 0x807fffffu) | 0x3f000000u);
    if (mantissa < .707106781186547524f) { exponent -= 1; mantissa = mantissa + mantissa - 1; }
    else mantissa -= 1;
    const float square = mantissa * mantissa;
    float result = 7.0376836292e-2f * mantissa - 1.1514610310e-1f;
    result = result * mantissa + 1.1676998740e-1f;
    result = result * mantissa - 1.2420140846e-1f;
    result = result * mantissa + 1.4249322787e-1f;
    result = result * mantissa - 1.6668057665e-1f;
    result = result * mantissa + 2.0000714765e-1f;
    result = result * mantissa - 2.4999993993e-1f;
    result = result * mantissa + 3.3333331174e-1f;
    result = result * mantissa * square;
    result = result + -2.12194440e-4f * exponent;
    result = result + -.5f * square;
    return mantissa + result + .693359375f * exponent;
}
__device__ float SinPolynomial(float square, float value) {
    float result = -1.9515295891e-4f * square + 8.3321608736e-3f;
    result = result * square - 1.6666654611e-1f;
    return result * square * value + value;
}
__device__ float CosPolynomial(float square) {
    float result = 2.443315711809948e-5f * square - 1.388731625493765e-3f;
    result = result * square + 4.166664568298827e-2f;
    return result * square * square - .5f * square + 1;
}
__device__ float NoiseSin(float input) {
    float sign = 1, value = input;
    if (value < 0) { value = -value; sign = -1; }
    int quadrant = int(1.27323954473516f * value);
    float rounded = float(quadrant);
    if (quadrant & 1) { ++quadrant; rounded += 1; }
    quadrant &= 7;
    if (quadrant > 3) { sign = -sign; quadrant -= 4; }
    value = ((value - rounded * .78515625f) - rounded * 2.4187564849853515625e-4f) -
            rounded * 3.77489497744594108e-8f;
    const float square = value * value;
    const float result = (quadrant == 1 || quadrant == 2) ? CosPolynomial(square) :
                                                            SinPolynomial(square, value);
    return sign < 0 ? -result : result;
}
__device__ float NoiseCos(float input) {
    float value = input < 0 ? -input : input;
    int quadrant = int(1.27323954473516f * value);
    float rounded = float(quadrant);
    if (quadrant & 1) { ++quadrant; rounded += 1; }
    quadrant &= 7;
    float sign = 1;
    if (quadrant > 3) { quadrant -= 4; sign = -sign; }
    if (quadrant > 1) sign = -sign;
    value = ((value - rounded * .78515625f) - rounded * 2.4187564849853515625e-4f) -
            rounded * 3.77489497744594108e-8f;
    const float square = value * value;
    const float result = (quadrant == 1 || quadrant == 2) ? SinPolynomial(square, value) :
                                                            CosPolynomial(square);
    return sign < 0 ? -result : result;
}
__device__ Axis MakeAxis(float uv, int dimension) {
    const float pixel = uv * dimension - .5f;
    const float base = floorf(pixel);
    const float t = fminf(1.f, fmaxf(0.f, pixel - base));
    const float square = t * t, cube = square * t;
    const float w0 = -.5f * t + square - .5f * cube;
    const float w1 = 1.f - 2.5f * square + 1.5f * cube;
    const float w2 = .5f * t + 2.f * square - 1.5f * cube;
    const float w3 = -.5f * square + .5f * cube;
    const float grouped = w1 + w2;
    Axis result{{base - .5f, base + .5f + w2 / grouped, base + 2.5f},
                {w0, grouped, w3}};
    for (int index = 0; index < 3; ++index)
        result.center[index] = fminf(dimension - .5f, fmaxf(.5f, result.center[index]));
    return result;
}
__device__ float HistoryLinear(const float* history, int channel, float px, float py,
                               int width, int height) {
    px -= .5f; py -= .5f;
    const int x0 = max(0, min(width - 1, int(floorf(px))));
    const int y0 = max(0, min(height - 1, int(floorf(py))));
    const int x1 = min(x0 + 1, width - 1), y1 = min(y0 + 1, height - 1);
    const float tx = fminf(1.f, fmaxf(0.f, px - x0));
    const float ty = fminf(1.f, fmaxf(0.f, py - y0));
    const float top = history[(y0 * width + x0) * 3 + channel] * (1 - tx) +
                      history[(y0 * width + x1) * 3 + channel] * tx;
    const float bottom = history[(y1 * width + x0) * 3 + channel] * (1 - tx) +
                         history[(y1 * width + x1) * 3 + channel] * tx;
    return top * (1 - ty) + bottom * ty;
}
__device__ float SampleHistory(const float* history, float motionX, float motionY,
                               int x, int y, int width, int height, int channel) {
    const float u = (x + .5f) / width + motionX;
    const float v = (y + .5f) / height + motionY;
    const Axis ax = MakeAxis(u, width), ay = MakeAxis(v, height);
    const int taps[5][2] = {{0,1},{1,0},{1,1},{1,2},{2,1}};
    float numerator = 0, denominator = 0;
    for (int index = 0; index < 5; ++index) {
        const int a = taps[index][0], b = taps[index][1];
        const float weight = ax.weight[a] * ay.weight[b];
        numerator += HistoryLinear(history, channel, ax.center[a], ay.center[b],
                                   width, height) * weight;
        denominator += weight;
    }
    return numerator / denominator;
}
__device__ __half Scaled(float value) {
    __half result = __float2half(value);
    result = __hsub(result, __float2half(.5f));
    return __hmul(result, __float2half(.125f));
}

__global__ void PrepareFeatures(const unsigned char* color, std::size_t colorPitch,
                                const unsigned char* motion, std::size_t motionPitch,
                                const unsigned char* mask, std::size_t maskPitch,
                                const float* history, __half* output, int logicalHeight,
                                int logicalWidth, int networkHeight, int networkWidth,
                                int colorWidth, int colorHeight, int maskWidth,
                                int maskHeight, bool hasMask,
                                std::uint32_t noiseIndex, bool temporal,
                                int motionWidth, int motionHeight, int effectiveWidth,
                                int effectiveHeight, float motionScaleX, float motionScaleY,
                                float jitterX, float jitterY,
                                float style, float tone, float local, float skin, float automatic,
                                const NrScheduled::Decision* scheduledDecision) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= networkHeight * networkWidth) return;
    if (scheduledDecision) {
        noiseIndex=scheduledDecision->noise;
        temporal=scheduledDecision->useHistory!=0;
    }
    const int nx = index % networkWidth, ny = index / networkWidth;
    const int x = nx < logicalWidth ? nx : max(0, 2 * logicalWidth - 2 - nx);
    const int y = ny < logicalHeight ? ny : max(0, 2 * logicalHeight - 2 - ny);
    std::uint32_t seed = (std::uint32_t(ny) * 0xd8163841u) ^
                         (std::uint32_t(nx) * 0x8da6b343u) ^
                         (noiseIndex * 0x9e3779b9u) ^ 0x243f6a88u;
    std::uint32_t mixed = Mix(seed); mixed ^= mixed >> 22;
    const std::uint32_t a[4] = {0xcaa5b80du,0x83232c31u,0x2c9277b5u,0xfa6dc5f9u};
    const std::uint32_t b[4] = {0x21dd796bu,0x3463e0acu,0xac564b05u,0x4712a88eu};
    float uniform[4];
    for (int q = 0; q < 4; ++q) {
        const std::uint32_t value = Mix(mixed * a[q] + b[q]);
        uniform[q] = float(((value >> 30) ^ (value >> 8)) + 1) * 5.960464477539063e-8f;
    }
    const float radiusA = sqrtf(-2 * NoiseLog(uniform[0]));
    const float radiusB = sqrtf(-2 * NoiseLog(uniform[2]));
    const float angleA = 6.2831854820251465f * uniform[3];
    const float angleB = 6.2831854820251465f * uniform[1];
    __half* destination = output + std::size_t(index) * 16;
    destination[0] = __float2half(radiusB * NoiseCos(angleA));
    destination[1] = __float2half(radiusB * NoiseSin(angleA));
    destination[2] = __float2half(radiusA * NoiseCos(angleB));
    destination[3] = __float2half(1);
    const int colorX = min(colorWidth - 1, max(0, int((2ll * x + 1) * colorWidth /
                                                       (2ll * logicalWidth))));
    const int colorY = min(colorHeight - 1, max(0, int((2ll * y + 1) * colorHeight /
                                                        (2ll * logicalHeight))));
    const auto* colorRow = reinterpret_cast<const __half*>(color + std::size_t(colorY) * colorPitch);
    float normalizedX = 0, normalizedY = 0;
    if (temporal) {
        const int mx = min(motionWidth - 1, max(0, int((2ll * x + 1) * motionWidth /
                                                        (2ll * logicalWidth))));
        const int my = min(motionHeight - 1, max(0, int((2ll * y + 1) * motionHeight /
                                                         (2ll * logicalHeight))));
        const auto* motionRow = reinterpret_cast<const __half*>(motion + std::size_t(my) * motionPitch);
        normalizedX = __half2float(motionRow[mx * 2]) * motionScaleX / effectiveWidth +
                      jitterX / effectiveWidth;
        normalizedY = __half2float(motionRow[mx * 2 + 1]) * motionScaleY / effectiveHeight +
                      jitterY / effectiveHeight;
    }
    for (int channel = 0; channel < 3; ++channel) {
        const float current = __half2float(colorRow[colorX * 4 + channel]);
        destination[4 + channel] = Scaled(current);
        const float previous = temporal ? SampleHistory(history, normalizedX, normalizedY,
                                                        x, y, logicalWidth, logicalHeight,
                                                        channel) : current;
        destination[7 + channel] = Scaled(previous);
    }
    destination[10] = __float2half(style / 128.f);
    if (hasMask) {
        const int maskX = min(maskWidth - 1, max(0, int((2ll * x + 1) * maskWidth /
                                                        (2ll * logicalWidth))));
        const int maskY = min(maskHeight - 1, max(0, int((2ll * y + 1) * maskHeight /
                                                         (2ll * logicalHeight))));
        const auto* maskRow = reinterpret_cast<const __half*>(mask + std::size_t(maskY) * maskPitch);
        destination[11] = __float2half(__half2float(maskRow[maskX * 4 + 1]) * tone);
        destination[12] = __float2half(__half2float(maskRow[maskX * 4 + 2]) * local);
        destination[13] = __float2half(0);
        destination[14] = __float2half(0);
    } else {
        destination[11] = __float2half(tone);
        destination[12] = __float2half(local);
        destination[13] = __float2half(skin);
        destination[14] = __float2half(automatic);
    }
    destination[15] = __float2half(0);
}

__global__ void Compose(const unsigned char* color, std::size_t colorPitch,
                        const unsigned char* mask, std::size_t maskPitch,
                        const __half* head, const __half* features,
                        const __half* blendScale, unsigned char* output,
                        std::size_t outputPitch, float* history, int logicalHeight,
                        int logicalWidth, int networkWidth, int colorWidth,
                        int colorHeight, int maskWidth, int maskHeight, bool hasMask,
                        bool temporal, float intensity,
                        const NrScheduled::Decision* scheduledDecision) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= logicalHeight * logicalWidth) return;
    if (scheduledDecision) temporal=scheduledDecision->useHistory!=0;
    const int x = index % logicalWidth, y = index / logicalWidth;
    const int colorX = min(colorWidth - 1, max(0, int((2ll * x + 1) * colorWidth /
                                                       (2ll * logicalWidth))));
    const int colorY = min(colorHeight - 1, max(0, int((2ll * y + 1) * colorHeight /
                                                        (2ll * logicalHeight))));
    const auto* colorRow = reinterpret_cast<const __half*>(color + std::size_t(colorY) * colorPitch);
    auto* outputRow = reinterpret_cast<__half*>(output + std::size_t(y) * outputPitch);
    const __half* prediction = head + (std::size_t(y) * networkWidth + x) * 4;
    const __half* feature = features + (std::size_t(y) * networkWidth + x) * 16;
    float alpha = 0;
    if (temporal) {
        const float logit = __half2float(prediction[3]);
        const float exponential = expf(-fabsf(logit));
        const float sigmoid = logit >= 0 ? 1 / (1 + exponential) : exponential / (1 + exponential);
        alpha = fminf(1.f, fmaxf(0.f, sigmoid * __half2float(*blendScale)));
    }
    for (int channel = 0; channel < 3; ++channel) {
        const float base = __half2float(colorRow[colorX * 4 + channel]);
        const float predicted = fminf(1.f, fmaxf(0.f, base + .25f *
                                                __half2float(prediction[channel])));
        const float previous = __half2float(feature[7 + channel]) * 8 + .5f;
        const float temporalValue = temporal ? predicted + alpha * (previous - predicted) : predicted;
        float factor = intensity;
        if (hasMask) {
            const int maskX = min(maskWidth - 1, max(0, int((2ll * x + 1) * maskWidth /
                                                            (2ll * logicalWidth))));
            const int maskY = min(maskHeight - 1, max(0, int((2ll * y + 1) * maskHeight /
                                                             (2ll * logicalHeight))));
            const auto* maskRow = reinterpret_cast<const __half*>(mask + std::size_t(maskY) * maskPitch);
            factor *= __half2float(maskRow[maskX * 4]);
        }
        const float value = temporal && !hasMask && intensity == 1.f ? temporalValue :
            fminf(1.f, fmaxf(0.f, base + fminf(1.f, fmaxf(0.f, factor)) *
                                            (temporalValue - base)));
        history[index * 3 + channel] = value;
        outputRow[x * 4 + channel] = __float2half(value);
    }
    outputRow[x * 4 + 3] = __float2half(1);
}

__global__ void ValidateFiniteRgba16(const unsigned char* output,
                                     std::size_t outputPitch,
                                     unsigned width, unsigned height,
                                     std::uint32_t* invalid) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) return;
    const unsigned x = index % width, y = index / width;
    const auto* row = reinterpret_cast<const std::uint16_t*>(output +
                                                             std::size_t(y) * outputPitch);
    for (unsigned channel = 0; channel < 4; ++channel)
        if ((row[x * 4 + channel] & 0x7c00u) == 0x7c00u) atomicOr(invalid, 1u);
}

__global__ void InjectNonfiniteRgba16(unsigned char* output) {
    if (!blockIdx.x && !threadIdx.x)
        *reinterpret_cast<std::uint16_t*>(output) = 0x7c00u;
}

// Composition uses fmin/fmax, which can hide NaN/Inf from the head. Inspect
// retained stage buffers as well as the final image before publishing history.
__global__ void ValidateNeuralState(const std::uint16_t* features,
                                  const std::uint16_t* head, const float* history,
                                  unsigned networkPixels, unsigned logicalPixels,
                                  std::uint32_t* invalid) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t bits = 0;
    if (index < networkPixels * 16 && (features[index] & 0x7c00u) == 0x7c00u) bits |= 4u;
    if (index < networkPixels * 4 && (head[index] & 0x7c00u) == 0x7c00u) bits |= 2u;
    if (index < logicalPixels * 3 && !isfinite(history[index])) bits |= 8u;
    if (bits) atomicOr(invalid, bits);
}
} // namespace

extern "C" hipError_t launch_v2_prepare_features(
    const void* color, std::size_t colorPitch, const void* motion, std::size_t motionPitch,
    const void* mask, std::size_t maskPitch, const float* history, void* output,
    unsigned logicalHeight, unsigned logicalWidth, unsigned networkHeight,
    unsigned networkWidth, unsigned colorWidth, unsigned colorHeight,
    unsigned maskWidth, unsigned maskHeight, bool hasMask, std::uint32_t noiseIndex, bool temporal,
    unsigned motionWidth, unsigned motionHeight, unsigned effectiveWidth,
    unsigned effectiveHeight, float motionScaleX, float motionScaleY, float jitterX,
    float jitterY, float style, float tone, float local, float skin, float automatic,
    hipStream_t stream) {
    const unsigned count = networkHeight * networkWidth;
    PrepareFeatures<<<(count + 127) / 128, 128, 0, stream>>>(
        static_cast<const unsigned char*>(color), colorPitch,
        static_cast<const unsigned char*>(motion), motionPitch,
        static_cast<const unsigned char*>(mask), maskPitch, history,
        static_cast<__half*>(output), logicalHeight, logicalWidth, networkHeight,
        networkWidth, colorWidth, colorHeight, maskWidth, maskHeight, hasMask,
        noiseIndex, temporal, motionWidth, motionHeight, effectiveWidth,
        effectiveHeight, motionScaleX, motionScaleY, jitterX, jitterY, style, tone,
        local, skin, automatic,nullptr);
    return hipGetLastError();
}

// The scheduled variant reads temporal validity/noise from the preceding GPU
// resolve. It uses the same feature arithmetic as the frozen v2/v3 path.
extern "C" hipError_t launch_scheduled_prepare_features(
    const void* color, std::size_t colorPitch, const void* motion, std::size_t motionPitch,
    const void* mask, std::size_t maskPitch, const float* history, void* output,
    unsigned logicalHeight, unsigned logicalWidth, unsigned networkHeight,
    unsigned networkWidth, unsigned colorWidth, unsigned colorHeight,
    unsigned maskWidth, unsigned maskHeight, bool hasMask,
    const NrScheduled::Decision* decision,
    unsigned motionWidth, unsigned motionHeight, unsigned effectiveWidth,
    unsigned effectiveHeight, float motionScaleX, float motionScaleY, float jitterX,
    float jitterY, float style, float tone, float local, float skin, float automatic,
    hipStream_t stream) {
    const unsigned count=networkHeight*networkWidth;
    PrepareFeatures<<<(count+127)/128,128,0,stream>>>(
        static_cast<const unsigned char*>(color),colorPitch,
        static_cast<const unsigned char*>(motion),motionPitch,
        static_cast<const unsigned char*>(mask),maskPitch,history,
        static_cast<__half*>(output),logicalHeight,logicalWidth,networkHeight,
        networkWidth,colorWidth,colorHeight,maskWidth,maskHeight,hasMask,
        0,false,motionWidth,motionHeight,effectiveWidth,effectiveHeight,
        motionScaleX,motionScaleY,jitterX,jitterY,style,tone,local,skin,automatic,decision);
    return hipGetLastError();
}

extern "C" hipError_t launch_v2_compose(
    const void* color, std::size_t colorPitch, const void* mask, std::size_t maskPitch,
    const void* head, const void* features, const void* blendScale, void* output,
    std::size_t outputPitch, float* history, unsigned logicalHeight,
    unsigned logicalWidth, unsigned networkWidth, unsigned colorWidth,
    unsigned colorHeight, unsigned maskWidth, unsigned maskHeight, bool hasMask,
    bool temporal, float intensity, hipStream_t stream) {
    const unsigned count = logicalHeight * logicalWidth;
    Compose<<<(count + 127) / 128, 128, 0, stream>>>(
        static_cast<const unsigned char*>(color), colorPitch,
        static_cast<const unsigned char*>(mask), maskPitch, static_cast<const __half*>(head),
        static_cast<const __half*>(features),
        static_cast<const __half*>(blendScale), static_cast<unsigned char*>(output),
        outputPitch, history, logicalHeight, logicalWidth, networkWidth,
        colorWidth, colorHeight, maskWidth, maskHeight, hasMask, temporal, intensity,nullptr);
    return hipGetLastError();
}

extern "C" hipError_t launch_scheduled_compose(
    const void* color,std::size_t colorPitch,const void* mask,std::size_t maskPitch,
    const void* head,const void* features,const void* blendScale,void* output,
    std::size_t outputPitch,float* candidate,unsigned logicalHeight,unsigned logicalWidth,
    unsigned networkWidth,unsigned colorWidth,unsigned colorHeight,
    unsigned maskWidth,unsigned maskHeight,bool hasMask,
    const NrScheduled::Decision* decision,float intensity,hipStream_t stream) {
    const unsigned count=logicalHeight*logicalWidth;
    Compose<<<(count+127)/128,128,0,stream>>>(
        static_cast<const unsigned char*>(color),colorPitch,
        static_cast<const unsigned char*>(mask),maskPitch,static_cast<const __half*>(head),
        static_cast<const __half*>(features),static_cast<const __half*>(blendScale),
        static_cast<unsigned char*>(output),outputPitch,candidate,logicalHeight,logicalWidth,
        networkWidth,colorWidth,colorHeight,maskWidth,maskHeight,hasMask,false,intensity,decision);
    return hipGetLastError();
}

extern "C" hipError_t launch_v2_validate_output(
    const void* output, std::size_t outputPitch, unsigned width, unsigned height,
    std::uint32_t* invalid, hipStream_t stream) {
    const unsigned count = width * height;
    ValidateFiniteRgba16<<<(count + 255) / 256, 256, 0, stream>>>(
        static_cast<const unsigned char*>(output), outputPitch, width, height, invalid);
    return hipGetLastError();
}

extern "C" hipError_t launch_v2_inject_nonfinite(void* output, hipStream_t stream) {
    InjectNonfiniteRgba16<<<1, 1, 0, stream>>>(static_cast<unsigned char*>(output));
    return hipGetLastError();
}

extern "C" hipError_t launch_v3_inject_head_nonfinite(void* head, std::size_t bytes,
                                                        hipStream_t stream) {
    return hipMemsetAsync(head, 0x7e, bytes, stream);
}

extern "C" hipError_t launch_v2_validate_neural(
    const void* features, const void* head, const float* history,
    unsigned networkPixels, unsigned logicalPixels, std::uint32_t* invalid,
    hipStream_t stream) {
    ValidateNeuralState<<<(networkPixels * 16 + 255) / 256, 256, 0, stream>>>(
        static_cast<const std::uint16_t*>(features), static_cast<const std::uint16_t*>(head),
        history, networkPixels, logicalPixels, invalid);
    return hipGetLastError();
}
