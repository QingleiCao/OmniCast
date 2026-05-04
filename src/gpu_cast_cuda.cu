#include "omnicast/gpu_cast.hpp"

#if OMNICAST_HAS_CUDA

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if defined(__has_include)
#if __has_include(<cuda_fp8.h>)
#include <cuda_fp8.h>
#define OMNICAST_CUDA_HAS_FP8_TYPES 1
#endif
#endif
#ifndef OMNICAST_CUDA_HAS_FP8_TYPES
#define OMNICAST_CUDA_HAS_FP8_TYPES 0
#endif

// 1: faster staged conversion (double->float->low-precision), 0: stricter clamp-before-stage path
#ifndef OMNICAST_FP64_TO_LOW_FAST_STAGE
#define OMNICAST_FP64_TO_LOW_FAST_STAGE 1
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace omnicast {
namespace {

#define OMNICAST_DEVICE_INLINE __device__ __forceinline__
#define OMNICAST_HOST_DEVICE_INLINE __host__ __device__ __forceinline__

using LaunchFn = GpuStatus (*)(const void*, void*, std::size_t, cudaStream_t);
enum class Fp16Fp8Impl : std::uint8_t;
enum class Fp32Fp8Impl : std::uint8_t;

struct LaunchConfig {
    int block = 256;
    int max_blocks = 4096;
};

struct RuntimeConfig {
    bool available = false;
    int device = -1;
    int major = 0;
    int minor = 0;
    int sm_count = 0;
    bool native_fp8 = false;
    bool fp8_lut_ready = false;
    LaunchConfig generic_launch{};
    LaunchConfig half2_launch{};
    LaunchConfig fp8_launch{};
    LaunchConfig fp8_native_launch{};
    LaunchConfig fp8_lut_launch{};
    // Static fallback before/without autotune: packed path (enum value 2) for both E4/E5 directions.
    std::array<std::uint8_t, 2> auto_fp16_to_fp8_impl{{2U, 2U}};
    std::array<std::uint8_t, 2> auto_fp8_to_fp16_impl{{2U, 2U}};
    std::array<std::uint8_t, 2> auto_fp32_to_fp8_impl{{2U, 2U}};
    std::array<std::uint8_t, 2> auto_fp8_to_fp32_impl{{2U, 2U}};
    bool fp16_fp8_autotuned = false;
    bool fp32_fp8_autotuned = false;
    std::array<std::array<LaunchFn, 5>, 5> dispatch{};
    std::string backend{"cuda-unavailable"};
};

struct DeviceTuning {
    int sm_count = 80;
    int major = 8;
    int minor = 0;
};

enum class ArchFamily : std::uint8_t {
    generic = 0,
    a100 = 1,
    h100 = 2,
    b100 = 3,
};

thread_local const RuntimeConfig* g_active_runtime_cfg = nullptr;

const RuntimeConfig& get_runtime_config();

template <bool DstE5M2>
GpuStatus launch_fp32_to_fp8_with_impl(
    const void* input,
    void* output,
    std::size_t count,
    cudaStream_t stream,
    Fp32Fp8Impl impl
);

template <bool SrcE5M2>
GpuStatus launch_fp8_to_fp32_with_impl(
    const void* input,
    void* output,
    std::size_t count,
    cudaStream_t stream,
    Fp32Fp8Impl impl
);

__constant__ std::uint8_t k_fp8_e4_to_e5_lut[256];
__constant__ std::uint8_t k_fp8_e5_to_e4_lut[256];
__device__ const std::uint16_t* k_fp8_e4_to_e5_lut16 = nullptr;
__device__ const std::uint16_t* k_fp8_e5_to_e4_lut16 = nullptr;
__device__ const std::uint32_t* k_fp8_e4_to_e5_lut32 = nullptr;
__device__ const std::uint32_t* k_fp8_e5_to_e4_lut32 = nullptr;
__device__ const std::uint16_t* k_fp8_e4_to_fp16_lut = nullptr;
__device__ const std::uint16_t* k_fp8_e5_to_fp16_lut = nullptr;
__device__ const std::uint8_t* k_fp16_to_fp8_e4_lut = nullptr;
__device__ const std::uint8_t* k_fp16_to_fp8_e5_lut = nullptr;
__device__ const float* k_fp8_e4_to_fp32_lut = nullptr;
__device__ const float* k_fp8_e5_to_fp32_lut = nullptr;

enum class Fp8LutMode : std::uint8_t {
    auto_mode = 0,
    force_x4 = 1,
    force_x16 = 2,
};

enum class Fp8RemapImpl : std::uint8_t {
    auto_mode = 0,
    scalar = 1,
    lut8_const = 2,
    lut8_shared = 3,
    lut32 = 4,
    lut16 = 5,
    lutx16 = 6,
    copy32 = 7,
    copy64 = 8,
    copy128 = 9,
    lut32_u64 = 10,
    lut32_u128 = 11,
};

enum class Fp16Fp8Impl : std::uint8_t {
    auto_mode = 0,
    native = 1,
    packed = 2,
    packed_wide = 3,
    lut8_to_half = 4,
    lut16_to_fp8 = 5,
    scalar = 6,
    copyshape = 7,
};

enum class Fp32Fp8Impl : std::uint8_t {
    auto_mode = 0,
    native = 1,
    packed = 2,
    packed_wide = 3,
    lut8_to_fp32 = 4,
    scalar = 5,
    copyshape = 6,
};

OMNICAST_HOST_DEVICE_INLINE std::uint8_t fp16_fp8_impl_to_u8(Fp16Fp8Impl impl) {
    return static_cast<std::uint8_t>(impl);
}

inline Fp16Fp8Impl fp16_fp8_impl_from_u8(std::uint8_t v) {
    switch (v) {
        case 1:
            return Fp16Fp8Impl::native;
        case 2:
            return Fp16Fp8Impl::packed;
        case 3:
            return Fp16Fp8Impl::packed_wide;
        case 4:
            return Fp16Fp8Impl::lut8_to_half;
        case 5:
            return Fp16Fp8Impl::lut16_to_fp8;
        case 6:
            return Fp16Fp8Impl::scalar;
        case 7:
            return Fp16Fp8Impl::copyshape;
        case 0:
        default:
            return Fp16Fp8Impl::auto_mode;
    }
}

OMNICAST_HOST_DEVICE_INLINE std::uint8_t fp32_fp8_impl_to_u8(Fp32Fp8Impl impl) {
    return static_cast<std::uint8_t>(impl);
}

inline Fp32Fp8Impl fp32_fp8_impl_from_u8(std::uint8_t v) {
    switch (v) {
        case 1:
            return Fp32Fp8Impl::native;
        case 2:
            return Fp32Fp8Impl::packed;
        case 3:
            return Fp32Fp8Impl::packed_wide;
        case 4:
            return Fp32Fp8Impl::lut8_to_fp32;
        case 5:
            return Fp32Fp8Impl::scalar;
        case 6:
            return Fp32Fp8Impl::copyshape;
        case 0:
        default:
            return Fp32Fp8Impl::auto_mode;
    }
}

#if OMNICAST_CUDA_HAS_FP8_TYPES
using fp8_storage_t = __nv_fp8_storage_t;
using fp8x2_storage_t = __nv_fp8x2_storage_t;
using fp8x4_storage_t = __nv_fp8x4_storage_t;
static_assert(sizeof(fp8_storage_t) == 1, "Unexpected __nv_fp8_storage_t size");
static_assert(sizeof(fp8x2_storage_t) == 2, "Unexpected __nv_fp8x2_storage_t size");
static_assert(sizeof(fp8x4_storage_t) == 4, "Unexpected __nv_fp8x4_storage_t size");
#endif

OMNICAST_DEVICE_INLINE float decode_fp8_e4m3(std::uint8_t x) {
    const std::int8_t sx = static_cast<std::int8_t>(x);
    return static_cast<float>(sx) * (1.0f / 16.0f);
}

OMNICAST_DEVICE_INLINE float decode_fp8_e5m2(std::uint8_t x) {
    const std::int8_t sx = static_cast<std::int8_t>(x);
    return static_cast<float>(sx) * (1.0f / 8.0f);
}

OMNICAST_DEVICE_INLINE std::uint8_t encode_fp8_e4m3(float value) {
    float clamped = fminf(fmaxf(value, -8.0f), 7.9375f);
    int q = __float2int_rn(clamped * 16.0f);
    q = (q < -128) ? -128 : q;
    q = (q > 127) ? 127 : q;
    return static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
}

OMNICAST_DEVICE_INLINE std::uint8_t encode_fp8_e5m2(float value) {
    float clamped = fminf(fmaxf(value, -16.0f), 15.875f);
    int q = __float2int_rn(clamped * 8.0f);
    q = (q < -128) ? -128 : q;
    q = (q > 127) ? 127 : q;
    return static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
}

inline float decode_fp8_e4m3_host(std::uint8_t x) {
    return static_cast<float>(static_cast<std::int8_t>(x)) * (1.0f / 16.0f);
}

inline float decode_fp8_e5m2_host(std::uint8_t x) {
    return static_cast<float>(static_cast<std::int8_t>(x)) * (1.0f / 8.0f);
}

inline std::uint8_t encode_fp8_e4m3_host(float value) {
    float clamped = std::fmin(std::fmax(value, -8.0f), 7.9375f);
    int q = static_cast<int>(std::lrint(clamped * 16.0f));
    q = (q < -128) ? -128 : q;
    q = (q > 127) ? 127 : q;
    return static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
}

inline std::uint8_t encode_fp8_e5m2_host(float value) {
    float clamped = std::fmin(std::fmax(value, -16.0f), 15.875f);
    int q = static_cast<int>(std::lrint(clamped * 8.0f));
    q = (q < -128) ? -128 : q;
    q = (q > 127) ? 127 : q;
    return static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
}

OMNICAST_DEVICE_INLINE half2 half2_from_u32_bits(std::uint32_t bits) {
    __half_raw lo{};
    __half_raw hi{};
    lo.x = static_cast<std::uint16_t>(bits & 0xFFFFU);
    hi.x = static_cast<std::uint16_t>((bits >> 16U) & 0xFFFFU);
    return __halves2half2(static_cast<__half>(lo), static_cast<__half>(hi));
}

OMNICAST_DEVICE_INLINE std::uint32_t half2_to_u32_bits(half2 value) {
    const __half lo = __low2half(value);
    const __half hi = __high2half(value);
    const __half_raw lo_raw = static_cast<__half_raw>(lo);
    const __half_raw hi_raw = static_cast<__half_raw>(hi);
    return static_cast<std::uint32_t>(lo_raw.x) | (static_cast<std::uint32_t>(hi_raw.x) << 16U);
}

OMNICAST_DEVICE_INLINE float fp64_stage_to_float(double v) {
#if OMNICAST_FP64_TO_LOW_FAST_STAGE
    return static_cast<float>(v);
#else
    constexpr double kHi = 65504.0;
    constexpr double kLo = -65504.0;
    const double clamped = (v > kHi) ? kHi : ((v < kLo) ? kLo : v);
    return static_cast<float>(clamped);
#endif
}

OMNICAST_DEVICE_INLINE float fp64_stage_to_fp8_e4_input(double v) {
#if OMNICAST_FP64_TO_LOW_FAST_STAGE
    return static_cast<float>(v);
#else
    constexpr double kHi = 7.9375;
    constexpr double kLo = -8.0;
    const double clamped = (v > kHi) ? kHi : ((v < kLo) ? kLo : v);
    return static_cast<float>(clamped);
#endif
}

OMNICAST_DEVICE_INLINE float fp64_stage_to_fp8_e5_input(double v) {
#if OMNICAST_FP64_TO_LOW_FAST_STAGE
    return static_cast<float>(v);
#else
    constexpr double kHi = 15.875;
    constexpr double kLo = -16.0;
    const double clamped = (v > kHi) ? kHi : ((v < kLo) ? kLo : v);
    return static_cast<float>(clamped);
#endif
}

template <typename T>
OMNICAST_DEVICE_INLINE float decode_value(T value);

template <>
OMNICAST_DEVICE_INLINE float decode_value<double>(double value) {
    return fp64_stage_to_float(value);
}

template <>
OMNICAST_DEVICE_INLINE float decode_value<float>(float value) {
    return value;
}

template <>
OMNICAST_DEVICE_INLINE float decode_value<std::uint16_t>(std::uint16_t value) {
    __half_raw raw;
    raw.x = value;
    return __half2float(static_cast<__half>(raw));
}

template <typename T>
OMNICAST_DEVICE_INLINE T encode_value(float value);

template <>
OMNICAST_DEVICE_INLINE double encode_value<double>(float value) {
    return static_cast<double>(value);
}

template <>
OMNICAST_DEVICE_INLINE float encode_value<float>(float value) {
    return value;
}

template <>
OMNICAST_DEVICE_INLINE std::uint16_t encode_value<std::uint16_t>(float value) {
    const __half half_value = __float2half_rn(value);
    const __half_raw raw = static_cast<__half_raw>(half_value);
    return raw.x;
}

template <typename SrcT, typename DstT, bool SrcE5M2, bool DstE5M2>
__global__ void cast_kernel(const SrcT* input, DstT* output, std::size_t count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);

    for (; idx < count; idx += stride * 4) {
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const std::size_t i = idx + static_cast<std::size_t>(u) * stride;
            if (i >= count) {
                break;
            }

            float x = 0.0f;
            if constexpr (std::is_same<SrcT, std::uint8_t>::value) {
                x = SrcE5M2 ? decode_fp8_e5m2(input[i]) : decode_fp8_e4m3(input[i]);
            } else {
                x = decode_value<SrcT>(input[i]);
            }

            if constexpr (std::is_same<DstT, std::uint8_t>::value) {
                output[i] = DstE5M2 ? encode_fp8_e5m2(x) : encode_fp8_e4m3(x);
            } else {
                output[i] = encode_value<DstT>(x);
            }
        }
    }
}

__global__ void fp32_to_fp16_half2_kernel(const float2* input, half2* output, std::size_t pair_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < pair_count; idx += stride) {
        output[idx] = __float22half2_rn(input[idx]);
    }
}

__global__ void fp16_to_fp32_half2_kernel(const half2* input, float2* output, std::size_t pair_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < pair_count; idx += stride) {
        output[idx] = __half22float2(input[idx]);
    }
}

__global__ void fp64_to_fp32x2_kernel(const double2* input, float2* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const double2 v = input[idx];
        output[idx] = make_float2(static_cast<float>(v.x), static_cast<float>(v.y));
    }
}

__global__ void fp32_to_fp64x2_kernel(const float2* input, double2* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const float2 v = input[idx];
        output[idx] = make_double2(static_cast<double>(v.x), static_cast<double>(v.y));
    }
}

__global__ void fp64_to_fp16x2_kernel(const double2* input, half2* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const double2 v = input[idx];
        const float2 f = make_float2(fp64_stage_to_float(v.x), fp64_stage_to_float(v.y));
        output[idx] = __float22half2_rn(f);
    }
}

__global__ void fp16_to_fp64x2_kernel(const half2* input, double2* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const float2 f = __half22float2(input[idx]);
        output[idx] = make_double2(static_cast<double>(f.x), static_cast<double>(f.y));
    }
}

template <bool E5M2>
__global__ void fp16_to_fp8x2_kernel(const half2* input, std::uint16_t* output, std::size_t pair_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < pair_count; idx += stride) {
        const float2 f = __half22float2(input[idx]);
        const std::uint8_t b0 = E5M2 ? encode_fp8_e5m2(f.x) : encode_fp8_e4m3(f.x);
        const std::uint8_t b1 = E5M2 ? encode_fp8_e5m2(f.y) : encode_fp8_e4m3(f.y);
        output[idx] = static_cast<std::uint16_t>(b0) | (static_cast<std::uint16_t>(b1) << 8U);
    }
}

template <bool E5M2>
__global__ void fp8_to_fp16x2_kernel(const std::uint16_t* input, half2* output, std::size_t pair_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < pair_count; idx += stride) {
        const std::uint16_t packed = input[idx];
        const std::uint8_t b0 = static_cast<std::uint8_t>(packed & 0xFFU);
        const std::uint8_t b1 = static_cast<std::uint8_t>((packed >> 8U) & 0xFFU);
        const float2 f = make_float2(
            E5M2 ? decode_fp8_e5m2(b0) : decode_fp8_e4m3(b0),
            E5M2 ? decode_fp8_e5m2(b1) : decode_fp8_e4m3(b1)
        );
        output[idx] = __float22half2_rn(f);
    }
}

template <bool E5M2>
__global__ void fp16_to_fp8x4_kernel(const half2* input, std::uint32_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const half2 h0 = input[idx * 2];
        const half2 h1 = input[idx * 2 + 1];
        const float2 f0 = __half22float2(h0);
        const float2 f1 = __half22float2(h1);
        const std::uint8_t b0 = E5M2 ? encode_fp8_e5m2(f0.x) : encode_fp8_e4m3(f0.x);
        const std::uint8_t b1 = E5M2 ? encode_fp8_e5m2(f0.y) : encode_fp8_e4m3(f0.y);
        const std::uint8_t b2 = E5M2 ? encode_fp8_e5m2(f1.x) : encode_fp8_e4m3(f1.x);
        const std::uint8_t b3 = E5M2 ? encode_fp8_e5m2(f1.y) : encode_fp8_e4m3(f1.y);
        output[idx] = static_cast<std::uint32_t>(b0) | (static_cast<std::uint32_t>(b1) << 8U) |
                      (static_cast<std::uint32_t>(b2) << 16U) | (static_cast<std::uint32_t>(b3) << 24U);
    }
}

template <bool E5M2>
__global__ void fp8_to_fp16x4_kernel(const std::uint32_t* input, half2* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t packed = input[idx];
        const std::uint8_t b0 = static_cast<std::uint8_t>(packed & 0xFFU);
        const std::uint8_t b1 = static_cast<std::uint8_t>((packed >> 8U) & 0xFFU);
        const std::uint8_t b2 = static_cast<std::uint8_t>((packed >> 16U) & 0xFFU);
        const std::uint8_t b3 = static_cast<std::uint8_t>((packed >> 24U) & 0xFFU);
        const float2 f0 = make_float2(
            E5M2 ? decode_fp8_e5m2(b0) : decode_fp8_e4m3(b0),
            E5M2 ? decode_fp8_e5m2(b1) : decode_fp8_e4m3(b1)
        );
        const float2 f1 = make_float2(
            E5M2 ? decode_fp8_e5m2(b2) : decode_fp8_e4m3(b2),
            E5M2 ? decode_fp8_e5m2(b3) : decode_fp8_e4m3(b3)
        );
        output[idx * 2] = __float22half2_rn(f0);
        output[idx * 2 + 1] = __float22half2_rn(f1);
    }
}

template <bool E5M2>
__global__ void fp16_to_fp8x8_kernel(const uint4* input, std::uint64_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const uint4 in = input[idx];
        const half2 h0 = half2_from_u32_bits(in.x);
        const half2 h1 = half2_from_u32_bits(in.y);
        const half2 h2 = half2_from_u32_bits(in.z);
        const half2 h3 = half2_from_u32_bits(in.w);
        const float2 f0 = __half22float2(h0);
        const float2 f1 = __half22float2(h1);
        const float2 f2 = __half22float2(h2);
        const float2 f3 = __half22float2(h3);
        const std::uint8_t b0 = E5M2 ? encode_fp8_e5m2(f0.x) : encode_fp8_e4m3(f0.x);
        const std::uint8_t b1 = E5M2 ? encode_fp8_e5m2(f0.y) : encode_fp8_e4m3(f0.y);
        const std::uint8_t b2 = E5M2 ? encode_fp8_e5m2(f1.x) : encode_fp8_e4m3(f1.x);
        const std::uint8_t b3 = E5M2 ? encode_fp8_e5m2(f1.y) : encode_fp8_e4m3(f1.y);
        const std::uint8_t b4 = E5M2 ? encode_fp8_e5m2(f2.x) : encode_fp8_e4m3(f2.x);
        const std::uint8_t b5 = E5M2 ? encode_fp8_e5m2(f2.y) : encode_fp8_e4m3(f2.y);
        const std::uint8_t b6 = E5M2 ? encode_fp8_e5m2(f3.x) : encode_fp8_e4m3(f3.x);
        const std::uint8_t b7 = E5M2 ? encode_fp8_e5m2(f3.y) : encode_fp8_e4m3(f3.y);
        output[idx] = static_cast<std::uint64_t>(b0) | (static_cast<std::uint64_t>(b1) << 8U) |
                      (static_cast<std::uint64_t>(b2) << 16U) | (static_cast<std::uint64_t>(b3) << 24U) |
                      (static_cast<std::uint64_t>(b4) << 32U) | (static_cast<std::uint64_t>(b5) << 40U) |
                      (static_cast<std::uint64_t>(b6) << 48U) | (static_cast<std::uint64_t>(b7) << 56U);
    }
}

template <bool E5M2>
__global__ void fp8_to_fp16x8_kernel(const std::uint64_t* input, uint4* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint64_t packed = input[idx];
        const std::uint8_t b0 = static_cast<std::uint8_t>((packed >> 0U) & 0xFFU);
        const std::uint8_t b1 = static_cast<std::uint8_t>((packed >> 8U) & 0xFFU);
        const std::uint8_t b2 = static_cast<std::uint8_t>((packed >> 16U) & 0xFFU);
        const std::uint8_t b3 = static_cast<std::uint8_t>((packed >> 24U) & 0xFFU);
        const std::uint8_t b4 = static_cast<std::uint8_t>((packed >> 32U) & 0xFFU);
        const std::uint8_t b5 = static_cast<std::uint8_t>((packed >> 40U) & 0xFFU);
        const std::uint8_t b6 = static_cast<std::uint8_t>((packed >> 48U) & 0xFFU);
        const std::uint8_t b7 = static_cast<std::uint8_t>((packed >> 56U) & 0xFFU);
        const half2 h0 = __float22half2_rn(make_float2(
            E5M2 ? decode_fp8_e5m2(b0) : decode_fp8_e4m3(b0),
            E5M2 ? decode_fp8_e5m2(b1) : decode_fp8_e4m3(b1)
        ));
        const half2 h1 = __float22half2_rn(make_float2(
            E5M2 ? decode_fp8_e5m2(b2) : decode_fp8_e4m3(b2),
            E5M2 ? decode_fp8_e5m2(b3) : decode_fp8_e4m3(b3)
        ));
        const half2 h2 = __float22half2_rn(make_float2(
            E5M2 ? decode_fp8_e5m2(b4) : decode_fp8_e4m3(b4),
            E5M2 ? decode_fp8_e5m2(b5) : decode_fp8_e4m3(b5)
        ));
        const half2 h3 = __float22half2_rn(make_float2(
            E5M2 ? decode_fp8_e5m2(b6) : decode_fp8_e4m3(b6),
            E5M2 ? decode_fp8_e5m2(b7) : decode_fp8_e4m3(b7)
        ));
        uint4 out{};
        out.x = half2_to_u32_bits(h0);
        out.y = half2_to_u32_bits(h1);
        out.z = half2_to_u32_bits(h2);
        out.w = half2_to_u32_bits(h3);
        output[idx] = out;
    }
}

template <bool SrcE5M2>
__global__ void fp8_to_fp16_lutx4_kernel(const std::uint32_t* input, std::uint64_t* output, std::size_t vec_count) {
    const std::uint16_t* lut = SrcE5M2 ? k_fp8_e5_to_fp16_lut : k_fp8_e4_to_fp16_lut;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t packed = input[idx];
        const std::uint16_t h0 = __ldg(lut + ((packed >> 0U) & 0xFFU));
        const std::uint16_t h1 = __ldg(lut + ((packed >> 8U) & 0xFFU));
        const std::uint16_t h2 = __ldg(lut + ((packed >> 16U) & 0xFFU));
        const std::uint16_t h3 = __ldg(lut + ((packed >> 24U) & 0xFFU));
        output[idx] = static_cast<std::uint64_t>(h0) | (static_cast<std::uint64_t>(h1) << 16U) |
                      (static_cast<std::uint64_t>(h2) << 32U) | (static_cast<std::uint64_t>(h3) << 48U);
    }
}

template <bool DstE5M2>
__global__ void fp16_to_fp8_lutx4_kernel(const std::uint64_t* input, std::uint32_t* output, std::size_t vec_count) {
    const std::uint8_t* lut = DstE5M2 ? k_fp16_to_fp8_e5_lut : k_fp16_to_fp8_e4_lut;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint64_t packed = input[idx];
        const std::uint16_t h0 = static_cast<std::uint16_t>((packed >> 0U) & 0xFFFFU);
        const std::uint16_t h1 = static_cast<std::uint16_t>((packed >> 16U) & 0xFFFFU);
        const std::uint16_t h2 = static_cast<std::uint16_t>((packed >> 32U) & 0xFFFFU);
        const std::uint16_t h3 = static_cast<std::uint16_t>((packed >> 48U) & 0xFFFFU);
        const std::uint8_t b0 = __ldg(lut + h0);
        const std::uint8_t b1 = __ldg(lut + h1);
        const std::uint8_t b2 = __ldg(lut + h2);
        const std::uint8_t b3 = __ldg(lut + h3);
        output[idx] = static_cast<std::uint32_t>(b0) | (static_cast<std::uint32_t>(b1) << 8U) |
                      (static_cast<std::uint32_t>(b2) << 16U) | (static_cast<std::uint32_t>(b3) << 24U);
    }
}

template <bool E5M2>
__global__ void fp64_to_fp8x2_kernel(const double2* input, std::uint16_t* output, std::size_t pair_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < pair_count; idx += stride) {
        const double2 v = input[idx];
        const float x0 = E5M2 ? fp64_stage_to_fp8_e5_input(v.x) : fp64_stage_to_fp8_e4_input(v.x);
        const float x1 = E5M2 ? fp64_stage_to_fp8_e5_input(v.y) : fp64_stage_to_fp8_e4_input(v.y);
        const std::uint8_t b0 = E5M2 ? encode_fp8_e5m2(x0) : encode_fp8_e4m3(x0);
        const std::uint8_t b1 = E5M2 ? encode_fp8_e5m2(x1) : encode_fp8_e4m3(x1);
        output[idx] = static_cast<std::uint16_t>(b0) | (static_cast<std::uint16_t>(b1) << 8U);
    }
}

template <bool E5M2>
__global__ void fp8_to_fp64x2_kernel(const std::uint16_t* input, double2* output, std::size_t pair_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < pair_count; idx += stride) {
        const std::uint16_t packed = input[idx];
        const std::uint8_t b0 = static_cast<std::uint8_t>(packed & 0xFFU);
        const std::uint8_t b1 = static_cast<std::uint8_t>((packed >> 8U) & 0xFFU);
        output[idx] = make_double2(
            static_cast<double>(E5M2 ? decode_fp8_e5m2(b0) : decode_fp8_e4m3(b0)),
            static_cast<double>(E5M2 ? decode_fp8_e5m2(b1) : decode_fp8_e4m3(b1))
        );
    }
}

template <bool E5ToE4>
__global__ void fp8_lut_transform_x4_kernel(const std::uint32_t* input, std::uint32_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t packed = input[idx];
        const std::uint8_t b0 = static_cast<std::uint8_t>(packed & 0xFFU);
        const std::uint8_t b1 = static_cast<std::uint8_t>((packed >> 8U) & 0xFFU);
        const std::uint8_t b2 = static_cast<std::uint8_t>((packed >> 16U) & 0xFFU);
        const std::uint8_t b3 = static_cast<std::uint8_t>((packed >> 24U) & 0xFFU);
        const std::uint8_t o0 = E5ToE4 ? k_fp8_e5_to_e4_lut[b0] : k_fp8_e4_to_e5_lut[b0];
        const std::uint8_t o1 = E5ToE4 ? k_fp8_e5_to_e4_lut[b1] : k_fp8_e4_to_e5_lut[b1];
        const std::uint8_t o2 = E5ToE4 ? k_fp8_e5_to_e4_lut[b2] : k_fp8_e4_to_e5_lut[b2];
        const std::uint8_t o3 = E5ToE4 ? k_fp8_e5_to_e4_lut[b3] : k_fp8_e4_to_e5_lut[b3];
        output[idx] = static_cast<std::uint32_t>(o0) | (static_cast<std::uint32_t>(o1) << 8U) |
                      (static_cast<std::uint32_t>(o2) << 16U) | (static_cast<std::uint32_t>(o3) << 24U);
    }
}

template <bool E5ToE4>
__global__ void fp8_lut_transform_shared_x4_kernel(const std::uint32_t* input, std::uint32_t* output, std::size_t vec_count) {
    __shared__ std::uint8_t s_lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) {
        s_lut[i] = E5ToE4 ? k_fp8_e5_to_e4_lut[i] : k_fp8_e4_to_e5_lut[i];
    }
    __syncthreads();

    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t packed = input[idx];
        const std::uint8_t b0 = static_cast<std::uint8_t>(packed & 0xFFU);
        const std::uint8_t b1 = static_cast<std::uint8_t>((packed >> 8U) & 0xFFU);
        const std::uint8_t b2 = static_cast<std::uint8_t>((packed >> 16U) & 0xFFU);
        const std::uint8_t b3 = static_cast<std::uint8_t>((packed >> 24U) & 0xFFU);
        const std::uint8_t o0 = s_lut[b0];
        const std::uint8_t o1 = s_lut[b1];
        const std::uint8_t o2 = s_lut[b2];
        const std::uint8_t o3 = s_lut[b3];
        output[idx] = static_cast<std::uint32_t>(o0) | (static_cast<std::uint32_t>(o1) << 8U) |
                      (static_cast<std::uint32_t>(o2) << 16U) | (static_cast<std::uint32_t>(o3) << 24U);
    }
}

template <bool E5ToE4>
__device__ inline std::uint32_t fp8_lut_transform_u32(std::uint32_t packed) {
    const std::uint8_t b0 = static_cast<std::uint8_t>(packed & 0xFFU);
    const std::uint8_t b1 = static_cast<std::uint8_t>((packed >> 8U) & 0xFFU);
    const std::uint8_t b2 = static_cast<std::uint8_t>((packed >> 16U) & 0xFFU);
    const std::uint8_t b3 = static_cast<std::uint8_t>((packed >> 24U) & 0xFFU);
    const std::uint8_t o0 = E5ToE4 ? k_fp8_e5_to_e4_lut[b0] : k_fp8_e4_to_e5_lut[b0];
    const std::uint8_t o1 = E5ToE4 ? k_fp8_e5_to_e4_lut[b1] : k_fp8_e4_to_e5_lut[b1];
    const std::uint8_t o2 = E5ToE4 ? k_fp8_e5_to_e4_lut[b2] : k_fp8_e4_to_e5_lut[b2];
    const std::uint8_t o3 = E5ToE4 ? k_fp8_e5_to_e4_lut[b3] : k_fp8_e4_to_e5_lut[b3];
    return static_cast<std::uint32_t>(o0) | (static_cast<std::uint32_t>(o1) << 8U) |
           (static_cast<std::uint32_t>(o2) << 16U) | (static_cast<std::uint32_t>(o3) << 24U);
}

template <bool E5ToE4>
__global__ void fp8_lut_transform_x16_kernel(const uint4* input, uint4* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const uint4 p = input[idx];
        uint4 o{};
        o.x = fp8_lut_transform_u32<E5ToE4>(p.x);
        o.y = fp8_lut_transform_u32<E5ToE4>(p.y);
        o.z = fp8_lut_transform_u32<E5ToE4>(p.z);
        o.w = fp8_lut_transform_u32<E5ToE4>(p.w);
        output[idx] = o;
    }
}

template <bool E5ToE4>
__global__ void fp8_lut16_transform_x4_kernel(const std::uint32_t* input, std::uint32_t* output, std::size_t vec_count) {
    const std::uint16_t* lut = E5ToE4 ? k_fp8_e5_to_e4_lut16 : k_fp8_e4_to_e5_lut16;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t packed = input[idx];
        const std::uint16_t lo = static_cast<std::uint16_t>(packed & 0xFFFFU);
        const std::uint16_t hi = static_cast<std::uint16_t>((packed >> 16U) & 0xFFFFU);
        const std::uint16_t out_lo = __ldg(lut + lo);
        const std::uint16_t out_hi = __ldg(lut + hi);
        output[idx] = static_cast<std::uint32_t>(out_lo) | (static_cast<std::uint32_t>(out_hi) << 16U);
    }
}

template <bool E5ToE4>
__global__ void fp8_lut32_transform_x4_kernel(const std::uint32_t* input, std::uint32_t* output, std::size_t vec_count) {
    const std::uint32_t* lut = E5ToE4 ? k_fp8_e5_to_e4_lut32 : k_fp8_e4_to_e5_lut32;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t packed = input[idx];
        const std::uint32_t y0 = __ldg(lut + ((packed >> 0U) & 0xFFU));
        const std::uint32_t y1 = __ldg(lut + ((packed >> 8U) & 0xFFU));
        const std::uint32_t y2 = __ldg(lut + ((packed >> 16U) & 0xFFU));
        const std::uint32_t y3 = __ldg(lut + ((packed >> 24U) & 0xFFU));
        output[idx] = y0 | (y1 << 8U) | (y2 << 16U) | (y3 << 24U);
    }
}

template <bool E5ToE4>
OMNICAST_DEVICE_INLINE std::uint32_t fp8_lut32_transform_word(std::uint32_t packed, const std::uint32_t* lut) {
    const std::uint32_t y0 = __ldg(lut + ((packed >> 0U) & 0xFFU));
    const std::uint32_t y1 = __ldg(lut + ((packed >> 8U) & 0xFFU));
    const std::uint32_t y2 = __ldg(lut + ((packed >> 16U) & 0xFFU));
    const std::uint32_t y3 = __ldg(lut + ((packed >> 24U) & 0xFFU));
    return y0 | (y1 << 8U) | (y2 << 16U) | (y3 << 24U);
}

template <bool E5ToE4>
__global__ void fp8_lut32_transform_x8_kernel(const std::uint64_t* input, std::uint64_t* output, std::size_t vec_count) {
    const std::uint32_t* lut = E5ToE4 ? k_fp8_e5_to_e4_lut32 : k_fp8_e4_to_e5_lut32;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint64_t in = input[idx];
        const std::uint32_t lo = static_cast<std::uint32_t>(in & 0xFFFFFFFFULL);
        const std::uint32_t hi = static_cast<std::uint32_t>((in >> 32U) & 0xFFFFFFFFULL);
        const std::uint64_t out =
            static_cast<std::uint64_t>(fp8_lut32_transform_word<E5ToE4>(lo, lut)) |
            (static_cast<std::uint64_t>(fp8_lut32_transform_word<E5ToE4>(hi, lut)) << 32U);
        output[idx] = out;
    }
}

template <bool E5ToE4>
__global__ void fp8_lut32_transform_x16_kernel(const uint4* input, uint4* output, std::size_t vec_count) {
    const std::uint32_t* lut = E5ToE4 ? k_fp8_e5_to_e4_lut32 : k_fp8_e4_to_e5_lut32;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const uint4 in = input[idx];
        uint4 out{};
        out.x = fp8_lut32_transform_word<E5ToE4>(in.x, lut);
        out.y = fp8_lut32_transform_word<E5ToE4>(in.y, lut);
        out.z = fp8_lut32_transform_word<E5ToE4>(in.z, lut);
        out.w = fp8_lut32_transform_word<E5ToE4>(in.w, lut);
        output[idx] = out;
    }
}

__global__ void fp8_copy_u32_kernel(const std::uint32_t* input, std::uint32_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        output[idx] = input[idx];
    }
}

__global__ void fp8_copy_u64_kernel(const std::uint64_t* input, std::uint64_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        output[idx] = input[idx];
    }
}

__global__ void fp8_copy_u128_kernel(const uint4* input, uint4* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        output[idx] = input[idx];
    }
}

__global__ void fp16_to_fp8_copyshape_x4_kernel(const std::uint64_t* input, std::uint32_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint64_t in = input[idx];
        output[idx] = static_cast<std::uint32_t>((in >> 0U) & 0xFFU) |
                      (static_cast<std::uint32_t>((in >> 16U) & 0xFFU) << 8U) |
                      (static_cast<std::uint32_t>((in >> 32U) & 0xFFU) << 16U) |
                      (static_cast<std::uint32_t>((in >> 48U) & 0xFFU) << 24U);
    }
}

__global__ void fp8_to_fp16_copyshape_x4_kernel(const std::uint32_t* input, std::uint64_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t in = input[idx];
        output[idx] = static_cast<std::uint64_t>((in >> 0U) & 0xFFU) |
                      (static_cast<std::uint64_t>((in >> 8U) & 0xFFU) << 16U) |
                      (static_cast<std::uint64_t>((in >> 16U) & 0xFFU) << 32U) |
                      (static_cast<std::uint64_t>((in >> 24U) & 0xFFU) << 48U);
    }
}

bool ensure_fp8_luts_initialized() {
    static std::mutex mu;
    static bool initialized = false;
    static std::uint16_t* e4_to_e5_lut16_dev = nullptr;
    static std::uint16_t* e5_to_e4_lut16_dev = nullptr;
    static std::uint32_t* e4_to_e5_lut32_dev = nullptr;
    static std::uint32_t* e5_to_e4_lut32_dev = nullptr;
    static std::uint16_t* e4_to_fp16_dev = nullptr;
    static std::uint16_t* e5_to_fp16_dev = nullptr;
    static std::uint8_t* fp16_to_e4_dev = nullptr;
    static std::uint8_t* fp16_to_e5_dev = nullptr;
    static float* e4_to_fp32_dev = nullptr;
    static float* e5_to_fp32_dev = nullptr;
    std::lock_guard<std::mutex> lock(mu);
    if (initialized) {
        return true;
    }

    std::array<std::uint8_t, 256> e4_to_e5{};
    std::array<std::uint8_t, 256> e5_to_e4{};
    std::array<std::uint16_t, 256> e4_to_fp16{};
    std::array<std::uint16_t, 256> e5_to_fp16{};
    std::array<float, 256> e4_to_fp32{};
    std::array<float, 256> e5_to_fp32{};
    std::array<std::uint32_t, 256> e4_to_e5_32{};
    std::array<std::uint32_t, 256> e5_to_e4_32{};
    std::vector<std::uint16_t> e4_to_e5_16(1U << 16U);
    std::vector<std::uint16_t> e5_to_e4_16(1U << 16U);
    std::vector<std::uint8_t> fp16_to_e4(1U << 16U);
    std::vector<std::uint8_t> fp16_to_e5(1U << 16U);
    for (int i = 0; i < 256; ++i) {
        const std::uint8_t x = static_cast<std::uint8_t>(i);
        e4_to_e5[i] = encode_fp8_e5m2_host(decode_fp8_e4m3_host(x));
        e5_to_e4[i] = encode_fp8_e4m3_host(decode_fp8_e5m2_host(x));
        const float e4 = decode_fp8_e4m3_host(x);
        const float e5 = decode_fp8_e5m2_host(x);
        e4_to_fp32[i] = e4;
        e5_to_fp32[i] = e5;
        const __half_raw e4_raw = static_cast<__half_raw>(__float2half_rn(e4));
        const __half_raw e5_raw = static_cast<__half_raw>(__float2half_rn(e5));
        e4_to_fp16[i] = e4_raw.x;
        e5_to_fp16[i] = e5_raw.x;
        e4_to_e5_32[i] = static_cast<std::uint32_t>(e4_to_e5[i]);
        e5_to_e4_32[i] = static_cast<std::uint32_t>(e5_to_e4[i]);
    }
    for (int i = 0; i < (1 << 16); ++i) {
        const std::uint16_t in = static_cast<std::uint16_t>(i);
        const std::uint8_t b0 = static_cast<std::uint8_t>(in & 0xFFU);
        const std::uint8_t b1 = static_cast<std::uint8_t>((in >> 8U) & 0xFFU);
        const std::uint16_t out_e4_to_e5 =
            static_cast<std::uint16_t>(e4_to_e5[static_cast<std::size_t>(b0)]) |
            (static_cast<std::uint16_t>(e4_to_e5[static_cast<std::size_t>(b1)]) << 8U);
        const std::uint16_t out_e5_to_e4 =
            static_cast<std::uint16_t>(e5_to_e4[static_cast<std::size_t>(b0)]) |
            (static_cast<std::uint16_t>(e5_to_e4[static_cast<std::size_t>(b1)]) << 8U);
        e4_to_e5_16[static_cast<std::size_t>(i)] = out_e4_to_e5;
        e5_to_e4_16[static_cast<std::size_t>(i)] = out_e5_to_e4;
        __half_raw in_half{};
        in_half.x = in;
        const float in_f = __half2float(static_cast<__half>(in_half));
        fp16_to_e4[static_cast<std::size_t>(i)] = encode_fp8_e4m3_host(in_f);
        fp16_to_e5[static_cast<std::size_t>(i)] = encode_fp8_e5m2_host(in_f);
    }

    if (cudaMemcpyToSymbol(k_fp8_e4_to_e5_lut, e4_to_e5.data(), e4_to_e5.size()) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp8_e5_to_e4_lut, e5_to_e4.data(), e5_to_e4.size()) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&e4_to_fp16_dev, e4_to_fp16.size() * sizeof(std::uint16_t)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&e5_to_fp16_dev, e5_to_fp16.size() * sizeof(std::uint16_t)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(e4_to_fp16_dev, e4_to_fp16.data(), e4_to_fp16.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(e5_to_fp16_dev, e5_to_fp16.data(), e5_to_fp16.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp8_e4_to_fp16_lut, &e4_to_fp16_dev, sizeof(e4_to_fp16_dev)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp8_e5_to_fp16_lut, &e5_to_fp16_dev, sizeof(e5_to_fp16_dev)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&e4_to_fp32_dev, e4_to_fp32.size() * sizeof(float)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&e5_to_fp32_dev, e5_to_fp32.size() * sizeof(float)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(e4_to_fp32_dev, e4_to_fp32.data(), e4_to_fp32.size() * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(e5_to_fp32_dev, e5_to_fp32.data(), e5_to_fp32.size() * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp8_e4_to_fp32_lut, &e4_to_fp32_dev, sizeof(e4_to_fp32_dev)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp8_e5_to_fp32_lut, &e5_to_fp32_dev, sizeof(e5_to_fp32_dev)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&fp16_to_e4_dev, fp16_to_e4.size() * sizeof(std::uint8_t)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&fp16_to_e5_dev, fp16_to_e5.size() * sizeof(std::uint8_t)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(fp16_to_e4_dev, fp16_to_e4.data(), fp16_to_e4.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(fp16_to_e5_dev, fp16_to_e5.data(), fp16_to_e5.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp16_to_fp8_e4_lut, &fp16_to_e4_dev, sizeof(fp16_to_e4_dev)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp16_to_fp8_e5_lut, &fp16_to_e5_dev, sizeof(fp16_to_e5_dev)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&e4_to_e5_lut16_dev, e4_to_e5_16.size() * sizeof(std::uint16_t)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&e5_to_e4_lut16_dev, e5_to_e4_16.size() * sizeof(std::uint16_t)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(
            e4_to_e5_lut16_dev,
            e4_to_e5_16.data(),
            e4_to_e5_16.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice
        ) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(
            e5_to_e4_lut16_dev,
            e5_to_e4_16.data(),
            e5_to_e4_16.size() * sizeof(std::uint16_t),
            cudaMemcpyHostToDevice
        ) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp8_e4_to_e5_lut16, &e4_to_e5_lut16_dev, sizeof(e4_to_e5_lut16_dev)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp8_e5_to_e4_lut16, &e5_to_e4_lut16_dev, sizeof(e5_to_e4_lut16_dev)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&e4_to_e5_lut32_dev, e4_to_e5_32.size() * sizeof(std::uint32_t)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&e5_to_e4_lut32_dev, e5_to_e4_32.size() * sizeof(std::uint32_t)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(
            e4_to_e5_lut32_dev,
            e4_to_e5_32.data(),
            e4_to_e5_32.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice
        ) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpy(
            e5_to_e4_lut32_dev,
            e5_to_e4_32.data(),
            e5_to_e4_32.size() * sizeof(std::uint32_t),
            cudaMemcpyHostToDevice
        ) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp8_e4_to_e5_lut32, &e4_to_e5_lut32_dev, sizeof(e4_to_e5_lut32_dev)) != cudaSuccess) {
        return false;
    }
    if (cudaMemcpyToSymbol(k_fp8_e5_to_e4_lut32, &e5_to_e4_lut32_dev, sizeof(e5_to_e4_lut32_dev)) != cudaSuccess) {
        return false;
    }
    initialized = true;
    return true;
}

Fp8LutMode fp8_lut_mode_from_env() {
    Fp8LutMode mode = Fp8LutMode::auto_mode;
    const char* env = std::getenv("OMNICAST_FP8_LUT_MODE");
    if (env == nullptr || env[0] == '\0') {
        return mode;
    }
    if (std::strcmp(env, "x4") == 0) {
        mode = Fp8LutMode::force_x4;
    } else if (std::strcmp(env, "x16") == 0) {
        mode = Fp8LutMode::force_x16;
    }
    return mode;
}

Fp8RemapImpl fp8_remap_impl_from_env() {
    const char* env = std::getenv("OMNICAST_FP8_REMAP_IMPL");
    if (env == nullptr || env[0] == '\0') {
        return Fp8RemapImpl::auto_mode;
    }
    if (std::strcmp(env, "scalar") == 0) {
        return Fp8RemapImpl::scalar;
    }
    if (std::strcmp(env, "lut8_const") == 0 || std::strcmp(env, "lut32") == 0) {
        return Fp8RemapImpl::lut8_const;
    }
    if (std::strcmp(env, "lut8_shared") == 0) {
        return Fp8RemapImpl::lut8_shared;
    }
    if (std::strcmp(env, "lut32_u32") == 0 || std::strcmp(env, "lut32wide") == 0) {
        return Fp8RemapImpl::lut32;
    }
    if (std::strcmp(env, "lut32_u64") == 0) {
        return Fp8RemapImpl::lut32_u64;
    }
    if (std::strcmp(env, "lut32_u128") == 0) {
        return Fp8RemapImpl::lut32_u128;
    }
    if (std::strcmp(env, "lutx16") == 0) {
        return Fp8RemapImpl::lutx16;
    }
    if (std::strcmp(env, "lut16") == 0) {
        return Fp8RemapImpl::lut16;
    }
    if (std::strcmp(env, "copy32") == 0) {
        return Fp8RemapImpl::copy32;
    }
    if (std::strcmp(env, "copy64") == 0) {
        return Fp8RemapImpl::copy64;
    }
    if (std::strcmp(env, "copy128") == 0) {
        return Fp8RemapImpl::copy128;
    }
    return Fp8RemapImpl::auto_mode;
}

Fp16Fp8Impl fp16_fp8_impl_from_env() {
    const char* env = std::getenv("OMNICAST_FP16_FP8_IMPL");
    if (env == nullptr || env[0] == '\0') {
        return Fp16Fp8Impl::auto_mode;
    }
    if (std::strcmp(env, "native") == 0) {
        return Fp16Fp8Impl::native;
    }
    if (std::strcmp(env, "packed") == 0) {
        return Fp16Fp8Impl::packed;
    }
    if (std::strcmp(env, "packed_wide") == 0 || std::strcmp(env, "wide") == 0) {
        return Fp16Fp8Impl::packed_wide;
    }
    if (std::strcmp(env, "lut8_to_half") == 0) {
        return Fp16Fp8Impl::lut8_to_half;
    }
    if (std::strcmp(env, "lut16_to_fp8") == 0) {
        return Fp16Fp8Impl::lut16_to_fp8;
    }
    if (std::strcmp(env, "scalar") == 0) {
        return Fp16Fp8Impl::scalar;
    }
    if (std::strcmp(env, "copyshape") == 0) {
        return Fp16Fp8Impl::copyshape;
    }
    return Fp16Fp8Impl::auto_mode;
}

Fp32Fp8Impl fp32_fp8_impl_from_env() {
    const char* env = std::getenv("OMNICAST_FP32_FP8_IMPL");
    if (env == nullptr || env[0] == '\0') {
        return Fp32Fp8Impl::auto_mode;
    }
    if (std::strcmp(env, "native") == 0) {
        return Fp32Fp8Impl::native;
    }
    if (std::strcmp(env, "packed") == 0) {
        return Fp32Fp8Impl::packed;
    }
    if (std::strcmp(env, "packed_wide") == 0 || std::strcmp(env, "wide") == 0) {
        return Fp32Fp8Impl::packed_wide;
    }
    if (std::strcmp(env, "lut8_to_fp32") == 0) {
        return Fp32Fp8Impl::lut8_to_fp32;
    }
    if (std::strcmp(env, "scalar") == 0) {
        return Fp32Fp8Impl::scalar;
    }
    if (std::strcmp(env, "copyshape") == 0) {
        return Fp32Fp8Impl::copyshape;
    }
    return Fp32Fp8Impl::auto_mode;
}

template <bool E5M2>
__global__ void fp32_to_fp8x4_kernel(const float4* input, std::uint32_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const float4 v = input[idx];
        const std::uint8_t b0 = E5M2 ? encode_fp8_e5m2(v.x) : encode_fp8_e4m3(v.x);
        const std::uint8_t b1 = E5M2 ? encode_fp8_e5m2(v.y) : encode_fp8_e4m3(v.y);
        const std::uint8_t b2 = E5M2 ? encode_fp8_e5m2(v.z) : encode_fp8_e4m3(v.z);
        const std::uint8_t b3 = E5M2 ? encode_fp8_e5m2(v.w) : encode_fp8_e4m3(v.w);
        output[idx] = static_cast<std::uint32_t>(b0) | (static_cast<std::uint32_t>(b1) << 8U) |
                      (static_cast<std::uint32_t>(b2) << 16U) | (static_cast<std::uint32_t>(b3) << 24U);
    }
}

template <bool E5M2>
__global__ void fp8_to_fp32x4_kernel(const std::uint32_t* input, float4* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t packed = input[idx];
        const std::uint8_t b0 = static_cast<std::uint8_t>(packed & 0xFFU);
        const std::uint8_t b1 = static_cast<std::uint8_t>((packed >> 8U) & 0xFFU);
        const std::uint8_t b2 = static_cast<std::uint8_t>((packed >> 16U) & 0xFFU);
        const std::uint8_t b3 = static_cast<std::uint8_t>((packed >> 24U) & 0xFFU);
        float4 out{};
        out.x = E5M2 ? decode_fp8_e5m2(b0) : decode_fp8_e4m3(b0);
        out.y = E5M2 ? decode_fp8_e5m2(b1) : decode_fp8_e4m3(b1);
        out.z = E5M2 ? decode_fp8_e5m2(b2) : decode_fp8_e4m3(b2);
        out.w = E5M2 ? decode_fp8_e5m2(b3) : decode_fp8_e4m3(b3);
        output[idx] = out;
    }
}

template <bool E5M2>
__global__ void fp32_to_fp8x8_kernel(const float4* input, std::uint64_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const float4 v0 = input[idx * 2];
        const float4 v1 = input[idx * 2 + 1];
        const std::uint8_t b0 = E5M2 ? encode_fp8_e5m2(v0.x) : encode_fp8_e4m3(v0.x);
        const std::uint8_t b1 = E5M2 ? encode_fp8_e5m2(v0.y) : encode_fp8_e4m3(v0.y);
        const std::uint8_t b2 = E5M2 ? encode_fp8_e5m2(v0.z) : encode_fp8_e4m3(v0.z);
        const std::uint8_t b3 = E5M2 ? encode_fp8_e5m2(v0.w) : encode_fp8_e4m3(v0.w);
        const std::uint8_t b4 = E5M2 ? encode_fp8_e5m2(v1.x) : encode_fp8_e4m3(v1.x);
        const std::uint8_t b5 = E5M2 ? encode_fp8_e5m2(v1.y) : encode_fp8_e4m3(v1.y);
        const std::uint8_t b6 = E5M2 ? encode_fp8_e5m2(v1.z) : encode_fp8_e4m3(v1.z);
        const std::uint8_t b7 = E5M2 ? encode_fp8_e5m2(v1.w) : encode_fp8_e4m3(v1.w);
        output[idx] = static_cast<std::uint64_t>(b0) | (static_cast<std::uint64_t>(b1) << 8U) |
                      (static_cast<std::uint64_t>(b2) << 16U) | (static_cast<std::uint64_t>(b3) << 24U) |
                      (static_cast<std::uint64_t>(b4) << 32U) | (static_cast<std::uint64_t>(b5) << 40U) |
                      (static_cast<std::uint64_t>(b6) << 48U) | (static_cast<std::uint64_t>(b7) << 56U);
    }
}

template <bool E5M2>
__global__ void fp8_to_fp32x8_kernel(const std::uint64_t* input, float4* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint64_t packed = input[idx];
        const std::uint8_t b0 = static_cast<std::uint8_t>((packed >> 0U) & 0xFFU);
        const std::uint8_t b1 = static_cast<std::uint8_t>((packed >> 8U) & 0xFFU);
        const std::uint8_t b2 = static_cast<std::uint8_t>((packed >> 16U) & 0xFFU);
        const std::uint8_t b3 = static_cast<std::uint8_t>((packed >> 24U) & 0xFFU);
        const std::uint8_t b4 = static_cast<std::uint8_t>((packed >> 32U) & 0xFFU);
        const std::uint8_t b5 = static_cast<std::uint8_t>((packed >> 40U) & 0xFFU);
        const std::uint8_t b6 = static_cast<std::uint8_t>((packed >> 48U) & 0xFFU);
        const std::uint8_t b7 = static_cast<std::uint8_t>((packed >> 56U) & 0xFFU);
        float4 out0{};
        out0.x = E5M2 ? decode_fp8_e5m2(b0) : decode_fp8_e4m3(b0);
        out0.y = E5M2 ? decode_fp8_e5m2(b1) : decode_fp8_e4m3(b1);
        out0.z = E5M2 ? decode_fp8_e5m2(b2) : decode_fp8_e4m3(b2);
        out0.w = E5M2 ? decode_fp8_e5m2(b3) : decode_fp8_e4m3(b3);
        float4 out1{};
        out1.x = E5M2 ? decode_fp8_e5m2(b4) : decode_fp8_e4m3(b4);
        out1.y = E5M2 ? decode_fp8_e5m2(b5) : decode_fp8_e4m3(b5);
        out1.z = E5M2 ? decode_fp8_e5m2(b6) : decode_fp8_e4m3(b6);
        out1.w = E5M2 ? decode_fp8_e5m2(b7) : decode_fp8_e4m3(b7);
        output[idx * 2] = out0;
        output[idx * 2 + 1] = out1;
    }
}

template <bool E5M2>
__global__ void fp8_to_fp32_lutx4_kernel(const std::uint32_t* input, float4* output, std::size_t vec_count) {
    const float* lut = E5M2 ? k_fp8_e5_to_fp32_lut : k_fp8_e4_to_fp32_lut;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t packed = input[idx];
        float4 out{};
        out.x = __ldg(lut + ((packed >> 0U) & 0xFFU));
        out.y = __ldg(lut + ((packed >> 8U) & 0xFFU));
        out.z = __ldg(lut + ((packed >> 16U) & 0xFFU));
        out.w = __ldg(lut + ((packed >> 24U) & 0xFFU));
        output[idx] = out;
    }
}

__global__ void fp32_to_fp8_copyshape_x4_kernel(const uint4* input, std::uint32_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const uint4 in = input[idx];
        output[idx] = static_cast<std::uint32_t>(in.x & 0xFFU) | (static_cast<std::uint32_t>(in.y & 0xFFU) << 8U) |
                      (static_cast<std::uint32_t>(in.z & 0xFFU) << 16U) |
                      (static_cast<std::uint32_t>(in.w & 0xFFU) << 24U);
    }
}

__global__ void fp8_to_fp32_copyshape_x4_kernel(const std::uint32_t* input, uint4* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t in = input[idx];
        uint4 out{};
        out.x = in & 0xFFU;
        out.y = (in >> 8U) & 0xFFU;
        out.z = (in >> 16U) & 0xFFU;
        out.w = (in >> 24U) & 0xFFU;
        output[idx] = out;
    }
}

#if OMNICAST_CUDA_HAS_FP8_TYPES
template <bool E5M2>
__device__ inline std::uint8_t encode_fp8_native(float value) {
    union Fp8Bits {
        __nv_fp8_e4m3 e4;
        __nv_fp8_e5m2 e5;
        std::uint8_t u8;
    } bits{};
    if constexpr (E5M2) {
        bits.e5 = __nv_fp8_e5m2(value);
    } else {
        bits.e4 = __nv_fp8_e4m3(value);
    }
    return bits.u8;
}

template <bool E5M2>
__device__ inline float decode_fp8_native(std::uint8_t value) {
    union Fp8Bits {
        __nv_fp8_e4m3 e4;
        __nv_fp8_e5m2 e5;
        std::uint8_t u8;
    } bits{};
    bits.u8 = value;
    if constexpr (E5M2) {
        return static_cast<float>(bits.e5);
    } else {
        return static_cast<float>(bits.e4);
    }
}

template <bool E5M2>
__global__ void fp32_to_fp8x4_native_kernel(const float4* input, std::uint32_t* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const float4 v = input[idx];
        const std::uint8_t b0 = encode_fp8_native<E5M2>(v.x);
        const std::uint8_t b1 = encode_fp8_native<E5M2>(v.y);
        const std::uint8_t b2 = encode_fp8_native<E5M2>(v.z);
        const std::uint8_t b3 = encode_fp8_native<E5M2>(v.w);
        output[idx] = static_cast<std::uint32_t>(b0) | (static_cast<std::uint32_t>(b1) << 8U) |
                      (static_cast<std::uint32_t>(b2) << 16U) | (static_cast<std::uint32_t>(b3) << 24U);
    }
}

template <bool E5M2>
__global__ void fp8_to_fp32x4_native_kernel(const std::uint32_t* input, float4* output, std::size_t vec_count) {
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(gridDim.x);
    std::size_t idx = static_cast<std::size_t>(blockDim.x) * static_cast<std::size_t>(blockIdx.x) +
                      static_cast<std::size_t>(threadIdx.x);
    for (; idx < vec_count; idx += stride) {
        const std::uint32_t packed = input[idx];
        const std::uint8_t b0 = static_cast<std::uint8_t>(packed & 0xFFU);
        const std::uint8_t b1 = static_cast<std::uint8_t>((packed >> 8U) & 0xFFU);
        const std::uint8_t b2 = static_cast<std::uint8_t>((packed >> 16U) & 0xFFU);
        const std::uint8_t b3 = static_cast<std::uint8_t>((packed >> 24U) & 0xFFU);
        float4 out{};
        out.x = decode_fp8_native<E5M2>(b0);
        out.y = decode_fp8_native<E5M2>(b1);
        out.z = decode_fp8_native<E5M2>(b2);
        out.w = decode_fp8_native<E5M2>(b3);
        output[idx] = out;
    }
}
#endif

ArchFamily detect_arch_family(int major, int minor) {
    if (major == 8 && minor == 0) {
        return ArchFamily::a100;
    }
    if (major == 9 && minor == 0) {
        return ArchFamily::h100;
    }
    if (major >= 10) {
        return ArchFamily::b100;
    }
    return ArchFamily::generic;
}

int tuned_waves(const DeviceTuning& tuning) {
    switch (detect_arch_family(tuning.major, tuning.minor)) {
        case ArchFamily::a100:
            return 32;
        case ArchFamily::h100:
            return 40;
        case ArchFamily::b100:
            return 48;
        case ArchFamily::generic:
        default:
            return (tuning.major >= 9) ? 36 : 24;
    }
}

template <typename KernelPtr>
LaunchConfig select_launch_cfg(KernelPtr kernel, const DeviceTuning& tuning) {
    int min_grid = 0;
    int block = 0;
    LaunchConfig cfg{};
    if (cudaOccupancyMaxPotentialBlockSize(&min_grid, &block, kernel, 0, 0) == cudaSuccess && block > 0) {
        cfg.block = block;
    }
    const int tuned_max = tuning.sm_count * tuned_waves(tuning);
    cfg.max_blocks = std::max(256, tuned_max);
    return cfg;
}

LaunchConfig retune_launch_cfg(LaunchConfig base, int block_override, float max_blocks_scale) {
    if (block_override > 0) {
        base.block = block_override;
    }
    base.max_blocks = std::max(256, static_cast<int>(static_cast<float>(base.max_blocks) * max_blocks_scale));
    return base;
}

const RuntimeConfig& active_runtime_cfg() {
    if (g_active_runtime_cfg != nullptr) {
        return *g_active_runtime_cfg;
    }
    return get_runtime_config();
}

GpuStatus launch_copy(const void* input, void* output, std::size_t count, cudaStream_t stream, std::size_t item_size) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const cudaError_t err = cudaMemcpyAsync(output, input, count * item_size, cudaMemcpyDeviceToDevice, stream);
    return (err == cudaSuccess) ? GpuStatus::success : GpuStatus::cuda_runtime_error;
}

template <typename SrcT, typename DstT, bool SrcE5M2 = false, bool DstE5M2 = false>
GpuStatus launch_kernel(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const LaunchConfig cfg = active_runtime_cfg().generic_launch;
    const int blocks =
        static_cast<int>(std::min<std::size_t>((count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks));
    cast_kernel<SrcT, DstT, SrcE5M2, DstE5M2><<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const SrcT*>(input),
        reinterpret_cast<DstT*>(output),
        count
    );
    const cudaError_t err = cudaPeekAtLastError();
    return (err == cudaSuccess) ? GpuStatus::success : GpuStatus::cuda_runtime_error;
}

GpuStatus launch_fp32_to_fp16_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize = ((in_addr & 0x3U) == 0U) && ((out_addr & 0x3U) == 0U) && (count >= 2);
    if (!can_vectorize) {
        return launch_kernel<float, std::uint16_t>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().half2_launch;

    const std::size_t pair_count = count / 2;
    const int blocks = static_cast<int>(
        std::min<std::size_t>((pair_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
    );
    fp32_to_fp16_half2_kernel<<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const float2*>(input),
        reinterpret_cast<half2*>(output),
        pair_count
    );
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return GpuStatus::cuda_runtime_error;
    }

    if ((count & 1U) != 0U) {
        const float* tail_in = reinterpret_cast<const float*>(input) + (count - 1);
        std::uint16_t* tail_out = reinterpret_cast<std::uint16_t*>(output) + (count - 1);
        const GpuStatus tail_status = launch_kernel<float, std::uint16_t>(tail_in, tail_out, 1, stream);
        if (tail_status != GpuStatus::success) {
            return tail_status;
        }
    }
    return GpuStatus::success;
}

template <bool DstE5M2>
GpuStatus launch_fp32_to_fp8_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    Fp32Fp8Impl impl = fp32_fp8_impl_from_env();
    if (impl == Fp32Fp8Impl::auto_mode) {
        const RuntimeConfig& runtime = active_runtime_cfg();
        impl = fp32_fp8_impl_from_u8(runtime.auto_fp32_to_fp8_impl[DstE5M2 ? 1U : 0U]);
    }
    return launch_fp32_to_fp8_with_impl<DstE5M2>(input, output, count, stream, impl);
}

GpuStatus launch_fp16_to_fp32_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize = ((in_addr & 0x3U) == 0U) && ((out_addr & 0x3U) == 0U) && (count >= 2);
    if (!can_vectorize) {
        return launch_kernel<std::uint16_t, float>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().half2_launch;

    const std::size_t pair_count = count / 2;
    const int blocks = static_cast<int>(
        std::min<std::size_t>((pair_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
    );
    fp16_to_fp32_half2_kernel<<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const half2*>(input),
        reinterpret_cast<float2*>(output),
        pair_count
    );
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return GpuStatus::cuda_runtime_error;
    }

    if ((count & 1U) != 0U) {
        const std::uint16_t* tail_in = reinterpret_cast<const std::uint16_t*>(input) + (count - 1);
        float* tail_out = reinterpret_cast<float*>(output) + (count - 1);
        const GpuStatus tail_status = launch_kernel<std::uint16_t, float>(tail_in, tail_out, 1, stream);
        if (tail_status != GpuStatus::success) {
            return tail_status;
        }
    }
    return GpuStatus::success;
}

template <bool SrcE5M2>
GpuStatus launch_fp8_to_fp32_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    Fp32Fp8Impl impl = fp32_fp8_impl_from_env();
    if (impl == Fp32Fp8Impl::auto_mode) {
        const RuntimeConfig& runtime = active_runtime_cfg();
        impl = fp32_fp8_impl_from_u8(runtime.auto_fp8_to_fp32_impl[SrcE5M2 ? 1U : 0U]);
    }
    return launch_fp8_to_fp32_with_impl<SrcE5M2>(input, output, count, stream, impl);
}

GpuStatus launch_fp64_to_fp32_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize = ((in_addr & 0xFU) == 0U) && ((out_addr & 0x7U) == 0U) && (count >= 2);
    if (!can_vectorize) {
        return launch_kernel<double, float>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().generic_launch;
    const std::size_t vec_count = count / 2;
    const int blocks = static_cast<int>(
        std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
    );
    fp64_to_fp32x2_kernel<<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const double2*>(input),
        reinterpret_cast<float2*>(output),
        vec_count
    );
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return GpuStatus::cuda_runtime_error;
    }
    if ((count & 1U) != 0U) {
        const double* tail_in = reinterpret_cast<const double*>(input) + (count - 1);
        float* tail_out = reinterpret_cast<float*>(output) + (count - 1);
        return launch_kernel<double, float>(tail_in, tail_out, 1, stream);
    }
    return GpuStatus::success;
}

GpuStatus launch_fp32_to_fp64_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize = ((in_addr & 0x7U) == 0U) && ((out_addr & 0xFU) == 0U) && (count >= 2);
    if (!can_vectorize) {
        return launch_kernel<float, double>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().generic_launch;
    const std::size_t vec_count = count / 2;
    const int blocks = static_cast<int>(
        std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
    );
    fp32_to_fp64x2_kernel<<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const float2*>(input),
        reinterpret_cast<double2*>(output),
        vec_count
    );
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return GpuStatus::cuda_runtime_error;
    }
    if ((count & 1U) != 0U) {
        const float* tail_in = reinterpret_cast<const float*>(input) + (count - 1);
        double* tail_out = reinterpret_cast<double*>(output) + (count - 1);
        return launch_kernel<float, double>(tail_in, tail_out, 1, stream);
    }
    return GpuStatus::success;
}

GpuStatus launch_fp64_to_fp16_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize = ((in_addr & 0xFU) == 0U) && ((out_addr & 0x3U) == 0U) && (count >= 2);
    if (!can_vectorize) {
        return launch_kernel<double, std::uint16_t>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().half2_launch;
    const std::size_t vec_count = count / 2;
    const int blocks = static_cast<int>(
        std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
    );
    fp64_to_fp16x2_kernel<<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const double2*>(input),
        reinterpret_cast<half2*>(output),
        vec_count
    );
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return GpuStatus::cuda_runtime_error;
    }
    if ((count & 1U) != 0U) {
        const double* tail_in = reinterpret_cast<const double*>(input) + (count - 1);
        std::uint16_t* tail_out = reinterpret_cast<std::uint16_t*>(output) + (count - 1);
        return launch_kernel<double, std::uint16_t>(tail_in, tail_out, 1, stream);
    }
    return GpuStatus::success;
}

GpuStatus launch_fp16_to_fp64_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize = ((in_addr & 0x3U) == 0U) && ((out_addr & 0xFU) == 0U) && (count >= 2);
    if (!can_vectorize) {
        return launch_kernel<std::uint16_t, double>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().half2_launch;
    const std::size_t vec_count = count / 2;
    const int blocks = static_cast<int>(
        std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
    );
    fp16_to_fp64x2_kernel<<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const half2*>(input),
        reinterpret_cast<double2*>(output),
        vec_count
    );
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return GpuStatus::cuda_runtime_error;
    }
    if ((count & 1U) != 0U) {
        const std::uint16_t* tail_in = reinterpret_cast<const std::uint16_t*>(input) + (count - 1);
        double* tail_out = reinterpret_cast<double*>(output) + (count - 1);
        return launch_kernel<std::uint16_t, double>(tail_in, tail_out, 1, stream);
    }
    return GpuStatus::success;
}

template <bool DstE5M2>
GpuStatus launch_fp64_to_fp8_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize = ((in_addr & 0xFU) == 0U) && ((out_addr & 0x1U) == 0U) && (count >= 2);
    if (!can_vectorize) {
        return launch_kernel<double, std::uint8_t, false, DstE5M2>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().fp8_launch;
    const std::size_t pair_count = count / 2;
    const int blocks = static_cast<int>(
        std::min<std::size_t>((pair_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
    );
    fp64_to_fp8x2_kernel<DstE5M2><<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const double2*>(input),
        reinterpret_cast<std::uint16_t*>(output),
        pair_count
    );
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return GpuStatus::cuda_runtime_error;
    }
    if ((count & 1U) != 0U) {
        const double* tail_in = reinterpret_cast<const double*>(input) + (count - 1);
        std::uint8_t* tail_out = reinterpret_cast<std::uint8_t*>(output) + (count - 1);
        return launch_kernel<double, std::uint8_t, false, DstE5M2>(tail_in, tail_out, 1, stream);
    }
    return GpuStatus::success;
}

template <bool SrcE5M2>
GpuStatus launch_fp8_to_fp64_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize = ((in_addr & 0x1U) == 0U) && ((out_addr & 0xFU) == 0U) && (count >= 2);
    if (!can_vectorize) {
        return launch_kernel<std::uint8_t, double, SrcE5M2, false>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().fp8_launch;
    const std::size_t pair_count = count / 2;
    const int blocks = static_cast<int>(
        std::min<std::size_t>((pair_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
    );
    fp8_to_fp64x2_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const std::uint16_t*>(input),
        reinterpret_cast<double2*>(output),
        pair_count
    );
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return GpuStatus::cuda_runtime_error;
    }
    if ((count & 1U) != 0U) {
        const std::uint8_t* tail_in = reinterpret_cast<const std::uint8_t*>(input) + (count - 1);
        double* tail_out = reinterpret_cast<double*>(output) + (count - 1);
        return launch_kernel<std::uint8_t, double, SrcE5M2, false>(tail_in, tail_out, 1, stream);
    }
    return GpuStatus::success;
}

template <bool E5ToE4>
GpuStatus launch_fp8_lut_transform_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const RuntimeConfig& runtime = active_runtime_cfg();
    if (!runtime.fp8_lut_ready) {
        if constexpr (E5ToE4) {
            return launch_kernel<std::uint8_t, std::uint8_t, true, false>(input, output, count, stream);
        }
        return launch_kernel<std::uint8_t, std::uint8_t, false, true>(input, output, count, stream);
    }

    Fp8RemapImpl remap_impl = fp8_remap_impl_from_env();
    if (remap_impl == Fp8RemapImpl::auto_mode) {
        // Default to LUT32 for FP8<->FP8 remap; best throughput in benchmarks.
        remap_impl = Fp8RemapImpl::lut32;
    }
    if (remap_impl == Fp8RemapImpl::scalar) {
        if constexpr (E5ToE4) {
            return launch_kernel<std::uint8_t, std::uint8_t, true, false>(input, output, count, stream);
        }
        return launch_kernel<std::uint8_t, std::uint8_t, false, true>(input, output, count, stream);
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize_x16 = ((in_addr & 0xFU) == 0U) && ((out_addr & 0xFU) == 0U) && (count >= 16);
    const bool can_vectorize_x4 = ((in_addr & 0x3U) == 0U) && ((out_addr & 0x3U) == 0U) && (count >= 4);
    if (!can_vectorize_x4) {
        if constexpr (E5ToE4) {
            return launch_kernel<std::uint8_t, std::uint8_t, true, false>(input, output, count, stream);
        } else {
            return launch_kernel<std::uint8_t, std::uint8_t, false, true>(input, output, count, stream);
        }
    }

    const LaunchConfig cfg = E5ToE4 ? runtime.fp8_lut_launch : runtime.fp8_launch;
    const Fp8LutMode lut_mode = fp8_lut_mode_from_env();
    bool prefer_x16 = false;
    if (remap_impl == Fp8RemapImpl::lutx16 || remap_impl == Fp8RemapImpl::copy128 ||
        remap_impl == Fp8RemapImpl::lut32_u128) {
        prefer_x16 = true;
    } else if (remap_impl == Fp8RemapImpl::lut8_const || remap_impl == Fp8RemapImpl::lut8_shared ||
               remap_impl == Fp8RemapImpl::lut32 || remap_impl == Fp8RemapImpl::lut32_u64 ||
               remap_impl == Fp8RemapImpl::lut16 ||
               remap_impl == Fp8RemapImpl::copy32 || remap_impl == Fp8RemapImpl::copy64) {
        prefer_x16 = false;
    } else if (lut_mode == Fp8LutMode::force_x16) {
        prefer_x16 = true;
    } else if (lut_mode == Fp8LutMode::force_x4) {
        prefer_x16 = false;
    } else {
        // Auto-mode: A100 prefers x4 for this kernel family.
        const bool is_a100 = (runtime.major == 8 && runtime.minor == 0);
        if (!is_a100 && E5ToE4 && count >= (1U << 20U)) {
            prefer_x16 = true;
        }
    }
    std::size_t consumed = 0;
    if (prefer_x16 && can_vectorize_x16) {
        const std::size_t vec16_count = count / 16;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec16_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        if (remap_impl == Fp8RemapImpl::copy128) {
            fp8_copy_u128_kernel<<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const uint4*>(input),
                reinterpret_cast<uint4*>(output),
                vec16_count
            );
        } else if (remap_impl == Fp8RemapImpl::lut32_u128) {
            fp8_lut32_transform_x16_kernel<E5ToE4><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const uint4*>(input),
                reinterpret_cast<uint4*>(output),
                vec16_count
            );
        } else {
            fp8_lut_transform_x16_kernel<E5ToE4><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const uint4*>(input),
                reinterpret_cast<uint4*>(output),
                vec16_count
            );
        }
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed = vec16_count * 16;
    }

    const std::size_t rem_after_x16 = count - consumed;
    if (rem_after_x16 >= 4 && can_vectorize_x4) {
        const std::size_t vec4_count = rem_after_x16 / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec4_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        if (remap_impl == Fp8RemapImpl::copy128 && can_vectorize_x16) {
            // already consumed by x16 path above for copy128
        } else if (remap_impl == Fp8RemapImpl::copy64 &&
                   ((reinterpret_cast<std::uintptr_t>(input) + consumed) & 0x7U) == 0U &&
                   ((reinterpret_cast<std::uintptr_t>(output) + consumed) & 0x7U) == 0U &&
                   vec4_count >= 2) {
            const std::size_t vec8_count = rem_after_x16 / 8;
            const int copy_blocks = static_cast<int>(
                std::min<std::size_t>((vec8_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
            );
            fp8_copy_u64_kernel<<<copy_blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint64_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<std::uint64_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
                vec8_count
            );
            const cudaError_t err = cudaPeekAtLastError();
            if (err != cudaSuccess) {
                return GpuStatus::cuda_runtime_error;
            }
            consumed += vec8_count * 8;
        } else if (remap_impl == Fp8RemapImpl::lut32_u64 &&
                   ((reinterpret_cast<std::uintptr_t>(input) + consumed) & 0x7U) == 0U &&
                   ((reinterpret_cast<std::uintptr_t>(output) + consumed) & 0x7U) == 0U &&
                   vec4_count >= 2) {
            const std::size_t vec8_count = rem_after_x16 / 8;
            const int copy_blocks = static_cast<int>(
                std::min<std::size_t>((vec8_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
            );
            fp8_lut32_transform_x8_kernel<E5ToE4><<<copy_blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint64_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<std::uint64_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
                vec8_count
            );
            const cudaError_t err = cudaPeekAtLastError();
            if (err != cudaSuccess) {
                return GpuStatus::cuda_runtime_error;
            }
            consumed += vec8_count * 8;
        } else if (remap_impl == Fp8RemapImpl::copy32 || remap_impl == Fp8RemapImpl::copy128) {
            fp8_copy_u32_kernel<<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
                vec4_count
            );
            const cudaError_t err = cudaPeekAtLastError();
            if (err != cudaSuccess) {
                return GpuStatus::cuda_runtime_error;
            }
            consumed += vec4_count * 4;
        } else if (remap_impl == Fp8RemapImpl::lut16) {
            fp8_lut16_transform_x4_kernel<E5ToE4><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
                vec4_count
            );
            const cudaError_t err = cudaPeekAtLastError();
            if (err != cudaSuccess) {
                return GpuStatus::cuda_runtime_error;
            }
            consumed += vec4_count * 4;
        } else if (remap_impl == Fp8RemapImpl::lut32 || remap_impl == Fp8RemapImpl::lut32_u128 ||
                   remap_impl == Fp8RemapImpl::auto_mode) {
            fp8_lut32_transform_x4_kernel<E5ToE4><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
                vec4_count
            );
            const cudaError_t err = cudaPeekAtLastError();
            if (err != cudaSuccess) {
                return GpuStatus::cuda_runtime_error;
            }
            consumed += vec4_count * 4;
        } else if (remap_impl == Fp8RemapImpl::lut8_shared) {
            fp8_lut_transform_shared_x4_kernel<E5ToE4><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
                vec4_count
            );
            const cudaError_t err = cudaPeekAtLastError();
            if (err != cudaSuccess) {
                return GpuStatus::cuda_runtime_error;
            }
            consumed += vec4_count * 4;
        } else {
            fp8_lut_transform_x4_kernel<E5ToE4><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
                vec4_count
            );
            const cudaError_t err = cudaPeekAtLastError();
            if (err != cudaSuccess) {
                return GpuStatus::cuda_runtime_error;
            }
            consumed += vec4_count * 4;
        }
    }

    const std::size_t tail = count - consumed;
    if (tail > 0) {
        const std::uint8_t* tail_in = reinterpret_cast<const std::uint8_t*>(input) + consumed;
        std::uint8_t* tail_out = reinterpret_cast<std::uint8_t*>(output) + consumed;
        if (remap_impl == Fp8RemapImpl::copy32 || remap_impl == Fp8RemapImpl::copy64 ||
            remap_impl == Fp8RemapImpl::copy128) {
            return launch_copy(tail_in, tail_out, tail, stream, sizeof(std::uint8_t));
        }
        if constexpr (E5ToE4) {
            return launch_kernel<std::uint8_t, std::uint8_t, true, false>(tail_in, tail_out, tail, stream);
        } else {
            return launch_kernel<std::uint8_t, std::uint8_t, false, true>(tail_in, tail_out, tail, stream);
        }
    }
    return GpuStatus::success;
}

template <bool DstE5M2>
GpuStatus launch_fp32_to_fp8_with_impl(
    const void* input,
    void* output,
    std::size_t count,
    cudaStream_t stream,
    Fp32Fp8Impl impl
) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize_x8 = ((in_addr & 0x1FU) == 0U) && ((out_addr & 0x7U) == 0U) && (count >= 8);
    const bool can_vectorize_x4 = ((in_addr & 0xFU) == 0U) && ((out_addr & 0x3U) == 0U) && (count >= 4);
    if (impl == Fp32Fp8Impl::scalar || !can_vectorize_x4) {
        return launch_kernel<float, std::uint8_t, false, DstE5M2>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().fp8_launch;
    std::size_t consumed = 0;
    if (impl == Fp32Fp8Impl::copyshape && can_vectorize_x4) {
        const std::size_t vec_count = count / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp32_to_fp8_copyshape_x4_kernel<<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const uint4*>(input),
            reinterpret_cast<std::uint32_t*>(output),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed = vec_count * 4;
    } else if (impl == Fp32Fp8Impl::packed_wide && can_vectorize_x8) {
        const std::size_t vec_count = count / 8;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp32_to_fp8x8_kernel<DstE5M2><<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const float4*>(input),
            reinterpret_cast<std::uint64_t*>(output),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed = vec_count * 8;
    }

    const std::size_t rem = count - consumed;
    if (rem >= 4 && (((in_addr + consumed * sizeof(float)) & 0xFU) == 0U) &&
        (((out_addr + consumed * sizeof(std::uint8_t)) & 0x3U) == 0U)) {
        const std::size_t vec_count = rem / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        if (impl == Fp32Fp8Impl::native && active_runtime_cfg().native_fp8) {
#if OMNICAST_CUDA_HAS_FP8_TYPES
            fp32_to_fp8x4_native_kernel<DstE5M2><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const float4*>(reinterpret_cast<const float*>(input) + consumed),
                reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
                vec_count
            );
#else
            fp32_to_fp8x4_kernel<DstE5M2><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const float4*>(reinterpret_cast<const float*>(input) + consumed),
                reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
                vec_count
            );
#endif
        } else {
            fp32_to_fp8x4_kernel<DstE5M2><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const float4*>(reinterpret_cast<const float*>(input) + consumed),
                reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
                vec_count
            );
        }
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed += vec_count * 4;
    }
    if (consumed < count) {
        const float* tail_in = reinterpret_cast<const float*>(input) + consumed;
        std::uint8_t* tail_out = reinterpret_cast<std::uint8_t*>(output) + consumed;
        return launch_kernel<float, std::uint8_t, false, DstE5M2>(tail_in, tail_out, count - consumed, stream);
    }
    return GpuStatus::success;
}

template <bool DstE5M2>
GpuStatus launch_fp16_to_fp8_with_impl(
    const void* input,
    void* output,
    std::size_t count,
    cudaStream_t stream,
    Fp16Fp8Impl impl
) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize_x8 = ((in_addr & 0xFU) == 0U) && ((out_addr & 0x7U) == 0U) && (count >= 8);
    const bool can_vectorize_x2 = ((in_addr & 0x3U) == 0U) && ((out_addr & 0x1U) == 0U) && (count >= 2);
    if (impl == Fp16Fp8Impl::scalar || !can_vectorize_x2) {
        return launch_kernel<std::uint16_t, std::uint8_t, false, DstE5M2>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().fp8_launch;
    std::size_t consumed = 0;
    if (impl == Fp16Fp8Impl::copyshape && ((in_addr & 0x7U) == 0U) && ((out_addr & 0x3U) == 0U) && count >= 4) {
        const std::size_t vec_count = count / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp16_to_fp8_copyshape_x4_kernel<<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const std::uint64_t*>(input),
            reinterpret_cast<std::uint32_t*>(output),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed = vec_count * 4;
    }
    if (impl == Fp16Fp8Impl::packed_wide && can_vectorize_x8) {
        const std::size_t vec_count = count / 8;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp16_to_fp8x8_kernel<DstE5M2><<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const uint4*>(input),
            reinterpret_cast<std::uint64_t*>(output),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed = vec_count * 8;
    }
    if ((impl == Fp16Fp8Impl::packed || impl == Fp16Fp8Impl::packed_wide || impl == Fp16Fp8Impl::native ||
         impl == Fp16Fp8Impl::lut8_to_half) &&
        (count - consumed) >= 4 &&
        (((in_addr + consumed * sizeof(std::uint16_t)) & 0x7U) == 0U) &&
        (((out_addr + consumed * sizeof(std::uint8_t)) & 0x3U) == 0U)) {
        const std::size_t vec_count = (count - consumed) / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp16_to_fp8x4_kernel<DstE5M2><<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const half2*>(reinterpret_cast<const std::uint16_t*>(input) + consumed),
            reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed += vec_count * 4;
    } else if (impl == Fp16Fp8Impl::lut16_to_fp8 &&
               (count - consumed) >= 4 &&
               (((in_addr + consumed * sizeof(std::uint16_t)) & 0x7U) == 0U) &&
               (((out_addr + consumed * sizeof(std::uint8_t)) & 0x3U) == 0U)) {
        const std::size_t vec_count = (count - consumed) / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp16_to_fp8_lutx4_kernel<DstE5M2><<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const std::uint64_t*>(reinterpret_cast<const std::uint16_t*>(input) + consumed),
            reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed += vec_count * 4;
    }

    const std::size_t rem = count - consumed;
    if (rem >= 2) {
        const std::size_t pair_count = rem / 2;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((pair_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp16_to_fp8x2_kernel<DstE5M2><<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const half2*>(reinterpret_cast<const std::uint16_t*>(input) + consumed),
            reinterpret_cast<std::uint16_t*>(reinterpret_cast<std::uint8_t*>(output) + consumed),
            pair_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed += pair_count * 2;
    }
    if (consumed < count) {
        const std::uint16_t* tail_in = reinterpret_cast<const std::uint16_t*>(input) + consumed;
        std::uint8_t* tail_out = reinterpret_cast<std::uint8_t*>(output) + consumed;
        return launch_kernel<std::uint16_t, std::uint8_t, false, DstE5M2>(tail_in, tail_out, count - consumed, stream);
    }
    return GpuStatus::success;
}

template <bool DstE5M2>
GpuStatus launch_fp16_to_fp8_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    Fp16Fp8Impl impl = fp16_fp8_impl_from_env();
    if (impl == Fp16Fp8Impl::auto_mode) {
        const RuntimeConfig& runtime = active_runtime_cfg();
        impl = fp16_fp8_impl_from_u8(runtime.auto_fp16_to_fp8_impl[DstE5M2 ? 1U : 0U]);
    }
    return launch_fp16_to_fp8_with_impl<DstE5M2>(input, output, count, stream, impl);
}

template <bool SrcE5M2>
GpuStatus launch_fp8_to_fp32_with_impl(
    const void* input,
    void* output,
    std::size_t count,
    cudaStream_t stream,
    Fp32Fp8Impl impl
) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize_x8 = ((in_addr & 0x7U) == 0U) && ((out_addr & 0x1FU) == 0U) && (count >= 8);
    const bool can_vectorize_x4 = ((in_addr & 0x3U) == 0U) && ((out_addr & 0xFU) == 0U) && (count >= 4);
    if (impl == Fp32Fp8Impl::scalar || !can_vectorize_x4) {
        return launch_kernel<std::uint8_t, float, SrcE5M2, false>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().fp8_launch;
    std::size_t consumed = 0;
    if (impl == Fp32Fp8Impl::copyshape && can_vectorize_x4) {
        const std::size_t vec_count = count / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp8_to_fp32_copyshape_x4_kernel<<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const std::uint32_t*>(input),
            reinterpret_cast<uint4*>(output),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed = vec_count * 4;
    } else if (impl == Fp32Fp8Impl::packed_wide && can_vectorize_x8) {
        const std::size_t vec_count = count / 8;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp8_to_fp32x8_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const std::uint64_t*>(input),
            reinterpret_cast<float4*>(output),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed = vec_count * 8;
    }

    const std::size_t rem = count - consumed;
    if (rem >= 4 && (((in_addr + consumed * sizeof(std::uint8_t)) & 0x3U) == 0U) &&
        (((out_addr + consumed * sizeof(float)) & 0xFU) == 0U)) {
        const std::size_t vec_count = rem / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        if (impl == Fp32Fp8Impl::lut8_to_fp32) {
            fp8_to_fp32_lutx4_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<float4*>(reinterpret_cast<float*>(output) + consumed),
                vec_count
            );
        } else if (impl == Fp32Fp8Impl::native && active_runtime_cfg().native_fp8) {
#if OMNICAST_CUDA_HAS_FP8_TYPES
            fp8_to_fp32x4_native_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<float4*>(reinterpret_cast<float*>(output) + consumed),
                vec_count
            );
#else
            fp8_to_fp32x4_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<float4*>(reinterpret_cast<float*>(output) + consumed),
                vec_count
            );
#endif
        } else {
            fp8_to_fp32x4_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
                reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
                reinterpret_cast<float4*>(reinterpret_cast<float*>(output) + consumed),
                vec_count
            );
        }
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed += vec_count * 4;
    }
    if (consumed < count) {
        const std::uint8_t* tail_in = reinterpret_cast<const std::uint8_t*>(input) + consumed;
        float* tail_out = reinterpret_cast<float*>(output) + consumed;
        return launch_kernel<std::uint8_t, float, SrcE5M2, false>(tail_in, tail_out, count - consumed, stream);
    }
    return GpuStatus::success;
}

template <bool SrcE5M2>
GpuStatus launch_fp8_to_fp16_with_impl(
    const void* input,
    void* output,
    std::size_t count,
    cudaStream_t stream,
    Fp16Fp8Impl impl
) {
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize_x8 = ((in_addr & 0x7U) == 0U) && ((out_addr & 0xFU) == 0U) && (count >= 8);
    const bool can_vectorize_x2 = ((in_addr & 0x1U) == 0U) && ((out_addr & 0x3U) == 0U) && (count >= 2);
    if (impl == Fp16Fp8Impl::scalar || !can_vectorize_x2) {
        return launch_kernel<std::uint8_t, std::uint16_t, SrcE5M2, false>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().fp8_launch;
    std::size_t consumed = 0;
    if (impl == Fp16Fp8Impl::copyshape && ((in_addr & 0x3U) == 0U) && ((out_addr & 0x7U) == 0U) && count >= 4) {
        const std::size_t vec_count = count / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp8_to_fp16_copyshape_x4_kernel<<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const std::uint32_t*>(input),
            reinterpret_cast<std::uint64_t*>(output),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed = vec_count * 4;
    }
    if (impl == Fp16Fp8Impl::packed_wide && can_vectorize_x8) {
        const std::size_t vec_count = count / 8;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp8_to_fp16x8_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const std::uint64_t*>(input),
            reinterpret_cast<uint4*>(output),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed = vec_count * 8;
    }
    if (impl == Fp16Fp8Impl::lut8_to_half &&
        (count - consumed) >= 4 &&
        (((in_addr + consumed * sizeof(std::uint8_t)) & 0x3U) == 0U) &&
        (((out_addr + consumed * sizeof(std::uint16_t)) & 0x7U) == 0U)) {
        const std::size_t vec_count = (count - consumed) / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp8_to_fp16_lutx4_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
            reinterpret_cast<std::uint64_t*>(reinterpret_cast<std::uint16_t*>(output) + consumed),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed += vec_count * 4;
    } else if ((impl == Fp16Fp8Impl::packed || impl == Fp16Fp8Impl::packed_wide || impl == Fp16Fp8Impl::native ||
                impl == Fp16Fp8Impl::lut16_to_fp8) &&
               (count - consumed) >= 4 &&
               (((in_addr + consumed * sizeof(std::uint8_t)) & 0x3U) == 0U) &&
               (((out_addr + consumed * sizeof(std::uint16_t)) & 0x3U) == 0U)) {
        const std::size_t vec_count = (count - consumed) / 4;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp8_to_fp16x4_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const std::uint32_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
            reinterpret_cast<half2*>(reinterpret_cast<std::uint16_t*>(output) + consumed),
            vec_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed += vec_count * 4;
    }

    const std::size_t rem = count - consumed;
    if (rem >= 2) {
        const std::size_t pair_count = rem / 2;
        const int blocks = static_cast<int>(
            std::min<std::size_t>((pair_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
        );
        fp8_to_fp16x2_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
            reinterpret_cast<const std::uint16_t*>(reinterpret_cast<const std::uint8_t*>(input) + consumed),
            reinterpret_cast<half2*>(reinterpret_cast<std::uint16_t*>(output) + consumed),
            pair_count
        );
        const cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess) {
            return GpuStatus::cuda_runtime_error;
        }
        consumed += pair_count * 2;
    }
    if (consumed < count) {
        const std::uint8_t* tail_in = reinterpret_cast<const std::uint8_t*>(input) + consumed;
        std::uint16_t* tail_out = reinterpret_cast<std::uint16_t*>(output) + consumed;
        return launch_kernel<std::uint8_t, std::uint16_t, SrcE5M2, false>(tail_in, tail_out, count - consumed, stream);
    }
    return GpuStatus::success;
}

template <bool SrcE5M2>
GpuStatus launch_fp8_to_fp16_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
    Fp16Fp8Impl impl = fp16_fp8_impl_from_env();
    if (impl == Fp16Fp8Impl::auto_mode) {
        const RuntimeConfig& runtime = active_runtime_cfg();
        impl = fp16_fp8_impl_from_u8(runtime.auto_fp8_to_fp16_impl[SrcE5M2 ? 1U : 0U]);
    }
    return launch_fp8_to_fp16_with_impl<SrcE5M2>(input, output, count, stream, impl);
}

template <bool DstE5M2>
GpuStatus launch_fp32_to_fp8_native_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
#if !OMNICAST_CUDA_HAS_FP8_TYPES
    return launch_fp32_to_fp8_opt<DstE5M2>(input, output, count, stream);
#else
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize = ((in_addr & 0xFU) == 0U) && ((out_addr & 0x3U) == 0U) && (count >= 4);
    if (!can_vectorize) {
        return launch_fp32_to_fp8_opt<DstE5M2>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().fp8_native_launch;
    const std::size_t vec_count = count / 4;
    const int blocks = static_cast<int>(
        std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
    );
    fp32_to_fp8x4_native_kernel<DstE5M2><<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<std::uint32_t*>(output),
        vec_count
    );
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return GpuStatus::cuda_runtime_error;
    }
    const std::size_t tail = count - vec_count * 4;
    if (tail > 0) {
        const float* tail_in = reinterpret_cast<const float*>(input) + vec_count * 4;
        std::uint8_t* tail_out = reinterpret_cast<std::uint8_t*>(output) + vec_count * 4;
        return launch_kernel<float, std::uint8_t, false, DstE5M2>(tail_in, tail_out, tail, stream);
    }
    return GpuStatus::success;
#endif
}

template <bool SrcE5M2>
GpuStatus launch_fp8_to_fp32_native_opt(const void* input, void* output, std::size_t count, cudaStream_t stream) {
#if !OMNICAST_CUDA_HAS_FP8_TYPES
    return launch_fp8_to_fp32_opt<SrcE5M2>(input, output, count, stream);
#else
    if (count == 0) {
        return GpuStatus::success;
    }
    const auto in_addr = reinterpret_cast<std::uintptr_t>(input);
    const auto out_addr = reinterpret_cast<std::uintptr_t>(output);
    const bool can_vectorize = ((in_addr & 0x3U) == 0U) && ((out_addr & 0xFU) == 0U) && (count >= 4);
    if (!can_vectorize) {
        return launch_fp8_to_fp32_opt<SrcE5M2>(input, output, count, stream);
    }

    const LaunchConfig cfg = active_runtime_cfg().fp8_native_launch;
    const std::size_t vec_count = count / 4;
    const int blocks = static_cast<int>(
        std::min<std::size_t>((vec_count + static_cast<std::size_t>(cfg.block) - 1) / cfg.block, cfg.max_blocks)
    );
    fp8_to_fp32x4_native_kernel<SrcE5M2><<<blocks, cfg.block, 0, stream>>>(
        reinterpret_cast<const std::uint32_t*>(input),
        reinterpret_cast<float4*>(output),
        vec_count
    );
    const cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        return GpuStatus::cuda_runtime_error;
    }
    const std::size_t tail = count - vec_count * 4;
    if (tail > 0) {
        const std::uint8_t* tail_in = reinterpret_cast<const std::uint8_t*>(input) + vec_count * 4;
        float* tail_out = reinterpret_cast<float*>(output) + vec_count * 4;
        return launch_kernel<std::uint8_t, float, SrcE5M2, false>(tail_in, tail_out, tail, stream);
    }
    return GpuStatus::success;
#endif
}

GpuStatus unsupported_launch(const void*, void*, std::size_t, cudaStream_t) {
    return GpuStatus::unsupported_conversion;
}

bool fp16_fp8_autotune_enabled() {
    const char* env = std::getenv("OMNICAST_FP16_FP8_AUTOTUNE");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return std::strcmp(env, "0") != 0;
}

template <typename LaunchFnT>
double benchmark_impl_ms(LaunchFnT&& fn, int iters) {
    for (int i = 0; i < 3; ++i) {
        if (fn() != GpuStatus::success) {
            return std::numeric_limits<double>::infinity();
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        return std::numeric_limits<double>::infinity();
    }
    cudaEvent_t start_evt{};
    cudaEvent_t stop_evt{};
    if (cudaEventCreate(&start_evt) != cudaSuccess || cudaEventCreate(&stop_evt) != cudaSuccess) {
        return std::numeric_limits<double>::infinity();
    }
    if (cudaEventRecord(start_evt, nullptr) != cudaSuccess) {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
        return std::numeric_limits<double>::infinity();
    }
    for (int i = 0; i < iters; ++i) {
        if (fn() != GpuStatus::success) {
            cudaEventDestroy(start_evt);
            cudaEventDestroy(stop_evt);
            return std::numeric_limits<double>::infinity();
        }
    }
    if (cudaEventRecord(stop_evt, nullptr) != cudaSuccess || cudaEventSynchronize(stop_evt) != cudaSuccess) {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
        return std::numeric_limits<double>::infinity();
    }
    float elapsed_ms = 0.0f;
    if (cudaEventElapsedTime(&elapsed_ms, start_evt, stop_evt) != cudaSuccess) {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
        return std::numeric_limits<double>::infinity();
    }
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    return static_cast<double>(elapsed_ms) / static_cast<double>(iters);
}

void autotune_fp16_fp8(RuntimeConfig& cfg) {
    if (!fp16_fp8_autotune_enabled()) {
        cfg.fp16_fp8_autotuned = false;
        return;
    }
    constexpr std::size_t kCount = 1U << 20U;
    constexpr int kIters = 12;
    std::uint16_t* d_fp16_in = nullptr;
    std::uint8_t* d_fp8_e4 = nullptr;
    std::uint8_t* d_fp8_e5 = nullptr;
    std::uint16_t* d_fp16_out = nullptr;
    if (cudaMalloc(&d_fp16_in, kCount * sizeof(std::uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_fp8_e4, kCount * sizeof(std::uint8_t)) != cudaSuccess ||
        cudaMalloc(&d_fp8_e5, kCount * sizeof(std::uint8_t)) != cudaSuccess ||
        cudaMalloc(&d_fp16_out, kCount * sizeof(std::uint16_t)) != cudaSuccess) {
        if (d_fp16_in != nullptr) cudaFree(d_fp16_in);
        if (d_fp8_e4 != nullptr) cudaFree(d_fp8_e4);
        if (d_fp8_e5 != nullptr) cudaFree(d_fp8_e5);
        if (d_fp16_out != nullptr) cudaFree(d_fp16_out);
        cfg.fp16_fp8_autotuned = false;
        return;
    }
    cudaMemset(d_fp16_in, 0x2A, kCount * sizeof(std::uint16_t));
    cudaMemset(d_fp8_e4, 0x6B, kCount * sizeof(std::uint8_t));
    cudaMemset(d_fp8_e5, 0x94, kCount * sizeof(std::uint8_t));

    const RuntimeConfig* prev_cfg = g_active_runtime_cfg;
    g_active_runtime_cfg = &cfg;

    auto pick_best = [&](const auto& modes, auto&& bench_fn) -> Fp16Fp8Impl {
        constexpr double kSwitchEpsilon = 0.01;  // require >=1% improvement to switch
        Fp16Fp8Impl best = modes[0];
        double best_ms = std::numeric_limits<double>::infinity();
        for (const Fp16Fp8Impl mode : modes) {
            const double ms = bench_fn(mode);
            if (!std::isfinite(ms)) {
                continue;
            }
            if (best_ms == std::numeric_limits<double>::infinity() || ms < best_ms * (1.0 - kSwitchEpsilon)) {
                best_ms = ms;
                best = mode;
            }
        }
        return best;
    };

    const std::array<Fp16Fp8Impl, 2> fp16_to_fp8_modes = {
        Fp16Fp8Impl::packed, Fp16Fp8Impl::native
    };
    const std::array<Fp16Fp8Impl, 2> fp8_to_fp16_modes = {
        Fp16Fp8Impl::packed, Fp16Fp8Impl::native
    };

    const Fp16Fp8Impl fp16_to_e4 = pick_best(
        fp16_to_fp8_modes,
        [&](Fp16Fp8Impl mode) {
            return benchmark_impl_ms(
                [&]() { return launch_fp16_to_fp8_with_impl<false>(d_fp16_in, d_fp8_e4, kCount, nullptr, mode); }, kIters
            );
        }
    );
    const Fp16Fp8Impl fp16_to_e5 = pick_best(
        fp16_to_fp8_modes,
        [&](Fp16Fp8Impl mode) {
            return benchmark_impl_ms(
                [&]() { return launch_fp16_to_fp8_with_impl<true>(d_fp16_in, d_fp8_e5, kCount, nullptr, mode); }, kIters
            );
        }
    );
    const Fp16Fp8Impl e4_to_fp16 = pick_best(
        fp8_to_fp16_modes,
        [&](Fp16Fp8Impl mode) {
            return benchmark_impl_ms(
                [&]() { return launch_fp8_to_fp16_with_impl<false>(d_fp8_e4, d_fp16_out, kCount, nullptr, mode); }, kIters
            );
        }
    );
    const Fp16Fp8Impl e5_to_fp16 = pick_best(
        fp8_to_fp16_modes,
        [&](Fp16Fp8Impl mode) {
            return benchmark_impl_ms(
                [&]() { return launch_fp8_to_fp16_with_impl<true>(d_fp8_e5, d_fp16_out, kCount, nullptr, mode); }, kIters
            );
        }
    );

    g_active_runtime_cfg = prev_cfg;
    cudaFree(d_fp16_in);
    cudaFree(d_fp8_e4);
    cudaFree(d_fp8_e5);
    cudaFree(d_fp16_out);

    cfg.auto_fp16_to_fp8_impl[0] = fp16_fp8_impl_to_u8(fp16_to_e4);
    cfg.auto_fp16_to_fp8_impl[1] = fp16_fp8_impl_to_u8(fp16_to_e5);
    cfg.auto_fp8_to_fp16_impl[0] = fp16_fp8_impl_to_u8(e4_to_fp16);
    cfg.auto_fp8_to_fp16_impl[1] = fp16_fp8_impl_to_u8(e5_to_fp16);
    cfg.fp16_fp8_autotuned = true;
}

bool fp32_fp8_autotune_enabled() {
    const char* env = std::getenv("OMNICAST_FP32_FP8_AUTOTUNE");
    if (env == nullptr || env[0] == '\0') {
        return true;
    }
    return std::strcmp(env, "0") != 0;
}

void autotune_fp32_fp8(RuntimeConfig& cfg) {
    if (!fp32_fp8_autotune_enabled()) {
        cfg.fp32_fp8_autotuned = false;
        return;
    }
    constexpr std::size_t kCount = 1U << 20U;
    constexpr int kIters = 12;
    float* d_fp32 = nullptr;
    std::uint8_t* d_fp8e4 = nullptr;
    std::uint8_t* d_fp8e5 = nullptr;
    float* d_fp32_out = nullptr;
    if (cudaMalloc(&d_fp32, kCount * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_fp8e4, kCount * sizeof(std::uint8_t)) != cudaSuccess ||
        cudaMalloc(&d_fp8e5, kCount * sizeof(std::uint8_t)) != cudaSuccess ||
        cudaMalloc(&d_fp32_out, kCount * sizeof(float)) != cudaSuccess) {
        if (d_fp32 != nullptr) cudaFree(d_fp32);
        if (d_fp8e4 != nullptr) cudaFree(d_fp8e4);
        if (d_fp8e5 != nullptr) cudaFree(d_fp8e5);
        if (d_fp32_out != nullptr) cudaFree(d_fp32_out);
        cfg.fp32_fp8_autotuned = false;
        return;
    }
    cudaMemset(d_fp32, 0x3A, kCount * sizeof(float));
    cudaMemset(d_fp8e4, 0x51, kCount * sizeof(std::uint8_t));
    cudaMemset(d_fp8e5, 0xA3, kCount * sizeof(std::uint8_t));

    const RuntimeConfig* prev_cfg = g_active_runtime_cfg;
    g_active_runtime_cfg = &cfg;
    auto pick_best = [&](const auto& modes, auto&& bench_fn) -> Fp32Fp8Impl {
        constexpr double kSwitchEpsilon = 0.01;  // require >=1% improvement to switch
        Fp32Fp8Impl best = modes[0];
        double best_ms = std::numeric_limits<double>::infinity();
        for (const Fp32Fp8Impl mode : modes) {
            const double ms = bench_fn(mode);
            if (std::isfinite(ms) &&
                (best_ms == std::numeric_limits<double>::infinity() || ms < best_ms * (1.0 - kSwitchEpsilon))) {
                best_ms = ms;
                best = mode;
            }
        }
        return best;
    };

    const std::array<Fp32Fp8Impl, 2> fp32_to_fp8_modes = {
        Fp32Fp8Impl::packed, Fp32Fp8Impl::native
    };
    const std::array<Fp32Fp8Impl, 2> fp8_to_fp32_modes = {
        Fp32Fp8Impl::packed, Fp32Fp8Impl::native
    };

    const Fp32Fp8Impl fp32_to_e4 = pick_best(fp32_to_fp8_modes, [&](Fp32Fp8Impl mode) {
        return benchmark_impl_ms(
            [&]() { return launch_fp32_to_fp8_with_impl<false>(d_fp32, d_fp8e4, kCount, nullptr, mode); }, kIters
        );
    });
    const Fp32Fp8Impl fp32_to_e5 = pick_best(fp32_to_fp8_modes, [&](Fp32Fp8Impl mode) {
        return benchmark_impl_ms(
            [&]() { return launch_fp32_to_fp8_with_impl<true>(d_fp32, d_fp8e5, kCount, nullptr, mode); }, kIters
        );
    });
    const Fp32Fp8Impl e4_to_fp32 = pick_best(fp8_to_fp32_modes, [&](Fp32Fp8Impl mode) {
        return benchmark_impl_ms(
            [&]() { return launch_fp8_to_fp32_with_impl<false>(d_fp8e4, d_fp32_out, kCount, nullptr, mode); }, kIters
        );
    });
    const Fp32Fp8Impl e5_to_fp32 = pick_best(fp8_to_fp32_modes, [&](Fp32Fp8Impl mode) {
        return benchmark_impl_ms(
            [&]() { return launch_fp8_to_fp32_with_impl<true>(d_fp8e5, d_fp32_out, kCount, nullptr, mode); }, kIters
        );
    });

    g_active_runtime_cfg = prev_cfg;
    cudaFree(d_fp32);
    cudaFree(d_fp8e4);
    cudaFree(d_fp8e5);
    cudaFree(d_fp32_out);

    cfg.auto_fp32_to_fp8_impl[0] = fp32_fp8_impl_to_u8(fp32_to_e4);
    cfg.auto_fp32_to_fp8_impl[1] = fp32_fp8_impl_to_u8(fp32_to_e5);
    cfg.auto_fp8_to_fp32_impl[0] = fp32_fp8_impl_to_u8(e4_to_fp32);
    cfg.auto_fp8_to_fp32_impl[1] = fp32_fp8_impl_to_u8(e5_to_fp32);
    cfg.fp32_fp8_autotuned = true;
}

void fill_dispatch(RuntimeConfig& cfg) {
    for (std::size_t i = 0; i < cfg.dispatch.size(); ++i) {
        for (std::size_t j = 0; j < cfg.dispatch[i].size(); ++j) {
            cfg.dispatch[i][j] = unsupported_launch;
        }
    }

    cfg.dispatch[0][0] = [](const void* in, void* out, std::size_t n, cudaStream_t s) {
        return launch_copy(in, out, n, s, sizeof(std::uint16_t));
    };
    cfg.dispatch[1][1] = [](const void* in, void* out, std::size_t n, cudaStream_t s) {
        return launch_copy(in, out, n, s, sizeof(float));
    };
    cfg.dispatch[2][2] = [](const void* in, void* out, std::size_t n, cudaStream_t s) {
        return launch_copy(in, out, n, s, sizeof(double));
    };
    cfg.dispatch[3][3] = [](const void* in, void* out, std::size_t n, cudaStream_t s) {
        return launch_copy(in, out, n, s, sizeof(std::uint8_t));
    };
    cfg.dispatch[4][4] = [](const void* in, void* out, std::size_t n, cudaStream_t s) {
        return launch_copy(in, out, n, s, sizeof(std::uint8_t));
    };

    cfg.dispatch[2][1] = launch_fp64_to_fp32_opt;
    cfg.dispatch[2][0] = launch_fp64_to_fp16_opt;
    cfg.dispatch[2][3] = launch_fp64_to_fp8_opt<false>;
    cfg.dispatch[2][4] = launch_fp64_to_fp8_opt<true>;

    cfg.dispatch[1][2] = launch_fp32_to_fp64_opt;
    cfg.dispatch[1][0] = launch_fp32_to_fp16_opt;
    cfg.dispatch[1][3] = launch_fp32_to_fp8_opt<false>;
    cfg.dispatch[1][4] = launch_fp32_to_fp8_opt<true>;

    cfg.dispatch[0][2] = launch_fp16_to_fp64_opt;
    cfg.dispatch[0][1] = launch_fp16_to_fp32_opt;
    cfg.dispatch[0][3] = launch_fp16_to_fp8_opt<false>;
    cfg.dispatch[0][4] = launch_fp16_to_fp8_opt<true>;

    cfg.dispatch[3][2] = launch_fp8_to_fp64_opt<false>;
    cfg.dispatch[3][1] = launch_fp8_to_fp32_opt<false>;
    cfg.dispatch[3][0] = launch_fp8_to_fp16_opt<false>;
    cfg.dispatch[3][4] = cfg.fp8_lut_ready ? launch_fp8_lut_transform_opt<false>
                                           : launch_kernel<std::uint8_t, std::uint8_t, false, true>;

    cfg.dispatch[4][2] = launch_fp8_to_fp64_opt<true>;
    cfg.dispatch[4][1] = launch_fp8_to_fp32_opt<true>;
    cfg.dispatch[4][0] = launch_fp8_to_fp16_opt<true>;
    cfg.dispatch[4][3] = cfg.fp8_lut_ready ? launch_fp8_lut_transform_opt<true>
                                           : launch_kernel<std::uint8_t, std::uint8_t, true, false>;
}

RuntimeConfig make_runtime_config_for_device(int device) {
    RuntimeConfig cfg;
    for (auto& row : cfg.dispatch) {
        row.fill(nullptr);
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return cfg;
    }

    cfg.available = true;
    cfg.device = device;
    cfg.major = prop.major;
    cfg.minor = prop.minor;
    cfg.sm_count = prop.multiProcessorCount;
    DeviceTuning tuning{};
    tuning.sm_count = std::max(1, prop.multiProcessorCount);
    tuning.major = prop.major;
    tuning.minor = prop.minor;

    const LaunchConfig generic_base = select_launch_cfg(cast_kernel<float, float, false, false>, tuning);
    const LaunchConfig half2_base = select_launch_cfg(fp32_to_fp16_half2_kernel, tuning);
    const LaunchConfig fp8_base = select_launch_cfg(fp32_to_fp8x4_kernel<false>, tuning);
#if OMNICAST_CUDA_HAS_FP8_TYPES
    const LaunchConfig fp8_native_base = select_launch_cfg(fp32_to_fp8x4_native_kernel<false>, tuning);
#else
    const LaunchConfig fp8_native_base = fp8_base;
#endif

    cfg.generic_launch = retune_launch_cfg(generic_base, 0, 1.0f);
    cfg.half2_launch = retune_launch_cfg(half2_base, 256, 1.10f);
    cfg.fp8_launch = retune_launch_cfg(fp8_base, 256, 1.15f);
    cfg.fp8_native_launch = retune_launch_cfg(fp8_native_base, 256, 1.20f);
    cfg.fp8_lut_launch = retune_launch_cfg(fp8_base, 192, 1.30f);
    cfg.fp8_lut_ready = ensure_fp8_luts_initialized();

    cfg.native_fp8 = (prop.major > 8) || (prop.major == 8 && prop.minor >= 9);
    switch (detect_arch_family(prop.major, prop.minor)) {
        case ArchFamily::a100:
            cfg.backend = cfg.native_fp8 ? "a100-fp8-native" : "a100-fp8-emulated";
            break;
        case ArchFamily::h100:
            cfg.backend = cfg.native_fp8 ? "h100-fp8-native" : "h100-fp8-emulated";
            break;
        case ArchFamily::b100:
            cfg.backend = cfg.native_fp8 ? "b100-fp8-native" : "b100-fp8-emulated";
            break;
        case ArchFamily::generic:
        default:
            cfg.backend = cfg.native_fp8 ? "cuda-fp8-native" : "cuda-fp8-emulated";
            break;
    }
    fill_dispatch(cfg);
    autotune_fp16_fp8(cfg);
    autotune_fp32_fp8(cfg);
    return cfg;
}

const RuntimeConfig& get_runtime_config() {
    static RuntimeConfig unavailable_cfg;
    static std::mutex mu;
    static std::map<int, RuntimeConfig> per_device;
    thread_local int cached_device = -1;
    thread_local const RuntimeConfig* cached_cfg = nullptr;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
        return unavailable_cfg;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return unavailable_cfg;
    }

    if (cached_cfg != nullptr && cached_device == device) {
        return *cached_cfg;
    }

    std::lock_guard<std::mutex> lock(mu);
    auto it = per_device.find(device);
    if (it != per_device.end()) {
        cached_device = device;
        cached_cfg = &it->second;
        return *cached_cfg;
    }

    const auto inserted = per_device.emplace(device, make_runtime_config_for_device(device));
    cached_device = device;
    cached_cfg = &inserted.first->second;
    return *cached_cfg;
}

std::size_t dtype_index(GpuDType type) {
    switch (type) {
        case GpuDType::fp16:
            return 0;
        case GpuDType::fp32:
            return 1;
        case GpuDType::fp64:
            return 2;
        case GpuDType::fp8_e4m3:
            return 3;
        case GpuDType::fp8_e5m2:
            return 4;
        default:
            return 5;
    }
}

bool validate_dtype(GpuDType type) {
    return dtype_index(type) < 5;
}

}  // namespace

bool cuda_available() {
    const RuntimeConfig& cfg = get_runtime_config();
    return cfg.available;
}

const char* cuda_backend() {
    static thread_local std::string backend_copy;
    backend_copy = get_runtime_config().backend;
    return backend_copy.c_str();
}

int cuda_device_major() {
    const RuntimeConfig& cfg = get_runtime_config();
    return cfg.major;
}

int cuda_device_minor() {
    const RuntimeConfig& cfg = get_runtime_config();
    return cfg.minor;
}

GpuStatus convert_device(
    const void* input,
    GpuDType src_type,
    void* output,
    GpuDType dst_type,
    std::size_t count,
    cuda_stream_handle_t stream
) {
    const RuntimeConfig& cfg = get_runtime_config();
    if (!cfg.available) {
        return GpuStatus::cuda_unavailable;
    }
    if (!validate_dtype(src_type) || !validate_dtype(dst_type)) {
        return GpuStatus::invalid_argument;
    }
    if (count > 0 && (input == nullptr || output == nullptr)) {
        return GpuStatus::null_pointer;
    }

    const std::size_t src_index = dtype_index(src_type);
    const std::size_t dst_index = dtype_index(dst_type);
    const LaunchFn launch = cfg.dispatch[src_index][dst_index];
    if (launch == nullptr) {
        return GpuStatus::unsupported_conversion;
    }

    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const RuntimeConfig* prev_cfg = g_active_runtime_cfg;
    g_active_runtime_cfg = &cfg;
    const GpuStatus status = launch(input, output, count, cuda_stream);
    g_active_runtime_cfg = prev_cfg;
    return status;
}

}  // namespace omnicast

#endif
