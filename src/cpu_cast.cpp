#include "omnicast/cpu_cast.hpp"

#include <cstdlib>
#include <cstring>

#if defined(OMNICAST_ENABLE_CPU_SIMD) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#endif

namespace omnicast {
namespace {

std::uint32_t bit_cast_u32(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(float));
    return bits;
}

float bit_cast_f32(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
}

fp16_t float_to_half(float value) {
    const std::uint32_t x = bit_cast_u32(value);
    const std::uint32_t sign = (x >> 16U) & 0x8000U;
    std::uint32_t mantissa = x & 0x007FFFFFU;
    const std::int32_t exp = static_cast<std::int32_t>((x >> 23U) & 0xFFU) - 127;

    if (exp > 15) {
        if (((x >> 23U) & 0xFFU) == 0xFFU && mantissa != 0) {
            return static_cast<fp16_t>(sign | 0x7E00U);
        }
        return static_cast<fp16_t>(sign | 0x7C00U);
    }

    if (exp < -14) {
        if (exp < -24) {
            return static_cast<fp16_t>(sign);
        }

        mantissa |= 0x00800000U;
        const std::uint32_t shift = static_cast<std::uint32_t>(-exp - 1);
        std::uint32_t half_mantissa = mantissa >> (shift + 13U);
        const std::uint32_t round_bit = (mantissa >> (shift + 12U)) & 0x1U;
        const std::uint32_t sticky = mantissa & ((1U << (shift + 12U)) - 1U);
        if (round_bit && (sticky || (half_mantissa & 0x1U))) {
            ++half_mantissa;
        }
        return static_cast<fp16_t>(sign | half_mantissa);
    }

    std::uint32_t half_exp = static_cast<std::uint32_t>(exp + 15);
    std::uint32_t half_mantissa = mantissa >> 13U;
    const std::uint32_t round_bit = (mantissa >> 12U) & 0x1U;
    const std::uint32_t sticky = mantissa & 0xFFFU;

    if (round_bit && (sticky || (half_mantissa & 0x1U))) {
        ++half_mantissa;
        if (half_mantissa == 0x400U) {
            half_mantissa = 0U;
            ++half_exp;
            if (half_exp >= 0x1FU) {
                return static_cast<fp16_t>(sign | 0x7C00U);
            }
        }
    }

    return static_cast<fp16_t>(sign | (half_exp << 10U) | half_mantissa);
}

float half_to_float(fp16_t value) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(value & 0x8000U)) << 16U;
    std::uint32_t exp = (value >> 10U) & 0x1FU;
    std::uint32_t mantissa = value & 0x03FFU;

    std::uint32_t out = 0;
    if (exp == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --exp;
            }
            mantissa &= 0x03FFU;
            const std::uint32_t exp_f32 = exp + (127U - 15U);
            out = sign | (exp_f32 << 23U) | (mantissa << 13U);
        }
    } else if (exp == 0x1FU) {
        out = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        const std::uint32_t exp_f32 = exp + (127U - 15U);
        out = sign | (exp_f32 << 23U) | (mantissa << 13U);
    }

    return bit_cast_f32(out);
}

#if defined(OMNICAST_ENABLE_CPU_SIMD) && OMNICAST_ENABLE_CPU_SIMD && \
    (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#define OMNICAST_CAN_BUILD_X86_SIMD 1
#define OMNICAST_TARGET_AVX __attribute__((target("avx,f16c")))
#define OMNICAST_TARGET_AVX2 __attribute__((target("avx2,f16c")))
#define OMNICAST_TARGET_AVX512 __attribute__((target("avx512f,avx2,f16c")))
#else
#define OMNICAST_CAN_BUILD_X86_SIMD 0
#define OMNICAST_TARGET_AVX
#define OMNICAST_TARGET_AVX2
#define OMNICAST_TARGET_AVX512
#endif

enum class CpuBackend { scalar, avx_f16c, avx2_f16c, avx512_f16c };

bool env_truthy(const char* name) {
    const char* env = std::getenv(name);
    return env != nullptr && env[0] != '\0' && env[0] != '0';
}

std::size_t dtype_size(DType t) {
    switch (t) {
        case DType::fp16:
            return sizeof(fp16_t);
        case DType::fp32:
            return sizeof(float);
        case DType::fp64:
            return sizeof(double);
        default:
            return 0;
    }
}

void convert_fp64_to_fp32_scalar_impl(const double* input, float* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = static_cast<float>(input[i]);
    }
}

void convert_fp64_to_fp16_scalar_impl(const double* input, fp16_t* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = float_to_half(static_cast<float>(input[i]));
    }
}

void convert_fp32_to_fp64_scalar_impl(const float* input, double* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = static_cast<double>(input[i]);
    }
}

void convert_fp32_to_fp16_scalar_impl(const float* input, fp16_t* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = float_to_half(input[i]);
    }
}

void convert_fp16_to_fp32_scalar_impl(const fp16_t* input, float* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = half_to_float(input[i]);
    }
}

void convert_fp16_to_fp64_scalar_impl(const fp16_t* input, double* output, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        output[i] = static_cast<double>(half_to_float(input[i]));
    }
}

#if OMNICAST_CAN_BUILD_X86_SIMD
OMNICAST_TARGET_AVX void convert_fp64_to_fp32_avx(const double* input, float* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const __m256d v = _mm256_loadu_pd(input + i);
        const __m128 out = _mm256_cvtpd_ps(v);
        _mm_storeu_ps(output + i, out);
    }
    for (; i < count; ++i) {
        output[i] = static_cast<float>(input[i]);
    }
}

OMNICAST_TARGET_AVX void convert_fp64_to_fp16_avx(const double* input, fp16_t* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const __m256d v64 = _mm256_loadu_pd(input + i);
        const __m128 v32 = _mm256_cvtpd_ps(v64);
        const __m128i h = _mm_cvtps_ph(v32, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + i), h);
    }
    for (; i < count; ++i) {
        output[i] = float_to_half(static_cast<float>(input[i]));
    }
}

OMNICAST_TARGET_AVX void convert_fp32_to_fp64_avx(const float* input, double* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const __m128 v = _mm_loadu_ps(input + i);
        const __m256d out = _mm256_cvtps_pd(v);
        _mm256_storeu_pd(output + i, out);
    }
    for (; i < count; ++i) {
        output[i] = static_cast<double>(input[i]);
    }
}

OMNICAST_TARGET_AVX void convert_fp32_to_fp16_avx(const float* input, fp16_t* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256 v = _mm256_loadu_ps(input + i);
        const __m128i out = _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), out);
    }
    for (; i < count; ++i) {
        output[i] = float_to_half(input[i]);
    }
}

OMNICAST_TARGET_AVX void convert_fp16_to_fp32_avx(const fp16_t* input, float* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        const __m256 out = _mm256_cvtph_ps(v);
        _mm256_storeu_ps(output + i, out);
    }
    for (; i < count; ++i) {
        output[i] = half_to_float(input[i]);
    }
}

OMNICAST_TARGET_AVX void convert_fp16_to_fp64_avx(const fp16_t* input, double* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        std::uint64_t packed = 0;
        std::memcpy(&packed, input + i, sizeof(std::uint64_t));
        const __m128i h = _mm_cvtsi64_si128(static_cast<long long>(packed));
        const __m128 f = _mm_cvtph_ps(h);
        const __m256d out = _mm256_cvtps_pd(f);
        _mm256_storeu_pd(output + i, out);
    }
    for (; i < count; ++i) {
        output[i] = static_cast<double>(half_to_float(input[i]));
    }
}

OMNICAST_TARGET_AVX2 void convert_fp64_to_fp32_avx2(const double* input, float* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256d v0 = _mm256_loadu_pd(input + i);
        const __m256d v1 = _mm256_loadu_pd(input + i + 4);
        const __m128 out0 = _mm256_cvtpd_ps(v0);
        const __m128 out1 = _mm256_cvtpd_ps(v1);
        _mm_storeu_ps(output + i, out0);
        _mm_storeu_ps(output + i + 4, out1);
    }
    for (; i < count; ++i) {
        output[i] = static_cast<float>(input[i]);
    }
}

OMNICAST_TARGET_AVX2 void convert_fp64_to_fp16_avx2(const double* input, fp16_t* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m256d v64_0 = _mm256_loadu_pd(input + i);
        const __m256d v64_1 = _mm256_loadu_pd(input + i + 4);
        const __m128 v32_0 = _mm256_cvtpd_ps(v64_0);
        const __m128 v32_1 = _mm256_cvtpd_ps(v64_1);
        const __m128i h0 = _mm_cvtps_ph(v32_0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m128i h1 = _mm_cvtps_ph(v32_1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + i), h0);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(output + i + 4), h1);
    }
    for (; i < count; ++i) {
        output[i] = float_to_half(static_cast<float>(input[i]));
    }
}

OMNICAST_TARGET_AVX2 void convert_fp32_to_fp64_avx2(const float* input, double* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        const __m128 v0 = _mm_loadu_ps(input + i);
        const __m128 v1 = _mm_loadu_ps(input + i + 4);
        const __m256d out0 = _mm256_cvtps_pd(v0);
        const __m256d out1 = _mm256_cvtps_pd(v1);
        _mm256_storeu_pd(output + i, out0);
        _mm256_storeu_pd(output + i + 4, out1);
    }
    for (; i < count; ++i) {
        output[i] = static_cast<double>(input[i]);
    }
}

OMNICAST_TARGET_AVX2 void convert_fp32_to_fp16_avx2(const float* input, fp16_t* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256 v0 = _mm256_loadu_ps(input + i);
        const __m256 v1 = _mm256_loadu_ps(input + i + 8);
        const __m128i h0 = _mm256_cvtps_ph(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        const __m128i h1 = _mm256_cvtps_ph(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), h0);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i + 8), h1);
    }
    for (; i < count; ++i) {
        output[i] = float_to_half(input[i]);
    }
}

OMNICAST_TARGET_AVX2 void convert_fp16_to_fp32_avx2(const fp16_t* input, float* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        const __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i + 8));
        const __m256 out0 = _mm256_cvtph_ps(h0);
        const __m256 out1 = _mm256_cvtph_ps(h1);
        _mm256_storeu_ps(output + i, out0);
        _mm256_storeu_ps(output + i + 8, out1);
    }
    for (; i < count; ++i) {
        output[i] = half_to_float(input[i]);
    }
}

OMNICAST_TARGET_AVX2 void convert_fp16_to_fp64_avx2(const fp16_t* input, double* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        std::uint64_t packed0 = 0;
        std::uint64_t packed1 = 0;
        std::memcpy(&packed0, input + i, sizeof(std::uint64_t));
        std::memcpy(&packed1, input + i + 4, sizeof(std::uint64_t));
        const __m128i h0 = _mm_cvtsi64_si128(static_cast<long long>(packed0));
        const __m128i h1 = _mm_cvtsi64_si128(static_cast<long long>(packed1));
        const __m128 f0 = _mm_cvtph_ps(h0);
        const __m128 f1 = _mm_cvtph_ps(h1);
        const __m256d out0 = _mm256_cvtps_pd(f0);
        const __m256d out1 = _mm256_cvtps_pd(f1);
        _mm256_storeu_pd(output + i, out0);
        _mm256_storeu_pd(output + i + 4, out1);
    }
    for (; i < count; ++i) {
        output[i] = static_cast<double>(half_to_float(input[i]));
    }
}

OMNICAST_TARGET_AVX512 void convert_fp64_to_fp32_avx512(const double* input, float* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m512d v0 = _mm512_loadu_pd(input + i);
        const __m512d v1 = _mm512_loadu_pd(input + i + 8);
        const __m256 out0 = _mm512_cvtpd_ps(v0);
        const __m256 out1 = _mm512_cvtpd_ps(v1);
        _mm256_storeu_ps(output + i, out0);
        _mm256_storeu_ps(output + i + 8, out1);
    }
    for (; i + 8 <= count; i += 8) {
        const __m512d v = _mm512_loadu_pd(input + i);
        const __m256 out = _mm512_cvtpd_ps(v);
        _mm256_storeu_ps(output + i, out);
    }
    for (; i < count; ++i) {
        output[i] = static_cast<float>(input[i]);
    }
}

OMNICAST_TARGET_AVX512 void convert_fp32_to_fp64_avx512(const float* input, double* output, std::size_t count) {
    std::size_t i = 0;
    for (; i + 16 <= count; i += 16) {
        const __m256 v0 = _mm256_loadu_ps(input + i);
        const __m256 v1 = _mm256_loadu_ps(input + i + 8);
        const __m512d out0 = _mm512_cvtps_pd(v0);
        const __m512d out1 = _mm512_cvtps_pd(v1);
        _mm512_storeu_pd(output + i, out0);
        _mm512_storeu_pd(output + i + 8, out1);
    }
    for (; i + 8 <= count; i += 8) {
        const __m256 v = _mm256_loadu_ps(input + i);
        const __m512d out = _mm512_cvtps_pd(v);
        _mm512_storeu_pd(output + i, out);
    }
    for (; i < count; ++i) {
        output[i] = static_cast<double>(input[i]);
    }
}
#endif

using F64ToF32Fn = void (*)(const double*, float*, std::size_t);
using F64ToF16Fn = void (*)(const double*, fp16_t*, std::size_t);
using F32ToF64Fn = void (*)(const float*, double*, std::size_t);
using F32ToF16Fn = void (*)(const float*, fp16_t*, std::size_t);
using F16ToF32Fn = void (*)(const fp16_t*, float*, std::size_t);
using F16ToF64Fn = void (*)(const fp16_t*, double*, std::size_t);

struct DispatchTable {
    CpuBackend backend = CpuBackend::scalar;
    bool force_scalar = false;
    bool force_simd = false;
    F64ToF32Fn fp64_to_fp32 = convert_fp64_to_fp32_scalar_impl;
    F64ToF16Fn fp64_to_fp16 = convert_fp64_to_fp16_scalar_impl;
    F32ToF64Fn fp32_to_fp64 = convert_fp32_to_fp64_scalar_impl;
    F32ToF16Fn fp32_to_fp16 = convert_fp32_to_fp16_scalar_impl;
    F16ToF32Fn fp16_to_fp32 = convert_fp16_to_fp32_scalar_impl;
    F16ToF64Fn fp16_to_fp64 = convert_fp16_to_fp64_scalar_impl;
};

DispatchTable make_dispatch() {
    DispatchTable d{};
    d.force_scalar = env_truthy("OMNICAST_FORCE_SCALAR");
    d.force_simd = env_truthy("OMNICAST_FORCE_SIMD");

#if OMNICAST_CAN_BUILD_X86_SIMD
    __builtin_cpu_init();
    const bool has_avx = __builtin_cpu_supports("avx");
    const bool has_avx2 = __builtin_cpu_supports("avx2");
    const bool has_avx512f = __builtin_cpu_supports("avx512f");
    const bool has_f16c = __builtin_cpu_supports("f16c");

    if (!d.force_scalar && has_f16c && has_avx512f) {
        d.backend = CpuBackend::avx512_f16c;
        d.fp64_to_fp32 = convert_fp64_to_fp32_avx512;
        d.fp64_to_fp16 = convert_fp64_to_fp16_avx2;
        d.fp32_to_fp64 = convert_fp32_to_fp64_avx512;
        d.fp32_to_fp16 = convert_fp32_to_fp16_avx2;
        d.fp16_to_fp32 = convert_fp16_to_fp32_avx2;
        d.fp16_to_fp64 = convert_fp16_to_fp64_avx2;
    } else if (!d.force_scalar && has_f16c && has_avx2) {
        d.backend = CpuBackend::avx2_f16c;
        d.fp64_to_fp32 = convert_fp64_to_fp32_avx2;
        d.fp64_to_fp16 = convert_fp64_to_fp16_avx2;
        d.fp32_to_fp64 = convert_fp32_to_fp64_avx2;
        d.fp32_to_fp16 = convert_fp32_to_fp16_avx2;
        d.fp16_to_fp32 = convert_fp16_to_fp32_avx2;
        d.fp16_to_fp64 = convert_fp16_to_fp64_avx2;
    } else if (!d.force_scalar && has_f16c && has_avx) {
        d.backend = CpuBackend::avx_f16c;
        d.fp64_to_fp32 = convert_fp64_to_fp32_avx;
        d.fp64_to_fp16 = convert_fp64_to_fp16_avx;
        d.fp32_to_fp64 = convert_fp32_to_fp64_avx;
        d.fp32_to_fp16 = convert_fp32_to_fp16_avx;
        d.fp16_to_fp32 = convert_fp16_to_fp32_avx;
        d.fp16_to_fp64 = convert_fp16_to_fp64_avx;
    }
#endif
    return d;
}

const DispatchTable& dispatch() {
    static const DispatchTable table = make_dispatch();
    return table;
}

const char* runtime_backend_name() {
    const DispatchTable& d = dispatch();
    if (d.force_scalar) {
        return "scalar(forced)";
    }
    if (d.force_simd && d.backend == CpuBackend::scalar) {
        return "scalar(force-simd-unavailable)";
    }

    switch (d.backend) {
        case CpuBackend::avx512_f16c:
            return d.force_simd ? "AVX512+F16C(forced)" : "AVX512+F16C(runtime)";
        case CpuBackend::avx2_f16c:
            return d.force_simd ? "AVX2+F16C(forced)" : "AVX2+F16C(runtime)";
        case CpuBackend::avx_f16c:
            return d.force_simd ? "AVX+F16C(forced)" : "AVX+F16C(runtime)";
        case CpuBackend::scalar:
        default:
            return "scalar";
    }
}

InplaceStatus convert_expand_backward(
    unsigned char* bytes,
    std::size_t count,
    DType src_type,
    DType dst_type
) {
    constexpr std::size_t kInplaceChunk = 1024;

    if (src_type == DType::fp16 && dst_type == DType::fp32) {
        std::size_t end = count;
        while (end > 0) {
            const std::size_t chunk = end > kInplaceChunk ? kInplaceChunk : end;
            const std::size_t begin = end - chunk;
            fp16_t src_chunk[kInplaceChunk];
            std::memcpy(src_chunk, bytes + begin * sizeof(fp16_t), chunk * sizeof(fp16_t));
            convert_fp16_to_fp32(src_chunk, reinterpret_cast<float*>(bytes) + begin, chunk);
            end = begin;
        }
        return InplaceStatus::success;
    }
    if (src_type == DType::fp16 && dst_type == DType::fp64) {
        std::size_t end = count;
        while (end > 0) {
            const std::size_t chunk = end > kInplaceChunk ? kInplaceChunk : end;
            const std::size_t begin = end - chunk;
            fp16_t src_chunk[kInplaceChunk];
            std::memcpy(src_chunk, bytes + begin * sizeof(fp16_t), chunk * sizeof(fp16_t));
            convert_fp16_to_fp64(src_chunk, reinterpret_cast<double*>(bytes) + begin, chunk);
            end = begin;
        }
        return InplaceStatus::success;
    }
    if (src_type == DType::fp32 && dst_type == DType::fp64) {
        std::size_t end = count;
        while (end > 0) {
            const std::size_t chunk = end > kInplaceChunk ? kInplaceChunk : end;
            const std::size_t begin = end - chunk;
            float src_chunk[kInplaceChunk];
            std::memcpy(src_chunk, bytes + begin * sizeof(float), chunk * sizeof(float));
            convert_fp32_to_fp64(src_chunk, reinterpret_cast<double*>(bytes) + begin, chunk);
            end = begin;
        }
        return InplaceStatus::success;
    }

    return InplaceStatus::unsupported_conversion;
}

}  // namespace

float fp64_to_fp32(double value) {
    return static_cast<float>(value);
}

fp16_t fp64_to_fp16(double value) {
    return float_to_half(static_cast<float>(value));
}

double fp32_to_fp64(float value) {
    return static_cast<double>(value);
}

fp16_t fp32_to_fp16(float value) {
    return float_to_half(value);
}

float fp16_to_fp32(fp16_t value) {
    return half_to_float(value);
}

double fp16_to_fp64(fp16_t value) {
    return static_cast<double>(half_to_float(value));
}

bool simd_available() {
    return dispatch().backend != CpuBackend::scalar;
}

const char* simd_backend() {
    return runtime_backend_name();
}

void convert_fp64_to_fp32_scalar(const double* input, float* output, std::size_t count) {
    convert_fp64_to_fp32_scalar_impl(input, output, count);
}

void convert_fp64_to_fp16_scalar(const double* input, fp16_t* output, std::size_t count) {
    convert_fp64_to_fp16_scalar_impl(input, output, count);
}

void convert_fp32_to_fp64_scalar(const float* input, double* output, std::size_t count) {
    convert_fp32_to_fp64_scalar_impl(input, output, count);
}

void convert_fp32_to_fp16_scalar(const float* input, fp16_t* output, std::size_t count) {
    convert_fp32_to_fp16_scalar_impl(input, output, count);
}

void convert_fp16_to_fp32_scalar(const fp16_t* input, float* output, std::size_t count) {
    convert_fp16_to_fp32_scalar_impl(input, output, count);
}

void convert_fp16_to_fp64_scalar(const fp16_t* input, double* output, std::size_t count) {
    convert_fp16_to_fp64_scalar_impl(input, output, count);
}

void convert_fp64_to_fp32(const double* input, float* output, std::size_t count) {
    dispatch().fp64_to_fp32(input, output, count);
}

void convert_fp64_to_fp16(const double* input, fp16_t* output, std::size_t count) {
    dispatch().fp64_to_fp16(input, output, count);
}

void convert_fp32_to_fp64(const float* input, double* output, std::size_t count) {
    dispatch().fp32_to_fp64(input, output, count);
}

void convert_fp32_to_fp16(const float* input, fp16_t* output, std::size_t count) {
    dispatch().fp32_to_fp16(input, output, count);
}

void convert_fp16_to_fp32(const fp16_t* input, float* output, std::size_t count) {
    dispatch().fp16_to_fp32(input, output, count);
}

void convert_fp16_to_fp64(const fp16_t* input, double* output, std::size_t count) {
    dispatch().fp16_to_fp64(input, output, count);
}

InplaceStatus convert_inplace(
    void* buffer,
    std::size_t count,
    DType src_type,
    DType dst_type,
    std::size_t capacity_bytes
) {
    if (count == 0) {
        return InplaceStatus::success;
    }
    if (buffer == nullptr) {
        return InplaceStatus::null_buffer;
    }

    const std::size_t src_size = dtype_size(src_type);
    const std::size_t dst_size = dtype_size(dst_type);
    if (src_size == 0 || dst_size == 0) {
        return InplaceStatus::unsupported_conversion;
    }

    if (src_type == dst_type) {
        return InplaceStatus::success;
    }

    const std::size_t required_bytes = count * dst_size;
    if (required_bytes > capacity_bytes) {
        return InplaceStatus::insufficient_capacity;
    }

    if (dst_size < src_size) {
        if (src_type == DType::fp64 && dst_type == DType::fp32) {
            convert_fp64_to_fp32(reinterpret_cast<const double*>(buffer), reinterpret_cast<float*>(buffer), count);
            return InplaceStatus::success;
        }
        if (src_type == DType::fp64 && dst_type == DType::fp16) {
            convert_fp64_to_fp16(reinterpret_cast<const double*>(buffer), reinterpret_cast<fp16_t*>(buffer), count);
            return InplaceStatus::success;
        }
        if (src_type == DType::fp32 && dst_type == DType::fp16) {
            convert_fp32_to_fp16(reinterpret_cast<const float*>(buffer), reinterpret_cast<fp16_t*>(buffer), count);
            return InplaceStatus::success;
        }
        return InplaceStatus::unsupported_conversion;
    }

    return convert_expand_backward(reinterpret_cast<unsigned char*>(buffer), count, src_type, dst_type);
}

}  // namespace omnicast
