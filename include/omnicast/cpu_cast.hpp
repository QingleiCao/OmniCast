#ifndef OMNICAST_CPU_CAST_HPP
#define OMNICAST_CPU_CAST_HPP

#include <cstddef>
#include <cstdint>

namespace omnicast {

using fp16_t = std::uint16_t;

enum class DType : std::uint8_t {
    fp16 = 0,
    fp32 = 1,
    fp64 = 2,
};

enum class InplaceStatus : std::uint8_t {
    success = 0,
    null_buffer = 1,
    insufficient_capacity = 2,
    unsupported_conversion = 3,
};

float fp64_to_fp32(double value);
fp16_t fp64_to_fp16(double value);

double fp32_to_fp64(float value);
fp16_t fp32_to_fp16(float value);

float fp16_to_fp32(fp16_t value);
double fp16_to_fp64(fp16_t value);

// Alias names to match "PF32/PF64" phrasing if needed.
inline float fp16_to_pf32(fp16_t value) { return fp16_to_fp32(value); }
inline double fp16_to_pf64(fp16_t value) { return fp16_to_fp64(value); }

bool simd_available();
const char* simd_backend();

void convert_fp64_to_fp32_scalar(const double* input, float* output, std::size_t count);
void convert_fp64_to_fp16_scalar(const double* input, fp16_t* output, std::size_t count);
void convert_fp32_to_fp64_scalar(const float* input, double* output, std::size_t count);
void convert_fp32_to_fp16_scalar(const float* input, fp16_t* output, std::size_t count);
void convert_fp16_to_fp32_scalar(const fp16_t* input, float* output, std::size_t count);
void convert_fp16_to_fp64_scalar(const fp16_t* input, double* output, std::size_t count);

void convert_fp64_to_fp32(const double* input, float* output, std::size_t count);
void convert_fp64_to_fp16(const double* input, fp16_t* output, std::size_t count);
void convert_fp32_to_fp64(const float* input, double* output, std::size_t count);
void convert_fp32_to_fp16(const float* input, fp16_t* output, std::size_t count);
void convert_fp16_to_fp32(const fp16_t* input, float* output, std::size_t count);
void convert_fp16_to_fp64(const fp16_t* input, double* output, std::size_t count);

InplaceStatus convert_inplace(
    void* buffer,
    std::size_t count,
    DType src_type,
    DType dst_type,
    std::size_t capacity_bytes
);

}  // namespace omnicast

#endif  // OMNICAST_CPU_CAST_HPP
