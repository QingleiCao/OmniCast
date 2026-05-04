#ifndef OMNICAST_GPU_CAST_HPP
#define OMNICAST_GPU_CAST_HPP

#include <cstddef>
#include <cstdint>

namespace omnicast {

using fp16_t = std::uint16_t;
using fp8_t = std::uint8_t;
using cuda_stream_handle_t = void*;

enum class GpuDType : std::uint8_t {
    fp16 = 0,
    fp32 = 1,
    fp64 = 2,
    fp8_e4m3 = 3,
    fp8_e5m2 = 4,
};

enum class Fp8Format : std::uint8_t {
    e4m3 = 0,
    e5m2 = 1,
};

enum class GpuStatus : std::uint8_t {
    success = 0,
    cuda_unavailable = 1,
    null_pointer = 2,
    invalid_argument = 3,
    unsupported_conversion = 4,
    unsupported_arch = 5,
    cuda_runtime_error = 6,
};

bool cuda_available();
const char* cuda_backend();
int cuda_device_major();
int cuda_device_minor();

GpuStatus convert_device(
    const void* input,
    GpuDType src_type,
    void* output,
    GpuDType dst_type,
    std::size_t count,
    cuda_stream_handle_t stream
);

inline GpuDType fp8_format_to_dtype(Fp8Format format) {
    return (format == Fp8Format::e5m2) ? GpuDType::fp8_e5m2 : GpuDType::fp8_e4m3;
}

}  // namespace omnicast

#endif  // OMNICAST_GPU_CAST_HPP
