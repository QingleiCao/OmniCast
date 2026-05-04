#include "omnicast/c_api.h"

#include "omnicast/cpu_cast.hpp"
#include "omnicast/gpu_cast.hpp"

namespace {

omnicast::DType to_cpp_dtype(omnicast_dtype t) {
    switch (t) {
        case OMNICAST_DTYPE_FP16:
            return omnicast::DType::fp16;
        case OMNICAST_DTYPE_FP32:
            return omnicast::DType::fp32;
        case OMNICAST_DTYPE_FP64:
        default:
            return omnicast::DType::fp64;
    }
}

omnicast_inplace_status to_c_status(omnicast::InplaceStatus s) {
    switch (s) {
        case omnicast::InplaceStatus::success:
            return OMNICAST_INPLACE_SUCCESS;
        case omnicast::InplaceStatus::null_buffer:
            return OMNICAST_INPLACE_NULL_BUFFER;
        case omnicast::InplaceStatus::insufficient_capacity:
            return OMNICAST_INPLACE_INSUFFICIENT_CAPACITY;
        case omnicast::InplaceStatus::unsupported_conversion:
        default:
            return OMNICAST_INPLACE_UNSUPPORTED_CONVERSION;
    }
}

omnicast::GpuDType to_cpp_gpu_dtype(omnicast_gpu_dtype t) {
    switch (t) {
        case OMNICAST_GPU_DTYPE_FP16:
            return omnicast::GpuDType::fp16;
        case OMNICAST_GPU_DTYPE_FP32:
            return omnicast::GpuDType::fp32;
        case OMNICAST_GPU_DTYPE_FP8_E4M3:
            return omnicast::GpuDType::fp8_e4m3;
        case OMNICAST_GPU_DTYPE_FP8_E5M2:
            return omnicast::GpuDType::fp8_e5m2;
        case OMNICAST_GPU_DTYPE_FP64:
        default:
            return omnicast::GpuDType::fp64;
    }
}

omnicast_gpu_dtype to_c_gpu_dtype(omnicast::GpuDType t) {
    switch (t) {
        case omnicast::GpuDType::fp16:
            return OMNICAST_GPU_DTYPE_FP16;
        case omnicast::GpuDType::fp32:
            return OMNICAST_GPU_DTYPE_FP32;
        case omnicast::GpuDType::fp8_e4m3:
            return OMNICAST_GPU_DTYPE_FP8_E4M3;
        case omnicast::GpuDType::fp8_e5m2:
            return OMNICAST_GPU_DTYPE_FP8_E5M2;
        case omnicast::GpuDType::fp64:
        default:
            return OMNICAST_GPU_DTYPE_FP64;
    }
}

omnicast_gpu_status to_c_gpu_status(omnicast::GpuStatus status) {
    switch (status) {
        case omnicast::GpuStatus::success:
            return OMNICAST_GPU_SUCCESS;
        case omnicast::GpuStatus::cuda_unavailable:
            return OMNICAST_GPU_CUDA_UNAVAILABLE;
        case omnicast::GpuStatus::null_pointer:
            return OMNICAST_GPU_NULL_POINTER;
        case omnicast::GpuStatus::invalid_argument:
            return OMNICAST_GPU_INVALID_ARGUMENT;
        case omnicast::GpuStatus::unsupported_conversion:
            return OMNICAST_GPU_UNSUPPORTED_CONVERSION;
        case omnicast::GpuStatus::unsupported_arch:
            return OMNICAST_GPU_UNSUPPORTED_ARCH;
        case omnicast::GpuStatus::cuda_runtime_error:
        default:
            return OMNICAST_GPU_CUDA_RUNTIME_ERROR;
    }
}

}  // namespace

extern "C" {

float omnicast_fp64_to_fp32(double value) {
    return omnicast::fp64_to_fp32(value);
}

omnicast_fp16_t omnicast_fp64_to_fp16(double value) {
    return omnicast::fp64_to_fp16(value);
}

double omnicast_fp32_to_fp64(float value) {
    return omnicast::fp32_to_fp64(value);
}

omnicast_fp16_t omnicast_fp32_to_fp16(float value) {
    return omnicast::fp32_to_fp16(value);
}

float omnicast_fp16_to_fp32(omnicast_fp16_t value) {
    return omnicast::fp16_to_fp32(value);
}

double omnicast_fp16_to_fp64(omnicast_fp16_t value) {
    return omnicast::fp16_to_fp64(value);
}

int omnicast_simd_available(void) {
    return omnicast::simd_available() ? 1 : 0;
}

const char* omnicast_simd_backend(void) {
    return omnicast::simd_backend();
}

void omnicast_convert_fp64_to_fp32(const double* input, float* output, size_t count) {
    omnicast::convert_fp64_to_fp32(input, output, count);
}

void omnicast_convert_fp64_to_fp16(const double* input, omnicast_fp16_t* output, size_t count) {
    omnicast::convert_fp64_to_fp16(input, output, count);
}

void omnicast_convert_fp32_to_fp64(const float* input, double* output, size_t count) {
    omnicast::convert_fp32_to_fp64(input, output, count);
}

void omnicast_convert_fp32_to_fp16(const float* input, omnicast_fp16_t* output, size_t count) {
    omnicast::convert_fp32_to_fp16(input, output, count);
}

void omnicast_convert_fp16_to_fp32(const omnicast_fp16_t* input, float* output, size_t count) {
    omnicast::convert_fp16_to_fp32(input, output, count);
}

void omnicast_convert_fp16_to_fp64(const omnicast_fp16_t* input, double* output, size_t count) {
    omnicast::convert_fp16_to_fp64(input, output, count);
}

void omnicast_convert_fp64_to_fp32_scalar(const double* input, float* output, size_t count) {
    omnicast::convert_fp64_to_fp32_scalar(input, output, count);
}

void omnicast_convert_fp64_to_fp16_scalar(const double* input, omnicast_fp16_t* output, size_t count) {
    omnicast::convert_fp64_to_fp16_scalar(input, output, count);
}

void omnicast_convert_fp32_to_fp64_scalar(const float* input, double* output, size_t count) {
    omnicast::convert_fp32_to_fp64_scalar(input, output, count);
}

void omnicast_convert_fp32_to_fp16_scalar(const float* input, omnicast_fp16_t* output, size_t count) {
    omnicast::convert_fp32_to_fp16_scalar(input, output, count);
}

void omnicast_convert_fp16_to_fp32_scalar(const omnicast_fp16_t* input, float* output, size_t count) {
    omnicast::convert_fp16_to_fp32_scalar(input, output, count);
}

void omnicast_convert_fp16_to_fp64_scalar(const omnicast_fp16_t* input, double* output, size_t count) {
    omnicast::convert_fp16_to_fp64_scalar(input, output, count);
}

omnicast_inplace_status omnicast_convert_inplace(
    void* buffer,
    size_t count,
    omnicast_dtype src_type,
    omnicast_dtype dst_type,
    size_t capacity_bytes
) {
    return to_c_status(
        omnicast::convert_inplace(
            buffer,
            count,
            to_cpp_dtype(src_type),
            to_cpp_dtype(dst_type),
            capacity_bytes
        )
    );
}

int omnicast_cuda_available(void) {
    return omnicast::cuda_available() ? 1 : 0;
}

const char* omnicast_cuda_backend(void) {
    return omnicast::cuda_backend();
}

int omnicast_cuda_device_major(void) {
    return omnicast::cuda_device_major();
}

int omnicast_cuda_device_minor(void) {
    return omnicast::cuda_device_minor();
}

omnicast_gpu_dtype omnicast_gpu_dtype_from_fp8_format(omnicast_fp8_format format) {
    const omnicast::Fp8Format cpp_format =
        (format == OMNICAST_FP8_E5M2) ? omnicast::Fp8Format::e5m2 : omnicast::Fp8Format::e4m3;
    return to_c_gpu_dtype(omnicast::fp8_format_to_dtype(cpp_format));
}

omnicast_gpu_status omnicast_convert_device(
    const void* input,
    omnicast_gpu_dtype src_type,
    void* output,
    omnicast_gpu_dtype dst_type,
    size_t count,
    omnicast_cuda_stream stream
) {
    return to_c_gpu_status(
        omnicast::convert_device(input, to_cpp_gpu_dtype(src_type), output, to_cpp_gpu_dtype(dst_type), count, stream)
    );
}

}  // extern "C"
