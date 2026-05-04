#ifndef OMNICAST_C_API_H
#define OMNICAST_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint16_t omnicast_fp16_t;

typedef enum omnicast_dtype {
    OMNICAST_DTYPE_FP16 = 0,
    OMNICAST_DTYPE_FP32 = 1,
    OMNICAST_DTYPE_FP64 = 2
} omnicast_dtype;

typedef enum omnicast_gpu_dtype {
    OMNICAST_GPU_DTYPE_FP16 = 0,
    OMNICAST_GPU_DTYPE_FP32 = 1,
    OMNICAST_GPU_DTYPE_FP64 = 2,
    OMNICAST_GPU_DTYPE_FP8_E4M3 = 3,
    OMNICAST_GPU_DTYPE_FP8_E5M2 = 4
} omnicast_gpu_dtype;

typedef enum omnicast_fp8_format {
    OMNICAST_FP8_E4M3 = 0,
    OMNICAST_FP8_E5M2 = 1
} omnicast_fp8_format;

typedef enum omnicast_gpu_status {
    OMNICAST_GPU_SUCCESS = 0,
    OMNICAST_GPU_CUDA_UNAVAILABLE = 1,
    OMNICAST_GPU_NULL_POINTER = 2,
    OMNICAST_GPU_INVALID_ARGUMENT = 3,
    OMNICAST_GPU_UNSUPPORTED_CONVERSION = 4,
    OMNICAST_GPU_UNSUPPORTED_ARCH = 5,
    OMNICAST_GPU_CUDA_RUNTIME_ERROR = 6
} omnicast_gpu_status;

typedef void* omnicast_cuda_stream;

typedef enum omnicast_inplace_status {
    OMNICAST_INPLACE_SUCCESS = 0,
    OMNICAST_INPLACE_NULL_BUFFER = 1,
    OMNICAST_INPLACE_INSUFFICIENT_CAPACITY = 2,
    OMNICAST_INPLACE_UNSUPPORTED_CONVERSION = 3
} omnicast_inplace_status;

float omnicast_fp64_to_fp32(double value);
omnicast_fp16_t omnicast_fp64_to_fp16(double value);
double omnicast_fp32_to_fp64(float value);
omnicast_fp16_t omnicast_fp32_to_fp16(float value);
float omnicast_fp16_to_fp32(omnicast_fp16_t value);
double omnicast_fp16_to_fp64(omnicast_fp16_t value);

int omnicast_simd_available(void);
const char* omnicast_simd_backend(void);

void omnicast_convert_fp64_to_fp32(const double* input, float* output, size_t count);
void omnicast_convert_fp64_to_fp16(const double* input, omnicast_fp16_t* output, size_t count);
void omnicast_convert_fp32_to_fp64(const float* input, double* output, size_t count);
void omnicast_convert_fp32_to_fp16(const float* input, omnicast_fp16_t* output, size_t count);
void omnicast_convert_fp16_to_fp32(const omnicast_fp16_t* input, float* output, size_t count);
void omnicast_convert_fp16_to_fp64(const omnicast_fp16_t* input, double* output, size_t count);

void omnicast_convert_fp64_to_fp32_scalar(const double* input, float* output, size_t count);
void omnicast_convert_fp64_to_fp16_scalar(const double* input, omnicast_fp16_t* output, size_t count);
void omnicast_convert_fp32_to_fp64_scalar(const float* input, double* output, size_t count);
void omnicast_convert_fp32_to_fp16_scalar(const float* input, omnicast_fp16_t* output, size_t count);
void omnicast_convert_fp16_to_fp32_scalar(const omnicast_fp16_t* input, float* output, size_t count);
void omnicast_convert_fp16_to_fp64_scalar(const omnicast_fp16_t* input, double* output, size_t count);

omnicast_inplace_status omnicast_convert_inplace(
    void* buffer,
    size_t count,
    omnicast_dtype src_type,
    omnicast_dtype dst_type,
    size_t capacity_bytes
);

int omnicast_cuda_available(void);
const char* omnicast_cuda_backend(void);
int omnicast_cuda_device_major(void);
int omnicast_cuda_device_minor(void);

omnicast_gpu_dtype omnicast_gpu_dtype_from_fp8_format(omnicast_fp8_format format);

omnicast_gpu_status omnicast_convert_device(
    const void* input,
    omnicast_gpu_dtype src_type,
    void* output,
    omnicast_gpu_dtype dst_type,
    size_t count,
    omnicast_cuda_stream stream
);

#ifdef __cplusplus
}
#endif

#endif  // OMNICAST_C_API_H
