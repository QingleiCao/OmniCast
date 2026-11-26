/**
 * @file gpu_rt.hpp
 * @brief CUDA/HIP portability layer and the launch interface between the
 * public GPU API (gpu_api.cu) and the per-source-type kernel units
 * (gpu_inst_*.cu).
 */
#ifndef OMC_GPU_RT_HPP
#define OMC_GPU_RT_HPP

#include <stdint.h>
#include <stddef.h>
#include "omnicast/omnicast.h"

#if defined(OMC_BACKEND_HIP)
#include <hip/hip_runtime.h>
typedef hipStream_t gpuStream_t;
typedef hipError_t  gpuError_t;
#define gpuSuccess               hipSuccess
#define gpuErrorNotReady         hipErrorNotReady
#define gpuGetLastError          hipGetLastError
#define gpuPeekAtLastError       hipPeekAtLastError
#define gpuGetErrorName          hipGetErrorName
#define gpuGetErrorString        hipGetErrorString
#define OMC_RT                   "hip"     /* prefix of runtime call names in messages */
#define gpuMemcpyAsync           hipMemcpyAsync
#define gpuMemcpy2DAsync         hipMemcpy2DAsync
#define gpuMemsetAsync           hipMemsetAsync
#define gpuMemcpyDeviceToDevice  hipMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice    hipMemcpyHostToDevice
#define gpuMalloc                hipMalloc
#define gpuFree                  hipFree
#define gpuMemset                hipMemset
#define gpuGetDevice             hipGetDevice
#define gpuSetDevice             hipSetDevice
#define gpuDeviceSynchronize     hipDeviceSynchronize
#define gpuDeviceGetAttribute    hipDeviceGetAttribute
#define gpuDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define gpuDevAttrMemoryPoolsSupported hipDeviceAttributeMemoryPoolsSupported
#define gpuMallocAsync           hipMallocAsync
#define gpuFreeAsync             hipFreeAsync
#define gpuStreamSynchronize     hipStreamSynchronize
#define gpuDevAttrCooperativeLaunch hipDeviceAttributeCooperativeLaunch
#define gpuLaunchCooperativeKernel hipLaunchCooperativeKernel
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor hipOccupancyMaxActiveBlocksPerMultiprocessor
#else
#include <cuda_runtime.h>
typedef cudaStream_t gpuStream_t;
typedef cudaError_t  gpuError_t;
#define gpuSuccess               cudaSuccess
#define gpuErrorNotReady         cudaErrorNotReady
#define gpuGetLastError          cudaGetLastError
#define gpuPeekAtLastError       cudaPeekAtLastError
#define gpuGetErrorName          cudaGetErrorName
#define gpuGetErrorString        cudaGetErrorString
#define OMC_RT                   "cuda"    /* prefix of runtime call names in messages */
#define gpuMemcpyAsync           cudaMemcpyAsync
#define gpuMemcpy2DAsync         cudaMemcpy2DAsync
#define gpuMemsetAsync           cudaMemsetAsync
#define gpuMemcpyDeviceToDevice  cudaMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice    cudaMemcpyHostToDevice
#define gpuMalloc                cudaMalloc
#define gpuFree                  cudaFree
#define gpuMemset                cudaMemset
#define gpuGetDevice             cudaGetDevice
#define gpuSetDevice             cudaSetDevice
#define gpuDeviceSynchronize     cudaDeviceSynchronize
#define gpuDeviceGetAttribute    cudaDeviceGetAttribute
#define gpuDevAttrMultiProcessorCount cudaDevAttrMultiProcessorCount
#define gpuDevAttrMemoryPoolsSupported cudaDevAttrMemoryPoolsSupported
#define gpuMallocAsync           cudaMallocAsync
#define gpuFreeAsync             cudaFreeAsync
#define gpuStreamSynchronize     cudaStreamSynchronize
#define gpuDevAttrCooperativeLaunch cudaDevAttrCooperativeLaunch
#define gpuLaunchCooperativeKernel cudaLaunchCooperativeKernel
#define gpuOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor
#endif

namespace omc {
namespace gpu {

enum Op {
    OP_CVT = 0,      /* 1-D / 2-D, optional padding */
    OP_TRANSPOSE,    /* dst = cvt(src)^T, optional padding */
    OP_INPLACE,      /* ordered single pass (1-D, or 2-D with the same ld before and after) */
    OP_AMAX          /* amax (and the dynamic power-of-two scale) */
};

enum Mode { MODE_PLAIN = 0, MODE_SCALED = 1, MODE_SR = 2, MODE_SR_SCALED = 3 };

struct LaunchArgs {
    Op          op;
    Mode        mode;
    omc_dtype_t st, dt;
    const void *src;
    void       *dst;
    int64_t     lds, ldd;    /* OP_INPLACE: lds is the leading dimension before and after */
    int64_t     m, n;        /* source matrix */
    int64_t     dr, dc;      /* destination region written (>= converted part) */
    bool        sat, rne;
    uint64_t    seed;
    float       scale;       /* MODE_SCALED / MODE_SR_SCALED, used when scale_ptr is NULL */
    const float *scale_ptr;  /* scaled modes: device pointer, reads scale_ptr[scale_index] */
    int         scale_index;
    /* OP_AMAX */
    float      *amax_out;    /* device: {amax} or {amax, s, 1/s} */
    int         amax_scale;  /* write the scale triple */
    float       scale_max;
    /* workspace (zeroed, left zeroed) */
    void       *ws;
    size_t      ws_bytes;
    gpuStream_t stream;
    int         num_sms;
    int         device;
    /* OP_TRANSPOSE with OMC_SCALE_DYNAMIC_POW2: the amax pass is not run yet;
     * one cooperative kernel computes it and converts (amax_out, scale_max) */
    bool        fused_amax;
};

/* Workspace layout (32-bit words): [0] in-place ticket, [1] in-place done
 * count, [2] amax done count (also the fused transpose's departures), [3]
 * fused transpose's arrivals, [4, 4 + kMaxAmaxBlocks) amax partials (need not
 * be zero), then one in-place flag per chunk. Counters and flags are zero
 * between calls. */
constexpr int    kThreads       = 256;
constexpr int    kMaxAmaxBlocks = 1024;
constexpr int    kFlagsOffset   = 4 + kMaxAmaxBlocks;
inline size_t workspace_bytes_for(size_t count) {
    const size_t chunk = (size_t)kThreads * 2;           /* smallest chunk (ITEMS >= 2) */
    const size_t nchunks = (count + chunk - 1) / chunk;
    const size_t r = (kFlagsOffset + nchunks) * sizeof(uint32_t);
    return (r + 255) & ~(size_t)255;
}

using LaunchFn = omc_status_t (*)(const LaunchArgs &);

/** Records a failed runtime call for omc_last_error_detail ("<where>: <error
 *  name> (<description>)", where given in printf format) and returns s.
 *  Defined in gpu_api.cu. */
omc_status_t device_error(omc_status_t s, gpuError_t e, const char *where, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 3, 4)))
#endif
    ;

/* one per source type, defined in gpu_inst_<type>.cu */
#define OMC_DECLARE_LAUNCH(name) omc_status_t launch_src_##name(const LaunchArgs &a);
OMC_DECLARE_LAUNCH(fp64) OMC_DECLARE_LAUNCH(fp32) OMC_DECLARE_LAUNCH(fp16) OMC_DECLARE_LAUNCH(bf16)
OMC_DECLARE_LAUNCH(e4m3) OMC_DECLARE_LAUNCH(e5m2)
OMC_DECLARE_LAUNCH(int8) OMC_DECLARE_LAUNCH(int16) OMC_DECLARE_LAUNCH(int32) OMC_DECLARE_LAUNCH(int64)
OMC_DECLARE_LAUNCH(uint8) OMC_DECLARE_LAUNCH(uint16) OMC_DECLARE_LAUNCH(uint32) OMC_DECLARE_LAUNCH(uint64)
#undef OMC_DECLARE_LAUNCH

} /* namespace gpu */
} /* namespace omc */

#endif /* OMC_GPU_RT_HPP */
