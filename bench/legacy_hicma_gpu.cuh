/**
 * @file legacy_hicma_gpu.cuh
 * @brief The HiCMA-X 0.2 GPU conversion kernels that OmniCast replaces
 * (src/core/hicma_parsec.cu, hicma_parsec_stochastic_rounding.cu), kept
 * verbatim in substance as the baseline of the GPU benchmark: 32x32 thread
 * blocks, one element per thread; FP8 written transposed one byte per thread.
 */
#ifndef LEGACY_HICMA_GPU_CUH
#define LEGACY_HICMA_GPU_CUH

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
typedef hipStream_t lStream_t;
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
typedef cudaStream_t lStream_t;
#endif
#include <stdint.h>

#define L_CHUNK 32

#define L_2D_HEAD \
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    const int idy = blockIdx.y * blockDim.y + threadIdx.y; \
    if (idx >= nrows || idy >= ncols) return;
#define L_LAUNCH(kernel, ...) do { \
    dim3 b(L_CHUNK, L_CHUNK), g((nrows + L_CHUNK - 1) / L_CHUNK, (ncols + L_CHUNK - 1) / L_CHUNK); \
    kernel<<<g, b, 0, stream>>>(__VA_ARGS__); } while (0)

__global__ void l_double2float_k(int nrows, int ncols, const double *D, int ldh, float *F, int ldf) {
    L_2D_HEAD F[idy * ldf + idx] = __double2float_rn(D[idy * ldh + idx]);
}
__global__ void l_float2double_k(int nrows, int ncols, const float *F, int ldf, double *D, int ldh) {
    L_2D_HEAD D[idy * ldh + idx] = (double)F[idy * ldf + idx];
}
__global__ void l_float2half_k(int nrows, int ncols, const float *F, int ldf, __half *H, int ldh) {
    L_2D_HEAD H[idy * ldh + idx] = __float2half_rn(F[idy * ldf + idx]);
}
__global__ void l_half2float_k(int nrows, int ncols, const __half *H, int ldh, float *F, int ldf) {
    L_2D_HEAD F[idy * ldf + idx] = __half2float(H[idy * ldh + idx]);
}
__global__ void l_double2half_k(int nrows, int ncols, const double *F, int ldf, __half *H, int ldh) {
#if defined(__HIPCC__)   /* HiCMA-X 0.2 HIP: through FP32 (double rounding) */
    L_2D_HEAD H[idy * ldh + idx] = __float2half((float)(F[idy * ldf + idx]));
#else
    L_2D_HEAD H[idy * ldh + idx] = __double2half(F[idy * ldf + idx]);
#endif
}
__global__ void l_half2double_k(int nrows, int ncols, const __half *H, int ldh, double *F, int ldf) {
    L_2D_HEAD F[idy * ldf + idx] = (double)__half2float(H[idy * ldh + idx]);
}
template <class T>
__global__ void l_memcpy_k(int nrows, int ncols, const T *src, T *dest) {
    L_2D_HEAD dest[idy * nrows + idx] = src[idy * nrows + idx];
}

static void l_double2float(int nrows, int ncols, const double *D, int ldh, float *F, int ldf, lStream_t stream) { L_LAUNCH(l_double2float_k, nrows, ncols, D, ldh, F, ldf); }
static void l_float2double(int nrows, int ncols, const float *F, int ldf, double *D, int ldh, lStream_t stream) { L_LAUNCH(l_float2double_k, nrows, ncols, F, ldf, D, ldh); }
static void l_float2half(int nrows, int ncols, const float *F, int ldf, void *H, int ldh, lStream_t stream) { L_LAUNCH(l_float2half_k, nrows, ncols, F, ldf, (__half *)H, ldh); }
static void l_half2float(int nrows, int ncols, const void *H, int ldh, float *F, int ldf, lStream_t stream) { L_LAUNCH(l_half2float_k, nrows, ncols, (const __half *)H, ldh, F, ldf); }
static void l_double2half(int nrows, int ncols, const double *F, int ldf, void *H, int ldh, lStream_t stream) { L_LAUNCH(l_double2half_k, nrows, ncols, F, ldf, (__half *)H, ldh); }
static void l_half2double(int nrows, int ncols, const void *H, int ldh, double *F, int ldf, lStream_t stream) { L_LAUNCH(l_half2double_k, nrows, ncols, (const __half *)H, ldh, F, ldf); }
template <class T>
static void l_memcpy(int nrows, int ncols, const void *src, void *dst, lStream_t stream) { L_LAUNCH(l_memcpy_k<T>, nrows, ncols, (const T *)src, (T *)dst); }

#if !defined(__HIPCC__)
__global__ void l_double2bf_k(int nrows, int ncols, const double *F, int ldf, __nv_bfloat16 *H, int ldh) {
    L_2D_HEAD H[idy * ldh + idx] = __double2bfloat16(F[idy * ldf + idx]);
}
__global__ void l_float2bf_k(int nrows, int ncols, const float *F, int ldf, __nv_bfloat16 *H, int ldh) {
    L_2D_HEAD H[idy * ldh + idx] = __float2bfloat16(F[idy * ldf + idx]);
}
__global__ void l_bf2float_k(int nrows, int ncols, const __nv_bfloat16 *H, int ldh, float *F, int ldf) {
    L_2D_HEAD F[idy * ldf + idx] = __bfloat162float(H[idy * ldh + idx]);
}
static void l_double2bf(int nrows, int ncols, const double *F, int ldf, void *H, int ldh, lStream_t stream) { L_LAUNCH(l_double2bf_k, nrows, ncols, F, ldf, (__nv_bfloat16 *)H, ldh); }
static void l_float2bf(int nrows, int ncols, const float *F, int ldf, void *H, int ldh, lStream_t stream) { L_LAUNCH(l_float2bf_k, nrows, ncols, F, ldf, (__nv_bfloat16 *)H, ldh); }
static void l_bf2float(int nrows, int ncols, const void *H, int ldh, float *F, int ldf, lStream_t stream) { L_LAUNCH(l_bf2float_k, nrows, ncols, (const __nv_bfloat16 *)H, ldh, F, ldf); }

/* FP8, transposed (cublasLt TN) and zero-padded, one byte per thread */
template <class T> __device__ __forceinline__ float l_tof(T v) { return (float)v; }
template <> __device__ __forceinline__ float l_tof<__half>(__half v) { return __half2float(v); }
template <class T>
__global__ void l_fp8_padded_k(int nrows, int ncols, int padded_rows, int padded_cols, const T *S, int lds, uint8_t *Tt, int ldt) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= padded_rows || idy >= padded_cols) return;
    if (idx < nrows && idy < ncols)
        Tt[idx * ldt + idy] = (uint8_t)__nv_cvt_float_to_fp8(l_tof(S[idy * lds + idx]), __NV_SATFINITE, __NV_E4M3);
    else
        Tt[idx * ldt + idy] = 0;
}
template <class T>
static void l_fp8_padded(int nrows, int ncols, const T *S, int lds, void *Tt, int ldt, int pr, int pc, lStream_t stream) {
    dim3 b(L_CHUNK, L_CHUNK), g((pr + L_CHUNK - 1) / L_CHUNK, (pc + L_CHUNK - 1) / L_CHUNK);
    l_fp8_padded_k<T><<<g, b, 0, stream>>>(nrows, ncols, pr, pc, S, lds, (uint8_t *)Tt, ldt);
}

/* per-tile scaled FP8 (hicma_fp8_scaled_convert_GPU) */
__device__ __forceinline__ float l_abs(double v) { return (float)fabs(v); }
__device__ __forceinline__ float l_abs(float v) { return fabsf(v); }
__device__ __forceinline__ float l_scaled(double v, float s) { return (float)(v * (double)s); }
__device__ __forceinline__ float l_scaled(float v, float s) { return v * s; }
template <typename T>
__global__ void l_amax_k(int nrows, int ncols, const T *S, int lds, unsigned int *amax_bits) {
    __shared__ float warp_max[32];
    const long n = (long)nrows * ncols;
    float m = 0.0f;
    for (long e = (long)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (long)gridDim.x * blockDim.x)
        m = fmaxf(m, l_abs(S[(e / nrows) * lds + (e % nrows)]));
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (0 == lane) warp_max[warp] = m;
    __syncthreads();
    if (0 == warp) {
        m = (lane < (int)((blockDim.x + 31) / 32)) ? warp_max[lane] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
        if (0 == lane) atomicMax(amax_bits, __float_as_uint(m));
    }
}
__global__ void l_scale_k(float *scale) {
    const float amax = __uint_as_float(*(const unsigned int *)scale);
    float s = 1.0f;
    if (amax > 0.0f && isfinite(amax)) {
        const float r = 448.0f / amax;
        int e = 126;
        if (isfinite(r)) { frexpf(r, &e); e -= 1; }
        e = max(-126, min(126, e));
        s = ldexpf(1.0f, e);
    }
    scale[1] = s; scale[2] = 1.0f / s;
}
template <typename T>
__global__ void l_scaled_cvt_k(int nrows, int ncols, int padded_rows, int padded_cols, const T *S, int lds, uint8_t *dst, int ldt, const float *scale) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= padded_rows || idy >= padded_cols) return;
    uint8_t v = 0;
    if (idx < nrows && idy < ncols)
        v = (uint8_t)__nv_cvt_float_to_fp8(l_scaled(S[idy * lds + idx], scale[1]), __NV_SATFINITE, __NV_E4M3);
    dst[idx * ldt + idy] = v;
}
template <typename T>
static void l_fp8_scaled(int nrows, int ncols, const T *S, int lds, void *dst, int ldt, int pr, int pc, float *scale, lStream_t stream) {
    cudaMemsetAsync(scale, 0, sizeof(float), stream);
    const long n = (long)nrows * ncols;
    const int threads = 256;
    const int blocks = (int)((n + threads - 1) / threads < 1024 ? (n + threads - 1) / threads : 1024);
    l_amax_k<T><<<blocks, threads, 0, stream>>>(nrows, ncols, S, lds, (unsigned int *)scale);
    l_scale_k<<<1, 1, 0, stream>>>(scale);
    dim3 b(L_CHUNK, L_CHUNK), g((pr + L_CHUNK - 1) / L_CHUNK, (pc + L_CHUNK - 1) / L_CHUNK);
    l_scaled_cvt_k<T><<<g, b, 0, stream>>>(nrows, ncols, pr, pc, S, lds, (uint8_t *)dst, ldt, scale);
}
#endif

/* stochastic rounding (hicma_parsec_stochastic_rounding.cu) */
__device__ __forceinline__ uint64_t l_sm64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}
__device__ __forceinline__ float l_u01(int i, int j, uint64_t seed, uint64_t bits) {
    uint64_t pos = (uint64_t)(uint32_t)i | ((uint64_t)(uint32_t)j << 32);
    return (float)(l_sm64(bits ^ l_sm64(pos ^ seed)) >> 40) * (1.0f / 16777216.0f);
}
__device__ float l_sr32(double value, float random) {
    float nearest = (float)value;
    if (!isfinite(value) || !isfinite(nearest)) return nearest;
    double a = fabs(value); float n = fabsf(nearest); float lower, upper;
    if (a > (double)n) { lower = n; upper = nextafterf(n, INFINITY); }
    else if (a < (double)n) { lower = nextafterf(n, 0.0f); upper = n; }
    else return nearest;
    if (isinf(upper)) return nearest;
    double p_lower = ((double)upper - a) / ((double)upper - (double)lower);
    float r = (random < p_lower) ? lower : upper;
    return (value < 0.0) ? -r : r;
}
__device__ __half l_sr16(double value, float random) {
    if (isnan(value)) return __float2half((float)value);
    double a = fabs(value);
    __half lower = __float2half_rz((float)a);
    if ((double)__half2float(lower) > a) lower = __ushort_as_half(__half_as_ushort(lower) - 1);
    double lv = (double)__half2float(lower);
    if (lv == a) return (value < 0.0) ? __hneg(lower) : lower;
    __half upper = __ushort_as_half(__half_as_ushort(lower) + 1);
    double uv = (double)__half2float(upper);
    if (isinf(uv)) return __float2half((float)value);
    double p_lower = (uv - a) / (uv - lv);
    __half r = (random < p_lower) ? lower : upper;
    return (value < 0.0) ? __hneg(r) : r;
}
__global__ void l_d2s_round_k(int nrows, int ncols, const double *D, int ldd, float *F, int ldf, uint64_t seed) {
    L_2D_HEAD F[idy * ldf + idx] = l_sr32(D[idy * ldd + idx], l_u01(idx, idy, seed, (uint64_t)__double_as_longlong(D[idy * ldd + idx])));
}
__global__ void l_d2h_round_k(int nrows, int ncols, const double *D, int ldd, __half *H, int ldh, uint64_t seed) {
    L_2D_HEAD H[idy * ldh + idx] = l_sr16(D[idy * ldd + idx], l_u01(idx, idy, seed, (uint64_t)__double_as_longlong(D[idy * ldd + idx])));
}
__global__ void l_s2h_round_k(int nrows, int ncols, const float *F, int ldf, __half *H, int ldh, uint64_t seed) {
    L_2D_HEAD H[idy * ldh + idx] = l_sr16((double)F[idy * ldf + idx], l_u01(idx, idy, seed, (uint64_t)__float_as_uint(F[idy * ldf + idx])));
}
static void l_d2s_round(int nrows, int ncols, const double *D, int ldd, float *F, int ldf, uint64_t seed, lStream_t stream) { L_LAUNCH(l_d2s_round_k, nrows, ncols, D, ldd, F, ldf, seed); }
static void l_d2h_round(int nrows, int ncols, const double *D, int ldd, void *H, int ldh, uint64_t seed, lStream_t stream) { L_LAUNCH(l_d2h_round_k, nrows, ncols, D, ldd, (__half *)H, ldh, seed); }
static void l_s2h_round(int nrows, int ncols, const float *F, int ldf, void *H, int ldh, uint64_t seed, lStream_t stream) { L_LAUNCH(l_s2h_round_k, nrows, ncols, F, ldf, (__half *)H, ldh, seed); }

#endif
