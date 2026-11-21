/**
 * @file omnicast.h
 * @brief OmniCast: fast data-type conversion on CPUs and GPUs (NVIDIA, AMD).
 *
 * One C API converts arrays and column-major matrices between
 *
 *   FP64, FP32, FP16, BF16, FP8 (E4M3, E5M2) and 8/16/32/64-bit integers
 *
 * on the host (SIMD, chosen at run time for the CPU in use) and on a GPU
 * (CUDA or HIP, asynchronous on a stream).
 *
 * Semantics (the same on every backend, bit for bit, except NaN payloads):
 *   - float -> narrower float: round to nearest even (default) or stochastic
 *     rounding (OMC_ROUND_STOCHASTIC). FP64 sources are rounded once, directly
 *     to the target (no double rounding through FP32).
 *   - overflow: Inf for formats that have it (FP32, FP16, BF16, E5M2), NaN for
 *     E4M3 (which has no Inf); with OMC_FLAG_SATURATE the largest finite value.
 *   - float -> integer: truncation (C cast) by default, round to nearest even
 *     with OMC_ROUND_NEAREST_EVEN; always saturating, NaN -> 0.
 *   - integer -> float and widening float conversions: exact where the target
 *     can hold the value, otherwise round to nearest even.
 *
 * Matrices are column-major: element (i, j) of an m-by-n matrix with leading
 * dimension ld is at index i + j * ld, ld counted in elements of that matrix.
 *
 * All functions are thread-safe. GPU functions run on the current device and
 * the given stream (NULL: the default stream), and return once the work is
 * enqueued.
 */
#ifndef OMNICAST_H
#define OMNICAST_H

#include <stddef.h>
#include <stdint.h>

#define OMNICAST_VERSION_MAJOR 2
#define OMNICAST_VERSION_MINOR 0
#define OMNICAST_VERSION_PATCH 0

#ifdef __cplusplus
extern "C" {
#endif

/** Element types. FP8 types are the OCP formats (E4M3 = E4M3FN, no Inf). */
typedef enum omc_dtype {
    OMC_FP64 = 0,
    OMC_FP32,
    OMC_FP16,
    OMC_BF16,
    OMC_FP8_E4M3,
    OMC_FP8_E5M2,
    OMC_INT8,
    OMC_INT16,
    OMC_INT32,
    OMC_INT64,
    OMC_UINT8,
    OMC_UINT16,
    OMC_UINT32,
    OMC_UINT64,
    OMC_NUM_DTYPES
} omc_dtype_t;

typedef enum omc_status {
    OMC_SUCCESS = 0,
    OMC_ERROR_INVALID_VALUE,   /**< bad argument (NULL pointer, ld < m, ...) */
    OMC_ERROR_NOT_SUPPORTED,   /**< conversion/option not available (or no GPU backend built) */
    OMC_ERROR_DEVICE,          /**< CUDA/HIP runtime error */
    OMC_ERROR_ALLOCATION       /**< workspace allocation failed */
} omc_status_t;

typedef enum omc_round {
    OMC_ROUND_DEFAULT = 0,     /**< nearest even for float targets, truncation for integer targets */
    OMC_ROUND_NEAREST_EVEN,    /**< nearest even for every target */
    OMC_ROUND_TOWARD_ZERO,     /**< truncation (integer targets only) */
    OMC_ROUND_STOCHASTIC       /**< stochastic rounding (float targets narrower than the source) */
} omc_round_t;

/** Option flags (omc_options_t::flags) */
#define OMC_FLAG_SATURATE  0x1u  /**< clamp overflow (and +-Inf) to the largest finite value */
#define OMC_FLAG_TRANSPOSE 0x2u  /**< 2-D only: dst(j, i) = cvt(src(i, j)); dst is n-by-m */

typedef enum omc_scale_mode {
    OMC_SCALE_NONE = 0,        /**< dst = cvt(src) */
    OMC_SCALE_HOST,            /**< dst = cvt(src * scale) */
    OMC_SCALE_DEVICE,          /**< GPU: dst = cvt(src * scale_ptr[0]), scale_ptr a device pointer */
    OMC_SCALE_DYNAMIC_POW2     /**< s = largest power of two with s * amax(src) <= scale_max;
                                    dst = cvt(src * s); writes scale_ptr[0..2] = {amax, s, 1/s}
                                    (device memory for GPU calls, host memory for CPU calls) */
} omc_scale_mode_t;

/**
 * Conversion options. A zero-initialized struct (or a NULL pointer) gives the
 * defaults, so C callers can write `&(omc_options_t){ .flags = OMC_FLAG_SATURATE }`.
 */
typedef struct omc_options {
    uint32_t         flags;        /**< OMC_FLAG_* */
    omc_round_t      rounding;     /**< OMC_ROUND_* */
    uint64_t         seed;         /**< stochastic rounding seed */
    omc_scale_mode_t scale_mode;   /**< OMC_SCALE_* */
    float            scale;        /**< OMC_SCALE_HOST factor */
    float           *scale_ptr;    /**< OMC_SCALE_DEVICE input / OMC_SCALE_DYNAMIC_POW2 output */
    float            scale_max;    /**< OMC_SCALE_DYNAMIC_POW2 target (0: largest finite value of dst) */
    int64_t          dst_rows;     /**< 2-D: rows of dst to write (0: m, or n when transposed); */
    int64_t          dst_cols;     /**<   the part outside the converted matrix is set to zero */
    int              num_threads;  /**< CPU: threads to use (0 or 1: the calling thread only) */
    void            *workspace;    /**< GPU: optional zeroed device workspace (NULL: internal) */
    size_t           workspace_bytes;
} omc_options_t;

/* ------------------------------------------------------------------------ */
/* Queries                                                                   */
/* ------------------------------------------------------------------------ */

size_t      omc_dtype_size(omc_dtype_t t);           /**< bytes per element (0 if invalid) */
const char *omc_dtype_name(omc_dtype_t t);           /**< "fp64", "fp32", "fp16", "bf16", "e4m3", ... */
omc_dtype_t omc_dtype_from_name(const char *name);   /**< inverse of omc_dtype_name, also "double",
                                                          "float", "half", "int", ...; OMC_NUM_DTYPES if unknown */
int         omc_dtype_is_float(omc_dtype_t t);
const char *omc_status_string(omc_status_t s);
const char *omc_version_string(void);

/** Name of the CPU code path in use: "avx512", "avx2", "neon", "generic". */
const char *omc_cpu_isa(void);
/** Force a CPU code path by name (for testing); returns 0 on success.
 *  The OMNICAST_CPU_ISA environment variable does the same at start-up. */
int         omc_cpu_set_isa(const char *name);

/** GPU backend compiled in: "cuda", "hip" or "none". */
const char *omc_gpu_backend(void);

/* ------------------------------------------------------------------------ */
/* Host (CPU) conversions                                                     */
/* ------------------------------------------------------------------------ */

/** dst[k] = cvt(src[k]) for k < count. src and dst must not overlap. */
omc_status_t omc_convert(omc_dtype_t src_type, const void *src,
                         omc_dtype_t dst_type, void *dst,
                         size_t count, const omc_options_t *opts);

/** m-by-n column-major matrices. With OMC_FLAG_TRANSPOSE dst is n-by-m. */
omc_status_t omc_convert_2d(omc_dtype_t src_type, const void *src, int64_t lds,
                            omc_dtype_t dst_type, void *dst, int64_t ldd,
                            int64_t m, int64_t n, const omc_options_t *opts);

/** In place: buf holds count src_type elements on entry, count dst_type
 *  elements on exit (the buffer must be large enough for both). */
omc_status_t omc_convert_inplace(omc_dtype_t src_type, omc_dtype_t dst_type,
                                 void *buf, size_t count, const omc_options_t *opts);

/** In place, m-by-n matrix with the same leading dimension ld (in elements)
 *  before and after. Transposition and padding are not available in place. */
omc_status_t omc_convert_inplace_2d(omc_dtype_t src_type, omc_dtype_t dst_type,
                                    void *buf, int64_t ld, int64_t m, int64_t n,
                                    const omc_options_t *opts);

/** Largest |src(i, j)| (NaN entries are ignored). Float types only. */
omc_status_t omc_amax_2d(omc_dtype_t type, const void *src, int64_t ld,
                         int64_t m, int64_t n, double *amax);

/* ------------------------------------------------------------------------ */
/* GPU conversions (device pointers, asynchronous on `stream`)               */
/* ------------------------------------------------------------------------ */

typedef void *omc_stream_t;   /**< cudaStream_t or hipStream_t */

omc_status_t omc_gpu_convert(omc_dtype_t src_type, const void *src,
                             omc_dtype_t dst_type, void *dst,
                             size_t count, const omc_options_t *opts,
                             omc_stream_t stream);

omc_status_t omc_gpu_convert_2d(omc_dtype_t src_type, const void *src, int64_t lds,
                                omc_dtype_t dst_type, void *dst, int64_t ldd,
                                int64_t m, int64_t n, const omc_options_t *opts,
                                omc_stream_t stream);

/** Single-pass in-place conversion of a contiguous buffer (one kernel, no
 *  temporary copy of the data). Uses a small internal per-stream workspace
 *  unless opts->workspace is given (see omc_gpu_inplace_workspace_size). */
omc_status_t omc_gpu_convert_inplace(omc_dtype_t src_type, omc_dtype_t dst_type,
                                     void *buf, size_t count,
                                     const omc_options_t *opts, omc_stream_t stream);

/** In place for an m-by-n matrix with leading dimension ld. Only contiguous
 *  matrices (ld == m or n == 1) are supported when the element size changes. */
omc_status_t omc_gpu_convert_inplace_2d(omc_dtype_t src_type, omc_dtype_t dst_type,
                                        void *buf, int64_t ld, int64_t m, int64_t n,
                                        const omc_options_t *opts, omc_stream_t stream);

/** d_amax[0] = largest |src(i, j)| as a float (device pointer). Float types only. */
omc_status_t omc_gpu_amax_2d(omc_dtype_t type, const void *src, int64_t ld,
                             int64_t m, int64_t n, float *d_amax, omc_stream_t stream);

/** Bytes of zeroed workspace that an in-place or dynamically scaled call on
 *  `count` elements needs when the caller provides it. The workspace must be
 *  zero-filled once (e.g. cudaMemset) and must not be used by two calls that
 *  can run at the same time; OmniCast leaves it zeroed after each call. */
size_t omc_gpu_workspace_size(size_t count);

/** Release the internal per-stream workspaces (e.g. before a stream is destroyed). */
void   omc_gpu_release_workspaces(void);

/* ------------------------------------------------------------------------ */
/* Scalars (host)                                                            */
/* ------------------------------------------------------------------------ */

uint16_t omc_fp32_to_fp16(float x);
uint16_t omc_fp64_to_fp16(double x);
float    omc_fp16_to_fp32(uint16_t h);
uint16_t omc_fp32_to_bf16(float x);
uint16_t omc_fp64_to_bf16(double x);
float    omc_bf16_to_fp32(uint16_t h);
uint8_t  omc_fp32_to_e4m3(float x, int saturate);
uint8_t  omc_fp64_to_e4m3(double x, int saturate);
float    omc_e4m3_to_fp32(uint8_t h);
uint8_t  omc_fp32_to_e5m2(float x, int saturate);
uint8_t  omc_fp64_to_e5m2(double x, int saturate);
float    omc_e5m2_to_fp32(uint8_t h);

/** Convert one value; both pointers address host memory. */
omc_status_t omc_convert_scalar(omc_dtype_t src_type, const void *src,
                                omc_dtype_t dst_type, void *dst,
                                const omc_options_t *opts);

#ifdef __cplusplus
}
#endif

#endif /* OMNICAST_H */
