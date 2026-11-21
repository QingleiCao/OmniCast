/**
 * @file gpu_kernels.cuh
 * @brief GPU conversion kernels (CUDA and HIP from one source).
 *
 *  k_cvt      1-D and 2-D (strided, padded) conversion. Each thread converts
 *             ITEMS consecutive elements with 16-byte vector loads and stores
 *             (ITEMS = 16 / size of the smaller type), so both the reads and
 *             the writes are full 16-byte transactions.
 *  k_cvtT     transposed conversion through a shared-memory tile: reads and
 *             writes are both coalesced (the HiCMA-X FP8 kernels wrote one
 *             byte per thread at a stride of ld).
 *  k_inplace  single-pass in-place conversion of a contiguous buffer. Blocks
 *             take chunks in address order (front to back when elements
 *             shrink, back to front when they grow), publish a flag once their
 *             input is in registers, and write only after the chunks their
 *             output overlaps have published theirs. Dependencies always point
 *             to earlier tickets, so the wait cannot deadlock; the last block
 *             zeroes the flags for the next call.
 *  k_amax     max |x| with a last-block finish that can also write the dynamic
 *             power-of-two scale {amax, s, 1/s}.
 */
#ifndef OMC_GPU_KERNELS_CUH
#define OMC_GPU_KERNELS_CUH

#include "gpu_rt.hpp"
#include "../core/omc_scalar.hpp"

namespace omc {
namespace gpu {

template <class T> struct Zero { __device__ __forceinline__ static T get() { return (T)0; } };
template <> struct Zero<f16>  { __device__ __forceinline__ static f16  get() { return make<f16>(0); } };
template <> struct Zero<bf16> { __device__ __forceinline__ static bf16 get() { return make<bf16>(0); } };
template <> struct Zero<e4m3> { __device__ __forceinline__ static e4m3 get() { return make<e4m3>(0); } };
template <> struct Zero<e5m2> { __device__ __forceinline__ static e5m2 get() { return make<e5m2>(0); } };

template <class S, class D> struct Items {
    static constexpr int a = 16 / (int)sizeof(S), b = 16 / (int)sizeof(D);
    static constexpr int value = a > b ? a : b;
};

template <class T, int N>
union VecU {
    static constexpr int W = (int)(N * sizeof(T) / 16);
    uint4 w[W];
    T     v[N];
};

template <class T, int N>
__device__ __forceinline__ void vload(const T *p, T (&v)[N]) {
    static_assert((N * sizeof(T)) % 16 == 0, "vector of 16-byte words");
    VecU<T, N> u;
    const uint4 *q = reinterpret_cast<const uint4 *>(p);
#pragma unroll
    for (int k = 0; k < VecU<T, N>::W; k++) u.w[k] = q[k];
#pragma unroll
    for (int k = 0; k < N; k++) v[k] = u.v[k];
}

template <class T, int N>
__device__ __forceinline__ void vstore(T *p, const T (&v)[N]) {
    VecU<T, N> u;
#pragma unroll
    for (int k = 0; k < N; k++) u.v[k] = v[k];
    uint4 *q = reinterpret_cast<uint4 *>(p);
#pragma unroll
    for (int k = 0; k < VecU<T, N>::W; k++) q[k] = u.w[k];
}

/** The element function of a kernel */
template <class S, class D, int MODE>
struct ElemOp {
    ElemOpts o;
    float    scale;
    uint64_t seed;
    __device__ __forceinline__ D operator()(S s, uint32_t i, uint32_t j) const {
        if constexpr (MODE == MODE_PLAIN)       return cvt<D>(s, o);
        else if constexpr (MODE == MODE_SCALED) return cvt_scaled<D>(s, scale, o);
        else                                    return SR<D, S>::go(s, i, j, seed, o);
    }
};

/* ------------------------------------------------------------------------ */

template <class S, class D, int MODE>
__global__ void __launch_bounds__(kThreads)
k_cvt(const S *src, int64_t lds, D *dst, int64_t ldd, int64_t m, int64_t n, int64_t dr, int64_t dc,
      ElemOp<S, D, MODE> op, const float *scale_ptr, int scale_index, int vec) {
    constexpr int IT = Items<S, D>::value;
    if (MODE == MODE_SCALED && scale_ptr) op.scale = scale_ptr[scale_index];
    const int64_t i0 = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * IT;
    if (i0 >= dr) return;
    for (int64_t j = (int64_t)blockIdx.y * blockDim.y + threadIdx.y; j < dc; j += (int64_t)gridDim.y * blockDim.y) {
        D *drow = dst + j * ldd;
        if (vec && j < n && i0 + IT <= m) {
            S v[IT];
            vload(src + j * lds + i0, v);
            D w[IT];
#pragma unroll
            for (int k = 0; k < IT; k++) w[k] = op(v[k], (uint32_t)(i0 + k), (uint32_t)j);
            vstore(drow + i0, w);
        } else {
#pragma unroll
            for (int k = 0; k < IT; k++) {
                const int64_t i = i0 + k;
                if (i < dr)
                    drow[i] = (i < m && j < n) ? op(src[j * lds + i], (uint32_t)i, (uint32_t)j) : Zero<D>::get();
            }
        }
    }
}

template <class S, class D, int MODE>
__global__ void __launch_bounds__(kThreads)
k_cvtT(const S *src, int64_t lds, D *dst, int64_t ldd, int64_t m, int64_t n, int64_t dr, int64_t dc,
       ElemOp<S, D, MODE> op, const float *scale_ptr, int scale_index) {
    __shared__ D tile[32][33];
    if (MODE == MODE_SCALED && scale_ptr) op.scale = scale_ptr[scale_index];
    const int tx = threadIdx.x, ty = threadIdx.y;          /* 32 x 8 */
    const int64_t i0 = (int64_t)blockIdx.x * 32;            /* source rows = dst columns */
    for (int64_t j0 = (int64_t)blockIdx.y * 32; j0 < dr; j0 += (int64_t)gridDim.y * 32) {   /* source columns = dst rows */
#pragma unroll
        for (int k = 0; k < 32; k += 8) {
            const int64_t i = i0 + tx, j = j0 + ty + k;
            tile[ty + k][tx] = (i < m && j < n) ? op(src[j * lds + i], (uint32_t)i, (uint32_t)j) : Zero<D>::get();
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < 32; k += 8) {
            const int64_t c = i0 + ty + k, r = j0 + tx;
            if (c < dc && r < dr) dst[c * ldd + r] = tile[tx][ty + k];
        }
        __syncthreads();
    }
}

template <class S, class D, int MODE>
__global__ void __launch_bounds__(kThreads)
k_inplace(char *buf, int64_t count, int64_t m, ElemOp<S, D, MODE> op, const float *scale_ptr, int scale_index,
          uint32_t *ws, uint32_t nchunks) {
    constexpr int IT = Items<S, D>::value;
    constexpr int64_t C = (int64_t)kThreads * IT;
    constexpr bool shrink = sizeof(D) < sizeof(S);
    __shared__ uint32_t s_ticket;
    __shared__ int s_last;
    if (MODE == MODE_SCALED && scale_ptr) op.scale = scale_ptr[scale_index];
    if (threadIdx.x == 0) s_ticket = atomicAdd(&ws[0], 1u);
    __syncthreads();
    const uint32_t chunk = shrink ? s_ticket : (nchunks - 1u - s_ticket);
    const S *src = reinterpret_cast<const S *>(buf);
    D *dst = reinterpret_cast<D *>(buf);
    const int64_t e0 = (int64_t)chunk * C + (int64_t)threadIdx.x * IT;
    const bool full = e0 + IT <= count;
    /* position (i, j) for stochastic rounding: column-major m-by-(count/m) */
    uint32_t ci = 0, cj = 0;
    if (MODE == MODE_SR) { const int64_t jj = e0 / m; cj = (uint32_t)jj; ci = (uint32_t)(e0 - jj * m); }
    D w[IT];
    if (full) {
        S v[IT];
        vload(src + e0, v);
#pragma unroll
        for (int k = 0; k < IT; k++) {
            w[k] = op(v[k], ci, cj);
            if (MODE == MODE_SR) { if (++ci == (uint32_t)m) { ci = 0; cj++; } }
        }
    } else {
#pragma unroll
        for (int k = 0; k < IT; k++) {
            if (e0 + k < count) w[k] = op(src[e0 + k], ci, cj);
            if (MODE == MODE_SR) { if (++ci == (uint32_t)m) { ci = 0; cj++; } }
        }
    }
    __syncthreads();                       /* the whole chunk is in registers */
    uint32_t *flags = ws + kFlagsOffset;
    if (threadIdx.x == 0) {
        __threadfence();
        atomicExch(&flags[chunk], 1u);
        /* input chunks overlapped by this chunk's output */
        const int64_t ob0 = (int64_t)chunk * C * (int64_t)sizeof(D);
        const int64_t e1  = ((int64_t)chunk + 1) * C < count ? ((int64_t)chunk + 1) * C : count;
        const int64_t ob1 = e1 * (int64_t)sizeof(D);
        const int64_t cb  = C * (int64_t)sizeof(S);
        uint32_t lo = (uint32_t)(ob0 / cb), hi = (uint32_t)((ob1 - 1) / cb);
        if (hi > nchunks - 1u) hi = nchunks - 1u;
        for (uint32_t q = lo; q <= hi; q++)
            if (q != chunk)
                while (atomicAdd(&flags[q], 0u) == 0u) { }
        __threadfence();
    }
    __syncthreads();
    if (full) {
        vstore(dst + e0, w);
    } else {
#pragma unroll
        for (int k = 0; k < IT; k++) if (e0 + k < count) dst[e0 + k] = w[k];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence();
        s_last = (atomicAdd(&ws[1], 1u) == gridDim.x - 1u);
    }
    __syncthreads();
    if (s_last) {                          /* every other block is done: reset */
        for (uint32_t q = threadIdx.x; q < nchunks; q += blockDim.x) flags[q] = 0u;
        if (threadIdx.x == 0) { ws[0] = 0u; ws[1] = 0u; }
        __threadfence();
    }
}

template <class S>
__global__ void __launch_bounds__(kThreads)
k_amax(const S *src, int64_t lds, int64_t m, int64_t n, uint32_t *ws, float *out, int write_scale, float target) {
    __shared__ float red[kThreads];
    __shared__ int s_last;
    float mx = 0.0f;
    for (int64_t j = blockIdx.y; j < n; j += gridDim.y)
        for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < m; i += (int64_t)gridDim.x * blockDim.x)
            mx = fmaxf(mx, abs_f32(src[j * lds + i]));
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    const uint32_t nblocks = gridDim.x * gridDim.y;
    uint32_t *part = ws + 4;
    if (threadIdx.x == 0) {
        part[blockIdx.y * gridDim.x + blockIdx.x] = __float_as_uint(red[0]);
        __threadfence();
        s_last = (atomicAdd(&ws[2], 1u) == nblocks - 1u);
    }
    __syncthreads();
    if (s_last) {
        float v = 0.0f;
        for (uint32_t q = threadIdx.x; q < nblocks; q += blockDim.x) v = fmaxf(v, __uint_as_float(atomicAdd(&part[q], 0u)));
        red[threadIdx.x] = v;
        __syncthreads();
        for (int s = kThreads / 2; s > 0; s >>= 1) {
            if ((int)threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const float amax = red[0];
            out[0] = amax;
            if (write_scale) {
                const float s = pow2_scale(amax, target);
                out[1] = s;
                out[2] = 1.0f / s;
            }
            ws[2] = 0u;
            __threadfence();
        }
    }
}

/* ------------------------------------------------------------------------ */
/* Host-side launch                                                          */
/* ------------------------------------------------------------------------ */

inline omc_status_t check_launch() {
    return (gpuGetLastError() == gpuSuccess) ? OMC_SUCCESS : OMC_ERROR_DEVICE;
}

inline bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

template <class S, class D, int MODE>
omc_status_t launch_mode(const LaunchArgs &a) {
    ElemOp<S, D, MODE> op;
    op.o = ElemOpts{a.sat, a.rne};
    op.scale = a.scale;
    op.seed = a.seed;
    constexpr int IT = Items<S, D>::value;
    const S *src = static_cast<const S *>(a.src);
    D *dst = static_cast<D *>(a.dst);
    switch (a.op) {
    case OP_CVT: {
        const int vec = aligned16(src) && aligned16(dst) &&
                        (a.n <= 1 || (((a.lds * (int64_t)sizeof(S)) % 16) == 0 && ((a.ldd * (int64_t)sizeof(D)) % 16) == 0));
        const int64_t groups = (a.dr + IT - 1) / IT;      /* thread groups along a column */
        int tx = 32;
        while (tx < kThreads && tx < groups) tx *= 2;
        const int ty = kThreads / tx;
        const int64_t gx = (groups + tx - 1) / tx;
        int64_t gy = (a.dc + ty - 1) / ty;
        if (gy > 65535) gy = 65535;
        if (gx > 0x7fffffff) return OMC_ERROR_NOT_SUPPORTED;
        k_cvt<S, D, MODE><<<dim3((unsigned)gx, (unsigned)gy), dim3(tx, ty), 0, a.stream>>>(
            src, a.lds, dst, a.ldd, a.m, a.n, a.dr, a.dc, op, a.scale_ptr, a.scale_index, vec);
        return check_launch();
    }
    case OP_TRANSPOSE: {
        const int64_t gx = (a.dc + 31) / 32;
        int64_t gy = (a.dr + 31) / 32;
        if (gy > 65535) gy = 65535;
        if (gx > 0x7fffffff) return OMC_ERROR_NOT_SUPPORTED;
        k_cvtT<S, D, MODE><<<dim3((unsigned)gx, (unsigned)gy), dim3(32, 8), 0, a.stream>>>(
            src, a.lds, dst, a.ldd, a.m, a.n, a.dr, a.dc, op, a.scale_ptr, a.scale_index);
        return check_launch();
    }
    case OP_INPLACE: {
        if constexpr (sizeof(S) == sizeof(D)) {
            return OMC_ERROR_NOT_SUPPORTED;    /* the API layer uses k_cvt in place */
        } else {
            const int64_t count = a.m * a.n;
            const int64_t C = (int64_t)kThreads * IT;
            const int64_t nchunks = (count + C - 1) / C;
            if (!aligned16(a.dst)) return OMC_ERROR_INVALID_VALUE;
            if (nchunks > 0x7fffffff || (size_t)(nchunks + kFlagsOffset) * 4 > a.ws_bytes) return OMC_ERROR_INVALID_VALUE;
            k_inplace<S, D, MODE><<<(unsigned)nchunks, kThreads, 0, a.stream>>>(
                static_cast<char *>(a.dst), count, a.m, op, a.scale_ptr, a.scale_index,
                static_cast<uint32_t *>(a.ws), (uint32_t)nchunks);
            return check_launch();
        }
    }
    default:
        return OMC_ERROR_NOT_SUPPORTED;
    }
}

template <class S, class D>
omc_status_t launch_pair(const LaunchArgs &a) {
    switch (a.mode) {
    case MODE_PLAIN:
        return launch_mode<S, D, MODE_PLAIN>(a);
    case MODE_SCALED:
        if constexpr (Traits<S>::is_float && Traits<D>::is_float) return launch_mode<S, D, MODE_SCALED>(a);
        else return OMC_ERROR_NOT_SUPPORTED;
    case MODE_SR:
        if constexpr (sr_ok<D, S>::value) return launch_mode<S, D, MODE_SR>(a);
        else return OMC_ERROR_NOT_SUPPORTED;
    }
    return OMC_ERROR_NOT_SUPPORTED;
}

template <class S>
omc_status_t launch_for_src(const LaunchArgs &a) {
    if (a.op == OP_AMAX) {
        if constexpr (Traits<S>::is_float) {
            int64_t gx = (a.m + kThreads - 1) / kThreads;
            if (gx > 32) gx = 32;
            if (gx < 1) gx = 1;
            int64_t gy = kMaxAmaxBlocks / gx;
            if (gy > a.n) gy = a.n;
            if (gy < 1) gy = 1;
            if ((size_t)(kMaxAmaxBlocks + 4) * 4 > a.ws_bytes) return OMC_ERROR_INVALID_VALUE;
            k_amax<S><<<dim3((unsigned)gx, (unsigned)gy), kThreads, 0, a.stream>>>(
                static_cast<const S *>(a.src), a.lds, a.m, a.n, static_cast<uint32_t *>(a.ws),
                a.amax_out, a.amax_scale, a.scale_max);
            return check_launch();
        } else {
            return OMC_ERROR_NOT_SUPPORTED;
        }
    }
    switch (a.dt) {
    case OMC_FP64:     return launch_pair<S, double>(a);
    case OMC_FP32:     return launch_pair<S, float>(a);
    case OMC_FP16:     return launch_pair<S, f16>(a);
    case OMC_BF16:     return launch_pair<S, bf16>(a);
    case OMC_FP8_E4M3: return launch_pair<S, e4m3>(a);
    case OMC_FP8_E5M2: return launch_pair<S, e5m2>(a);
    case OMC_INT8:     return launch_pair<S, int8_t>(a);
    case OMC_INT16:    return launch_pair<S, int16_t>(a);
    case OMC_INT32:    return launch_pair<S, int32_t>(a);
    case OMC_INT64:    return launch_pair<S, int64_t>(a);
    case OMC_UINT8:    return launch_pair<S, uint8_t>(a);
    case OMC_UINT16:   return launch_pair<S, uint16_t>(a);
    case OMC_UINT32:   return launch_pair<S, uint32_t>(a);
    case OMC_UINT64:   return launch_pair<S, uint64_t>(a);
    default:           return OMC_ERROR_INVALID_VALUE;
    }
}

} /* namespace gpu */
} /* namespace omc */

#endif /* OMC_GPU_KERNELS_CUH */
