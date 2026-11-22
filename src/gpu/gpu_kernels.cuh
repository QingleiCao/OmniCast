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
 *  k_inplace  single-pass in-place conversion of a buffer (1-D, or a matrix
 *             with the same leading dimension before and after). Blocks
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

/** Elements per vector: 16 bytes of the wider type */
template <class S, class D> struct Vw {
    static constexpr int ws = (int)(sizeof(S) > sizeof(D) ? sizeof(S) : sizeof(D));
    static constexpr int value = 16 / ws;
};
/* Vectors per thread, loaded together before any is converted (memory-level
 * parallelism). Measured on V100 and H100 (docs/PROFILING.md): 4 rounds for
 * plain and scaled conversions (more rounds for widening ones, which read
 * fewer bytes per round, made them 2-9 % slower), 2 for stochastic rounding
 * (compute bound: fewer registers, more resident warps). The macros are for
 * such experiments. */
#ifndef OMC_ROUNDS_NARROW
#define OMC_ROUNDS_NARROW 4
#endif
#ifndef OMC_ROUNDS_WIDEN_MAX
#define OMC_ROUNDS_WIDEN_MAX 4
#endif
#ifndef OMC_ROUNDS_SR
#define OMC_ROUNDS_SR 2
#endif
template <class S, class D, int MODE> struct Rounds {
    static constexpr int grow = sizeof(D) > sizeof(S) ? (int)(sizeof(D) / sizeof(S)) : 1;
    static constexpr int widen = (4 * grow < OMC_ROUNDS_WIDEN_MAX) ? 4 * grow : OMC_ROUNDS_WIDEN_MAX;
    static constexpr int plain = grow > 1 ? (widen > OMC_ROUNDS_NARROW ? widen : OMC_ROUNDS_NARROW) : OMC_ROUNDS_NARROW;
    static constexpr int value = (MODE == 2 || MODE == 3) ? OMC_ROUNDS_SR : plain;
};

/* N consecutive elements as ONE aligned load / store of N * sizeof(T) bytes.
 * The 32-bit words are assembled explicitly: through a union the compiler
 * split 8-byte stores of two floats into two 4-byte stores. */
template <class T> __device__ __forceinline__ uint32_t bits32(T v) {
    uint32_t u = 0; memcpy(&u, &v, sizeof(T)); return u;
}
template <class T> __device__ __forceinline__ T from_bits32(uint32_t u) {
    T v; memcpy(&v, &u, sizeof(T)); return v;
}

template <class T, int N>
__device__ __forceinline__ void pack_words(const T (&v)[N], uint32_t (&w)[(N * sizeof(T) + 3) / 4]) {
    if constexpr (sizeof(T) >= 4) {
#pragma unroll
        for (int k = 0; k < N; k++) memcpy(&w[k * sizeof(T) / 4], &v[k], sizeof(T));
    } else {
        constexpr int PER = 4 / (int)sizeof(T);
#pragma unroll
        for (int i = 0; i < (N * (int)sizeof(T) + 3) / 4; i++) {
            uint32_t x = 0;
#pragma unroll
            for (int q = 0; q < PER; q++)
                if (i * PER + q < N) x |= bits32(v[i * PER + q]) << (8 * (int)sizeof(T) * q);
            w[i] = x;
        }
    }
}
template <class T, int N>
__device__ __forceinline__ void unpack_words(const uint32_t (&w)[(N * sizeof(T) + 3) / 4], T (&v)[N]) {
    if constexpr (sizeof(T) >= 4) {
#pragma unroll
        for (int k = 0; k < N; k++) memcpy(&v[k], &w[k * sizeof(T) / 4], sizeof(T));
    } else {
        constexpr int PER = 4 / (int)sizeof(T);
        constexpr uint32_t mask = (sizeof(T) == 1) ? 0xffu : 0xffffu;
#pragma unroll
        for (int k = 0; k < N; k++) v[k] = from_bits32<T>((w[k / PER] >> (8 * (int)sizeof(T) * (k % PER))) & mask);
    }
}

template <class T, int N>
__device__ __forceinline__ void vload(const T *p, T (&v)[N]) {
    constexpr int B = N * (int)sizeof(T);
    uint32_t w[(B + 3) / 4];
    if constexpr (B == 16) { const uint4 x = *reinterpret_cast<const uint4 *>(p); w[0] = x.x; w[1] = x.y; w[2] = x.z; w[3] = x.w; }
    else if constexpr (B == 8) { const uint2 x = *reinterpret_cast<const uint2 *>(p); w[0] = x.x; w[1] = x.y; }
    else if constexpr (B == 4) { w[0] = *reinterpret_cast<const uint32_t *>(p); }
    else if constexpr (B == 2) { w[0] = *reinterpret_cast<const uint16_t *>(p); }
    else { w[0] = *reinterpret_cast<const uint8_t *>(p); }
    unpack_words<T, N>(w, v);
}
/* 8- and 16-byte stores in inline PTX: the scalar fallback path stores to the
 * same addresses, and NVVM otherwise sinks both into one store with the
 * alignment of the fallback (two st.global.u32 instead of one v2). */
template <class T, int N>
__device__ __forceinline__ void vstore(T *p, const T (&v)[N]) {
    constexpr int B = N * (int)sizeof(T);
    uint32_t w[(B + 3) / 4];
    pack_words<T, N>(v, w);
#if defined(__CUDA_ARCH__)
    if constexpr (B == 16)
        asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" :: "l"(p), "r"(w[0]), "r"(w[1]), "r"(w[2]), "r"(w[3]) : "memory");
    else if constexpr (B == 8)
        asm volatile("st.global.v2.u32 [%0], {%1, %2};" :: "l"(p), "r"(w[0]), "r"(w[1]) : "memory");
#else
    if constexpr (B == 16) *reinterpret_cast<uint4 *>(p) = make_uint4(w[0], w[1], w[2], w[3]);
    else if constexpr (B == 8) *reinterpret_cast<uint2 *>(p) = make_uint2(w[0], w[1]);
#endif
    else if constexpr (B == 4) *reinterpret_cast<uint32_t *>(p) = w[0];
    else if constexpr (B == 2) *reinterpret_cast<uint16_t *>(p) = (uint16_t)w[0];
    else *reinterpret_cast<uint8_t *>(p) = (uint8_t)w[0];
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
        else if constexpr (MODE == MODE_SR)     return SR<D, S>::go(s, i, j, seed, o);
        else                                    return sr_scaled<D, S>(s, scale, i, j, seed, o);
    }
};

/* ------------------------------------------------------------------------ */

/* Rows of a column are split in block tiles of blockDim.x * VW * kRounds
 * elements; in round r, thread t converts the VW elements at
 * (r * blockDim.x + t) * VW, so every load and store instruction of a warp
 * covers one contiguous range, on both the narrow and the wide side. */
template <class S, class D, int MODE>
__global__ void __launch_bounds__(kThreads)
k_cvt(const S *src, int64_t lds, D *dst, int64_t ldd, int64_t m, int64_t n, int64_t dr, int64_t dc,
      ElemOp<S, D, MODE> op, const float *scale_ptr, int scale_index, int vec) {
    constexpr int VW = Vw<S, D>::value;
    constexpr int kRounds = Rounds<S, D, MODE>::value;
    if ((MODE == MODE_SCALED || MODE == MODE_SR_SCALED) && scale_ptr) op.scale = scale_ptr[scale_index];
    const int64_t tile0 = (int64_t)blockIdx.x * blockDim.x * VW * kRounds;
    if (tile0 >= dr) return;
    for (int64_t j = (int64_t)blockIdx.y * blockDim.y + threadIdx.y; j < dc; j += (int64_t)gridDim.y * blockDim.y) {
        const S *srow = src + j * lds;
        D *drow = dst + j * ldd;
        const bool jin = j < n;
        S v[kRounds][VW];
        bool full[kRounds];
#pragma unroll
        for (int r = 0; r < kRounds; r++) {
            const int64_t i = tile0 + ((int64_t)r * blockDim.x + threadIdx.x) * VW;
            full[r] = vec && jin && (i + VW <= m);
            if (full[r]) vload(srow + i, v[r]);
        }
#pragma unroll
        for (int r = 0; r < kRounds; r++) {
            const int64_t i = tile0 + ((int64_t)r * blockDim.x + threadIdx.x) * VW;
            if (full[r]) {
                D w[VW];
#pragma unroll
                for (int k = 0; k < VW; k++) w[k] = op(v[r][k], (uint32_t)(i + k), (uint32_t)j);
                vstore(drow + i, w);
            } else {
#pragma unroll
                for (int k = 0; k < VW; k++) {
                    const int64_t ii = i + k;
                    if (ii < dr)
                        drow[ii] = (ii < m && jin) ? op(srow[ii], (uint32_t)ii, (uint32_t)j) : Zero<D>::get();
                }
            }
        }
    }
}

/* 64 x 64 tiles. Read: 64 threads along the source rows, 4 source columns
 * per pass. The tile is kept as tile[source row][source column] with a row
 * pitch of 64 elements plus 4 bytes (8 for 8-byte types): an odd number of
 * bank words, and aligned for the element and the 4-byte writes. Write:
 * each thread stores W = 4 / sizeof(D) consecutive destination elements (one
 * 4-byte word for FP8 and 16-bit targets). */
template <class D> struct TP {
    static constexpr int W     = sizeof(D) >= 4 ? 1 : (int)(4 / sizeof(D));
    static constexpr int pitch = (int)(64 * sizeof(D) + (sizeof(D) > 4 ? sizeof(D) : 4));    /* bytes */
};

template <class S, class D, int MODE>
__global__ void __launch_bounds__(kThreads)
k_cvtT(const S *src, int64_t lds, D *dst, int64_t ldd, int64_t m, int64_t n, int64_t dr, int64_t dc,
       ElemOp<S, D, MODE> op, const float *scale_ptr, int scale_index, int vec) {
    constexpr int W = TP<D>::W, P = TP<D>::pitch;
    __shared__ __align__(16) unsigned char tile_raw[64 * P];
    if ((MODE == MODE_SCALED || MODE == MODE_SR_SCALED) && scale_ptr) op.scale = scale_ptr[scale_index];
    const int t = threadIdx.x;
    const int64_t i0 = (int64_t)blockIdx.x * 64;            /* source rows = dst columns */
    for (int64_t j0 = (int64_t)blockIdx.y * 64; j0 < dr; j0 += (int64_t)gridDim.y * 64) {   /* source columns = dst rows */
        {   /* read: thread (ti = t % 64, tj = t / 64), columns tj, tj + 4, ...;
             * the OMC_T_LOADS loads of a thread are issued before any
             * conversion, so that they are in flight together */
#ifndef OMC_T_LOADS
#define OMC_T_LOADS 16
#endif
            const int ti = t & 63, tj = t >> 6;
            const int64_t i = i0 + ti;
            constexpr int NL = OMC_T_LOADS;
#pragma unroll
            for (int jb = 0; jb < 16; jb += NL) {
                S v[NL];
                bool in[NL];
#pragma unroll
                for (int q = 0; q < NL; q++) {
                    const int64_t j = j0 + tj + 4 * (jb + q);
                    in[q] = (i < m && j < n);
                    if (in[q]) v[q] = src[j * lds + i];
                }
#pragma unroll
                for (int q = 0; q < NL; q++) {
                    const int jj = tj + 4 * (jb + q);
                    const int64_t j = j0 + jj;
                    const D w = in[q] ? op(v[q], (uint32_t)i, (uint32_t)j) : Zero<D>::get();
                    *reinterpret_cast<D *>(tile_raw + ti * P + jj * (int)sizeof(D)) = w;
                }
            }
        }
        __syncthreads();
        {   /* write: 64 / W threads per destination column */
            constexpr int TPC = 64 / W;              /* threads per column */
            constexpr int CPP = kThreads / TPC;      /* columns per pass */
            const int tr = t % TPC, tc = t / TPC;
            const int64_t r0 = j0 + (int64_t)tr * W;
#pragma unroll 4
            for (int cc = tc; cc < 64; cc += CPP) {
                const int64_t c = i0 + cc;
                if (c >= dc) break;
                const unsigned char *tp = tile_raw + cc * P + tr * W * (int)sizeof(D);
                D *dp = dst + c * ldd + r0;
                if (W > 1 && vec && r0 + W <= dr) {
                    *reinterpret_cast<uint32_t *>(dp) = *reinterpret_cast<const uint32_t *>(tp);
                } else {
#pragma unroll
                    for (int k = 0; k < W; k++)
                        if (r0 + k < dr) dp[k] = *reinterpret_cast<const D *>(tp + k * (int)sizeof(D));
                }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ uint32_t load_flag(const uint32_t *p) {
#if defined(__HIPCC__)
    return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
#else
    return *reinterpret_cast<const volatile uint32_t *>(p);
#endif
}

/* Element e of the linear range [0, count) is (i, j) = (e % ld, e / ld); with
 * ld > m (strided) the elements with i >= m are gaps, neither read nor
 * written. Chunks are ranges of the linear index, so the ordering argument is
 * the same as for a contiguous buffer. */
template <class S, class D, int MODE>
__global__ void __launch_bounds__(kThreads)
k_inplace(char *buf, int64_t count, int64_t m, int64_t ld, int strided, int vec, ElemOp<S, D, MODE> op,
          const float *scale_ptr, int scale_index, uint32_t *ws, uint32_t nchunks) {
    constexpr int VW = Vw<S, D>::value;
    constexpr int kRounds = Rounds<S, D, MODE>::value;
    constexpr int64_t C = (int64_t)kThreads * VW * kRounds;
    constexpr int64_t STEP = (int64_t)kThreads * VW;     /* between the rounds of a thread */
    constexpr bool shrink = sizeof(D) < sizeof(S);
    constexpr bool POS = (MODE == MODE_SR || MODE == MODE_SR_SCALED);
    __shared__ uint32_t s_ticket;
    __shared__ int s_last;
    if ((MODE == MODE_SCALED || MODE == MODE_SR_SCALED) && scale_ptr) op.scale = scale_ptr[scale_index];
    if (threadIdx.x == 0) s_ticket = atomicAdd(&ws[0], 1u);
    __syncthreads();
    const uint32_t chunk = shrink ? s_ticket : (nchunks - 1u - s_ticket);
    const S *src = reinterpret_cast<const S *>(buf);
    D *dst = reinterpret_cast<D *>(buf);
    const int64_t c0 = (int64_t)chunk * C;
    const bool need_pos = POS || strided;
    /* (i, j) of the first element of round 0; later rounds step by STEP */
    int64_t pi = 0, pj = 0;
    if (need_pos) { const int64_t e = c0 + (int64_t)threadIdx.x * VW; pj = e / ld; pi = e - pj * ld; }
    D w[kRounds][VW];
    bool vfull[kRounds];
    int64_t ri[kRounds];                   /* row of the round's first element (strided stores) */
#pragma unroll
    for (int r = 0; r < kRounds; r++) {
        const int64_t e0 = c0 + ((int64_t)r * kThreads + threadIdx.x) * VW;
        if (need_pos && r > 0) {
            pi += STEP;
            if (ld >= STEP) { if (pi >= ld) { pi -= ld; pj++; } }
            else            { pj += pi / ld; pi %= ld; }
        }
        uint32_t ci = (uint32_t)pi, cj = (uint32_t)pj;
        ri[r] = pi;
        vfull[r] = vec && (e0 + VW <= count) && (!strided || pi + VW <= m);
        if (vfull[r]) {
            S v[VW];
            vload(src + e0, v);
#pragma unroll
            for (int k = 0; k < VW; k++) {
                w[r][k] = op(v[k], ci, cj);
                if (POS) { if (++ci == (uint32_t)ld) { ci = 0; cj++; } }    /* contiguous: may cross a column */
            }
        } else {
#pragma unroll
            for (int k = 0; k < VW; k++) {
                if (e0 + k < count && (!strided || ci < (uint32_t)m)) w[r][k] = op(src[e0 + k], ci, cj);
                if (need_pos) { if (++ci == (uint32_t)ld) { ci = 0; cj++; } }
            }
        }
    }
    __syncthreads();                       /* the whole chunk is in registers (the conversion consumed the loads) */
    uint32_t *flags = ws + kFlagsOffset;
    if (threadIdx.x == 0) {
        /* no fence before the flag: this block wrote nothing yet, and its
         * loads are complete */
        atomicExch(&flags[chunk], 1u);
        /* input chunks overlapped by this chunk's output */
        const int64_t ob0 = c0 * (int64_t)sizeof(D);
        const int64_t e1  = c0 + C < count ? c0 + C : count;
        const int64_t ob1 = e1 * (int64_t)sizeof(D);
        const int64_t cb  = C * (int64_t)sizeof(S);
        uint32_t lo = (uint32_t)(ob0 / cb), hi = (uint32_t)((ob1 - 1) / cb);
        if (hi > nchunks - 1u) hi = nchunks - 1u;
        for (uint32_t q = lo; q <= hi; q++)     /* device-coherent loads, not read-modify-write atomics */
            if (q != chunk)
                while (load_flag(&flags[q]) == 0u) { }
        __threadfence();                   /* the stores below after the flags were seen */
    }
    __syncthreads();
#pragma unroll
    for (int r = 0; r < kRounds; r++) {
        const int64_t e0 = c0 + ((int64_t)r * kThreads + threadIdx.x) * VW;
        if (vfull[r]) {
            vstore(dst + e0, w[r]);
        } else {
            uint32_t ci = (uint32_t)ri[r];
#pragma unroll
            for (int k = 0; k < VW; k++) {
                if (e0 + k < count && (!strided || ci < (uint32_t)m)) dst[e0 + k] = w[r][k];
                if (strided) { if (++ci == (uint32_t)ld) ci = 0; }
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0)                  /* this block has seen all the flags it needs */
        s_last = (atomicAdd(&ws[1], 1u) == gridDim.x - 1u);
    __syncthreads();
    if (s_last) {                          /* every other block is done: reset */
        for (uint32_t q = threadIdx.x; q < nchunks; q += blockDim.x) flags[q] = 0u;
        if (threadIdx.x == 0) { ws[0] = 0u; ws[1] = 0u; }
        __threadfence();
    }
}

template <class S>
__global__ void __launch_bounds__(kThreads)
k_amax(const S *src, int64_t lds, int64_t m, int64_t n, uint32_t *ws, float *out, int write_scale, float target, int vec) {
    constexpr int VW = 16 / (int)sizeof(S);
    __shared__ float red[kThreads];
    __shared__ int s_last;
    float mx = 0.0f;
    for (int64_t j = blockIdx.y; j < n; j += gridDim.y) {
        const S *col = src + j * lds;
        int64_t i = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * VW;
        const int64_t stride = (int64_t)gridDim.x * blockDim.x * VW;
        if (vec) {
            for (; i + VW <= m; i += stride) {
                S v[VW];
                vload(col + i, v);
#pragma unroll
                for (int k = 0; k < VW; k++) mx = fmaxf(mx, abs_f32(v[k]));
            }
        }
        for (; i < m; i += stride)
            for (int k = 0; k < VW && i + k < m; k++) mx = fmaxf(mx, abs_f32(col[i + k]));
    }
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
    constexpr int VW = Vw<S, D>::value;
    constexpr int kRounds = Rounds<S, D, MODE>::value;
    const S *src = static_cast<const S *>(a.src);
    D *dst = static_cast<D *>(a.dst);
    switch (a.op) {
    case OP_CVT: {
        const int vec = aligned16(src) && aligned16(dst) &&
                        (a.n <= 1 || (((a.lds * (int64_t)sizeof(S)) % 16) == 0 && ((a.ldd * (int64_t)sizeof(D)) % 16) == 0));
        const int64_t groups = (a.dr + VW * kRounds - 1) / (VW * kRounds);   /* threads needed along a column */
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
        const int64_t gx = (a.dc + 63) / 64;
        int64_t gy = (a.dr + 63) / 64;
        if (gy > 65535) gy = 65535;
        if (gx > 0x7fffffff) return OMC_ERROR_NOT_SUPPORTED;
        const int vec = (((uintptr_t)dst & 3u) == 0) && (((a.ldd * (int64_t)sizeof(D)) & 3) == 0);
        k_cvtT<S, D, MODE><<<dim3((unsigned)gx, (unsigned)gy), kThreads, 0, a.stream>>>(
            src, a.lds, dst, a.ldd, a.m, a.n, a.dr, a.dc, op, a.scale_ptr, a.scale_index, vec);
        return check_launch();
    }
    case OP_INPLACE: {
        if constexpr (sizeof(S) == sizeof(D)) {
            return OMC_ERROR_NOT_SUPPORTED;    /* the API layer uses k_cvt in place */
        } else {
            const int64_t ld = a.lds;
            const int64_t count = (a.n - 1) * ld + a.m;          /* linear range covering the matrix */
            const int strided = (ld != a.m) && (a.n > 1);
            const int64_t C = (int64_t)kThreads * VW * kRounds;
            const int64_t nchunks = (count + C - 1) / C;
            const int vec = aligned16(a.dst);             /* else element by element */
            if (nchunks > 0x7fffffff || (size_t)(nchunks + kFlagsOffset) * 4 > a.ws_bytes) return OMC_ERROR_INVALID_VALUE;
            k_inplace<S, D, MODE><<<(unsigned)nchunks, kThreads, 0, a.stream>>>(
                static_cast<char *>(a.dst), count, a.m, a.n > 1 ? ld : a.m, strided, vec, op,
                a.scale_ptr, a.scale_index, static_cast<uint32_t *>(a.ws), (uint32_t)nchunks);
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
        return launch_mode<S, D, MODE_SCALED>(a);
    case MODE_SR:
        if constexpr (sr_ok<D, S>::value) return launch_mode<S, D, MODE_SR>(a);
        else return OMC_ERROR_NOT_SUPPORTED;
    case MODE_SR_SCALED:
        if constexpr (sr_ok<D, S>::value) return launch_mode<S, D, MODE_SR_SCALED>(a);
        else return OMC_ERROR_NOT_SUPPORTED;
    }
    return OMC_ERROR_NOT_SUPPORTED;
}

template <class S>
omc_status_t launch_for_src(const LaunchArgs &a) {
    if (a.op == OP_AMAX) {
        if constexpr (Traits<S>::is_float) {
            constexpr int AV = 16 / (int)sizeof(S);
            int64_t gx = (a.m + kThreads * AV - 1) / (kThreads * AV);
            if (gx > 64) gx = 64;
            if (gx < 1) gx = 1;
            int64_t gy = kMaxAmaxBlocks / gx;
            if (gy > a.n) gy = a.n;
            if (gy < 1) gy = 1;
            if ((size_t)(kMaxAmaxBlocks + 4) * 4 > a.ws_bytes) return OMC_ERROR_INVALID_VALUE;
            const int vec = aligned16(a.src) && (a.n <= 1 || ((a.lds * (int64_t)sizeof(S)) % 16) == 0);
            k_amax<S><<<dim3((unsigned)gx, (unsigned)gy), kThreads, 0, a.stream>>>(
                static_cast<const S *>(a.src), a.lds, a.m, a.n, static_cast<uint32_t *>(a.ws),
                a.amax_out, a.amax_scale, a.scale_max, vec);
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
