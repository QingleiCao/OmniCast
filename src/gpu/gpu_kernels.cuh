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
 *  k_cvtT_dyn k_amax and k_cvtT in one cooperative kernel for the dynamic
 *             scale: amax over each block's tiles, a grid barrier, then the
 *             transposed conversion of the same tiles in reverse order, the
 *             last read first, mostly from L2. Opt-in (OMNICAST_FUSED_DYNAMIC=1,
 *             see coop_launch in gpu_api.cu).
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
 * from 4- and 2-byte sources (compute bound: fewer registers, more resident
 * warps). The macros are for such experiments. */
/* Defaults measured on V100/H100 (CUDA) and the Radeon VII (HIP), see
 * docs/REPORT.md section 3.3. gfx906 prefers one vector per thread (fewer
 * VGPRs, more waves in flight): 1.1-1.3x over 4 rounds on narrowing and
 * widening conversions. */
#if defined(OMC_BACKEND_HIP)
#define OMC_DEF_NARROW 1
#define OMC_DEF_WIDEN 1
#define OMC_DEF_SR64 2
#else
#define OMC_DEF_NARROW 4
#define OMC_DEF_WIDEN 4
#define OMC_DEF_SR64 4
#endif
#ifndef OMC_ROUNDS_NARROW
#define OMC_ROUNDS_NARROW OMC_DEF_NARROW
#endif
#ifndef OMC_ROUNDS_WIDEN_MAX
#define OMC_ROUNDS_WIDEN_MAX OMC_DEF_WIDEN
#endif
#ifndef OMC_ROUNDS_SR
#define OMC_ROUNDS_SR 2
#endif
#ifndef OMC_ROUNDS_SR64
#define OMC_ROUNDS_SR64 OMC_DEF_SR64
#endif
#ifndef OMC_ROUNDS_INPLACE
#define OMC_ROUNDS_INPLACE 4    /* bigger chunks: fewer tickets and flags */
#endif
template <class S, class D, int MODE, int NARROW = OMC_ROUNDS_NARROW> struct Rounds {
    static constexpr int grow = sizeof(D) > sizeof(S) ? (int)(sizeof(D) / sizeof(S)) : 1;
    static constexpr int widen = (4 * grow < OMC_ROUNDS_WIDEN_MAX) ? 4 * grow : OMC_ROUNDS_WIDEN_MAX;
    static constexpr int plain = grow > 1 ? (widen > NARROW ? widen : NARROW) : NARROW;
    /* stochastic rounding: 2 rounds (compute bound); from FP64 the 8-byte
     * reads dominate: 4 rounds on NVIDIA (4-5 % faster), 2 on gfx906 */
    static constexpr int sr = sizeof(S) == 8 ? OMC_ROUNDS_SR64 : OMC_ROUNDS_SR;
    static constexpr int value = (MODE == 2 || MODE == 3) ? sr : plain;
};
/* k_inplace: chunks of OMC_ROUNDS_INPLACE rounds on every backend */
template <class S, class D, int MODE> using RoundsIP = Rounds<S, D, MODE, OMC_ROUNDS_INPLACE>;

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

/* 1-byte sources have 256 values: each block computes the 256 results in
 * shared memory and converts an element with one lookup. FP8 -> FP8/BF16/
 * int8 conversions were compute bound (about 40 integer operations per
 * element for 2 bytes moved). */
template <class S, int MODE> struct UseLut {
    static constexpr bool value = sizeof(S) == 1 && (MODE == MODE_PLAIN || MODE == MODE_SCALED);
};
template <class S, class D, int MODE>
struct Conv {
    ElemOp<S, D, MODE> op;
    const D *lut;
    __device__ __forceinline__ D operator()(S s, uint32_t i, uint32_t j) const {
        if constexpr (UseLut<S, MODE>::value) return lut[bits32(s) & 0xffu];
        else return op(s, i, j);
    }
};
/* fills the table (all threads of the block, then a barrier) */
template <class S, class D, int MODE>
__device__ __forceinline__ void fill_lut(D *lut, const ElemOp<S, D, MODE> &op) {
    if constexpr (UseLut<S, MODE>::value) {
        const int t = (int)(threadIdx.y * blockDim.x + threadIdx.x), nt = (int)(blockDim.x * blockDim.y);
        for (int q = t; q < 256; q += nt) lut[q] = op(from_bits32<S>((uint32_t)q), 0u, 0u);
        __syncthreads();
    }
}
#define OMC_LUT_DECL(S, D, MODE) \
    __shared__ __align__(16) unsigned char lut_raw_[UseLut<S, MODE>::value ? 256 * sizeof(D) : 16]; \
    D *lut_ = reinterpret_cast<D *>(lut_raw_)

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
    if (tile0 >= dr) return;                 /* the whole block */
    OMC_LUT_DECL(S, D, MODE);
    fill_lut<S, D, MODE>(lut_, op);
    const Conv<S, D, MODE> cv{op, lut_};
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
                for (int k = 0; k < VW; k++) w[k] = cv(v[r][k], (uint32_t)(i + k), (uint32_t)j);
                vstore(drow + i, w);
            } else {
#pragma unroll
                for (int k = 0; k < VW; k++) {
                    const int64_t ii = i + k;
                    if (ii < dr)
                        drow[ii] = (ii < m && jin) ? cv(srow[ii], (uint32_t)ii, (uint32_t)j) : Zero<D>::get();
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

/** One 64 x 64 tile of k_cvtT: source rows i0.., source columns j0.. */
template <class S, class D, int MODE>
__device__ __forceinline__ void cvtT_tile(const S *src, int64_t lds, D *dst, int64_t ldd, int64_t m, int64_t n,
                                          int64_t dr, int64_t dc, const Conv<S, D, MODE> &cv, int64_t i0,
                                          int64_t j0, int vec, unsigned char *tile_raw) {
    constexpr int W = TP<D>::W, P = TP<D>::pitch;
    const int t = threadIdx.x;
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
                    const D w = in[q] ? cv(v[q], (uint32_t)i, (uint32_t)j) : Zero<D>::get();
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

template <class S, class D, int MODE>
__global__ void __launch_bounds__(kThreads)
k_cvtT(const S *src, int64_t lds, D *dst, int64_t ldd, int64_t m, int64_t n, int64_t dr, int64_t dc,
       ElemOp<S, D, MODE> op, const float *scale_ptr, int scale_index, int vec) {
    __shared__ __align__(16) unsigned char tile_raw[64 * TP<D>::pitch];
    if ((MODE == MODE_SCALED || MODE == MODE_SR_SCALED) && scale_ptr) op.scale = scale_ptr[scale_index];
    OMC_LUT_DECL(S, D, MODE);
    fill_lut<S, D, MODE>(lut_, op);
    const Conv<S, D, MODE> cv{op, lut_};
    const int64_t i0 = (int64_t)blockIdx.x * 64;            /* source rows = dst columns */
    for (int64_t j0 = (int64_t)blockIdx.y * 64; j0 < dr; j0 += (int64_t)gridDim.y * 64)   /* source columns = dst rows */
        cvtT_tile<S, D, MODE>(src, lds, dst, ldd, m, n, dr, dc, cv, i0, j0, vec, tile_raw);
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
    constexpr int kRounds = RoundsIP<S, D, MODE>::value;
    constexpr int64_t C = (int64_t)kThreads * VW * kRounds;
    constexpr int64_t STEP = (int64_t)kThreads * VW;     /* between the rounds of a thread */
    constexpr bool shrink = sizeof(D) < sizeof(S);
    constexpr bool POS = (MODE == MODE_SR || MODE == MODE_SR_SCALED);
    __shared__ uint32_t s_ticket;
    __shared__ int s_last;
    if ((MODE == MODE_SCALED || MODE == MODE_SR_SCALED) && scale_ptr) op.scale = scale_ptr[scale_index];
    OMC_LUT_DECL(S, D, MODE);
    if constexpr (UseLut<S, MODE>::value)    /* the barrier below covers the table */
        for (int q = (int)threadIdx.x; q < 256; q += (int)blockDim.x) lut_[q] = op(from_bits32<S>((uint32_t)q), 0u, 0u);
    const Conv<S, D, MODE> cv{op, lut_};
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
                w[r][k] = cv(v[k], ci, cj);
                if (POS) { if (++ci == (uint32_t)ld) { ci = 0; cj++; } }    /* contiguous: may cross a column */
            }
        } else {
#pragma unroll
            for (int k = 0; k < VW; k++) {
                if (e0 + k < count && (!strided || ci < (uint32_t)m)) w[r][k] = cv(src[e0 + k], ci, cj);
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
    /* AU columns per pass, their loads issued together: one 16-byte load in
     * flight per thread was too few on the Radeon VII (the dynamic-scale
     * convert got 1.9x faster at 1024^2 with four) */
    constexpr int AU = 4;
    const int64_t stride = (int64_t)gridDim.x * blockDim.x * VW;
    const int64_t i0 = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) * VW;
    for (int64_t j0 = blockIdx.y; j0 < n; j0 += (int64_t)gridDim.y * AU) {
        int64_t i = i0;
        if (vec) {
            for (; i + VW <= m; i += stride) {
                S v[AU][VW];
#pragma unroll
                for (int u = 0; u < AU; u++) {
                    const int64_t j = j0 + (int64_t)u * gridDim.y;
                    if (j < n) vload(src + j * lds + i, v[u]);
                    else
#pragma unroll
                        for (int k = 0; k < VW; k++) v[u][k] = Zero<S>::get();
                }
#pragma unroll
                for (int u = 0; u < AU; u++)
#pragma unroll
                    for (int k = 0; k < VW; k++) mx = fmaxf(mx, abs_f32(v[u][k]));
            }
        }
        for (; i < m; i += stride)
            for (int u = 0; u < AU; u++) {
                const int64_t j = j0 + (int64_t)u * gridDim.y;
                if (j < n)
                    for (int k = 0; k < VW && i + k < m; k++) mx = fmaxf(mx, abs_f32(src[j * lds + i + k]));
            }
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

/* Grid barrier of k_cvtT_dyn: wait while another block has not arrived */
__device__ __forceinline__ void omc_sleep() {
#if defined(OMC_BACKEND_HIP)
    __builtin_amdgcn_s_sleep(1);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    __nanosleep(64);
#endif
}

/* Dynamic power-of-two scale and transposed conversion in one kernel. Needs
 * every block resident (cooperative launch, gridDim.x <= kMaxAmaxBlocks).
 * Block b owns tiles b, b + gridDim.x, ... of the ntx x nty tiles (k_cvtT's
 * tiles, x along the source rows). Same amax as k_amax (the maximum of the
 * same values), so the same scale and result as k_amax + k_cvtT. */
template <class S, class D, int MODE>
__global__ void __launch_bounds__(kThreads)
k_cvtT_dyn(const S *src, int64_t lds, D *dst, int64_t ldd, int64_t m, int64_t n, int64_t dr, int64_t dc,
           ElemOp<S, D, MODE> op, float *out, float target, uint32_t *ws, int vec, int vec_src, int64_t ntx,
           int64_t ntiles) {
    __shared__ __align__(16) unsigned char tile_raw[64 * TP<D>::pitch];
    __shared__ float red[kThreads];
    const int t = threadIdx.x;
    float mx = 0.0f;
    /* amax: 16-byte loads where the tile is whole and the source aligned */
    constexpr int VW = 16 / (int)sizeof(S);       /* elements per load */
    constexpr int VPC = 64 / VW;                  /* loads per tile column */
    constexpr int CPP = kThreads / VPC;           /* tile columns per pass */
    constexpr int NV = 64 / CPP;                  /* loads per thread and tile */
    const int vr = t % VPC, vcol = t / VPC;
    const int ti = t & 63, tj = t >> 6;
    for (int64_t q = blockIdx.x; q < ntiles; q += gridDim.x) {
        const int64_t i0 = (q % ntx) * 64, j0 = (q / ntx) * 64;
        if (vec_src && i0 + 64 <= m && j0 + 64 <= n) {
            S v[NV][VW];
#pragma unroll
            for (int c = 0; c < NV; c++) vload(src + (j0 + vcol + c * CPP) * lds + i0 + vr * VW, v[c]);
#pragma unroll
            for (int c = 0; c < NV; c++)
#pragma unroll
                for (int k = 0; k < VW; k++) mx = fmaxf(mx, abs_f32(v[c][k]));
        } else if (i0 + ti < m) {
            const int64_t i = i0 + ti;
            S v[16];
#pragma unroll
            for (int c = 0; c < 16; c++) {
                const int64_t j = j0 + tj + 4 * c;
                v[c] = (j < n) ? src[j * lds + i] : Zero<S>::get();
            }
#pragma unroll
            for (int c = 0; c < 16; c++) mx = fmaxf(mx, abs_f32(v[c]));
        }
    }
    red[t] = mx;
    __syncthreads();
    for (int r = kThreads / 2; r > 0; r >>= 1) {
        if (t < r) red[t] = fmaxf(red[t], red[t + r]);
        __syncthreads();
    }
    uint32_t *part = ws + 4;
    if (t == 0) {
        part[blockIdx.x] = __float_as_uint(red[0]);
        __threadfence();
        atomicAdd(&ws[3], 1u);
        while (*(volatile uint32_t *)&ws[3] < gridDim.x) omc_sleep();
        __threadfence();
    }
    __syncthreads();
    float v = 0.0f;
    for (uint32_t q = t; q < gridDim.x; q += kThreads) v = fmaxf(v, __uint_as_float(*(volatile uint32_t *)&part[q]));
    red[t] = v;
    __syncthreads();
    for (int r = kThreads / 2; r > 0; r >>= 1) {
        if (t < r) red[t] = fmaxf(red[t], red[t + r]);
        __syncthreads();
    }
    const float amax = red[0];
    const float s = pow2_scale(amax, target);
    if (t == 0) {
        if (blockIdx.x == 0) { out[0] = amax; out[1] = s; out[2] = 1.0f / s; }
        /* the last block past the barrier leaves the counters zero */
        if (atomicAdd(&ws[2], 1u) == gridDim.x - 1u) { ws[3] = 0u; ws[2] = 0u; __threadfence(); }
    }
    op.scale = s;
    OMC_LUT_DECL(S, D, MODE);
    fill_lut<S, D, MODE>(lut_, op);
    const Conv<S, D, MODE> cv{op, lut_};
    const int64_t last = blockIdx.x + ((ntiles - 1 - blockIdx.x) / gridDim.x) * gridDim.x;
    for (int64_t q = last; q >= (int64_t)blockIdx.x; q -= gridDim.x)
        cvtT_tile<S, D, MODE>(src, lds, dst, ldd, m, n, dr, dc, cv, (q % ntx) * 64, (q / ntx) * 64, vec, tile_raw);
}

/* ------------------------------------------------------------------------ */
/* Host-side launch                                                          */
/* ------------------------------------------------------------------------ */

inline omc_status_t check_launch() {
    return (gpuGetLastError() == gpuSuccess) ? OMC_SUCCESS : OMC_ERROR_DEVICE;
}

inline bool aligned16(const void *p) { return ((uintptr_t)p & 15u) == 0; }

/** k_amax over a's source, writing {amax} or {amax, s, 1/s} to out */
template <class S>
omc_status_t launch_amax(const LaunchArgs &a, float *out, int write_scale) {
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
        static_cast<const S *>(a.src), a.lds, a.m, a.n, static_cast<uint32_t *>(a.ws), out, write_scale,
        a.scale_max, vec);
    return check_launch();
}

/** k_cvtT_dyn as a cooperative kernel; an error (nothing launched) when the
 *  device cannot run it */
template <class S, class D, int MODE>
omc_status_t launch_cvtT_dyn(const LaunchArgs &a, const ElemOp<S, D, MODE> &op, int vec) {
    static int per_sm[64];   /* resident blocks per SM, by device (0: not known yet) */
    if (a.device < 0 || a.device >= 64) return OMC_ERROR_NOT_SUPPORTED;
    if (per_sm[a.device] == 0) {
        int v = 0;
        if (gpuOccupancyMaxActiveBlocksPerMultiprocessor(&v, k_cvtT_dyn<S, D, MODE>, kThreads, 0) != gpuSuccess) {
            gpuGetLastError();
            v = -1;
        }
        per_sm[a.device] = v > 0 ? v : -1;
    }
    if (per_sm[a.device] < 0) return OMC_ERROR_NOT_SUPPORTED;
    int64_t ntx = (a.dc + 63) / 64, ntiles = ntx * ((a.dr + 63) / 64);
    int64_t grid = (int64_t)per_sm[a.device] * a.num_sms;
    if (grid > kMaxAmaxBlocks) grid = kMaxAmaxBlocks;
    if (grid > ntiles) grid = ntiles;
    if (grid < 1 || (size_t)(4 + kMaxAmaxBlocks) * 4 > a.ws_bytes) return OMC_ERROR_NOT_SUPPORTED;
    const S *src = static_cast<const S *>(a.src);
    D *dst = static_cast<D *>(a.dst);
    int64_t lds = a.lds, ldd = a.ldd, m = a.m, n = a.n, dr = a.dr, dc = a.dc;
    ElemOp<S, D, MODE> opv = op;
    float *out = a.amax_out;
    float target = a.scale_max;
    uint32_t *ws = static_cast<uint32_t *>(a.ws);
    int vecv = vec;
    int vec_src = aligned16(src) && (n <= 1 || ((lds * (int64_t)sizeof(S)) % 16) == 0);
    void *args[] = {(void *)&src, (void *)&lds, (void *)&dst, (void *)&ldd, (void *)&m, (void *)&n,
                    (void *)&dr, (void *)&dc, (void *)&opv, (void *)&out, (void *)&target, (void *)&ws,
                    (void *)&vecv, (void *)&vec_src, (void *)&ntx, (void *)&ntiles};
    if (gpuLaunchCooperativeKernel((const void *)k_cvtT_dyn<S, D, MODE>, dim3((unsigned)grid), dim3(kThreads),
                                   args, 0, a.stream) != gpuSuccess) {
        gpuGetLastError();
        return OMC_ERROR_NOT_SUPPORTED;
    }
    return OMC_SUCCESS;
}

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
        if constexpr (Traits<S>::is_float && (MODE == MODE_SCALED || MODE == MODE_SR_SCALED)) {
            if (a.fused_amax) {
                if (launch_cvtT_dyn<S, D, MODE>(a, op, vec) == OMC_SUCCESS) return OMC_SUCCESS;
                /* no cooperative launch: amax pass first */
                const omc_status_t r = launch_amax<S>(a, a.amax_out, 1);
                if (r != OMC_SUCCESS) return r;
                k_cvtT<S, D, MODE><<<dim3((unsigned)gx, (unsigned)gy), kThreads, 0, a.stream>>>(
                    src, a.lds, dst, a.ldd, a.m, a.n, a.dr, a.dc, op, a.amax_out, 1, vec);
                return check_launch();
            }
        }
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
            const int64_t C = (int64_t)kThreads * VW * RoundsIP<S, D, MODE>::value;
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
            return launch_amax<S>(a, a.amax_out, a.amax_scale);
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
