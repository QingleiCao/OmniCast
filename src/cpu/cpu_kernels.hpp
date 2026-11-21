/**
 * @file cpu_kernels.hpp
 * @brief CPU kernel table and the generic (auto-vectorized) kernels.
 *
 * This header is compiled once per instruction set (cpu_isa_*.cpp), each time
 * with that ISA's compiler flags and inside its own namespace, so the same
 * branch-free element functions of omc_scalar.hpp vectorize for SSE2, AVX2,
 * AVX-512 or NEON. ISA files then replace the entries for which hand-written
 * intrinsics are faster (FP16 and FP64 conversions on x86 use the hardware
 * converts that compilers do not generate from the integer code).
 */
#ifndef OMC_CPU_KERNELS_HPP
#define OMC_CPU_KERNELS_HPP

#include <stddef.h>
#include <stdint.h>
#include "../core/omc_scalar.hpp"
#include "omnicast/omnicast.h"

namespace omc {
namespace cpu {

/** Parameters seen by a contiguous kernel. For stochastic rounding, element
 *  k of the call is at position (i0 + k, j). */
struct KParams {
    ElemOpts eo;
    float    scale;     /* scaled kernels */
    uint64_t seed;      /* stochastic kernels */
    uint32_t i0, j;
};

/** dst[k] = f(src[k]) for k < n; src and dst do not overlap */
using KFn    = void (*)(const void *src, void *dst, size_t n, const KParams &p);
/** max |src[k]| as double (NaN ignored) */
using AmaxFn = double (*)(const void *src, size_t n);

constexpr int NT = OMC_NUM_DTYPES;

struct Table {
    const char *name;
    KFn    cvt[NT][NT];      /* [src][dst] */
    KFn    scaled[NT][NT];
    KFn    sr[NT][NT];       /* NULL when the pair has no stochastic rounding */
    AmaxFn amax[NT];
};

/* C type of each omc_dtype_t, in enum order */
#define OMC_FOR_EACH_TYPE(X) \
    X(OMC_FP64, double)  X(OMC_FP32, float)   X(OMC_FP16, ::omc::f16) X(OMC_BF16, ::omc::bf16) \
    X(OMC_FP8_E4M3, ::omc::e4m3) X(OMC_FP8_E5M2, ::omc::e5m2) \
    X(OMC_INT8, int8_t)  X(OMC_INT16, int16_t) X(OMC_INT32, int32_t) X(OMC_INT64, int64_t) \
    X(OMC_UINT8, uint8_t) X(OMC_UINT16, uint16_t) X(OMC_UINT32, uint32_t) X(OMC_UINT64, uint64_t)

} /* namespace cpu */
} /* namespace omc */

#endif /* OMC_CPU_KERNELS_HPP */

/* ------------------------------------------------------------------------ */
/* Generic kernels: included by each ISA file with OMC_ISA_NS defined       */
/* ------------------------------------------------------------------------ */
#if defined(OMC_ISA_NS) && !defined(OMC_CPU_GENERIC_DONE)
#define OMC_CPU_GENERIC_DONE

namespace omc {
namespace cpu {
namespace OMC_ISA_NS {

/* The option flags are hoisted out of the loops so that each loop body is
 * constant and vectorizes. */
template <class D, class S, bool SAT, bool RNE>
static inline void loop_cvt(const S *__restrict s, D *__restrict d, size_t n) {
    const ElemOpts o{SAT, RNE};
    for (size_t k = 0; k < n; k++) d[k] = cvt<D>(s[k], o);
}

template <class D, class S>
static void k_cvt(const void *src, void *dst, size_t n, const KParams &p) {
    const S *s = (const S *)src;
    D *d = (D *)dst;
    if (p.eo.sat) {
        if (p.eo.rne) loop_cvt<D, S, true, true>(s, d, n);
        else          loop_cvt<D, S, true, false>(s, d, n);
    } else {
        if (p.eo.rne) loop_cvt<D, S, false, true>(s, d, n);
        else          loop_cvt<D, S, false, false>(s, d, n);
    }
}

template <class D, class S, bool SAT, bool RNE>
static inline void loop_scaled(const S *__restrict s, D *__restrict d, size_t n, float scale) {
    const ElemOpts o{SAT, RNE};
    for (size_t k = 0; k < n; k++) d[k] = cvt_scaled<D>(s[k], scale, o);
}

template <class D, class S>
static void k_scaled(const void *src, void *dst, size_t n, const KParams &p) {
    const S *s = (const S *)src;
    D *d = (D *)dst;
    if (p.eo.sat) { if (p.eo.rne) loop_scaled<D, S, true, true>(s, d, n, p.scale);
                    else          loop_scaled<D, S, true, false>(s, d, n, p.scale); }
    else          { if (p.eo.rne) loop_scaled<D, S, false, true>(s, d, n, p.scale);
                    else          loop_scaled<D, S, false, false>(s, d, n, p.scale); }
}

template <class D, class S, bool SAT>
static inline void loop_sr(const S *__restrict s, D *__restrict d, size_t n, uint32_t i0, uint32_t j, uint64_t seed) {
    const ElemOpts o{SAT, false};
    for (size_t k = 0; k < n; k++) d[k] = SR<D, S>::go(s[k], i0 + (uint32_t)k, j, seed, o);
}

template <class D, class S>
static void k_sr(const void *src, void *dst, size_t n, const KParams &p) {
    if (p.eo.sat) loop_sr<D, S, true>((const S *)src, (D *)dst, n, p.i0, p.j, p.seed);
    else          loop_sr<D, S, false>((const S *)src, (D *)dst, n, p.i0, p.j, p.seed);
}

template <class S>
static double k_amax(const void *src, size_t n) {
    const S *__restrict s = (const S *)src;
    float m = 0.0f;
    for (size_t k = 0; k < n; k++) {
        const float a = abs_f32(s[k]);
        m = (a > m) ? a : m;
    }
    return (double)m;
}

/* FP64 amax in FP64 (the host API reports a double) */
template <>
double k_amax<double>(const void *src, size_t n) {
    const double *__restrict s = (const double *)src;
    double m = 0.0;
    for (size_t k = 0; k < n; k++) {
        const double a = fabs(s[k]);
        m = (a > m) ? a : m;    /* NaN compares false: ignored */
    }
    return m;
}

template <class T> struct is_float_t { static constexpr bool value = Traits<T>::is_float; };

template <class S, class D>
static void fill_pair(Table &t, int si, int di) {
    t.cvt[si][di]    = &k_cvt<D, S>;
    t.scaled[si][di] = &k_scaled<D, S>;
    if constexpr (sr_ok<D, S>::value) t.sr[si][di] = &k_sr<D, S>;
    else                              t.sr[si][di] = nullptr;
}

template <class S>
static void fill_src(Table &t, int si) {
#define OMC_FILL_DST(E, T) fill_pair<S, T>(t, si, (int)E);
    OMC_FOR_EACH_TYPE(OMC_FILL_DST)
#undef OMC_FILL_DST
    if constexpr (Traits<S>::is_float) t.amax[si] = &k_amax<S>;
    else                              t.amax[si] = nullptr;
}

static void fill_generic(Table &t) {
#define OMC_FILL_SRC(E, T) fill_src<T>(t, (int)E);
    OMC_FOR_EACH_TYPE(OMC_FILL_SRC)
#undef OMC_FILL_SRC
}

} /* namespace OMC_ISA_NS */
} /* namespace cpu */
} /* namespace omc */

#endif
