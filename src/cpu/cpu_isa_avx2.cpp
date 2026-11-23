/** @file cpu_isa_avx2.cpp
 *  @brief AVX2 + F16C + FMA kernels (Haswell and later, Zen and later).
 *  Compiled with -mavx2 -mf16c -mfma. */
#define OMC_ISA_NS isa_avx2
#include "cpu_kernels.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

namespace omc {
namespace cpu {
namespace isa_avx2 {

/* ---- helpers ------------------------------------------------------------ */

/* FP16 saturation: +-Inf -> +-65504 */
static inline __m128i sat_f16(__m128i h) {
    const __m128i mag = _mm_and_si128(h, _mm_set1_epi16(0x7fff));
    const __m128i inf = _mm_cmpeq_epi16(mag, _mm_set1_epi16(0x7c00));
    return _mm_sub_epi16(h, _mm_and_si128(inf, _mm_set1_epi16(1)));   /* 0x7c00 - 1 = 0x7bff */
}

/* FP32 saturation: +-Inf -> +-FLT_MAX */
static inline __m128 sat_f32_128(__m128 f) {
    const __m128i u   = _mm_castps_si128(f);
    const __m128i mag = _mm_and_si128(u, _mm_set1_epi32(0x7fffffff));
    const __m128i inf = _mm_cmpeq_epi32(mag, _mm_set1_epi32(0x7f800000));
    return _mm_castsi128_ps(_mm_sub_epi32(u, _mm_and_si128(inf, _mm_set1_epi32(1))));
}

/* 4 doubles -> 4 floats rounded to odd */
static inline __m128 ro_pd_ps(__m256d d) {
    const __m128  f    = _mm256_cvtpd_ps(d);                       /* nearest */
    const __m256d b    = _mm256_cvtps_pd(f);
    const __m256d sgn  = _mm256_set1_pd(-0.0);
    const __m256d inex = _mm256_cmp_pd(b, d, _CMP_NEQ_OQ);         /* NaN: false */
    const __m256d down = _mm256_cmp_pd(_mm256_andnot_pd(sgn, b), _mm256_andnot_pd(sgn, d), _CMP_GT_OQ);
    /* 64-bit lane masks -> 32-bit lanes */
    const __m256i idx  = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    const __m128i inex32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(inex), idx));
    const __m128i down32 = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(down), idx));
    const __m128i u    = _mm_castps_si128(f);
    const __m128i even = _mm_cmpeq_epi32(_mm_and_si128(u, _mm_set1_epi32(1)), _mm_setzero_si128());
    const __m128i adj  = _mm_and_si128(inex32, even);
    const __m128i delta = _mm_and_si128(_mm_or_si128(down32, _mm_set1_epi32(1)), adj);   /* -1, +1 or 0 */
    return _mm_castsi128_ps(_mm_add_epi32(u, delta));
}

/* ---- FP64 <-> FP32 ------------------------------------------------------ */

template <bool SAT>
static void d2s(const void *src, void *dst, size_t n, const KParams &) {
    const double *s = (const double *)src; float *d = (float *)dst;
    size_t k = 0;
    for (; k + 16 <= n; k += 16) {
        __m128 f0 = _mm256_cvtpd_ps(_mm256_loadu_pd(s + k));
        __m128 f1 = _mm256_cvtpd_ps(_mm256_loadu_pd(s + k + 4));
        __m128 f2 = _mm256_cvtpd_ps(_mm256_loadu_pd(s + k + 8));
        __m128 f3 = _mm256_cvtpd_ps(_mm256_loadu_pd(s + k + 12));
        if (SAT) { f0 = sat_f32_128(f0); f1 = sat_f32_128(f1); f2 = sat_f32_128(f2); f3 = sat_f32_128(f3); }
        _mm256_storeu_ps(d + k,     _mm256_set_m128(f1, f0));
        _mm256_storeu_ps(d + k + 8, _mm256_set_m128(f3, f2));
    }
    const ElemOpts o{SAT, false};
    for (; k < n; k++) d[k] = cvt<float>(s[k], o);
}

static void s2d(const void *src, void *dst, size_t n, const KParams &) {
    const float *s = (const float *)src; double *d = (double *)dst;
    size_t k = 0;
    for (; k + 16 <= n; k += 16) {
        __m256 a = _mm256_loadu_ps(s + k), b = _mm256_loadu_ps(s + k + 8);
        _mm256_storeu_pd(d + k,      _mm256_cvtps_pd(_mm256_castps256_ps128(a)));
        _mm256_storeu_pd(d + k + 4,  _mm256_cvtps_pd(_mm256_extractf128_ps(a, 1)));
        _mm256_storeu_pd(d + k + 8,  _mm256_cvtps_pd(_mm256_castps256_ps128(b)));
        _mm256_storeu_pd(d + k + 12, _mm256_cvtps_pd(_mm256_extractf128_ps(b, 1)));
    }
    for (; k < n; k++) d[k] = (double)s[k];
}

/* ---- FP32 <-> FP16 ------------------------------------------------------ */

template <bool SAT>
static void s2h(const void *src, void *dst, size_t n, const KParams &) {
    const float *s = (const float *)src; uint16_t *d = (uint16_t *)dst;
    size_t k = 0;
    for (; k + 32 <= n; k += 32) {
        __m128i h0 = _mm256_cvtps_ph(_mm256_loadu_ps(s + k),      _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h1 = _mm256_cvtps_ph(_mm256_loadu_ps(s + k + 8),  _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h2 = _mm256_cvtps_ph(_mm256_loadu_ps(s + k + 16), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h3 = _mm256_cvtps_ph(_mm256_loadu_ps(s + k + 24), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        if (SAT) { h0 = sat_f16(h0); h1 = sat_f16(h1); h2 = sat_f16(h2); h3 = sat_f16(h3); }
        _mm_storeu_si128((__m128i *)(d + k),      h0);
        _mm_storeu_si128((__m128i *)(d + k + 8),  h1);
        _mm_storeu_si128((__m128i *)(d + k + 16), h2);
        _mm_storeu_si128((__m128i *)(d + k + 24), h3);
    }
    for (; k + 8 <= n; k += 8) {
        __m128i h = _mm256_cvtps_ph(_mm256_loadu_ps(s + k), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        if (SAT) h = sat_f16(h);
        _mm_storeu_si128((__m128i *)(d + k), h);
    }
    const ElemOpts o{SAT, false};
    for (; k < n; k++) d[k] = cvt<f16>(s[k], o).bits;
}

static void h2s(const void *src, void *dst, size_t n, const KParams &) {
    const uint16_t *s = (const uint16_t *)src; float *d = (float *)dst;
    size_t k = 0;
    for (; k + 32 <= n; k += 32) {
        _mm256_storeu_ps(d + k,      _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + k))));
        _mm256_storeu_ps(d + k + 8,  _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + k + 8))));
        _mm256_storeu_ps(d + k + 16, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + k + 16))));
        _mm256_storeu_ps(d + k + 24, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + k + 24))));
    }
    for (; k + 8 <= n; k += 8)
        _mm256_storeu_ps(d + k, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + k))));
    const ElemOpts o{false, false};
    for (; k < n; k++) d[k] = cvt<float>(make<f16>(s[k]), o);
}

/* ---- FP64 <-> FP16 ------------------------------------------------------ */

template <bool SAT>
static void d2h(const void *src, void *dst, size_t n, const KParams &) {
    const double *s = (const double *)src; uint16_t *d = (uint16_t *)dst;
    size_t k = 0;
    for (; k + 16 <= n; k += 16) {
        __m256 f0 = _mm256_set_m128(ro_pd_ps(_mm256_loadu_pd(s + k + 4)),  ro_pd_ps(_mm256_loadu_pd(s + k)));
        __m256 f1 = _mm256_set_m128(ro_pd_ps(_mm256_loadu_pd(s + k + 12)), ro_pd_ps(_mm256_loadu_pd(s + k + 8)));
        __m128i h0 = _mm256_cvtps_ph(f0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m128i h1 = _mm256_cvtps_ph(f1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        if (SAT) { h0 = sat_f16(h0); h1 = sat_f16(h1); }
        _mm_storeu_si128((__m128i *)(d + k),     h0);
        _mm_storeu_si128((__m128i *)(d + k + 8), h1);
    }
    const ElemOpts o{SAT, false};
    for (; k < n; k++) d[k] = cvt<f16>(s[k], o).bits;
}

static void h2d(const void *src, void *dst, size_t n, const KParams &) {
    const uint16_t *s = (const uint16_t *)src; double *d = (double *)dst;
    size_t k = 0;
    for (; k + 16 <= n; k += 16) {
        __m256 a = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + k)));
        __m256 b = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + k + 8)));
        _mm256_storeu_pd(d + k,      _mm256_cvtps_pd(_mm256_castps256_ps128(a)));
        _mm256_storeu_pd(d + k + 4,  _mm256_cvtps_pd(_mm256_extractf128_ps(a, 1)));
        _mm256_storeu_pd(d + k + 8,  _mm256_cvtps_pd(_mm256_castps256_ps128(b)));
        _mm256_storeu_pd(d + k + 12, _mm256_cvtps_pd(_mm256_extractf128_ps(b, 1)));
    }
    const ElemOpts o{false, false};
    for (; k < n; k++) d[k] = cvt<double>(make<f16>(s[k]), o);
}

/* ---- FP32 -> FP8 through F16C ------------------------------------------ */
/* x * 2^k (exact) to FP16 rounded to odd: truncation, then the last bit set
 * when that was inexact. With k = -8 for E4M3 (bias 7 against 15) and k = 0
 * for E5M2 (the FP16 exponent), the FP16 pattern lines up with the FP8 one,
 * subnormals on subnormals, so rounding to nearest even on the 16-bit
 * pattern finishes the job: about 2 operations per element against 20 for
 * the integer encoder. Rounding to odd first makes the two roundings one
 * (11 significand bits >= 4 + 2). Bit-identical to enc_f32 (test_cpu checks
 * every FP32 pattern). Assumes the default MXCSR (no flush to zero). */
static inline __m128i ro_ph(__m256 x) {
    const __m128i h = _mm256_cvtps_ph(x, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __m256 back = _mm256_cvtph_ps(h);
    const __m256i inex = _mm256_castps_si256(_mm256_cmp_ps(back, x, _CMP_NEQ_UQ));   /* NaN: set, then replaced */
    const __m128i m16 = _mm_packs_epi32(_mm256_castsi256_si128(inex), _mm256_extracti128_si256(inex, 1));
    return _mm_or_si128(h, _mm_and_si128(m16, _mm_set1_epi16(1)));
}
template <class D, bool SAT>
static inline __m128i fp8x16(__m256 a, __m256 b) {
    constexpr bool E4 = std::is_same<D, e4m3>::value;
    constexpr int SH = E4 ? 7 : 8;                          /* FP16 mantissa bits dropped */
    constexpr short MAXC = E4 ? 0x7e : 0x7b, OVF = SAT ? MAXC : (E4 ? 0x7f : 0x7c), NANC = E4 ? 0x7f : 0x7e;
    if (E4) { const __m256 sc = _mm256_set1_ps(0x1p-8f); a = _mm256_mul_ps(a, sc); b = _mm256_mul_ps(b, sc); }
    const __m256i h = _mm256_set_m128i(ro_ph(b), ro_ph(a));
    const __m256i mag = _mm256_and_si256(h, _mm256_set1_epi16(0x7fff));
    const __m256i odd = _mm256_and_si256(_mm256_srli_epi16(mag, SH), _mm256_set1_epi16(1));
    __m256i r = _mm256_srli_epi16(_mm256_add_epi16(_mm256_add_epi16(mag, _mm256_set1_epi16((1 << (SH - 1)) - 1)), odd), SH);
    r = _mm256_blendv_epi8(r, _mm256_set1_epi16(OVF), _mm256_cmpgt_epi16(r, _mm256_set1_epi16(MAXC)));
    r = _mm256_blendv_epi8(r, _mm256_set1_epi16(NANC), _mm256_cmpgt_epi16(mag, _mm256_set1_epi16(0x7c00)));
    r = _mm256_or_si256(r, _mm256_srli_epi16(_mm256_andnot_si256(_mm256_set1_epi16(0x7fff), h), 8));
    return _mm_packus_epi16(_mm256_castsi256_si128(r), _mm256_extracti128_si256(r, 1));
}
template <class D, bool SAT>
static void s2fp8(const void *src, void *dst, size_t n, const KParams &) {
    const float *s = (const float *)src; D *d = (D *)dst;
    size_t k = 0;
    for (; k + 16 <= n; k += 16)
        _mm_storeu_si128((__m128i *)(d + k), fp8x16<D, SAT>(_mm256_loadu_ps(s + k), _mm256_loadu_ps(s + k + 8)));
    const ElemOpts o{SAT, false};
    for (; k < n; k++) d[k] = cvt<D>(s[k], o);
}
/* FP16 sources: exactly to FP32 first */
template <class D, bool SAT>
static void h2fp8(const void *src, void *dst, size_t n, const KParams &) {
    const uint16_t *s = (const uint16_t *)src; D *d = (D *)dst;
    size_t k = 0;
    for (; k + 16 <= n; k += 16)
        _mm_storeu_si128((__m128i *)(d + k), fp8x16<D, SAT>(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + k))),
                                                            _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(s + k + 8)))));
    const ElemOpts o{SAT, false};
    for (; k < n; k++) d[k] = cvt<D>(make<f16>(s[k]), o);
}
/* ---- FP64 -> FP8: rounded to odd to FP32, then fp8x16 ------------------- */
/* With the rounding mode set to toward zero, rounding to odd is the convert
 * plus the last bit of an inexact result (5 operations per 4 elements
 * against 12 for ro_pd_ps). Rounding to odd twice is rounding to odd once.
 * The other operations of fp8x16 are exact or set their rounding. */
static inline __m256 ro_pd8_rz(const double *p) {
    const __m256i idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    __m128 f[2];
    for (int h = 0; h < 2; h++) {
        const __m256d x = _mm256_loadu_pd(p + 4 * h);
        const __m128 c = _mm256_cvtpd_ps(x);                                   /* toward zero */
        const __m256d inex = _mm256_cmp_pd(_mm256_cvtps_pd(c), x, _CMP_NEQ_UQ);  /* NaN stays NaN */
        const __m128i m = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(_mm256_castpd_si256(inex), idx));
        f[h] = _mm_castsi128_ps(_mm_or_si128(_mm_castps_si128(c), _mm_and_si128(m, _mm_set1_epi32(1))));
    }
    return _mm256_set_m128(f[1], f[0]);
}
template <class D, bool SAT>
static void d2fp8(const void *src, void *dst, size_t n, const KParams &) {
    const double *s = (const double *)src; D *d = (D *)dst;
    const unsigned int csr = _mm_getcsr();
    _mm_setcsr(csr | 0x6000u);                                                 /* RC = toward zero */
    size_t k = 0;
    for (; k + 16 <= n; k += 16)
        _mm_storeu_si128((__m128i *)(d + k), fp8x16<D, SAT>(ro_pd8_rz(s + k), ro_pd8_rz(s + k + 8)));
    _mm_setcsr(csr);
    const ElemOpts o{SAT, false};
    for (; k < n; k++) d[k] = cvt<D>(s[k], o);
}
template <class D> static void d2fp8_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? d2fp8<D, true>(a, b, n, p) : d2fp8<D, false>(a, b, n, p); }

/* ---- FP64 -> FP32 rounded to odd, then BF16 ----------------------------- */
/* The generic integer encoder vectorizes; only the FP64 step needs help. */
template <class D>
static void d2small(const void *src, void *dst, size_t n, const KParams &p) {
    const double *s = (const double *)src; D *d = (D *)dst;
    alignas(32) float buf[256];
    size_t k = 0;
    while (k < n) {
        const size_t m = (n - k < 256) ? n - k : 256;
        size_t q = 0;
        for (; q + 4 <= m; q += 4) _mm_store_ps(buf + q, ro_pd_ps(_mm256_loadu_pd(s + k + q)));
        for (; q < m; q++) buf[q] = f64_to_f32_ro(s[k + q]);
        k_cvt<D, float>(buf, d + k, m, p);
        k += m;
    }
}
/* FP32 -> E4M3 saturated: the integer encoder is as fast (0.95-1.06x on
 * Zen 2, Broadwell and Skylake-SP); 1.6-2.3x for the other FP32 and FP16
 * cases */
template <class D> static void s2fp8_any(const void *a, void *b, size_t n, const KParams &p) {
    if (std::is_same<D, e4m3>::value && p.eo.sat) k_cvt<D, float>(a, b, n, p);
    else if (p.eo.sat) s2fp8<D, true>(a, b, n, p);
    else s2fp8<D, false>(a, b, n, p);
}
template <class D> static void h2fp8_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? h2fp8<D, true>(a, b, n, p) : h2fp8<D, false>(a, b, n, p); }


/* ---- float -> integer (truncation), saturating, NaN -> 0 ----------------- */
/* Clamp in the source precision to [lo, hi], zero the NaNs, truncate with
 * cvtt (exact for in-range values), then narrow with saturating packs. */
static inline __m256i clamp_cvt_ps(__m256 x, float lo, float hi) {
    x = _mm256_and_ps(x, _mm256_cmp_ps(x, x, _CMP_EQ_OQ));          /* NaN -> 0 */
    x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(lo)), _mm256_set1_ps(hi));
    return _mm256_cvttps_epi32(x);
}
static inline __m128i clamp_cvt_pd(__m256d x, double lo, double hi) {
    x = _mm256_and_pd(x, _mm256_cmp_pd(x, x, _CMP_EQ_OQ));
    x = _mm256_min_pd(_mm256_max_pd(x, _mm256_set1_pd(lo)), _mm256_set1_pd(hi));
    return _mm256_cvttpd_epi32(x);
}
/* 32 int32 (4 x 8) -> 32 int8 / int16 in order */
static inline __m256i pack32_i8(__m256i a, __m256i b, __m256i c, __m256i d) {
    const __m256i ab = _mm256_packs_epi32(a, b), cd = _mm256_packs_epi32(c, d);   /* per-lane order */
    const __m256i r = _mm256_packs_epi16(ab, cd);
    return _mm256_permutevar8x32_epi32(r, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}
static inline __m256i pack16_i16(__m256i a, __m256i b) {
    return _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), 0xd8);
}

template <class D>
static void s2int(const void *src, void *dst, size_t n, const KParams &p) {
    if (p.eo.rne) { k_cvt<D, float>(src, dst, n, p); return; }
    const float *s = (const float *)src; D *d = (D *)dst;
    const float lo = (float)IntLimits<D>::lo, hi = (float)IntLimits<D>::hi;
    size_t k = 0;
    if (sizeof(D) == 1) {
        for (; k + 32 <= n; k += 32) {
            const __m256i r = pack32_i8(clamp_cvt_ps(_mm256_loadu_ps(s + k), lo, hi), clamp_cvt_ps(_mm256_loadu_ps(s + k + 8), lo, hi),
                                        clamp_cvt_ps(_mm256_loadu_ps(s + k + 16), lo, hi), clamp_cvt_ps(_mm256_loadu_ps(s + k + 24), lo, hi));
            _mm256_storeu_si256((__m256i *)(d + k), r);
        }
    } else if (sizeof(D) == 2) {
        for (; k + 16 <= n; k += 16) {
            const __m256i r = pack16_i16(clamp_cvt_ps(_mm256_loadu_ps(s + k), lo, hi), clamp_cvt_ps(_mm256_loadu_ps(s + k + 8), lo, hi));
            _mm256_storeu_si256((__m256i *)(d + k), r);
        }
    } else {
        /* int32: 2^31 - 1 is not a float; values >= 2^31 clamp to 2^31 and
         * cvtt gives 0x80000000: fix those up to INT32_MAX */
        for (; k + 8 <= n; k += 8) {
            __m256 x = _mm256_loadu_ps(s + k);
            const __m256 big = _mm256_cmp_ps(x, _mm256_set1_ps(2147483648.0f), _CMP_GE_OQ);
            __m256i r = clamp_cvt_ps(x, -2147483648.0f, 2147483648.0f);
            r = _mm256_blendv_epi8(r, _mm256_set1_epi32(0x7fffffff), _mm256_castps_si256(big));
            _mm256_storeu_si256((__m256i *)(d + k), r);
        }
    }
    const ElemOpts o{p.eo.sat, false};
    for (; k < n; k++) d[k] = cvt<D>(s[k], o);
}

template <class D>
static void d2int(const void *src, void *dst, size_t n, const KParams &p) {
    if (p.eo.rne) { k_cvt<D, double>(src, dst, n, p); return; }
    const double *s = (const double *)src; D *d = (D *)dst;
    const double lo = IntLimits<D>::lo, hi = IntLimits<D>::hi;
    size_t k = 0;
    if (sizeof(D) == 4) {
        for (; k + 8 <= n; k += 8) {
            const __m128i a = clamp_cvt_pd(_mm256_loadu_pd(s + k), lo, hi), b = clamp_cvt_pd(_mm256_loadu_pd(s + k + 4), lo, hi);
            _mm256_storeu_si256((__m256i *)(d + k), _mm256_set_m128i(b, a));
        }
    } else if (sizeof(D) == 2) {
        for (; k + 16 <= n; k += 16) {
            const __m256i a = _mm256_set_m128i(clamp_cvt_pd(_mm256_loadu_pd(s + k + 4), lo, hi), clamp_cvt_pd(_mm256_loadu_pd(s + k), lo, hi));
            const __m256i b = _mm256_set_m128i(clamp_cvt_pd(_mm256_loadu_pd(s + k + 12), lo, hi), clamp_cvt_pd(_mm256_loadu_pd(s + k + 8), lo, hi));
            _mm256_storeu_si256((__m256i *)(d + k), pack16_i16(a, b));
        }
    } else {
        for (; k + 32 <= n; k += 32) {
            __m256i q[4];
            for (int t = 0; t < 4; t++)
                q[t] = _mm256_set_m128i(clamp_cvt_pd(_mm256_loadu_pd(s + k + 8 * t + 4), lo, hi), clamp_cvt_pd(_mm256_loadu_pd(s + k + 8 * t), lo, hi));
            _mm256_storeu_si256((__m256i *)(d + k), pack32_i8(q[0], q[1], q[2], q[3]));
        }
    }
    const ElemOpts o{p.eo.sat, false};
    for (; k < n; k++) d[k] = cvt<D>(s[k], o);
}

static void d2s_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? d2s<true>(a, b, n, p) : d2s<false>(a, b, n, p); }
static void s2h_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? s2h<true>(a, b, n, p) : s2h<false>(a, b, n, p); }
static void d2h_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? d2h<true>(a, b, n, p) : d2h<false>(a, b, n, p); }
/* widening converts: Inf only appears from Inf, which saturation replaces */
static void s2d_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? k_cvt<double, float>(a, b, n, p) : s2d(a, b, n, p); }
static void h2s_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? k_cvt<float, f16>(a, b, n, p) : h2s(a, b, n, p); }
static void h2d_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? k_cvt<double, f16>(a, b, n, p) : h2d(a, b, n, p); }

} /* namespace isa_avx2 */

void fill_table_avx2(Table &t) {
    t.name = "avx2";
    isa_avx2::fill_generic(t);
    t.cvt[OMC_FP64][OMC_FP32] = &isa_avx2::d2s_any;
    t.cvt[OMC_FP32][OMC_FP64] = &isa_avx2::s2d_any;
    t.cvt[OMC_FP32][OMC_FP16] = &isa_avx2::s2h_any;
    t.cvt[OMC_FP16][OMC_FP32] = &isa_avx2::h2s_any;
    t.cvt[OMC_FP64][OMC_FP16] = &isa_avx2::d2h_any;
    t.cvt[OMC_FP16][OMC_FP64] = &isa_avx2::h2d_any;
    t.cvt[OMC_FP64][OMC_BF16]     = &isa_avx2::d2small<bf16>;
    t.cvt[OMC_FP64][OMC_FP8_E4M3] = &isa_avx2::d2fp8_any<e4m3>;
    t.cvt[OMC_FP64][OMC_FP8_E5M2] = &isa_avx2::d2fp8_any<e5m2>;
    t.cvt[OMC_FP32][OMC_FP8_E4M3] = &isa_avx2::s2fp8_any<e4m3>;
    t.cvt[OMC_FP32][OMC_FP8_E5M2] = &isa_avx2::s2fp8_any<e5m2>;
    t.cvt[OMC_FP16][OMC_FP8_E4M3] = &isa_avx2::h2fp8_any<e4m3>;
    t.cvt[OMC_FP16][OMC_FP8_E5M2] = &isa_avx2::h2fp8_any<e5m2>;
    t.cvt[OMC_FP32][OMC_INT8]  = &isa_avx2::s2int<int8_t>;
    t.cvt[OMC_FP32][OMC_INT16] = &isa_avx2::s2int<int16_t>;
    t.cvt[OMC_FP32][OMC_INT32] = &isa_avx2::s2int<int32_t>;
    t.cvt[OMC_FP64][OMC_INT8]  = &isa_avx2::d2int<int8_t>;
    t.cvt[OMC_FP64][OMC_INT16] = &isa_avx2::d2int<int16_t>;
    t.cvt[OMC_FP64][OMC_INT32] = &isa_avx2::d2int<int32_t>;
}

} /* namespace cpu */
} /* namespace omc */

#else
namespace omc { namespace cpu { void fill_table_avx2(Table &t) { t.name = nullptr; } } }
#endif
