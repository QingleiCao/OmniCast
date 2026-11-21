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

/* ---- FP64 -> FP32 rounded to odd, then a small format (BF16, FP8) ------- */
/* The generic integer encoders vectorize; only the FP64 step needs help. */
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
    t.cvt[OMC_FP64][OMC_FP8_E4M3] = &isa_avx2::d2small<e4m3>;
    t.cvt[OMC_FP64][OMC_FP8_E5M2] = &isa_avx2::d2small<e5m2>;
}

} /* namespace cpu */
} /* namespace omc */

#else
namespace omc { namespace cpu { void fill_table_avx2(Table &t) { t.name = nullptr; } } }
#endif
