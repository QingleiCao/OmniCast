/** @file cpu_isa_avx512.cpp
 *  @brief AVX-512 (F, BW, VL, DQ) kernels: Skylake-SP and later, Zen 4 and
 *  later. Compiled with -mavx512f -mavx512bw -mavx512vl -mavx512dq -mf16c
 *  -mfma -mprefer-vector-width=512. */
#define OMC_ISA_NS isa_avx512
#include "cpu_kernels.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

namespace omc {
namespace cpu {
namespace isa_avx512 {

#define OMC_RNE (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)

static inline __m256i sat_f16(__m256i h) {
    const __mmask16 inf = _mm256_cmpeq_epi16_mask(_mm256_and_si256(h, _mm256_set1_epi16(0x7fff)),
                                                  _mm256_set1_epi16(0x7c00));
    return _mm256_mask_sub_epi16(h, inf, h, _mm256_set1_epi16(1));
}
static inline __m256 sat_f32_256(__m256 f) {
    const __m256i u = _mm256_castps_si256(f);
    const __mmask8 inf = _mm256_cmpeq_epi32_mask(_mm256_and_si256(u, _mm256_set1_epi32(0x7fffffff)),
                                                 _mm256_set1_epi32(0x7f800000));
    return _mm256_castsi256_ps(_mm256_mask_sub_epi32(u, inf, u, _mm256_set1_epi32(1)));
}

/* 8 doubles -> 8 floats rounded to odd: truncate, set the last bit if inexact */
static inline __m256 ro_pd_ps(__m512d d) {
    const __m256  f = _mm512_cvt_roundpd_ps(d, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    const __mmask8 inex = _mm512_cmp_pd_mask(_mm512_cvtps_pd(f), d, _CMP_NEQ_UQ);
    const __m256i u = _mm256_castps_si256(f);
    return _mm256_castsi256_ps(_mm256_mask_or_epi32(u, inex, u, _mm256_set1_epi32(1)));
}

template <bool SAT>
static void d2s(const void *src, void *dst, size_t n, const KParams &) {
    const double *s = (const double *)src; float *d = (float *)dst;
    size_t k = 0;
    for (; k + 32 <= n; k += 32) {
        __m256 f0 = _mm512_cvtpd_ps(_mm512_loadu_pd(s + k));
        __m256 f1 = _mm512_cvtpd_ps(_mm512_loadu_pd(s + k + 8));
        __m256 f2 = _mm512_cvtpd_ps(_mm512_loadu_pd(s + k + 16));
        __m256 f3 = _mm512_cvtpd_ps(_mm512_loadu_pd(s + k + 24));
        if (SAT) { f0 = sat_f32_256(f0); f1 = sat_f32_256(f1); f2 = sat_f32_256(f2); f3 = sat_f32_256(f3); }
        _mm512_storeu_ps(d + k,      _mm512_insertf32x8(_mm512_castps256_ps512(f0), f1, 1));
        _mm512_storeu_ps(d + k + 16, _mm512_insertf32x8(_mm512_castps256_ps512(f2), f3, 1));
    }
    if (k < n) {   /* masked tail */
        for (; k < n; k += 8) {
            const size_t r = (n - k < 8) ? n - k : 8;
            const __mmask8 m = (__mmask8)((1u << r) - 1u);
            __m256 f = _mm512_cvtpd_ps(_mm512_maskz_loadu_pd(m, s + k));
            if (SAT) f = sat_f32_256(f);
            _mm256_mask_storeu_ps(d + k, m, f);
        }
    }
}

static void s2d(const void *src, void *dst, size_t n, const KParams &) {
    const float *s = (const float *)src; double *d = (double *)dst;
    size_t k = 0;
    for (; k + 32 <= n; k += 32) {
        __m512 a = _mm512_loadu_ps(s + k), b = _mm512_loadu_ps(s + k + 16);
        _mm512_storeu_pd(d + k,      _mm512_cvtps_pd(_mm512_castps512_ps256(a)));
        _mm512_storeu_pd(d + k + 8,  _mm512_cvtps_pd(_mm512_extractf32x8_ps(a, 1)));
        _mm512_storeu_pd(d + k + 16, _mm512_cvtps_pd(_mm512_castps512_ps256(b)));
        _mm512_storeu_pd(d + k + 24, _mm512_cvtps_pd(_mm512_extractf32x8_ps(b, 1)));
    }
    for (; k < n; k += 8) {
        const size_t r = (n - k < 8) ? n - k : 8;
        const __mmask8 m = (__mmask8)((1u << r) - 1u);
        _mm512_mask_storeu_pd(d + k, m, _mm512_cvtps_pd(_mm256_maskz_loadu_ps(m, s + k)));
    }
}

template <bool SAT>
static void s2h(const void *src, void *dst, size_t n, const KParams &) {
    const float *s = (const float *)src; uint16_t *d = (uint16_t *)dst;
    size_t k = 0;
    for (; k + 64 <= n; k += 64) {
        __m256i h0 = _mm512_cvtps_ph(_mm512_loadu_ps(s + k),      OMC_RNE);
        __m256i h1 = _mm512_cvtps_ph(_mm512_loadu_ps(s + k + 16), OMC_RNE);
        __m256i h2 = _mm512_cvtps_ph(_mm512_loadu_ps(s + k + 32), OMC_RNE);
        __m256i h3 = _mm512_cvtps_ph(_mm512_loadu_ps(s + k + 48), OMC_RNE);
        if (SAT) { h0 = sat_f16(h0); h1 = sat_f16(h1); h2 = sat_f16(h2); h3 = sat_f16(h3); }
        _mm256_storeu_si256((__m256i *)(d + k),      h0);
        _mm256_storeu_si256((__m256i *)(d + k + 16), h1);
        _mm256_storeu_si256((__m256i *)(d + k + 32), h2);
        _mm256_storeu_si256((__m256i *)(d + k + 48), h3);
    }
    for (; k < n; k += 16) {
        const size_t r = (n - k < 16) ? n - k : 16;
        const __mmask16 m = (__mmask16)((1u << r) - 1u);
        __m256i h = _mm512_cvtps_ph(_mm512_maskz_loadu_ps(m, s + k), OMC_RNE);
        if (SAT) h = sat_f16(h);
        _mm256_mask_storeu_epi16(d + k, m, h);
    }
}

static void h2s(const void *src, void *dst, size_t n, const KParams &) {
    const uint16_t *s = (const uint16_t *)src; float *d = (float *)dst;
    size_t k = 0;
    for (; k + 64 <= n; k += 64) {
        _mm512_storeu_ps(d + k,      _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(s + k))));
        _mm512_storeu_ps(d + k + 16, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(s + k + 16))));
        _mm512_storeu_ps(d + k + 32, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(s + k + 32))));
        _mm512_storeu_ps(d + k + 48, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(s + k + 48))));
    }
    for (; k < n; k += 16) {
        const size_t r = (n - k < 16) ? n - k : 16;
        const __mmask16 m = (__mmask16)((1u << r) - 1u);
        _mm512_mask_storeu_ps(d + k, m, _mm512_cvtph_ps(_mm256_maskz_loadu_epi16(m, s + k)));
    }
}

template <bool SAT>
static void d2h(const void *src, void *dst, size_t n, const KParams &) {
    const double *s = (const double *)src; uint16_t *d = (uint16_t *)dst;
    size_t k = 0;
    for (; k + 32 <= n; k += 32) {
        __m512 f0 = _mm512_insertf32x8(_mm512_castps256_ps512(ro_pd_ps(_mm512_loadu_pd(s + k))),
                                       ro_pd_ps(_mm512_loadu_pd(s + k + 8)), 1);
        __m512 f1 = _mm512_insertf32x8(_mm512_castps256_ps512(ro_pd_ps(_mm512_loadu_pd(s + k + 16))),
                                       ro_pd_ps(_mm512_loadu_pd(s + k + 24)), 1);
        __m256i h0 = _mm512_cvtps_ph(f0, OMC_RNE);
        __m256i h1 = _mm512_cvtps_ph(f1, OMC_RNE);
        if (SAT) { h0 = sat_f16(h0); h1 = sat_f16(h1); }
        _mm256_storeu_si256((__m256i *)(d + k),      h0);
        _mm256_storeu_si256((__m256i *)(d + k + 16), h1);
    }
    for (; k < n; k += 8) {
        const size_t r = (n - k < 8) ? n - k : 8;
        const __mmask8 m = (__mmask8)((1u << r) - 1u);
        __m128i h = _mm256_cvtps_ph(ro_pd_ps(_mm512_maskz_loadu_pd(m, s + k)), OMC_RNE);
        if (SAT) h = _mm256_castsi256_si128(sat_f16(_mm256_castsi128_si256(h)));
        _mm_mask_storeu_epi16(d + k, m, h);
    }
}

static void h2d(const void *src, void *dst, size_t n, const KParams &) {
    const uint16_t *s = (const uint16_t *)src; double *d = (double *)dst;
    size_t k = 0;
    for (; k + 32 <= n; k += 32) {
        __m512 a = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(s + k)));
        __m512 b = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(s + k + 16)));
        _mm512_storeu_pd(d + k,      _mm512_cvtps_pd(_mm512_castps512_ps256(a)));
        _mm512_storeu_pd(d + k + 8,  _mm512_cvtps_pd(_mm512_extractf32x8_ps(a, 1)));
        _mm512_storeu_pd(d + k + 16, _mm512_cvtps_pd(_mm512_castps512_ps256(b)));
        _mm512_storeu_pd(d + k + 24, _mm512_cvtps_pd(_mm512_extractf32x8_ps(b, 1)));
    }
    for (; k < n; k += 8) {
        const size_t r = (n - k < 8) ? n - k : 8;
        const __mmask8 m = (__mmask8)((1u << r) - 1u);
        _mm512_mask_storeu_pd(d + k, m, _mm512_cvtps_pd(_mm256_cvtph_ps(_mm_maskz_loadu_epi16(m, s + k))));
    }
}

/* FP64 -> BF16/FP8: round to odd in FP64 SIMD, then the generic encoder */
template <class D>
static void d2small(const void *src, void *dst, size_t n, const KParams &p) {
    const double *s = (const double *)src; D *d = (D *)dst;
    alignas(64) float buf[512];
    size_t k = 0;
    while (k < n) {
        const size_t m = (n - k < 512) ? n - k : 512;
        size_t q = 0;
        for (; q + 8 <= m; q += 8) _mm256_store_ps(buf + q, ro_pd_ps(_mm512_loadu_pd(s + k + q)));
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

} /* namespace isa_avx512 */

void fill_table_avx512(Table &t) {
    t.name = "avx512";
    isa_avx512::fill_generic(t);
    t.cvt[OMC_FP64][OMC_FP32] = &isa_avx512::d2s_any;
    t.cvt[OMC_FP32][OMC_FP64] = &isa_avx512::s2d_any;
    t.cvt[OMC_FP32][OMC_FP16] = &isa_avx512::s2h_any;
    t.cvt[OMC_FP16][OMC_FP32] = &isa_avx512::h2s_any;
    t.cvt[OMC_FP64][OMC_FP16] = &isa_avx512::d2h_any;
    t.cvt[OMC_FP16][OMC_FP64] = &isa_avx512::h2d_any;
    t.cvt[OMC_FP64][OMC_BF16]     = &isa_avx512::d2small<bf16>;
    t.cvt[OMC_FP64][OMC_FP8_E4M3] = &isa_avx512::d2small<e4m3>;
    t.cvt[OMC_FP64][OMC_FP8_E5M2] = &isa_avx512::d2small<e5m2>;
}

} /* namespace cpu */
} /* namespace omc */

#else
namespace omc { namespace cpu { void fill_table_avx512(Table &t) { t.name = nullptr; } } }
#endif
