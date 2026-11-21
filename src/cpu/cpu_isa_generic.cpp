/** @file cpu_isa_generic.cpp
 *  @brief Baseline kernels (SSE2 on x86-64, NEON on AArch64): the generic
 *  kernels, auto-vectorized for the compiler's default target. */
#define OMC_ISA_NS isa_generic
#include "cpu_kernels.hpp"

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace omc {
namespace cpu {

#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
/* AArch64: FCVTN/FCVTL convert FP32 <-> FP16 with round to nearest even and
 * full subnormal support; FCVTXN rounds FP64 -> FP32 to odd. */
namespace isa_generic {
template <bool SAT>
static void neon_s2h(const void *src, void *dst, size_t n, const KParams &) {
    const float *s = (const float *)src; uint16_t *d = (uint16_t *)dst;
    size_t k = 0;
    for (; k + 8 <= n; k += 8) {
        float16x4_t lo = vcvt_f16_f32(vld1q_f32(s + k));
        float16x8_t h  = vcvt_high_f16_f32(lo, vld1q_f32(s + k + 4));
        uint16x8_t  b  = vreinterpretq_u16_f16(h);
        if (SAT) {
            uint16x8_t mag = vandq_u16(b, vdupq_n_u16(0x7fff));
            uint16x8_t inf = vceqq_u16(mag, vdupq_n_u16(0x7c00));
            b = vbslq_u16(inf, vorrq_u16(vandq_u16(b, vdupq_n_u16(0x8000)), vdupq_n_u16(0x7bff)), b);
        }
        vst1q_u16(d + k, b);
    }
    const ElemOpts o{SAT, false};
    for (; k < n; k++) d[k] = cvt<f16>(s[k], o).bits;
}
static void neon_h2s(const void *src, void *dst, size_t n, const KParams &) {
    const uint16_t *s = (const uint16_t *)src; float *d = (float *)dst;
    size_t k = 0;
    for (; k + 8 <= n; k += 8) {
        float16x8_t h = vreinterpretq_f16_u16(vld1q_u16(s + k));
        vst1q_f32(d + k,     vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(d + k + 4, vcvt_high_f32_f16(h));
    }
    const ElemOpts o{false, false};
    for (; k < n; k++) d[k] = cvt<float>(make<f16>(s[k]), o);
}
template <bool SAT>
static void neon_d2h(const void *src, void *dst, size_t n, const KParams &) {
    const double *s = (const double *)src; uint16_t *d = (uint16_t *)dst;
    size_t k = 0;
    for (; k + 8 <= n; k += 8) {
        float32x4_t f0 = vcvtx_high_f32_f64(vcvtx_f32_f64(vld1q_f64(s + k)),     vld1q_f64(s + k + 2));
        float32x4_t f1 = vcvtx_high_f32_f64(vcvtx_f32_f64(vld1q_f64(s + k + 4)), vld1q_f64(s + k + 6));
        float16x8_t h  = vcvt_high_f16_f32(vcvt_f16_f32(f0), f1);
        uint16x8_t  b  = vreinterpretq_u16_f16(h);
        if (SAT) {
            uint16x8_t mag = vandq_u16(b, vdupq_n_u16(0x7fff));
            uint16x8_t inf = vceqq_u16(mag, vdupq_n_u16(0x7c00));
            b = vbslq_u16(inf, vorrq_u16(vandq_u16(b, vdupq_n_u16(0x8000)), vdupq_n_u16(0x7bff)), b);
        }
        vst1q_u16(d + k, b);
    }
    const ElemOpts o{SAT, false};
    for (; k < n; k++) d[k] = cvt<f16>(s[k], o).bits;
}
static void neon_h2d(const void *src, void *dst, size_t n, const KParams &) {
    const uint16_t *s = (const uint16_t *)src; double *d = (double *)dst;
    size_t k = 0;
    for (; k + 4 <= n; k += 4) {
        float32x4_t f = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(s + k)));
        vst1q_f64(d + k,     vcvt_f64_f32(vget_low_f32(f)));
        vst1q_f64(d + k + 2, vcvt_high_f64_f32(f));
    }
    const ElemOpts o{false, false};
    for (; k < n; k++) d[k] = cvt<double>(make<f16>(s[k]), o);
}
static void neon_s2h_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? neon_s2h<true>(a, b, n, p) : neon_s2h<false>(a, b, n, p); }
static void neon_d2h_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? neon_d2h<true>(a, b, n, p) : neon_d2h<false>(a, b, n, p); }
static void neon_h2s_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? k_cvt<float, f16>(a, b, n, p) : neon_h2s(a, b, n, p); }
static void neon_h2d_any(const void *a, void *b, size_t n, const KParams &p) { p.eo.sat ? k_cvt<double, f16>(a, b, n, p) : neon_h2d(a, b, n, p); }
} /* namespace isa_generic */
#endif

void fill_table_generic(Table &t) {
    t.name = "generic";
#if defined(__aarch64__)
    t.name = "neon";
#endif
    isa_generic::fill_generic(t);
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FP16_FORMAT_IEEE)
    t.cvt[OMC_FP32][OMC_FP16] = &isa_generic::neon_s2h_any;
    t.cvt[OMC_FP16][OMC_FP32] = &isa_generic::neon_h2s_any;
    t.cvt[OMC_FP64][OMC_FP16] = &isa_generic::neon_d2h_any;
    t.cvt[OMC_FP16][OMC_FP64] = &isa_generic::neon_h2d_any;
#endif
}

} /* namespace cpu */
} /* namespace omc */
