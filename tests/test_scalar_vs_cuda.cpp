/* Exhaustive check of the scalar core against the CUDA header (host) reference
 * implementations: every FP32 bit pattern -> FP16/BF16/E4M3/E5M2 (both
 * saturation modes), every small-format code -> FP32, and FP64 -> small
 * formats on edge cases and random values against __double2half and an
 * exact neighbour search. Build: g++ -O2 -fopenmp -I<cuda>/include */
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include "../src/core/omc_scalar.hpp"

using namespace omc;

static inline bool isnan16(uint32_t h, int E, int M, bool fn) {
    uint32_t em = h & ((1u << (E + M)) - 1);
    if (fn) return em == ((1u << (E + M)) - 1);
    return (em >> M) == ((1u << E) - 1) && (em & ((1u << M) - 1));
}

template <class F>
static long check_f32(const char *name, uint32_t (*ref)(float, bool), bool sat) {
    long bad = 0, first = -1;
    #pragma omp parallel for reduction(+:bad) schedule(static, 1 << 20)
    for (long long k = 0; k < (1LL << 32); k++) {
        float x = u2f((uint32_t)k);
        uint32_t a = enc_f32<F>(x, sat), b = ref(x, sat);
        if (a != b) {
            bool both_nan = isnan16(a, F::kE, F::kM, F::infcode == F::magmask) &&
                            isnan16(b, F::kE, F::kM, F::infcode == F::magmask);
            if (!both_nan) { bad++; if (first < 0) { first = k; } }
        }
    }
    printf("%-28s sat=%d : %ld mismatches%s", name, (int)sat, bad, bad ? "" : "\n");
    if (bad) printf(" (first 0x%08lx: omc 0x%x ref 0x%x)\n", first, enc_f32<F>(u2f((uint32_t)first), sat), ref(u2f((uint32_t)first), sat));
    return bad;
}

static uint32_t ref_f16(float x, bool sat) {
    __half h = __float2half_rn(x);
    uint32_t b = __half_as_ushort(h);
    if (sat && (b & 0x7fff) == 0x7c00) b = (b & 0x8000) | 0x7bff;
    return b;
}
static uint32_t ref_bf16(float x, bool sat) {
    __nv_bfloat16 h = __float2bfloat16_rn(x);
    uint32_t b = __bfloat16_as_ushort(h);
    if (sat && (b & 0x7fff) == 0x7f80) b = (b & 0x8000) | 0x7f7f;
    return b;
}
static uint32_t ref_e4m3(float x, bool sat) { return __nv_cvt_float_to_fp8(x, sat ? __NV_SATFINITE : __NV_NOSAT, __NV_E4M3); }
static uint32_t ref_e5m2(float x, bool sat) { return __nv_cvt_float_to_fp8(x, sat ? __NV_SATFINITE : __NV_NOSAT, __NV_E5M2); }

/* exact nearest-even reference for a double into a small format: scan codes */
template <class F>
static uint32_t ref_from_f64(double x, bool sat) {
    if (x != x) return F::nancode;
    uint32_t sign = std::signbit(x) ? F::signbit : 0;
    double a = fabs(x);
    double maxv = dec_f32<F>(F::maxcode);
    /* next value above max (virtual) for the overflow boundary */
    double ulp_top = maxv - (double)dec_f32<F>(F::maxcode - 1);
    if (F::infcode == F::magmask) {   /* FN: virtual next is maxv + ulp, the NaN code */
        if (a > maxv + ulp_top / 2 || (a == maxv + ulp_top / 2 && false)) return (sat ? F::maxcode : F::nancode) | sign;
        if (a == maxv + ulp_top / 2) return F::maxcode | sign;   /* tie: maxcode is even */
    } else {
        if (a >= maxv + ulp_top / 2) return (sat ? F::maxcode : F::infcode) | sign;
    }
    /* binary search the largest code with value <= a */
    uint32_t lo = 0, hi = F::maxcode;
    while (lo < hi) { uint32_t mid = (lo + hi + 1) / 2; if ((double)dec_f32<F>(mid) <= a) lo = mid; else hi = mid - 1; }
    if (lo == F::maxcode) return lo | sign;
    double lv = dec_f32<F>(lo), uv = dec_f32<F>(lo + 1);
    uint32_t r;
    if (a - lv < uv - a) r = lo; else if (a - lv > uv - a) r = lo + 1; else r = (lo & 1) ? lo + 1 : lo;
    return r | sign;
}

template <class F>
static long check_f64(const char *name, bool sat) {
    std::mt19937_64 g(12345);
    long bad = 0;
    auto test = [&](double x) {
        uint32_t a = enc_f32<F>(f64_to_f32_ro(x), sat);
        uint32_t b = ref_from_f64<F>(x, sat);
        if (a != b && !(isnan16(a, F::kE, F::kM, F::infcode == F::magmask) && isnan16(b, F::kE, F::kM, F::infcode == F::magmask))) {
            if (bad < 5) printf("  %s x=%.17g omc=0x%x ref=0x%x\n", name, x, a, b);
            bad++;
        }
    };
    /* every code, its midpoints, and +-1 ulp (FP64) around each */
    for (uint32_t c = 0; c <= F::maxcode; c++) {
        double v = dec_f32<F>(c);
        double nx = (c < F::maxcode) ? (double)dec_f32<F>(c + 1) : v * 2;
        double mid = (v + nx) / 2;
        for (double y : {v, mid, nextafter(mid, 0.0), nextafter(mid, 1e300), nextafter(v, 0.0), nextafter(v, 1e300)})
            { test(y); test(-y); }
    }
    for (int k = 0; k < 4000000; k++) {
        uint64_t u = g();
        double x = u2d(u);
        if (x != x) continue;
        /* bring the exponent near the format's range */
        int e = (int)(g() % 80) - 40;
        x = ldexp(frexp(x, &e) , (int)(g() % 60) - 30 + (F::kE == 8 ? 0 : 0));
        test(x);
    }
    test(1e300); test(-1e300); test(INFINITY); test(-INFINITY); test(5e-324); test(0.0); test(-0.0);
    printf("%-28s sat=%d : %ld mismatches (FP64 edge + random)\n", name, (int)sat, bad);
    return bad;
}

int main() {
    long bad = 0;
    for (bool sat : {false, true}) {
        bad += check_f32<FmtF16>("fp32->fp16 vs __float2half", ref_f16, sat);
        bad += check_f32<FmtBF16>("fp32->bf16 vs __float2bf16", ref_bf16, sat);
        bad += check_f32<FmtE4M3>("fp32->e4m3 vs cuda_fp8", ref_e4m3, sat);
        bad += check_f32<FmtE5M2>("fp32->e5m2 vs cuda_fp8", ref_e5m2, sat);
    }
    /* decode */
    long dbad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        float a = dec_f32<FmtF16>(h), b = __half2float(__ushort_as_half((unsigned short)h));
        if (f2u(a) != f2u(b) && !(a != a && b != b)) dbad++;
        a = dec_f32<FmtBF16>(h); b = __bfloat162float(__ushort_as_bfloat16((unsigned short)h));
        if (f2u(a) != f2u(b) && !(a != a && b != b)) dbad++;
    }
    for (uint32_t h = 0; h < 256; h++) {
        __half_raw r4 = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)h, __NV_E4M3);
        __half_raw r5 = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)h, __NV_E5M2);
        float b4 = __half2float(__half(r4)), b5 = __half2float(__half(r5));
        float a4 = dec_f32<FmtE4M3>(h), a5 = dec_f32<FmtE5M2>(h);
        if (f2u(a4) != f2u(b4) && !(a4 != a4 && b4 != b4)) { dbad++; printf("e4m3 dec 0x%x %g %g\n", h, a4, b4); }
        if (f2u(a5) != f2u(b5) && !(a5 != a5 && b5 != b5)) { dbad++; printf("e5m2 dec 0x%x %g %g\n", h, a5, b5); }
    }
    printf("%-28s       : %ld mismatches\n", "decode (all codes)", dbad);
    bad += dbad;
    for (bool sat : {false, true}) {
        bad += check_f64<FmtF16>("fp64->fp16 (exact ref)", sat);
        bad += check_f64<FmtBF16>("fp64->bf16 (exact ref)", sat);
        bad += check_f64<FmtE4M3>("fp64->e4m3 (exact ref)", sat);
        bad += check_f64<FmtE5M2>("fp64->e5m2 (exact ref)", sat);
    }
    /* fp64 -> fp16 against __double2half on random doubles */
    {
        std::mt19937_64 g(7); long b2 = 0;
        for (int k = 0; k < 20000000; k++) {
            double x = ldexp((double)(g() >> 11) / 9007199254740992.0, (int)(g() % 50) - 30);
            if (g() & 1) x = -x;
            uint32_t a = enc_f32<FmtF16>(f64_to_f32_ro(x), false);
            uint32_t r = __half_as_ushort(__double2half(x));
            if (a != r) { if (b2 < 5) printf("  d2h x=%.17g omc=0x%x cuda=0x%x\n", x, a, r); b2++; }
        }
        printf("%-28s       : %ld mismatches (20M random)\n", "fp64->fp16 vs __double2half", b2);
        bad += b2;
    }
    printf(bad ? "FAILED\n" : "PASSED\n");
    return bad ? 1 : 0;
}
