/**
 * @file legacy_hicma_sr.h
 * @brief Reference copies of the HiCMA-X 0.2 conversions that OmniCast
 * replaces (src/core/hicma_parsec_stochastic_rounding.{c,cu},
 * hicma_parsec_datatype_convert.c), used to check bit compatibility. The FP16
 * stochastic rounding was CUDA code; here it runs on the host with the
 * HiCMA-X ISPC half helpers and an independent round-toward-zero.
 *
 * HiCMA-X code: Copyright (c) 2023-2025 SLU, MIT, Nvidia; 2018-2025 KAUST;
 * 2018-2023 The University of Tennessee and The University of Tennessee Research
 * Foundation. BSD 3-Clause; see the section "Code from HiCMA-X" in LICENSE.
 */
#ifndef LEGACY_HICMA_SR_H
#define LEGACY_HICMA_SR_H
#include <stdint.h>
#include <string.h>
#include <math.h>

static inline uint64_t legacy_splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}
static inline float legacy_sr_u01(int i, int j, uint64_t seed, uint64_t bits) {
    uint64_t pos = (uint64_t)(uint32_t)i | ((uint64_t)(uint32_t)j << 32);
    return (float)(legacy_splitmix64(bits ^ legacy_splitmix64(pos ^ seed)) >> 40) * (1.0f / 16777216.0f);
}
static inline float legacy_stochastic_round_fp32(double value, float random) {
    float nearest = (float)value;
    if (!isfinite(value) || !isfinite(nearest)) return nearest;
    double a = fabs(value);
    float n = fabsf(nearest);
    float lower, upper;
    if (a > (double)n) { lower = n; upper = nextafterf(n, INFINITY); }
    else if (a < (double)n) { lower = nextafterf(n, 0.0f); upper = n; }
    else return nearest;
    if (isinf(upper)) return nearest;
    double p_lower = ((double)upper - a) / ((double)upper - (double)lower);
    float r = (random < p_lower) ? lower : upper;
    return (value < 0.0) ? -r : r;
}
static inline void legacy_double2float_round_CPU(int nrows, int ncols, const double *D, int ldd,
                                                 float *F, int ldf, uint64_t seed) {
    for (int j = 0; j < ncols; ++j)
        for (int i = 0; i < nrows; ++i) {
            double value = D[j * ldd + i];
            uint64_t bits; memcpy(&bits, &value, 8);
            F[j * ldf + i] = legacy_stochastic_round_fp32(value, legacy_sr_u01(i, j, seed, bits));
        }
}

/* HiCMA-X float_to_half_rtne (ISPC reference), verbatim logic */
static inline uint16_t legacy_float_to_half_rtne(float ff) {
    uint32_t fu; memcpy(&fu, &ff, 4);
    uint32_t fM = fu & 0x7fffff, fE = (fu >> 23) & 0xff, fS = fu >> 31;
    uint32_t oE = 0, oM = 0; uint16_t ou;
    if (fE == 0) { oE = 0; }
    else if (fE == 255) { oE = 31; oM = fM ? 0x200 : 0; }
    else {
        int newexp = (int)fE - 127 + 15;
        if (newexp >= 31) oE = 31;
        else if (newexp <= 0) {
            if ((14 - newexp) <= 24) {
                uint32_t mant = fM | 0x800000, shift = 14 - newexp;
                oM = mant >> shift;
                uint32_t lowmant = mant & ((1u << shift) - 1), halfway = 1u << (shift - 1);
                ou = (uint16_t)((oE << 10) | oM);
                if (lowmant >= halfway && (lowmant > halfway || (oM & 1))) ou++;
                return (uint16_t)(ou | (fS << 15));
            }
        } else {
            oE = (uint32_t)newexp; oM = fM >> 13;
            ou = (uint16_t)((oE << 10) | oM);
            if ((fM & 0x1000) && (((fM & 0x1fff) > 0x1000) || (oM & 1))) ou++;
            return (uint16_t)(ou | (fS << 15));
        }
    }
    return (uint16_t)((oE << 10) | oM | (fS << 15));
}
static inline float legacy_half_to_float(uint16_t h) {
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    float v;
    if (e == 0) v = ldexpf((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = ldexpf((float)(m | 0x400), (int)e - 25);
    return (h & 0x8000) ? -v : v;
}
/* round toward zero, independent of the code under test */
static inline uint16_t legacy_float2half_rz(float x) {
    uint16_t s = signbit(x) ? 0x8000 : 0;
    float a = fabsf(x);
    if (a != a) return 0x7e00;
    if (isinf(a)) return s | 0x7c00;
    if (a >= 65504.0f) return s | 0x7bff;
    if (a < 6.103515625e-05f) return s | (uint16_t)floor((double)a * 16777216.0);
    int e; double f = frexp((double)a, &e);           /* a = f 2^e, f in [0.5, 1) */
    uint32_t mant = (uint32_t)floor((f * 2.0 - 1.0) * 1024.0);
    return s | (uint16_t)(((uint32_t)(e - 1 + 15) << 10) | mant);
}
/* HiCMA-X stochastic_rounding_fp16 (CUDA), with the helpers above */
static inline uint16_t legacy_sr_fp16(double value, float random) {
    if (isnan(value)) return legacy_float_to_half_rtne((float)value);
    double a = fabs(value);
    uint16_t lower = legacy_float2half_rz((float)a);
    if ((double)legacy_half_to_float(lower) > a) lower = (uint16_t)(lower - 1);
    double lv = (double)legacy_half_to_float(lower);
    if (lv == a) return (value < 0.0) ? (uint16_t)(lower ^ 0x8000) : lower;
    uint16_t upper = (uint16_t)(lower + 1);
    double uv = (double)legacy_half_to_float(upper);
    if (isinf(uv)) return legacy_float_to_half_rtne((float)value);
    double p_lower = (uv - a) / (uv - lv);
    uint16_t r = (random < p_lower) ? lower : upper;
    return (value < 0.0) ? (uint16_t)(r ^ 0x8000) : r;
}
#endif
