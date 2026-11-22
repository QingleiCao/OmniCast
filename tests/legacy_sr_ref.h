/**
 * @file legacy_sr_ref.h
 * @brief The floating-point stochastic rounding of OmniCast 2.0.0 (itself a
 * branch-free form of HiCMA-X 0.2), kept as the reference for the integer
 * implementation in omc_scalar.hpp. Include after omc_scalar.hpp.
 */
#ifndef OMC_LEGACY_SR_REF_H
#define OMC_LEGACY_SR_REF_H
namespace omc {
namespace ref {

/** Uniform in [0, 1), a multiple of 2^-24, from the position (i, j), the
 *  seed and the bits of the value: independent of how the work is split. */
static inline float sr_u01(uint32_t i, uint32_t j, uint64_t seed, uint64_t bits) {
    const uint64_t pos = (uint64_t)i | ((uint64_t)j << 32);
    /* a 24-bit integer: converted through int32 (exact, and AVX2 has no
     * vector uint64 -> float conversion) */
    const int32_t r = (int32_t)(uint32_t)(splitmix64(bits ^ splitmix64(pos ^ seed)) >> 40);
    return (float)r * (1.0f / 16777216.0f);
}

/* sr_bits: the one of omc_scalar.hpp (unchanged) */

static inline float nextup_f32(float x)   { return u2f(f2u(x) + 1u); }   /* x >= 0 finite */
/** (u - a) / step for a power of two step: the product by 1/step is the
 *  same correctly rounded number as the quotient, without a division */
static inline double div_pow2(double num, double step) {
    return num * u2d(0x7fe0000000000000ULL - d2u(step));
}
static inline float nextdown_f32(float x) { return u2f(f2u(x) - 1u); }   /* x > 0 */

/** FP32 target: round down with probability (upper - a) / (upper - lower).
 *  Branch-free form of the HiCMA-X 0.2 code (same results), so that it
 *  vectorizes; the lanes it does not use may compute garbage. */
static inline float sr_f32(double value, float random) {
    const float nearest = (float)value;
    const double a = fabs(value);
    const float n = fabsf(nearest);
    const uint32_t nb = f2u(n);
    const bool gt = a > (double)n, lt = a < (double)n;
    const float lower = gt ? n : u2f(nb - 1u);
    const float upper = gt ? u2f(nb + 1u) : n;
    const double p_lower = div_pow2((double)upper - a, (double)upper - (double)lower);
    float r = (random < p_lower) ? lower : upper;
    r = (value < 0.0) ? -r : r;
    const bool fin = ((d2u(value) & 0x7fffffffffffffffULL) < 0x7ff0000000000000ULL)
                   & (nb < 0x7f800000u) & (f2u(upper) < 0x7f800000u);
    return (fin & (gt | lt)) ? r : nearest;
}

/** FP16 code -> FP32: the hardware convert on GPUs */
static inline float f16_bits_to_f32(uint32_t h) {
#if defined(OMC_ON_DEVICE)
    return __half2float(__ushort_as_half((unsigned short)h));
#else
    return dec_f32<FmtF16>(h);
#endif
}

/** FP16 target, as HiCMA-X 0.2 (including its round-to-nearest through FP32
 *  above 65504), in branch-free form. */
static inline uint32_t sr_f16(double value, float random, bool sat) {
    const double a = fabs(value);
    const uint32_t sign = (value < 0.0) ? 0x8000u : 0u;
    /* largest half <= a: truncate a to 11 significant bits, or count steps of
     * 2^-24 below 2^-14 */
    const uint64_t b = d2u(a);
    const uint32_t lo_n = ((uint32_t)(((int)(b >> 52) - 1023 + 15) & 31) << 10) | (uint32_t)((b >> 42) & 0x3ffu);
    const double as = (a < 6.103515625e-05) ? a : 0.0;
    const uint32_t lo_s = (uint32_t)(int32_t)(as * 16777216.0);
    const bool top = a >= 65504.0;
    const uint32_t lo = top ? 0x7bffu : ((a < 6.103515625e-05) ? lo_s : lo_n);
    const double lv = (double)f16_bits_to_f32(lo);
    const uint32_t up = lo + 1u;
    const double uv = (double)f16_bits_to_f32(up);
    const double p_lower = div_pow2(uv - a, uv - lv);
    uint32_t r = ((random < p_lower) ? lo : up) | sign;
    r = (lv == a) ? (lo | sign) : r;
    r = (top & (lv != a)) ? enc_f32<FmtF16>((float)value, sat) : r;   /* above the largest half */
    r = (value != value) ? (0x7e00u | (uint32_t)((d2u(value) >> 48) & 0x8000u)) : r;
    return r;
}

/** Other small formats: the same rule, round to nearest above the largest
 *  finite value. */
template <class F>
static inline uint32_t sr_small(double value, float random, bool sat) {
    if (value != value) return F::nancode;
    const double a = fabs(value);
    const uint32_t sign = (value < 0.0) ? F::signbit : 0u;
    /* lower = largest representable <= a, via FP32 round toward zero */
    const float maxf = dec_f32<F>(F::maxcode);
    if (a >= (double)maxf) {
        if (a == (double)maxf) return F::maxcode | sign;
        return enc_f32<F>(f64_to_f32_ro(value), sat);
    }
    float fa = f64_to_f32_ro(a);                  /* odd rounding keeps the order with codes */
    uint32_t lo = enc_f32<F>(fa, false);          /* nearest: lo or lo + 1 */
    double lv = (double)dec_f32<F>(lo);
    if (lv > a) { lo -= 1u; lv = (double)dec_f32<F>(lo); }
    if (lv == a) return lo | sign;
    const uint32_t up = lo + 1u;
    const double uv = (double)dec_f32<F>(up);
    const double p_lower = (uv - a) / (uv - lv);
    return ((random < p_lower) ? lo : up) | sign;
}

/** Stochastic rounding of one element at position (i, j). */
template <class D, class S> struct SR;
template <class S> struct SR<float, S> {
    static float go(S s, uint32_t i, uint32_t j, uint64_t seed, const ElemOpts &o) {
        const double v = (double)Src<S>::get(s);
        float r = sr_f32(v, sr_u01(i, j, seed, sr_bits(s)));
        return Enc<float>::from_f32(r, o);   /* saturation of +-Inf */
    }
};
template <class S> struct SR<f16, S> {
    static f16 go(S s, uint32_t i, uint32_t j, uint64_t seed, const ElemOpts &o) {
        const double v = (double)Src<S>::get(s);
        return make<f16>(sr_f16(v, sr_u01(i, j, seed, sr_bits(s)), o.sat));
    }
};
template <class D, class S> struct SRSmall {
    static D go(S s, uint32_t i, uint32_t j, uint64_t seed, const ElemOpts &o) {
        const double v = (double)Src<S>::get(s);
        return make<D>(sr_small<typename Traits<D>::F>(v, sr_u01(i, j, seed, sr_bits(s)), o.sat));
    }
};
template <class S> struct SR<bf16, S> : SRSmall<bf16, S> {};
template <class S> struct SR<e4m3, S> : SRSmall<e4m3, S> {};
template <class S> struct SR<e5m2, S> : SRSmall<e5m2, S> {};

} /* namespace ref */
} /* namespace omc */
#endif
