/**
 * @file omc_scalar.hpp
 * @brief Scalar conversion core of OmniCast, shared by the CPU and GPU backends.
 *
 * Every conversion is defined here once, as branch-free integer/float code that
 * compiles for the host (and vectorizes), for CUDA and for HIP. The SIMD and
 * GPU fast paths must give the same bits as these functions (NaN payloads
 * excepted); the tests check that.
 *
 * Small float formats are encoded from FP32 with round to nearest even:
 * normal numbers by integer rounding on the FP32 bit pattern, subnormals with
 * one float addition of a power of two (the FPU rounds at the subnormal
 * position). FP64 sources are first rounded to FP32 with round-to-odd, which
 * makes the second rounding exact (FP32 keeps more than p + 2 bits of every
 * small format), so FP64 -> FP16/BF16/FP8 is a single correct rounding.
 */
#ifndef OMC_SCALAR_HPP
#define OMC_SCALAR_HPP

#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__CUDACC__)
#include <cuda_fp16.h>
#elif defined(__HIPCC__)
#include <hip/hip_fp16.h>
#endif
#include <type_traits>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define OMC_HD __host__ __device__ __forceinline__
#define OMC_DEVICE_COMPILE 1
#else
#define OMC_HD inline __attribute__((always_inline))
#endif

#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define OMC_ON_DEVICE 1
#endif

namespace omc {

/* ------------------------------------------------------------------------ */
/* Storage types                                                             */
/* ------------------------------------------------------------------------ */

struct f16  { uint16_t bits; };
struct bf16 { uint16_t bits; };
struct e4m3 { uint8_t  bits; };
struct e5m2 { uint8_t  bits; };

/* ------------------------------------------------------------------------ */
/* Bit casts                                                                 */
/* ------------------------------------------------------------------------ */

OMC_HD uint32_t f2u(float f) {
#if defined(OMC_ON_DEVICE)
    return __float_as_uint(f);
#else
    uint32_t u; memcpy(&u, &f, 4); return u;
#endif
}
OMC_HD float u2f(uint32_t u) {
#if defined(OMC_ON_DEVICE)
    return __uint_as_float(u);
#else
    float f; memcpy(&f, &u, 4); return f;
#endif
}
OMC_HD uint64_t d2u(double d) {
#if defined(OMC_ON_DEVICE)
    return (uint64_t)__double_as_longlong(d);
#else
    uint64_t u; memcpy(&u, &d, 8); return u;
#endif
}
OMC_HD double u2d(uint64_t u) {
#if defined(OMC_ON_DEVICE)
    return __longlong_as_double((long long)u);
#else
    double d; memcpy(&d, &u, 8); return d;
#endif
}

/* ------------------------------------------------------------------------ */
/* Small float formats                                                       */
/* ------------------------------------------------------------------------ */

/** E exponent bits, M mantissa bits, FN: finite-only format (OCP E4M3: no
 *  Inf, a single NaN code per sign: all exponent and mantissa bits set). */
template <int E, int M, bool FN>
struct Fmt {
    static constexpr int      kE    = E;
    static constexpr int      kM    = M;
    static constexpr int      bias  = (1 << (E - 1)) - 1;
    static constexpr uint32_t magmask = (1u << (E + M)) - 1u;
    static constexpr uint32_t signbit = 1u << (E + M);
    static constexpr uint32_t maxcode = FN ? ((((1u << E) - 1u) << M) | ((1u << M) - 2u))
                                           : ((((1u << E) - 2u) << M) | ((1u << M) - 1u));
    static constexpr uint32_t infcode = FN ? magmask : (((1u << E) - 1u) << M);   /* overflow, not saturated */
    static constexpr uint32_t nancode = FN ? magmask : ((((1u << E) - 1u) << M) | (1u << (M - 1)));
    /* FP32 bit pattern of the smallest normal number 2^(1-bias) */
    static constexpr uint32_t f32_min_normal = (uint32_t)(127 - bias + 1) << 23;
    /* power of two whose FP32 ulp is the subnormal step 2^(1-bias-M) */
    static constexpr uint32_t f32_denorm_magic = (uint32_t)((127 - bias) + (23 - M) + 1) << 23;
};

using FmtF16  = Fmt<5, 10, false>;
using FmtBF16 = Fmt<8, 7,  false>;
using FmtE4M3 = Fmt<4, 3,  true>;
using FmtE5M2 = Fmt<5, 2,  false>;

/** FP32 -> small format, round to nearest even. */
template <class F>
OMC_HD uint32_t enc_f32(float x, bool sat) {
    const uint32_t u    = f2u(x);
    const uint32_t sign = (u >> 31) << (F::kE + F::kM);
    const uint32_t a    = u & 0x7fffffffu;
    /* normal numbers: rebias, then round on the integer pattern */
    const uint32_t odd  = (a >> (23 - F::kM)) & 1u;
    const uint32_t nrm  = (a + ((uint32_t)(F::bias - 127) << 23)
                             + ((1u << (23 - F::kM - 1)) - 1u) + odd) >> (23 - F::kM);
    /* subnormals: the FP32 addition rounds at the subnormal position */
    const float    dsum = u2f(a) + u2f(F::f32_denorm_magic);
    const uint32_t den  = f2u(dsum) - F::f32_denorm_magic;
    uint32_t r = (a < F::f32_min_normal) ? den : nrm;
    /* overflow (includes +-Inf) */
    const uint32_t ovf = sat ? F::maxcode : F::infcode;
    r = (r > F::maxcode) ? ovf : r;
    r = (a > 0x7f800000u) ? F::nancode : r;
    return r | sign;
}

/* BF16 shares the FP32 exponent: plain integer rounding covers subnormals */
template <>
OMC_HD uint32_t enc_f32<FmtBF16>(float x, bool sat) {
    const uint32_t u = f2u(x);
    const uint32_t a = u & 0x7fffffffu;
    uint32_t r = (a + 0x7fffu + ((a >> 16) & 1u)) >> 16;
    r = (sat && a <= 0x7f800000u && r > 0x7f7fu) ? 0x7f7fu : r;
    r = (a > 0x7f800000u) ? 0x7fc0u : r;
    return r | ((u >> 16) & 0x8000u);
}

/** small format -> FP32 (exact) */
template <class F>
OMC_HD float dec_f32(uint32_t h) {
    const uint32_t sign = (h & F::signbit) << (31 - F::kE - F::kM);
    const uint32_t em   = h & F::magmask;
    /* place exponent and mantissa in the FP32 fields and rescale: exact for
     * normal and subnormal values of the small format */
    float v = u2f(em << (23 - F::kM)) * u2f((uint32_t)(254 - F::bias) << 23);
    const bool special = (F::infcode == F::magmask) ? (em == F::magmask)                 /* FN: NaN */
                                                    : ((em >> F::kM) == ((1u << F::kE) - 1u));
    const uint32_t spec = (F::infcode == F::magmask) ? 0x7fc00000u
                         : (0x7f800000u | ((em & ((1u << F::kM) - 1u)) << (23 - F::kM)));
    return special ? u2f(spec | sign) : u2f(f2u(v) | sign);
}

template <>
OMC_HD float dec_f32<FmtBF16>(uint32_t h) { return u2f(h << 16); }

/* ------------------------------------------------------------------------ */
/* FP64 -> FP32                                                              */
/* ------------------------------------------------------------------------ */

/** Round to odd: truncate, and set the last bit when inexact. A second
 *  rounding to any format with at most 22 significand bits is then exact. */
OMC_HD float f64_to_f32_ro(double d) {
#if defined(__CUDA_ARCH__)
    const float f = __double2float_rz(d);
    return ((double)f != d) ? u2f(f2u(f) | 1u) : f;
#else
    const float  f = (float)d;
    const double b = (double)f;
    const uint32_t u = f2u(f);
    const bool adj  = (b != d) && ((u & 1u) == 0u) && (d == d);
    const bool down = fabs(b) > fabs(d);
    return u2f(adj ? (down ? u - 1u : u + 1u) : u);
#endif
}

/* ------------------------------------------------------------------------ */
/* Integers                                                                  */
/* ------------------------------------------------------------------------ */

template <class T> struct IntLimits;
#define OMC_INTLIM(T, LO, HI, ILO, IHI) \
    template <> struct IntLimits<T> { static constexpr double lo = LO, hi = HI; \
                                      static constexpr int64_t ilo = ILO; static constexpr uint64_t ihi = IHI; }
OMC_INTLIM(int8_t,   -128.0,                 127.0,                  -128,                       127u);
OMC_INTLIM(int16_t,  -32768.0,               32767.0,                -32768,                     32767u);
OMC_INTLIM(int32_t,  -2147483648.0,          2147483647.0,           -2147483647 - 1,            2147483647u);
OMC_INTLIM(int64_t,  -9223372036854775808.0, 9223372036854775807.0,  -9223372036854775807LL - 1, 9223372036854775807ULL);
OMC_INTLIM(uint8_t,  0.0,                    255.0,                  0,                          255u);
OMC_INTLIM(uint16_t, 0.0,                    65535.0,                0,                          65535u);
OMC_INTLIM(uint32_t, 0.0,                    4294967295.0,           0,                          4294967295u);
OMC_INTLIM(uint64_t, 0.0,                    18446744073709551615.0, 0,                          18446744073709551615ULL);
#undef OMC_INTLIM

/** double -> integer, truncating (rne = false) or to nearest even, saturating,
 *  NaN -> 0. The 64-bit limits are not representable in double: anything at
 *  or above 2^63 (2^64) clamps to the maximum. */
template <class T>
OMC_HD T f64_to_int(double x, bool rne) {
    double t = rne ? rint(x) : x;   /* the final conversion truncates */
    const double lo = IntLimits<T>::lo, hi = IntLimits<T>::hi;
    if constexpr (sizeof(T) == 8) {
        if (!(t == t)) return (T)0;
        if (t <= lo) return (T)IntLimits<T>::ilo;
        if (t >= hi) return (T)IntLimits<T>::ihi;   /* hi rounds up to 2^63 (2^64) */
        return (T)t;
    } else {
        t = (t == t) ? t : 0.0;
        t = (t < lo) ? lo : t;
        t = (t > hi) ? hi : t;
        if constexpr (std::is_same<T, uint32_t>::value) return (T)(int64_t)t;
        else                                           return (T)(int32_t)t;
    }
}

/** float -> integer of at most 16 bits (cheaper than going through double) */
template <class T>
OMC_HD T f32_to_small_int(float x, bool rne) {
    float t = rne ? rintf(x) : x;   /* the final conversion truncates */
    const float lo = (float)IntLimits<T>::lo, hi = (float)IntLimits<T>::hi;
    t = (t == t) ? t : 0.0f;
    t = (t < lo) ? lo : t;
    t = (t > hi) ? hi : t;
    return (T)(int32_t)t;
}

/** 64-bit integer -> double rounded to odd, so that a second rounding to
 *  FP32 or a smaller format is exact. */
template <class T>
OMC_HD double i64_to_f64_ro(T v) {
    const double d = (double)v;                      /* nearest */
    if (d > -9007199254740992.0 && d < 9007199254740992.0) return d;   /* |v| < 2^53: exact */
    bool exact, up;                                  /* up: |d| > |v| */
    if (d >= 18446744073709551616.0 || (std::is_signed<T>::value && d >= 9223372036854775808.0)) {
        exact = false; up = true;                    /* rounded up to 2^64 (2^63): not a T */
    } else {
        const T b = (T)d;
        exact = (b == v);
        if constexpr (std::is_signed<T>::value) up = (v < 0) ? (b < v) : (b > v);
        else                                    up = (b > v);
    }
    const uint64_t u = d2u(d);
    if (exact || (u & 1u)) return d;
    return u2d(up ? u - 1u : u + 1u);
}

/* ------------------------------------------------------------------------ */
/* Type traits                                                               */
/* ------------------------------------------------------------------------ */

template <class T> struct Traits;
template <> struct Traits<double>   { static constexpr bool is_float = true;  static constexpr int rank = 6; };
template <> struct Traits<float>    { static constexpr bool is_float = true;  static constexpr int rank = 5; };
template <> struct Traits<f16>      { static constexpr bool is_float = true;  static constexpr int rank = 4; using F = FmtF16; };
template <> struct Traits<bf16>     { static constexpr bool is_float = true;  static constexpr int rank = 4; using F = FmtBF16; };
template <> struct Traits<e4m3>     { static constexpr bool is_float = true;  static constexpr int rank = 2; using F = FmtE4M3; };
template <> struct Traits<e5m2>     { static constexpr bool is_float = true;  static constexpr int rank = 2; using F = FmtE5M2; };
template <> struct Traits<int8_t>   { static constexpr bool is_float = false; static constexpr int small = 1; };
template <> struct Traits<int16_t>  { static constexpr bool is_float = false; static constexpr int small = 1; };
template <> struct Traits<int32_t>  { static constexpr bool is_float = false; static constexpr int small = 0; };
template <> struct Traits<int64_t>  { static constexpr bool is_float = false; static constexpr int small = 0; };
template <> struct Traits<uint8_t>  { static constexpr bool is_float = false; static constexpr int small = 1; };
template <> struct Traits<uint16_t> { static constexpr bool is_float = false; static constexpr int small = 1; };
template <> struct Traits<uint32_t> { static constexpr bool is_float = false; static constexpr int small = 0; };
template <> struct Traits<uint64_t> { static constexpr bool is_float = false; static constexpr int small = 0; };

template <class T> struct is_small_float { static constexpr bool value = false; };
template <> struct is_small_float<f16>  { static constexpr bool value = true; };
template <> struct is_small_float<bf16> { static constexpr bool value = true; };
template <> struct is_small_float<e4m3> { static constexpr bool value = true; };
template <> struct is_small_float<e5m2> { static constexpr bool value = true; };

template <class T> OMC_HD uint32_t raw(T v) { return (uint32_t)v.bits; }
template <class T> OMC_HD T make(uint32_t b) { T t; t.bits = (decltype(t.bits))b; return t; }

/* ------------------------------------------------------------------------ */
/* Widening to FP32 / FP64 (exact for float types)                           */
/* ------------------------------------------------------------------------ */

template <class S> OMC_HD float to_f32_exact(S s) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    if constexpr (std::is_same<S, f16>::value) {     /* FP16: hardware convert */
        return __half2float(__ushort_as_half(s.bits));
    }
#endif
    return dec_f32<typename Traits<S>::F>(raw(s));
}
template <> OMC_HD float to_f32_exact<float>(float s) { return s; }

/* ------------------------------------------------------------------------ */
/* Conversion options seen by the element functions                          */
/* ------------------------------------------------------------------------ */

struct ElemOpts {
    bool sat;   /* saturate float overflow */
    bool rne;   /* float -> int: nearest even instead of truncation */
};

/* ------------------------------------------------------------------------ */
/* Element conversion: cvt<D>(S value)                                       */
/* ------------------------------------------------------------------------ */

template <class D> struct Enc;   /* encode from double / float */

template <> struct Enc<double> {
    OMC_HD static double from_f64(double x, const ElemOpts &o) {
        if (o.sat) {
            const uint64_t u = d2u(x);
            x = ((u & 0x7fffffffffffffffULL) == 0x7ff0000000000000ULL)
                ? u2d((u & 0x8000000000000000ULL) | 0x7fefffffffffffffULL) : x;
        }
        return x;
    }
    OMC_HD static double from_f32(float x, const ElemOpts &o) { return from_f64((double)x, o); }
};
template <> struct Enc<float> {
    OMC_HD static float from_f64(double x, const ElemOpts &o) {
        float f = (float)x;
        if (o.sat) {
            const uint32_t u = f2u(f);
            f = ((u & 0x7fffffffu) == 0x7f800000u) ? u2f((u & 0x80000000u) | 0x7f7fffffu) : f;
        }
        return f;
    }
    OMC_HD static float from_f32(float x, const ElemOpts &o) {
        if (o.sat) {
            const uint32_t u = f2u(x);
            x = ((u & 0x7fffffffu) == 0x7f800000u) ? u2f((u & 0x80000000u) | 0x7f7fffffu) : x;
        }
        return x;
    }
};

template <class D, class F>
struct EncSmall {
    OMC_HD static D from_f32(float x, const ElemOpts &o) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        if constexpr (F::kE == 5 && F::kM == 10) {   /* FP16: hardware convert, then saturate */
            uint32_t b = __half_as_ushort(__float2half_rn(x));
            if (o.sat && (b & 0x7fffu) == 0x7c00u) b = (b & 0x8000u) | 0x7bffu;
            return make<D>(b);
        }
#endif
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        /* FP8, saturated: the hardware convert of sm_89+ (round to nearest
         * even, overflow and infinity to the largest finite value), one
         * instruction for about 20 integer operations. It returns one
         * canonical NaN; keep the encoder's, with the sign. */
        if constexpr ((F::kE == 4 && F::kM == 3) || (F::kE == 5 && F::kM == 2)) {
            if (o.sat) {
                unsigned short r;
                if constexpr (F::kM == 3) asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(x));
                else                      asm("cvt.rn.satfinite.e5m2x2.f32 %0, %1, %2;" : "=h"(r) : "f"(0.0f), "f"(x));
                const uint32_t u = f2u(x);
                const uint32_t b = ((u & 0x7fffffffu) > 0x7f800000u) ? (F::nancode | ((u >> 31) << (F::kE + F::kM)))
                                                                    : (uint32_t)(r & 0xffu);
                return make<D>(b);
            }
        }
#endif
        return make<D>(enc_f32<F>(x, o.sat));
    }
    OMC_HD static D from_f64(double x, const ElemOpts &o) { return from_f32(f64_to_f32_ro(x), o); }
};
template <> struct Enc<f16>  : EncSmall<f16,  FmtF16>  {};
template <> struct Enc<bf16> : EncSmall<bf16, FmtBF16> {};
template <> struct Enc<e4m3> : EncSmall<e4m3, FmtE4M3> {};
template <> struct Enc<e5m2> : EncSmall<e5m2, FmtE5M2> {};

/* integer targets */
template <class D> struct EncInt {
    OMC_HD static D from_f64(double x, const ElemOpts &o) { return f64_to_int<D>(x, o.rne); }
    OMC_HD static D from_f32(float x, const ElemOpts &o) {
        if constexpr (Traits<D>::small) return f32_to_small_int<D>(x, o.rne);
        else                            return f64_to_int<D>((double)x, o.rne);
    }
};
template <> struct Enc<int8_t>   : EncInt<int8_t>   {};
template <> struct Enc<int16_t>  : EncInt<int16_t>  {};
template <> struct Enc<int32_t>  : EncInt<int32_t>  {};
template <> struct Enc<int64_t>  : EncInt<int64_t>  {};
template <> struct Enc<uint8_t>  : EncInt<uint8_t>  {};
template <> struct Enc<uint16_t> : EncInt<uint16_t> {};
template <> struct Enc<uint32_t> : EncInt<uint32_t> {};
template <> struct Enc<uint64_t> : EncInt<uint64_t> {};

/** Value of a source element in the cheapest exact intermediate */
template <class S> struct Src;
template <> struct Src<double>  { using T = double; OMC_HD static double get(double s) { return s; } };
template <> struct Src<float>   { using T = float;  OMC_HD static float get(float s) { return s; } };
template <> struct Src<f16>     { using T = float;  OMC_HD static float get(f16 s)  { return to_f32_exact(s); } };
template <> struct Src<bf16>    { using T = float;  OMC_HD static float get(bf16 s) { return to_f32_exact(s); } };
template <> struct Src<e4m3>    { using T = float;  OMC_HD static float get(e4m3 s) { return to_f32_exact(s); } };
template <> struct Src<e5m2>    { using T = float;  OMC_HD static float get(e5m2 s) { return to_f32_exact(s); } };
template <> struct Src<int8_t>  { using T = float;  OMC_HD static float get(int8_t s)   { return (float)s; } };
template <> struct Src<int16_t> { using T = float;  OMC_HD static float get(int16_t s)  { return (float)s; } };
template <> struct Src<uint8_t> { using T = float;  OMC_HD static float get(uint8_t s)  { return (float)s; } };
template <> struct Src<uint16_t>{ using T = float;  OMC_HD static float get(uint16_t s) { return (float)s; } };
template <> struct Src<int32_t> { using T = double; OMC_HD static double get(int32_t s)  { return (double)s; } };
template <> struct Src<uint32_t>{ using T = double; OMC_HD static double get(uint32_t s) { return (double)s; } };
template <> struct Src<int64_t> { using T = double; OMC_HD static double get(int64_t s)  { return i64_to_f64_ro(s); } };
template <> struct Src<uint64_t>{ using T = double; OMC_HD static double get(uint64_t s) { return i64_to_f64_ro(s); } };

template <class D, class X> struct EncFrom;
template <class D> struct EncFrom<D, double> { OMC_HD static D go(double x, const ElemOpts &o) { return Enc<D>::from_f64(x, o); } };
template <class D> struct EncFrom<D, float>  { OMC_HD static D go(float x,  const ElemOpts &o) { return Enc<D>::from_f32(x, o); } };

/** cvt<D>(s): the conversion of one element */
template <class D, class S>
OMC_HD D cvt(S s, const ElemOpts &o) {
    if constexpr (!Traits<S>::is_float && !Traits<D>::is_float) {
        /* integer -> integer: saturate */
        if constexpr (std::is_signed<S>::value) {
            const int64_t v = (int64_t)s;
            if constexpr (std::is_signed<D>::value) {
                return (D)(v < IntLimits<D>::ilo ? IntLimits<D>::ilo
                           : (v > (int64_t)IntLimits<D>::ihi ? (int64_t)IntLimits<D>::ihi : v));
            } else {
                return (D)(v < 0 ? 0u : ((uint64_t)v > IntLimits<D>::ihi ? IntLimits<D>::ihi : (uint64_t)v));
            }
        } else {
            const uint64_t v = (uint64_t)s;
            return (D)(v > IntLimits<D>::ihi ? IntLimits<D>::ihi : v);
        }
    } else if constexpr (std::is_same<S, e5m2>::value && std::is_same<D, f16>::value) {
        return make<f16>((uint32_t)s.bits << 8);            /* E5M2 is the top byte of FP16 */
    } else if constexpr (std::is_same<S, f16>::value && std::is_same<D, e5m2>::value) {
        const uint32_t a = s.bits & 0x7fffu, sign = (s.bits >> 8) & 0x80u;
        uint32_t r = (a + 0x7fu + ((a >> 8) & 1u)) >> 8;   /* same exponent: round the mantissa */
        r = (r > 0x7bu) ? (o.sat ? 0x7bu : 0x7cu) : r;
        r = (a > 0x7c00u) ? 0x7eu : r;
        return make<e5m2>(r | sign);
    } else {
        return EncFrom<D, typename Src<S>::T>::go(Src<S>::get(s), o);
    }
}

/* ------------------------------------------------------------------------ */
/* Stochastic rounding (bit-compatible with HiCMA-X 0.2)                     */
/* ------------------------------------------------------------------------ */

OMC_HD uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/** The 24-bit uniform integer of element (i, j): independent of how the work
 *  is split. u = u24 * 2^-24 is the uniform number in [0, 1) of HiCMA-X 0.2. */
OMC_HD uint32_t sr_u24(uint32_t i, uint32_t j, uint64_t seed, uint64_t bits) {
    const uint64_t pos = (uint64_t)i | ((uint64_t)j << 32);
    return (uint32_t)(splitmix64(bits ^ splitmix64(pos ^ seed)) >> 40);
}

template <class S> OMC_HD uint64_t sr_bits(S s);
template <> OMC_HD uint64_t sr_bits<double>(double s) { return d2u(s); }
template <> OMC_HD uint64_t sr_bits<float>(float s)   { return (uint64_t)f2u(s); }
template <> OMC_HD uint64_t sr_bits<f16>(f16 s)       { return (uint64_t)s.bits; }
template <> OMC_HD uint64_t sr_bits<bf16>(bf16 s)     { return (uint64_t)s.bits; }

/**
 * Stochastic rounding on the integer grid of the target.
 *
 * HiCMA-X 0.2 rounds a = |x| down to lo, the largest representable value
 * <= a, when u < p = (up - a) / (up - lo), up the next value. up - lo is a
 * power of two (the step), so with t = (a - lo) / step, an exact binary
 * fraction, u < 1 - t <=> u24 + 2^24 t < 2^24 <=> u24 + floor(2^24 t) < 2^24
 * (u24 is an integer). So: round up exactly when u24 + t24 carries into bit
 * 24, t24 the top 24 bits of the fraction t. lo and the bits of t come from
 * the bit pattern of a (source of MS mantissa bits and bias BS) by a shift:
 * no floating-point operation, no division, no branch, same results.
 *
 * a: bits of a finite value >= 0. Target with M mantissa bits and bias B.
 * Returns the code of the result (lo or lo + 1; up may be the next binade).
 */
template <class U, int MS, int BS, int M, int B>
OMC_HD uint32_t sr_code(U ab, uint32_t u24) {
    /* all the exponent arithmetic in the width of U, so that the variable
     * shifts vectorize (x86 shifts 64-bit lanes by 64-bit counts) */
    using I = typename std::conditional<sizeof(U) == 8, int64_t, int32_t>::type;
    constexpr I W = (I)sizeof(U) * 8;
    constexpr I EMIN = 1 - B;                           /* smallest normal exponent of the target */
    const I ea = (I)(ab >> MS);
    const U mant = ab & ((U(1) << MS) - 1u);
    const U sig = (ea != 0) ? (mant | (U(1) << MS)) : mant;
    const I e = ((ea != 0) ? ea : (I)1) - (I)BS;        /* a = sig * 2^(e - MS) */
    const I d = EMIN - e;                               /* > 0: subnormal in the target */
    const I k = (I)(MS - M) + (d > 0 ? d : (I)0);       /* bits of sig below the step */
    const U kc = (U)(k < W - 1 ? k : W - 1);
    const U ebase = (d > 0) ? U(0) : (U)(e + (I)B - 1);
    const U q = sig >> kc;
    const uint32_t lo = (uint32_t)((ebase << M) + q);
    const U r = sig - (q << kc);              /* not sig & ((1 << kc) - 1): GCC narrows that count to 32 bits */
    const I sh = k - 24;
    const U shr = (U)(sh > 0 ? (sh < W - 1 ? sh : W - 1) : (I)0), shl = (U)(sh < 0 ? -sh : (I)0);
    const uint32_t t24 = (uint32_t)((r >> shr) << shl);
    return lo + ((u24 + t24) >> 24);
}

/** Bits of |x| for the integer core: FP64 sources in 64 bits, the others as
 *  FP32 (exact) in 32 bits. */
template <class S> struct SrAbs {
    using U = uint32_t;
    static constexpr int MS = 23, BS = 127;
    OMC_HD static float val(S s) { return Src<S>::get(s); }
    OMC_HD static U bits(S s) { return f2u(fabsf(val(s))); }
    OMC_HD static bool neg(S s) { return (f2u(val(s)) >> 31) != 0; }
};
template <> struct SrAbs<double> {
    using U = uint64_t;
    static constexpr int MS = 52, BS = 1023;
    OMC_HD static double val(double s) { return s; }
    OMC_HD static U bits(double s) { return d2u(s) & 0x7fffffffffffffffULL; }
    OMC_HD static bool neg(double s) { return (d2u(s) >> 63) != 0; }
};

/** Stochastic rounding of one element at position (i, j). Above the largest
 *  finite value of the target, and for NaN and Inf, the rules of HiCMA-X 0.2
 *  (round to nearest) apply. */
template <class D, class S> struct SR;
/* FP32 target (from FP64): the floating-point form is faster than the integer
 * core on x86 (FP64 lanes; the integer form needs 64-bit shifts and selects)
 * and gives the same results. */
OMC_HD float nextup_f32(float x)   { return u2f(f2u(x) + 1u); }   /* x >= 0 finite */
/** (u - a) / step for a power of two step: the product by 1/step is the
 *  same correctly rounded number as the quotient, without a division */
OMC_HD double div_pow2(double num, double step) {
    return num * u2d(0x7fe0000000000000ULL - d2u(step));
}
/** round down with probability (upper - a) / (upper - lower), branch-free
 *  form of HiCMA-X 0.2; random = u24 * 2^-24 */
OMC_HD float sr_f32(double value, uint32_t u24) {
    const float random = (float)(int32_t)u24 * (1.0f / 16777216.0f);
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
template <class S> struct SR<float, S> {
    OMC_HD static float go(S s, uint32_t i, uint32_t j, uint64_t seed, const ElemOpts &o) {
        const double v = (double)Src<S>::get(s);
        const float r = sr_f32(v, sr_u24(i, j, seed, sr_bits(s)));
        return Enc<float>::from_f32(r, o);   /* saturation of +-Inf */
    }
};
/** FP16 target from FP64 on GPUs, where FP64 arithmetic is cheaper than the
 *  64-bit integer core: the floating-point form of HiCMA-X 0.2 (same
 *  results; random = u24 * 2^-24). */
OMC_HD float f16_bits_to_f32(uint32_t h) {
#if defined(OMC_ON_DEVICE)
    return __half2float(__ushort_as_half((unsigned short)h));
#else
    return dec_f32<FmtF16>(h);
#endif
}
OMC_HD uint32_t sr_f16_fp(double value, uint32_t u24, bool sat) {
    const float random = (float)(int32_t)u24 * (1.0f / 16777216.0f);
    const double a = fabs(value);
    const uint32_t sign = (value < 0.0) ? 0x8000u : 0u;
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
    r = (top & (lv != a)) ? enc_f32<FmtF16>((float)value, sat) : r;
    r = (value != value) ? (0x7e00u | (uint32_t)((d2u(value) >> 48) & 0x8000u)) : r;
    return r;
}
template <class S> struct SR<f16, S> {
    OMC_HD static f16 go(S s, uint32_t i, uint32_t j, uint64_t seed, const ElemOpts &o) {
#if defined(OMC_ON_DEVICE)
        if constexpr (sizeof(S) == 8)
            return make<f16>(sr_f16_fp((double)s, sr_u24(i, j, seed, sr_bits(s)), o.sat));
#endif
        using A = SrAbs<S>;
        const auto x = A::val(s);
        const double a = fabs((double)x);
        const uint32_t sign = (x < 0) ? 0x8000u : 0u;     /* as HiCMA-X 0.2: -0 gives +0 */
        const uint32_t u = sr_u24(i, j, seed, sr_bits(s));
        uint32_t r = sr_code<typename A::U, A::MS, A::BS, 10, 15>(A::bits(s), u) | sign;
        /* as HiCMA-X 0.2: above 65504 round to nearest through FP32 */
        r = (a > 65504.0) ? enc_f32<FmtF16>((float)x, o.sat) : r;
        r = (x != x) ? (0x7e00u | (A::neg(s) ? 0x8000u : 0u)) : r;
        return make<f16>(r);
    }
};
template <class D, class S> struct SRSmall {
    OMC_HD static D go(S s, uint32_t i, uint32_t j, uint64_t seed, const ElemOpts &o) {
        using F = typename Traits<D>::F;
        using A = SrAbs<S>;
        const auto x = A::val(s);
        const double a = fabs((double)x);
        const uint32_t sign = (x < 0) ? F::signbit : 0u;  /* -0 gives +0, as for FP16 */
        const uint32_t u = sr_u24(i, j, seed, sr_bits(s));
        uint32_t r = sr_code<typename A::U, A::MS, A::BS, F::kM, F::bias>(A::bits(s), u) | sign;
        /* above the largest finite value: round to nearest (from the exact value) */
        const double maxf = (double)dec_f32<F>(F::maxcode);
        r = (a > maxf) ? enc_f32<F>(f64_to_f32_ro((double)x), o.sat) : r;
        r = (x != x) ? F::nancode : r;
        return make<D>(r);
    }
};
template <class S> struct SR<bf16, S> : SRSmall<bf16, S> {};
template <class S> struct SR<e4m3, S> : SRSmall<e4m3, S> {};
template <class S> struct SR<e5m2, S> : SRSmall<e5m2, S> {};

/** Pairs that support stochastic rounding: float source, float target of
 *  lower precision (FP64 > FP32 > FP16, BF16 > FP8) */
template <class D, class S> struct sr_ok { static constexpr bool value = false; };
#define OMC_SR_OK(D, S) template <> struct sr_ok<D, S> { static constexpr bool value = true; }
OMC_SR_OK(float, double);
OMC_SR_OK(f16, double);  OMC_SR_OK(f16, float);
OMC_SR_OK(bf16, double); OMC_SR_OK(bf16, float);
OMC_SR_OK(e4m3, double); OMC_SR_OK(e4m3, float); OMC_SR_OK(e4m3, f16); OMC_SR_OK(e4m3, bf16);
OMC_SR_OK(e5m2, double); OMC_SR_OK(e5m2, float); OMC_SR_OK(e5m2, f16); OMC_SR_OK(e5m2, bf16);
#undef OMC_SR_OK

/* ------------------------------------------------------------------------ */
/* Scaling                                                                   */
/* ------------------------------------------------------------------------ */

/** cvt<D>(s * scale), the product in FP64 for FP64 and 32/64-bit integer
 *  sources, in FP32 otherwise. */
/** The product rounded to X. On gfx9 the AMDGPU backend fuses an FP32
 *  multiply and the FP16 conversion that follows into one v_mad_mix, which
 *  rounds once, even with -ffp-contract=off; an empty asm keeps the product. */
template <class X>
OMC_HD X scaled_product(X a, X b) {
    X x = a * b;
#if defined(__HIP_DEVICE_COMPILE__)
    asm volatile("" : "+v"(x));
#endif
    return x;
}

template <class D, class S>
OMC_HD D cvt_scaled(S s, float scale, const ElemOpts &o) {
    using X = typename Src<S>::T;
    const X x = scaled_product<X>(Src<S>::get(s), (X)scale);
    return EncFrom<D, X>::go(x, o);
}

/** Stochastic rounding of s * scale: the product as in cvt_scaled, then the
 *  stochastic rounding of that value (pairs with sr_ok<D, S>). */
template <class D, class S>
OMC_HD D sr_scaled(S s, float scale, uint32_t i, uint32_t j, uint64_t seed, const ElemOpts &o) {
    using X = typename Src<S>::T;
    const X x = scaled_product<X>(Src<S>::get(s), (X)scale);
    return SR<D, X>::go(x, i, j, seed, o);
}

/** |s| as float for amax (NaN -> 0) */
template <class S>
OMC_HD float abs_f32(S s) {
    const float v = fabsf((float)Src<S>::get(s));
    return (v == v) ? v : 0.0f;
}
template <>
OMC_HD float abs_f32<double>(double s) {
    /* nearest FP32, as HiCMA-X 0.2 (rounding up would cost the whole tile a
     * bit of resolution when 1/amax lands just above a power of two) */
    const double a = fabs(s);
    return (a == a) ? (float)a : 0.0f;
}

/** Largest finite value of a format (for dynamic scaling) */
template <class D> OMC_HD float max_finite();
template <> OMC_HD float max_finite<double>() { return 3.4028234663852886e38f; }
template <> OMC_HD float max_finite<float>()  { return 3.4028234663852886e38f; }
template <> OMC_HD float max_finite<f16>()    { return 65504.0f; }
template <> OMC_HD float max_finite<bf16>()   { return 3.3895313892515355e38f; }
template <> OMC_HD float max_finite<e4m3>()   { return 448.0f; }
template <> OMC_HD float max_finite<e5m2>()   { return 57344.0f; }

/** s = largest power of two with s * amax <= target, kept a normal float
 *  together with 1/s; amax = 0 (or not finite) gives s = 1. */
OMC_HD float pow2_scale(float amax, float target) {
    float s = 1.0f;
    if (amax > 0.0f && amax <= 3.4028234663852886e38f) {
        const float r = target / amax;
        int e = 126;
        if (r <= 3.4028234663852886e38f) {
            e = (int)((f2u(r) >> 23) & 0xffu) - 127;          /* floor(log2(r)) for normal r */
            if (e < -126) e = -126;
        }
        if (e > 126) e = 126;
        if (e < -126) e = -126;
        s = u2f((uint32_t)(e + 127) << 23);
    }
    return s;
}

} /* namespace omc */

#endif /* OMC_SCALAR_HPP */
