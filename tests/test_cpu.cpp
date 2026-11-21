/**
 * @file test_cpu.cpp
 * @brief Host API correctness: every type pair on every CPU code path against
 * the scalar core, 2-D layouts, transposition, padding, in place, scaling,
 * threads, and stochastic rounding against the HiCMA-X 0.2 implementation.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <string>
#include "omnicast/omnicast.h"
#include "../src/core/omc_scalar.hpp"
#include "../src/cpu/cpu_kernels.hpp"
#include "legacy_hicma_sr.h"

using namespace omc;

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; if (g_fail < 40) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

/* ---- reference: scalar core, element by element -------------------------- */

template <class D, class S>
static void ref_one(const void *s, void *d, const ElemOpts &o) { *(D *)d = cvt<D>(*(const S *)s, o); }

using RefFn = void (*)(const void *, void *, const ElemOpts &);
static RefFn g_ref[OMC_NUM_DTYPES][OMC_NUM_DTYPES];

template <class S, class D> static void reg_pair(int si, int di) { g_ref[si][di] = &ref_one<D, S>; }
template <class S> static void reg_src(int si) {
#define X(E, T) reg_pair<S, T>(si, (int)E);
    OMC_FOR_EACH_TYPE(X)
#undef X
}
static void reg_all() {
#define X(E, T) reg_src<T>((int)E);
    OMC_FOR_EACH_TYPE(X)
#undef X
}

/* value equality with NaN == NaN (any payload) */
static bool is_nan_bits(omc_dtype_t t, const void *p) {
    switch (t) {
    case OMC_FP64: { double v; memcpy(&v, p, 8); return v != v; }
    case OMC_FP32: { float v; memcpy(&v, p, 4); return v != v; }
    case OMC_FP16: { uint16_t h; memcpy(&h, p, 2); return (h & 0x7c00) == 0x7c00 && (h & 0x3ff); }
    case OMC_BF16: { uint16_t h; memcpy(&h, p, 2); return (h & 0x7f80) == 0x7f80 && (h & 0x7f); }
    case OMC_FP8_E4M3: { uint8_t h = *(const uint8_t *)p; return (h & 0x7f) == 0x7f; }
    case OMC_FP8_E5M2: { uint8_t h = *(const uint8_t *)p; return (h & 0x7c) == 0x7c && (h & 3); }
    default: return false;
    }
}
static bool same(omc_dtype_t t, const void *a, const void *b) {
    size_t sz = omc_dtype_size(t);
    if (!memcmp(a, b, sz)) return true;
    return is_nan_bits(t, a) && is_nan_bits(t, b);
}

/* ---- source data with edge cases ----------------------------------------- */

static std::vector<unsigned char> make_src(omc_dtype_t t, size_t n, std::mt19937_64 &g) {
    size_t sz = omc_dtype_size(t);
    std::vector<unsigned char> v(n * sz + 64);
    for (size_t k = 0; k < n; k++) {
        unsigned char *p = v.data() + k * sz;
        uint64_t r = g();
        if (t == OMC_FP64) {
            double x;
            switch (r % 8) {
            case 0: x = ldexp((double)(g() >> 11) / 9007199254740992.0, (int)(g() % 40) - 20); break;
            case 1: x = ldexp((double)(g() >> 11) / 9007199254740992.0, (int)(g() % 300) - 150); break;
            case 2: { static const double sp[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 1.7976931348623157e308, 65504.0, 65520.0, 448.0, 464.0,
                       57344.0, 1e300, -1e-300, 5e-324, 3.4028234663852886e38, 3.5e38, 6.103515625e-05, 5.9604644775390625e-08,
                       127.5, -128.5, 255.5, 2147483647.5, -2147483648.7, 9.3e18, -9.3e18, 1.9e19, 4294967296.0};
                      x = sp[g() % (sizeof(sp) / sizeof(sp[0]))]; break; }
            case 3: x = (double)(int64_t)(g() % 70000) - 35000.0 + 0.5 * (double)(g() % 3); break;
            default: x = ((double)(g() >> 11) / 9007199254740992.0 - 0.5) * 8.0; break;
            }
            if (r & 0x100) x = -x;
            memcpy(p, &x, 8);
        } else if (t == OMC_FP32) {
            float x;
            if (r % 4 == 0) { uint32_t u = (uint32_t)g(); memcpy(&x, &u, 4); }
            else if (r % 4 == 1) x = ldexpf((float)(g() >> 40) / 16777216.0f, (int)(g() % 60) - 30);
            else x = ((float)(g() >> 40) / 16777216.0f - 0.5f) * 1000.0f;
            memcpy(p, &x, 4);
        } else {
            memcpy(p, &r, sz);   /* every bit pattern of small floats and integers */
            if ((r >> 56) < 24) {   /* and their special codes, often */
                static const uint64_t sp16[] = {0x7c00, 0xfc00, 0x7e00, 0x7bff, 0x0001, 0x8000, 0x7f80, 0xff80, 0x7f7f, 0x0400};
                static const uint64_t sp8[]  = {0x7f, 0xff, 0x7e, 0x7c, 0xfc, 0x7b, 0x01, 0x80, 0x08, 0x04};
                uint64_t c = (sz == 2) ? sp16[(r >> 40) % 10] : (sz == 1) ? sp8[(r >> 40) % 10] : (r >> 63 ? ~0ULL : (1ULL << (8 * sz - 1)));
                memcpy(p, &c, sz);
            }
        }
    }
    return v;
}

static ElemOpts eopts(const omc_options_t &o) {
    return ElemOpts{(o.flags & OMC_FLAG_SATURATE) != 0, o.rounding == OMC_ROUND_NEAREST_EVEN};
}

/* ---- 1-D: every pair, every ISA, lengths that exercise the tails --------- */

static void test_pairs(const char *isa) {
    if (omc_cpu_set_isa(isa) != 0) { printf("  (ISA %s not available)\n", isa); return; }
    std::mt19937_64 g(42);
    const size_t lens[] = {1, 3, 7, 8, 15, 16, 17, 31, 33, 63, 64, 65, 127, 129, 255, 513, 1000, 4099};
    long checked = 0;
    for (int si = 0; si < OMC_NUM_DTYPES; si++)
        for (int di = 0; di < OMC_NUM_DTYPES; di++)
            for (int variant = 0; variant < 3; variant++) {
                omc_options_t o = {};
                if (variant == 1) o.flags = OMC_FLAG_SATURATE;
                if (variant == 2) o.rounding = OMC_ROUND_NEAREST_EVEN;
                const omc_dtype_t st = (omc_dtype_t)si, dt = (omc_dtype_t)di;
                const size_t ss = omc_dtype_size(st), ds = omc_dtype_size(dt);
                for (size_t n : lens) {
                    auto src = make_src(st, n, g);
                    /* misaligned source/destination too */
                    size_t off = (n % 3);
                    std::vector<unsigned char> dst(n * ds + 64, 0xcd), ref(n * ds + 64, 0xcd);
                    memmove(src.data() + off * ss, src.data(), n * ss);
                    omc_status_t r = omc_convert(st, src.data() + off * ss, dt, dst.data() + off * ds, n, &o);
                    CHECK(r == OMC_SUCCESS, "%s->%s status %d", omc_dtype_name(st), omc_dtype_name(dt), r);
                    ElemOpts eo = eopts(o);
                    for (size_t k = 0; k < n; k++)
                        g_ref[si][di](src.data() + (off + k) * ss, ref.data() + (off + k) * ds, eo);
                    int bad = 0;
                    for (size_t k = 0; k < n && bad < 3; k++) {
                        if (!same(dt, dst.data() + (off + k) * ds, ref.data() + (off + k) * ds)) {
                            bad++;
                            uint64_t sb = 0, db = 0, rb = 0;
                            memcpy(&sb, src.data() + (off + k) * ss, ss);
                            memcpy(&db, dst.data() + (off + k) * ds, ds);
                            memcpy(&rb, ref.data() + (off + k) * ds, ds);
                            CHECK(false, "[%s] %s->%s v%d n=%zu k=%zu src=0x%llx got=0x%llx ref=0x%llx", isa,
                                  omc_dtype_name(st), omc_dtype_name(dt), variant, n, k,
                                  (unsigned long long)sb, (unsigned long long)db, (unsigned long long)rb);
                        }
                    }
                    /* no write past the end */
                    CHECK(dst[(off + n) * ds] == 0xcd, "%s->%s overrun", omc_dtype_name(st), omc_dtype_name(dt));
                    checked += n;
                }
            }
    printf("  [%s] all %d pairs x 3 option sets: %ld elements checked\n", isa, OMC_NUM_DTYPES * OMC_NUM_DTYPES, checked);
}

/* ---- stochastic rounding vs HiCMA-X 0.2 ---------------------------------- */

static void test_sr() {
    std::mt19937_64 g(9);
    const int m = 37, n = 23, ld = 41;
    std::vector<double> A(ld * n);
    for (auto &x : A) x = ldexp((double)(g() >> 11) / 9007199254740992.0 - 0.5, (int)(g() % 40) - 25);
    A[0] = 1.0 / 3.0; A[1] = 65504.0; A[2] = 65510.0; A[3] = 1e-9; A[4] = -2.5e-6; A[5] = 0.0; A[6] = -0.0;
    const uint64_t seed = 0x1234abcdULL;
    omc_options_t o = {};
    o.rounding = OMC_ROUND_STOCHASTIC;
    o.seed = seed;
    /* FP64 -> FP32: HiCMA-X double2float_round_CPU */
    std::vector<float> got(ld * n, 0.f), ref(ld * n, 0.f);
    omc_status_t r = omc_convert_2d(OMC_FP64, A.data(), ld, OMC_FP32, got.data(), ld, m, n, &o);
    CHECK(r == OMC_SUCCESS, "SR d2s status %d", r);
    legacy_double2float_round_CPU(m, n, A.data(), ld, ref.data(), ld, seed);
    int bad = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++)
        if (memcmp(&got[j * ld + i], &ref[j * ld + i], 4)) bad++;
    CHECK(bad == 0, "SR fp64->fp32 differs from HiCMA-X at %d entries", bad);
    /* FP64/FP32 -> FP16: HiCMA-X stochastic_rounding_fp16 (GPU code, ported) */
    std::vector<uint16_t> h(ld * n), hr(ld * n);
    r = omc_convert_2d(OMC_FP64, A.data(), ld, OMC_FP16, h.data(), ld, m, n, &o);
    CHECK(r == OMC_SUCCESS, "SR d2h status %d", r);
    bad = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
        double v = A[j * ld + i];
        uint64_t bits; memcpy(&bits, &v, 8);
        uint16_t e = legacy_sr_fp16(v, legacy_sr_u01(i, j, seed, bits));
        if (e != h[j * ld + i]) bad++;
    }
    CHECK(bad == 0, "SR fp64->fp16 differs from HiCMA-X at %d entries", bad);
    std::vector<float> S(ld * n);
    for (int k = 0; k < ld * n; k++) S[k] = (float)A[k];
    r = omc_convert_2d(OMC_FP32, S.data(), ld, OMC_FP16, h.data(), ld, m, n, &o);
    bad = 0;
    for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
        float v = S[j * ld + i];
        uint32_t b32; memcpy(&b32, &v, 4);
        uint16_t e = legacy_sr_fp16((double)v, legacy_sr_u01(i, j, seed, (uint64_t)b32));
        if (e != h[j * ld + i]) bad++;
    }
    CHECK(bad == 0, "SR fp32->fp16 differs from HiCMA-X at %d entries", bad);
    /* unbiasedness: mean of SR(1/3 + tiny) over many seeds */
    double acc = 0; const int T = 200000;
    std::vector<double> one(T, 1.0 / 3.0); std::vector<uint16_t> oh(T);
    o.seed = 77;
    omc_convert(OMC_FP64, one.data(), OMC_FP16, oh.data(), T, &o);
    for (int k = 0; k < T; k++) acc += omc_fp16_to_fp32(oh[k]);
    acc /= T;
    CHECK(fabs(acc - 1.0 / 3.0) < 2e-6, "SR mean %.9f vs %.9f", acc, 1.0 / 3.0);
    /* SR for the other targets: every result is one of the two neighbours */
    for (omc_dtype_t dt : {OMC_BF16, OMC_FP8_E4M3, OMC_FP8_E5M2}) {
        const size_t ds = omc_dtype_size(dt);
        const uint32_t magmask = (dt == OMC_BF16) ? 0x7fff : 0x7f;
        std::vector<uint8_t> out(ld * n * 2);
        o.seed = seed;
        r = omc_convert_2d(OMC_FP64, A.data(), ld, dt, out.data(), ld, m, n, &o);
        CHECK(r == OMC_SUCCESS, "SR fp64->%s status %d", omc_dtype_name(dt), r);
        auto dec = [&](uint32_t c) { float f; omc_convert(dt, &c, OMC_FP32, &f, 1, nullptr); return (double)f; };
        bad = 0;
        for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
            const double a = fabs(A[j * ld + i]);
            uint32_t c = 0; memcpy(&c, out.data() + (j * ld + i) * ds, ds);
            c &= magmask;
            const double v = dec(c);
            /* no representable value strictly between v and a */
            if (v < a && c + 1 <= magmask && dec(c + 1) < a) bad++;
            if (v > a && c > 0 && dec(c - 1) > a) bad++;
        }
        CHECK(bad == 0, "SR fp64->%s: %d results not adjacent", omc_dtype_name(dt), bad);
    }
    printf("  stochastic rounding: bit-identical to HiCMA-X 0.2 (fp64->fp32, fp64->fp16, fp32->fp16), unbiased\n");
}

/* ---- 2-D, transposition, padding ----------------------------------------- */

static void test_2d() {
    std::mt19937_64 g(5);
    const omc_dtype_t pairs[][2] = {{OMC_FP64, OMC_FP32}, {OMC_FP64, OMC_FP8_E4M3}, {OMC_FP32, OMC_FP16},
                                    {OMC_FP16, OMC_FP64}, {OMC_FP32, OMC_FP8_E4M3}, {OMC_FP16, OMC_FP8_E5M2},
                                    {OMC_FP32, OMC_INT8}, {OMC_INT32, OMC_FP64}, {OMC_FP64, OMC_FP64}};
    int cases = 0;
    for (auto &pr : pairs) {
        const omc_dtype_t st = pr[0], dt = pr[1];
        const size_t ss = omc_dtype_size(st), ds = omc_dtype_size(dt);
        for (int tr = 0; tr < 2; tr++)
            for (int pad = 0; pad < 2; pad++)
                for (int m : {1, 5, 64, 70, 131}) for (int n : {1, 3, 64, 77}) {
                    const int lds = m + (int)(g() % 5);
                    const int rows = tr ? n : m, cols = tr ? m : n;
                    const int dr = rows + (pad ? 3 : 0), dc = cols + (pad ? 2 : 0);
                    const int ldd = dr + (int)(g() % 4);
                    auto src = make_src(st, (size_t)lds * n, g);
                    std::vector<unsigned char> dst((size_t)ldd * dc * ds + 16, 0xab), ref((size_t)ldd * dc * ds + 16, 0xab);
                    omc_options_t o = {};
                    o.flags = (tr ? OMC_FLAG_TRANSPOSE : 0) | OMC_FLAG_SATURATE;
                    if (pad) { o.dst_rows = dr; o.dst_cols = dc; }
                    omc_status_t r = omc_convert_2d(st, src.data(), lds, dt, dst.data(), ldd, m, n, &o);
                    CHECK(r == OMC_SUCCESS, "2d status %d", r);
                    ElemOpts eo{true, false};
                    for (int j = 0; j < dc; j++) for (int i = 0; i < dr; i++) {
                        unsigned char *rp = ref.data() + ((size_t)j * ldd + i) * ds;
                        int si_ = tr ? j : i, sj = tr ? i : j;   /* source (row, col) */
                        if (si_ < m && sj < n) g_ref[st][dt](src.data() + ((size_t)sj * lds + si_) * ss, rp, eo);
                        else memset(rp, 0, ds);
                    }
                    int bad = 0;
                    for (int j = 0; j < dc; j++) for (int i = 0; i < dr; i++)
                        if (!same(dt, dst.data() + ((size_t)j * ldd + i) * ds, ref.data() + ((size_t)j * ldd + i) * ds)) bad++;
                    /* rows between dr and ldd are untouched */
                    for (int j = 0; j < dc; j++) for (int i = dr; i < ldd; i++)
                        if (dst[((size_t)j * ldd + i) * ds] != 0xab) bad++;
                    CHECK(bad == 0, "2d %s->%s tr=%d pad=%d m=%d n=%d: %d bad", omc_dtype_name(st), omc_dtype_name(dt), tr, pad, m, n, bad);
                    cases++;
                }
    }
    printf("  2-D / transpose / padding: %d cases\n", cases);
}

/* ---- in place -------------------------------------------------------------- */

static void test_inplace() {
    std::mt19937_64 g(11);
    int cases = 0;
    for (int si = 0; si < OMC_NUM_DTYPES; si++) for (int di = 0; di < OMC_NUM_DTYPES; di++) {
        const omc_dtype_t st = (omc_dtype_t)si, dt = (omc_dtype_t)di;
        const size_t ss = omc_dtype_size(st), ds = omc_dtype_size(dt), mx = ss > ds ? ss : ds;
        for (size_t n : {1, 7, 1000, 1024, 1025, 5000}) {
            auto src = make_src(st, n, g);
            std::vector<unsigned char> buf(n * mx + 16), ref(n * ds + 16);
            memcpy(buf.data(), src.data(), n * ss);
            omc_convert(st, src.data(), dt, ref.data(), n, nullptr);
            omc_status_t r = omc_convert_inplace(st, dt, buf.data(), n, nullptr);
            CHECK(r == OMC_SUCCESS, "inplace status %d", r);
            int bad = 0;
            for (size_t k = 0; k < n; k++) if (!same(dt, buf.data() + k * ds, ref.data() + k * ds)) bad++;
            CHECK(bad == 0, "inplace %s->%s n=%zu: %d bad", omc_dtype_name(st), omc_dtype_name(dt), n, bad);
            cases++;
        }
        /* 2-D with ld > m: the rows between m and ld keep their bytes in neither layout, only the matrix counts */
        const int m = 300, n2 = 9, ld = 333;
        auto src = make_src(st, (size_t)ld * n2, g);
        std::vector<unsigned char> buf((size_t)ld * n2 * mx + 16), ref((size_t)ld * n2 * ds + 16);
        memcpy(buf.data(), src.data(), (size_t)ld * n2 * ss);
        omc_convert_2d(st, src.data(), ld, dt, ref.data(), ld, m, n2, nullptr);
        omc_status_t r = omc_convert_inplace_2d(st, dt, buf.data(), ld, m, n2, nullptr);
        CHECK(r == OMC_SUCCESS, "inplace2d status %d", r);
        int bad = 0;
        for (int j = 0; j < n2; j++) for (int i = 0; i < m; i++)
            if (!same(dt, buf.data() + ((size_t)j * ld + i) * ds, ref.data() + ((size_t)j * ld + i) * ds)) bad++;
        CHECK(bad == 0, "inplace2d %s->%s: %d bad", omc_dtype_name(st), omc_dtype_name(dt), bad);
        cases++;
    }
    /* in place with stochastic rounding equals out of place */
    {
        const int m = 50, n = 20;
        std::vector<double> A(m * n); for (auto &x : A) x = (double)(g() % 100000) / 7777.0;
        std::vector<float> ref(m * n);
        omc_options_t o = {}; o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 3;
        omc_convert_2d(OMC_FP64, A.data(), m, OMC_FP32, ref.data(), m, m, n, &o);
        omc_convert_inplace_2d(OMC_FP64, OMC_FP32, A.data(), m, m, n, &o);
        CHECK(!memcmp(A.data(), ref.data(), sizeof(float) * m * n), "inplace SR differs");
    }
    printf("  in place (1-D and 2-D, every pair): %d cases\n", cases);
}

/* ---- scaling, amax, threads, scalars, errors ------------------------------ */

static void test_misc() {
    std::mt19937_64 g(13);
    const int n = 10000;
    std::vector<double> A(n);
    for (auto &x : A) x = ((double)(g() >> 11) / 9007199254740992.0 - 0.5) * 1e-3;
    A[17] = -3.75e-3;
    double amax = 0;
    CHECK(omc_amax_2d(OMC_FP64, A.data(), n, n, 1, &amax) == OMC_SUCCESS && amax == 3.75e-3, "amax %g", amax);
    float sc[3] = {0, 0, 0};
    omc_options_t o = {};
    o.scale_mode = OMC_SCALE_DYNAMIC_POW2; o.scale_ptr = sc; o.flags = OMC_FLAG_SATURATE;
    std::vector<uint8_t> q(n), q2(n);
    CHECK(omc_convert(OMC_FP64, A.data(), OMC_FP8_E4M3, q.data(), n, &o) == OMC_SUCCESS, "dyn scale");
    CHECK(sc[0] == (float)3.75e-3 && sc[1] == 65536.0f && sc[2] == 1.0f / 65536.0f, "dyn scale %g %g %g", sc[0], sc[1], sc[2]);
    omc_options_t h = {}; h.scale_mode = OMC_SCALE_HOST; h.scale = 65536.0f; h.flags = OMC_FLAG_SATURATE;
    omc_convert(OMC_FP64, A.data(), OMC_FP8_E4M3, q2.data(), n, &h);
    CHECK(!memcmp(q.data(), q2.data(), n), "dynamic and host scale differ");
    int zeros = 0; for (auto c : q) zeros += ((c & 0x7f) == 0);
    CHECK(zeros < n / 100, "scaled fp8 has %d zeros", zeros);
    /* threads give the same bits */
    std::vector<float> f1(n), f2(n);
    omc_options_t t8 = {}; t8.num_threads = 8;
    omc_convert(OMC_FP64, A.data(), OMC_FP32, f1.data(), n, nullptr);
    omc_convert(OMC_FP64, A.data(), OMC_FP32, f2.data(), n, &t8);
    CHECK(!memcmp(f1.data(), f2.data(), 4 * n), "threads differ");
    /* scalars */
    CHECK(omc_fp32_to_fp16(1.0f) == 0x3c00 && omc_fp16_to_fp32(0x3c00) == 1.0f, "scalar fp16");
    CHECK(omc_fp32_to_bf16(1.0f) == 0x3f80 && omc_fp64_to_e4m3(448.0, 0) == 0x7e && omc_fp64_to_e4m3(500.0, 1) == 0x7e, "scalar small");
    CHECK(omc_fp64_to_e4m3(500.0, 0) == 0x7f && omc_fp32_to_e5m2(1e6f, 0) == 0x7c && omc_e5m2_to_fp32(0x3c) == 1.0f, "scalar fp8");
    /* names */
    CHECK(omc_dtype_from_name("double") == OMC_FP64 && omc_dtype_from_name("half") == OMC_FP16 &&
          omc_dtype_from_name("e4m3") == OMC_FP8_E4M3 && omc_dtype_from_name("bogus") == OMC_NUM_DTYPES, "names");
    /* errors */
    CHECK(omc_convert(OMC_FP64, nullptr, OMC_FP32, f1.data(), 4, nullptr) == OMC_ERROR_INVALID_VALUE, "null");
    CHECK(omc_convert_2d(OMC_FP64, A.data(), 2, OMC_FP32, f1.data(), 4, 4, 4, nullptr) == OMC_ERROR_INVALID_VALUE, "lds < m");
    omc_options_t bad = {}; bad.rounding = OMC_ROUND_STOCHASTIC;
    CHECK(omc_convert(OMC_FP32, f1.data(), OMC_FP64, A.data(), 4, &bad) == OMC_ERROR_NOT_SUPPORTED, "SR widening");
    printf("  scaling, amax, threads, scalars, errors: ok\n");
}

int main() {
    reg_all();
    const char *isa0 = omc_cpu_isa();
    printf("OmniCast %s, CPU path %s, GPU backend %s\n", omc_version_string(), isa0, omc_gpu_backend());
    for (const char *isa : {"generic", "avx2", "avx512"}) test_pairs(isa);
    omc_cpu_set_isa(isa0);
    test_sr();
    test_2d();
    test_inplace();
    test_misc();
    printf(g_fail ? "FAILED (%d)\n" : "PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
