/**
 * @file test_gpu.cu
 * @brief GPU API correctness: every type pair against the host API (itself
 * checked against the scalar core), 2-D / transposed / padded layouts,
 * single-pass in place (every size-changing pair, many sizes, concurrent
 * streams), stochastic rounding (bit-identical to the host), scaling, amax.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include "omnicast/omnicast.h"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gMalloc hipMalloc
#define gFree hipFree
#define gMemcpy hipMemcpy
#define gMemset hipMemset
#define gH2D hipMemcpyHostToDevice
#define gD2H hipMemcpyDeviceToHost
#define gSync hipDeviceSynchronize
#define gStream_t hipStream_t
#define gStreamCreate hipStreamCreate
#define gStreamDestroy hipStreamDestroy
#define gStreamSync hipStreamSynchronize
#define gGetDeviceProperties hipGetDeviceProperties
typedef hipDeviceProp_t gProp;
#else
#include <cuda_runtime.h>
#define gMalloc cudaMalloc
#define gFree cudaFree
#define gMemcpy cudaMemcpy
#define gMemset cudaMemset
#define gH2D cudaMemcpyHostToDevice
#define gD2H cudaMemcpyDeviceToHost
#define gSync cudaDeviceSynchronize
#define gStream_t cudaStream_t
#define gStreamCreate cudaStreamCreate
#define gStreamDestroy cudaStreamDestroy
#define gStreamSync cudaStreamSynchronize
#define gGetDeviceProperties cudaGetDeviceProperties
typedef cudaDeviceProp gProp;
#endif

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; if (g_fail < 40) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

static bool is_nan_bits(omc_dtype_t t, const unsigned char *p) {
    switch (t) {
    case OMC_FP64: { double v; memcpy(&v, p, 8); return v != v; }
    case OMC_FP32: { float v; memcpy(&v, p, 4); return v != v; }
    case OMC_FP16: { uint16_t h; memcpy(&h, p, 2); return (h & 0x7c00) == 0x7c00 && (h & 0x3ff); }
    case OMC_BF16: { uint16_t h; memcpy(&h, p, 2); return (h & 0x7f80) == 0x7f80 && (h & 0x7f); }
    case OMC_FP8_E4M3: return (p[0] & 0x7f) == 0x7f;
    case OMC_FP8_E5M2: return (p[0] & 0x7c) == 0x7c && (p[0] & 3);
    default: return false;
    }
}
static int count_diff(omc_dtype_t t, const unsigned char *a, const unsigned char *b, size_t n) {
    const size_t s = omc_dtype_size(t);
    int bad = 0;
    for (size_t k = 0; k < n; k++)
        if (memcmp(a + k * s, b + k * s, s) && !(is_nan_bits(t, a + k * s) && is_nan_bits(t, b + k * s))) bad++;
    return bad;
}

static std::vector<unsigned char> make_src(omc_dtype_t t, size_t n, std::mt19937_64 &g) {
    const size_t sz = omc_dtype_size(t);
    std::vector<unsigned char> v(n * sz + 16);
    for (size_t k = 0; k < n; k++) {
        uint64_t r = g();
        if (t == OMC_FP64) {
            double x;
            switch (r % 6) {
            case 0: x = ldexp((double)(g() >> 11) / 9007199254740992.0, (int)(g() % 300) - 150); break;
            case 1: { static const double sp[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 65504.0, 65520.0, 448.0, 464.0,
                       57344.0, 1e300, 5e-324, 3.5e38, 127.5, -128.5, 2147483647.5, 9.3e18, 1.9e19};
                      x = sp[g() % (sizeof(sp) / sizeof(sp[0]))]; break; }
            case 2: x = (double)(int64_t)(g() % 70000) - 35000.0 + 0.5 * (double)(g() % 3); break;
            default: x = ((double)(g() >> 11) / 9007199254740992.0 - 0.5) * 8.0; break;
            }
            if (r & 0x100) x = -x;
            memcpy(v.data() + k * sz, &x, 8);
        } else if (t == OMC_FP32 && (r % 3)) {
            float x = ((float)(g() >> 40) / 16777216.0f - 0.5f) * 1000.0f;
            memcpy(v.data() + k * sz, &x, 4);
        } else {
            memcpy(v.data() + k * sz, &r, sz);
        }
    }
    return v;
}

struct DBuf {
    void *p = nullptr;
    explicit DBuf(size_t b) { gMalloc(&p, b ? b : 1); gMemset(p, 0, b ? b : 1); }
    ~DBuf() { gFree(p); }
};

static void test_pairs() {
    std::mt19937_64 g(1);
    long checked = 0;
    for (int si = 0; si < OMC_NUM_DTYPES; si++) for (int di = 0; di < OMC_NUM_DTYPES; di++)
        for (int v = 0; v < 2; v++) {
            const omc_dtype_t st = (omc_dtype_t)si, dt = (omc_dtype_t)di;
            const size_t ss = omc_dtype_size(st), ds = omc_dtype_size(dt);
            omc_options_t o = {};
            if (v) { o.flags = OMC_FLAG_SATURATE; o.rounding = OMC_ROUND_NEAREST_EVEN; }
            for (size_t n : {1, 15, 16, 17, 1000, 4096, 100003}) {
                for (int off = 0; off < 2; off++) {   /* off = 1: misaligned pointers */
                    auto h = make_src(st, n, g);
                    std::vector<unsigned char> ref(n * ds), got(n * ds);
                    omc_convert(st, h.data(), dt, ref.data(), n, &o);
                    DBuf ds_(n * ss + 16), dd_(n * ds + 16);
                    char *s = (char *)ds_.p + off * ss, *d = (char *)dd_.p + off * ds;
                    gMemcpy(s, h.data(), n * ss, gH2D);
                    omc_status_t r = omc_gpu_convert(st, s, dt, d, n, &o, nullptr);
                    CHECK(r == OMC_SUCCESS, "%s->%s status %s", omc_dtype_name(st), omc_dtype_name(dt), omc_status_string(r));
                    gMemcpy(got.data(), d, n * ds, gD2H);
                    int bad = count_diff(dt, got.data(), ref.data(), n);
                    CHECK(bad == 0, "%s->%s v%d n=%zu off=%d: %d differ", omc_dtype_name(st), omc_dtype_name(dt), v, n, off, bad);
                    checked += n;
                }
            }
        }
    printf("  all %d pairs x 2 option sets, aligned and misaligned: %ld elements, bit-identical to the host\n",
           OMC_NUM_DTYPES * OMC_NUM_DTYPES, checked);
}

static void test_2d() {
    std::mt19937_64 g(2);
    const omc_dtype_t pairs[][2] = {{OMC_FP64, OMC_FP32}, {OMC_FP64, OMC_FP8_E4M3}, {OMC_FP32, OMC_FP8_E4M3},
                                    {OMC_FP16, OMC_FP8_E4M3}, {OMC_FP32, OMC_FP16}, {OMC_FP16, OMC_FP64},
                                    {OMC_FP32, OMC_INT8}, {OMC_FP64, OMC_FP64}, {OMC_BF16, OMC_FP32}};
    int cases = 0;
    for (auto &pr : pairs)
        for (int tr = 0; tr < 2; tr++) for (int pad = 0; pad < 2; pad++) for (int sr = 0; sr < 2; sr++)
            for (int m : {1, 33, 256, 300}) for (int n : {1, 31, 64, 129}) {
                const omc_dtype_t st = pr[0], dt = pr[1];
                omc_options_t o = {};
                o.flags = OMC_FLAG_SATURATE | (tr ? OMC_FLAG_TRANSPOSE : 0);
                if (sr) {
                    if (!(omc_dtype_is_float(st) && omc_dtype_is_float(dt) && omc_dtype_size(dt) < omc_dtype_size(st))) continue;
                    o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 99;
                }
                const int lds = m + (int)(g() % 9);
                const int rows = tr ? n : m, cols = tr ? m : n;
                const int dr = rows + (pad ? 16 - rows % 16 : 0), dc = cols + (pad ? 5 : 0);
                const int ldd = dr + (int)(g() % 3) * 8;
                if (pad) { o.dst_rows = dr; o.dst_cols = dc; }
                const size_t ss = omc_dtype_size(st), ds = omc_dtype_size(dt);
                auto h = make_src(st, (size_t)lds * n, g);
                std::vector<unsigned char> ref((size_t)ldd * dc * ds, 0x5a), got((size_t)ldd * dc * ds, 0x5a);
                omc_status_t r = omc_convert_2d(st, h.data(), lds, dt, ref.data(), ldd, m, n, &o);
                CHECK(r == OMC_SUCCESS, "host 2d %s", omc_status_string(r));
                DBuf s_((size_t)lds * n * ss), d_((size_t)ldd * dc * ds);
                gMemcpy(s_.p, h.data(), (size_t)lds * n * ss, gH2D);
                gMemcpy(d_.p, got.data(), got.size(), gH2D);
                r = omc_gpu_convert_2d(st, s_.p, lds, dt, d_.p, ldd, m, n, &o, nullptr);
                CHECK(r == OMC_SUCCESS, "gpu 2d %s", omc_status_string(r));
                gMemcpy(got.data(), d_.p, got.size(), gD2H);
                int bad = count_diff(dt, got.data(), ref.data(), (size_t)ldd * dc);
                CHECK(bad == 0, "2d %s->%s tr=%d pad=%d sr=%d m=%d n=%d: %d differ", omc_dtype_name(st), omc_dtype_name(dt),
                      tr, pad, sr, m, n, bad);
                cases++;
            }
    printf("  2-D / transposed / padded / stochastic: %d cases, bit-identical to the host\n", cases);
}

static void test_inplace() {
    std::mt19937_64 g(3);
    int cases = 0;
    for (int si = 0; si < OMC_NUM_DTYPES; si++) for (int di = 0; di < OMC_NUM_DTYPES; di++) {
        const omc_dtype_t st = (omc_dtype_t)si, dt = (omc_dtype_t)di;
        const size_t ss = omc_dtype_size(st), ds = omc_dtype_size(dt), mx = ss > ds ? ss : ds;
        for (size_t n : {1, 100, 511, 512, 513, 4097, 1000003}) {
            if (n > 5000 && !((st == OMC_FP64 || st == OMC_FP32 || st == OMC_FP16) && (dt == OMC_FP64 || dt == OMC_FP32 || dt == OMC_FP16 || dt == OMC_FP8_E4M3))) continue;
            auto h = make_src(st, n, g);
            std::vector<unsigned char> ref(n * ds), got(n * ds);
            omc_convert(st, h.data(), dt, ref.data(), n, nullptr);
            DBuf b(n * mx + 16);
            gMemcpy(b.p, h.data(), n * ss, gH2D);
            omc_status_t r = omc_gpu_convert_inplace(st, dt, b.p, n, nullptr, nullptr);
            CHECK(r == OMC_SUCCESS, "inplace %s->%s %s", omc_dtype_name(st), omc_dtype_name(dt), omc_status_string(r));
            gMemcpy(got.data(), b.p, n * ds, gD2H);
            int bad = count_diff(dt, got.data(), ref.data(), n);
            CHECK(bad == 0, "inplace %s->%s n=%zu: %d differ", omc_dtype_name(st), omc_dtype_name(dt), n, bad);
            cases++;
        }
    }
    /* repeated calls reuse the workspace (it must come back zeroed) and
     * concurrent streams each get their own */
    {
        const int NS = 4; const size_t n = 3000000;
        gStream_t st[NS]; DBuf *bufs[NS];
        std::vector<std::vector<unsigned char>> h(NS);
        std::vector<double> src(n);
        for (int k = 0; k < NS; k++) {
            gStreamCreate(&st[k]);
            bufs[k] = new DBuf(n * 8);
            for (size_t e = 0; e < n; e++) src[e] = (double)((e * 7919 + k) % 100003) / 77.0;
            gMemcpy(bufs[k]->p, src.data(), n * 8, gH2D);
        }
        for (int rep = 0; rep < 10; rep++)
            for (int k = 0; k < NS; k++) {
                omc_gpu_convert_inplace(OMC_FP64, OMC_FP32, bufs[k]->p, n, nullptr, st[k]);
                omc_gpu_convert_inplace(OMC_FP32, OMC_FP64, bufs[k]->p, n, nullptr, st[k]);
            }
        gSync();
        int bad = 0;
        std::vector<double> back(n);
        for (int k = 0; k < NS; k++) {
            gMemcpy(back.data(), bufs[k]->p, n * 8, gD2H);
            for (size_t e = 0; e < n; e++) {
                const double x = (double)((e * 7919 + k) % 100003) / 77.0;
                if (back[e] != (double)(float)x) bad++;
            }
            delete bufs[k];
            gStreamDestroy(st[k]);
        }
        CHECK(bad == 0, "concurrent in place round trips: %d differ", bad);
        cases++;
    }
    /* in place with stochastic rounding and 2-D coordinates */
    {
        const int m = 777, n = 33;
        std::vector<double> A((size_t)m * n);
        for (auto &x : A) x = (double)(g() % 1000000) / 7777.0;
        std::vector<uint16_t> ref((size_t)m * n), got((size_t)m * n);
        omc_options_t o = {}; o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 5;
        omc_convert_2d(OMC_FP64, A.data(), m, OMC_FP16, ref.data(), m, m, n, &o);
        DBuf b((size_t)m * n * 8);
        gMemcpy(b.p, A.data(), (size_t)m * n * 8, gH2D);
        omc_status_t r = omc_gpu_convert_inplace_2d(OMC_FP64, OMC_FP16, b.p, m, m, n, &o, nullptr);
        CHECK(r == OMC_SUCCESS, "inplace SR %s", omc_status_string(r));
        gMemcpy(got.data(), b.p, (size_t)m * n * 2, gD2H);
        CHECK(!memcmp(got.data(), ref.data(), (size_t)m * n * 2), "inplace SR fp64->fp16 differs from host");
        cases++;
    }
    printf("  in place (every pair, up to 1M elements, 4 concurrent streams x 20 calls): %d cases\n", cases);
}

static void test_scale() {
    std::mt19937_64 g(4);
    const int m = 1000, n = 300;
    std::vector<double> A((size_t)m * n);
    for (auto &x : A) x = ((double)(g() >> 11) / 9007199254740992.0 - 0.5) * 3e-4;
    A[12345] = 7.5e-4;
    DBuf a((size_t)m * n * 8), q((size_t)m * n), sc(16), am(4);
    gMemcpy(a.p, A.data(), (size_t)m * n * 8, gH2D);
    omc_options_t o = {};
    o.flags = OMC_FLAG_SATURATE | OMC_FLAG_TRANSPOSE; o.scale_mode = OMC_SCALE_DYNAMIC_POW2; o.scale_ptr = (float *)sc.p;
    o.dst_rows = n + 4; o.dst_cols = m + 8;
    omc_status_t r = omc_gpu_convert_2d(OMC_FP64, a.p, m, OMC_FP8_E4M3, q.p, n + 4, m, n, &o, nullptr);
    CHECK(r == OMC_SUCCESS, "dynamic scale %s", omc_status_string(r));
    float s3[3];
    gMemcpy(s3, sc.p, 12, gD2H);
    float hs[3] = {0, 0, 0};
    omc_options_t ho = o; ho.scale_ptr = hs;
    std::vector<uint8_t> ref((size_t)(n + 4) * (m + 8)), got(ref.size());
    DBuf q2(ref.size());
    r = omc_gpu_convert_2d(OMC_FP64, a.p, m, OMC_FP8_E4M3, q2.p, n + 4, m, n, &o, nullptr);
    omc_convert_2d(OMC_FP64, A.data(), m, OMC_FP8_E4M3, ref.data(), n + 4, m, n, &ho);
    gMemcpy(got.data(), q2.p, got.size(), gD2H);
    CHECK(s3[0] == hs[0] && s3[1] == hs[1] && s3[2] == hs[2], "scale triple gpu {%g %g %g} host {%g %g %g}", s3[0], s3[1], s3[2], hs[0], hs[1], hs[2]);
    CHECK(s3[1] == 524288.0f, "scale %g", s3[1]);
    CHECK(count_diff(OMC_FP8_E4M3, got.data(), ref.data(), got.size()) == 0, "dynamic scaled fp8 differs from host");
    r = omc_gpu_amax_2d(OMC_FP64, a.p, m, m, n, (float *)am.p, nullptr);
    float amax; gMemcpy(&amax, am.p, 4, gD2H);
    CHECK(r == OMC_SUCCESS && amax == (float)7.5e-4, "amax %g", amax);
    /* device scale pointer */
    omc_options_t dsc = {}; dsc.scale_mode = OMC_SCALE_DEVICE; dsc.scale_ptr = (float *)sc.p + 1;
    std::vector<uint16_t> h16((size_t)m * n), r16((size_t)m * n);
    DBuf o16((size_t)m * n * 2);
    omc_gpu_convert(OMC_FP64, a.p, OMC_FP16, o16.p, (size_t)m * n, &dsc, nullptr);
    omc_options_t hsc = {}; hsc.scale_mode = OMC_SCALE_HOST; hsc.scale = s3[1];
    omc_convert(OMC_FP64, A.data(), OMC_FP16, r16.data(), (size_t)m * n, &hsc);
    gMemcpy(h16.data(), o16.p, (size_t)m * n * 2, gD2H);
    CHECK(!memcmp(h16.data(), r16.data(), (size_t)m * n * 2), "device-scaled fp16 differs from host");
    printf("  scaling (host, device, dynamic power of two) and amax: ok\n");
}

/* Dynamic power-of-two scale with transposition and padding (the fused
 * cooperative kernel, enabled in main): every float source type,
 * round to nearest and stochastic, sizes from one element to many tiles per
 * block, NaN and infinity, then concurrent streams with a different scale per
 * call (counters left zero, partials not mixed up). Bit-identical to the host
 * and the same scale triple. */
static void test_dynamic_transpose() {
    std::mt19937_64 g(11);
    const omc_dtype_t srcs[] = {OMC_FP64, OMC_FP32, OMC_FP16, OMC_BF16};
    const int sizes[][2] = {{1, 1}, {63, 65}, {64, 64}, {1000, 300}, {2048, 2048}, {4100, 300}, {300, 4100}};
    int cases = 0;
    for (int special = 0; special < 2; special++)
    for (const auto &sz : sizes)
    for (omc_dtype_t st : srcs)
    for (int sr = 0; sr < 2; sr++) {
        const int m = sz[0], n = sz[1];
        omc_options_t o = {};
        o.flags = OMC_FLAG_SATURATE | OMC_FLAG_TRANSPOSE; o.scale_mode = OMC_SCALE_DYNAMIC_POW2;
        if (sr) { o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 77 + m; }
        o.dst_rows = n + 3; o.dst_cols = m + 5;
        const size_t es = omc_dtype_size(st);
        std::vector<double> A((size_t)m * n);
        const double mag = std::ldexp(1.0, (int)(g() % 40) - 20);
        for (auto &x : A) x = ((double)(g() >> 11) / 9007199254740992.0 - 0.5) * mag;
        A[g() % A.size()] = 3.0 * mag;
        if (special && A.size() > 4) {
            A[1] = NAN; A[A.size() - 2] = -INFINITY;
        }
        std::vector<unsigned char> S((size_t)m * n * es);
        omc_convert(OMC_FP64, A.data(), st, S.data(), A.size(), nullptr);
        const size_t qn = (size_t)(n + 3) * (m + 5);
        DBuf a(S.size()), q(qn), sc(16);
        gMemcpy(a.p, S.data(), S.size(), gH2D);
        o.scale_ptr = (float *)sc.p;
        omc_status_t r = omc_gpu_convert_2d(st, a.p, m, OMC_FP8_E4M3, q.p, n + 3, m, n, &o, nullptr);
        if (r == OMC_ERROR_NOT_SUPPORTED && sr) continue;    /* no stochastic rounding for this pair */
        CHECK(r == OMC_SUCCESS, "dynamic transpose %d x %d from %s: %s", m, n, omc_dtype_name(st), omc_status_string(r));
        float s3[3], hs[3] = {0, 0, 0};
        gMemcpy(s3, sc.p, 12, gD2H);
        std::vector<uint8_t> ref(qn), got(qn);
        omc_options_t ho = o; ho.scale_ptr = hs;
        omc_convert_2d(st, S.data(), m, OMC_FP8_E4M3, ref.data(), n + 3, m, n, &ho);
        gMemcpy(got.data(), q.p, qn, gD2H);
        CHECK(!memcmp(s3, hs, 12), "scale %d x %d %s%s: gpu {%g %g %g} host {%g %g %g}", m, n, omc_dtype_name(st),
              sr ? " SR" : "", s3[0], s3[1], s3[2], hs[0], hs[1], hs[2]);
        CHECK(count_diff(OMC_FP8_E4M3, got.data(), ref.data(), qn) == 0, "dynamic transpose %d x %d from %s%s%s differs",
              m, n, omc_dtype_name(st), sr ? " SR" : "", special ? " (NaN, inf)" : "");
        cases++;
    }
    /* concurrent streams, a different scale for every call */
    {
        const int NS = 4, CALLS = 8, m = 2048, n = 2048;
        gStream_t st[NS];
        for (int k = 0; k < NS; k++) gStreamCreate(&st[k]);
        std::vector<std::vector<double>> A(NS * CALLS);
        std::vector<DBuf *> in, out, scl;
        for (int c = 0; c < NS * CALLS; c++) {
            A[c].resize((size_t)m * n);
            const double mag = std::ldexp(1.0, c - 16);
            for (auto &x : A[c]) x = ((double)(g() >> 11) / 9007199254740992.0 - 0.5) * mag;
            in.push_back(new DBuf((size_t)m * n * 8)); out.push_back(new DBuf((size_t)m * n)); scl.push_back(new DBuf(16));
            gMemcpy(in[c]->p, A[c].data(), (size_t)m * n * 8, gH2D);
        }
        for (int call = 0; call < CALLS; call++)
            for (int k = 0; k < NS; k++) {
                const int c = call * NS + k;
                omc_options_t o = {};
                o.flags = OMC_FLAG_SATURATE | OMC_FLAG_TRANSPOSE; o.scale_mode = OMC_SCALE_DYNAMIC_POW2;
                o.scale_ptr = (float *)scl[c]->p;
                omc_status_t r = omc_gpu_convert_2d(OMC_FP64, in[c]->p, m, OMC_FP8_E4M3, out[c]->p, n, m, n, &o, st[k]);
                CHECK(r == OMC_SUCCESS, "concurrent dynamic transpose: %s", omc_status_string(r));
            }
        gSync();
        for (int c = 0; c < NS * CALLS; c++) {
            omc_options_t o = {};
            float hs[3] = {0, 0, 0}, s3[3];
            o.flags = OMC_FLAG_SATURATE | OMC_FLAG_TRANSPOSE; o.scale_mode = OMC_SCALE_DYNAMIC_POW2; o.scale_ptr = hs;
            std::vector<uint8_t> ref((size_t)m * n), got((size_t)m * n);
            omc_convert_2d(OMC_FP64, A[c].data(), m, OMC_FP8_E4M3, ref.data(), n, m, n, &o);
            gMemcpy(got.data(), out[c]->p, got.size(), gD2H);
            gMemcpy(s3, scl[c]->p, 12, gD2H);
            CHECK(!memcmp(s3, hs, 12), "concurrent call %d: scale gpu %g host %g", c, s3[1], hs[1]);
            CHECK(count_diff(OMC_FP8_E4M3, got.data(), ref.data(), got.size()) == 0, "concurrent call %d differs", c);
            delete in[c]; delete out[c]; delete scl[c];
        }
        for (int k = 0; k < NS; k++) gStreamDestroy(st[k]);
        cases += NS * CALLS;
    }
    printf("  dynamic scale + transpose (float sources, RN/SR, NaN/inf, %d concurrent streams): %d cases\n", 4, cases);
}

int main() {
    /* transposed conversions with a dynamic scale through the opt-in fused
     * kernel (omnicast_test_modes_gpu covers the default amax pass) */
    setenv("OMNICAST_FUSED_DYNAMIC", "1", 1);
    gProp p;
    gGetDeviceProperties(&p, 0);
    printf("OmniCast %s, GPU backend %s, device %s\n", omc_version_string(), omc_gpu_backend(), p.name);
    test_pairs();
    test_2d();
    test_inplace();
    test_scale();
    test_dynamic_transpose();
    omc_gpu_release_workspaces();
    printf(g_fail ? "FAILED (%d)\n" : "PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
