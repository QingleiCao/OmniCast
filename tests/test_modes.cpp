/**
 * @file test_modes.cpp
 * @brief Coverage: every type pair, every option set and every layout, out of
 * place, in place, and in place through dst == src, on every host code path
 * and (when compiled as test_modes_gpu.cu) on the GPU.
 *
 * The reference is the host out-of-place conversion on the generic code path,
 * which test_cpu checks element by element against the scalar core. A call
 * must give the reference status (stochastic rounding only exists for
 * narrowing float pairs, dynamic scaling only for float sources) and, when it
 * succeeds, the reference bits (NaN payloads aside) in the destination region.
 * The program prints the coverage table of chapter 3 of
 * docs/omnicast-2.1-report.pdf.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include "omnicast/omnicast.h"

#if defined(OMC_TEST_GPU)
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gMalloc hipMalloc
#define gFree hipFree
#define gMemcpy hipMemcpy
#define gH2D hipMemcpyHostToDevice
#define gD2H hipMemcpyDeviceToHost
#define gSync hipDeviceSynchronize
#define gGetLastError hipGetLastError
#define gSuccess hipSuccess
#else
#include <cuda_runtime.h>
#define gMalloc cudaMalloc
#define gFree cudaFree
#define gMemcpy cudaMemcpy
#define gH2D cudaMemcpyHostToDevice
#define gD2H cudaMemcpyDeviceToHost
#define gSync cudaDeviceSynchronize
#define gGetLastError cudaGetLastError
#define gSuccess cudaSuccess
#endif
#endif

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; if (g_fail < 30) { printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } } while (0)

typedef std::vector<unsigned char> Bytes;

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

/* random values: special values, values around the integer and FP8/FP16
 * ranges, random bit patterns of the type */
static void fill_src(omc_dtype_t t, unsigned char *p, size_t n, std::mt19937_64 &g) {
    const size_t sz = omc_dtype_size(t);
    for (size_t k = 0; k < n; k++) {
        const uint64_t r = g();
        if (t == OMC_FP64 || (t == OMC_FP32 && (r & 1))) {
            double x;
            switch ((r >> 8) % 5) {
            case 0: { static const double sp[] = {0.0, -0.0, INFINITY, -INFINITY, NAN, 65504.0, 65520.0, 448.0,
                       464.0, 57344.0, 1e300, 5e-324, 3.5e38, 127.5, -128.5, 255.5, 2147483647.5, 9.3e18, 1.9e19};
                      x = sp[g() % (sizeof(sp) / sizeof(sp[0]))]; break; }
            case 1: x = ldexp((double)(g() >> 11) / 9007199254740992.0, (int)(g() % 80) - 40); break;
            case 2: x = (double)(int64_t)(g() % 70000) - 35000.0 + 0.5 * (double)(g() % 3); break;
            default: x = ((double)(g() >> 11) / 9007199254740992.0 - 0.5) * 600.0; break;
            }
            if (r & 0x10000) x = -x;
            if (t == OMC_FP64) memcpy(p + k * sz, &x, 8);
            else { const float f = (float)x; memcpy(p + k * sz, &f, 4); }
        } else {
            memcpy(p + k * sz, &r, sz);
        }
    }
}

/* ------------------------------------------------------------------------ */

struct OptSet {
    const char *name;
    omc_options_t o;
    bool sr;
    bool dynamic;
    bool device_scale;
};

static std::vector<OptSet> option_sets() {
    std::vector<OptSet> v;
    auto add = [&](const char *name, omc_options_t o, bool sr, bool dyn, bool dev) { v.push_back({name, o, sr, dyn, dev}); };
    omc_options_t o;
    o = omc_options_t{};                                                  add("default", o, false, false, false);
    o = omc_options_t{}; o.flags = OMC_FLAG_SATURATE;                     add("saturate", o, false, false, false);
    o = omc_options_t{}; o.rounding = OMC_ROUND_NEAREST_EVEN;             add("nearest", o, false, false, false);
    o = omc_options_t{}; o.scale_mode = OMC_SCALE_HOST; o.scale = 0.37f; add("scale_host", o, false, false, false);
    o = omc_options_t{}; o.scale_mode = OMC_SCALE_DEVICE; o.flags = OMC_FLAG_SATURATE; add("scale_ptr+sat", o, false, false, true);
    o = omc_options_t{}; o.scale_mode = OMC_SCALE_DYNAMIC_POW2; o.flags = OMC_FLAG_SATURATE; add("dynamic+sat", o, false, true, false);
    o = omc_options_t{}; o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 0x5eedu; add("stochastic", o, true, false, false);
    o = omc_options_t{}; o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 77; o.scale_mode = OMC_SCALE_HOST; o.scale = 3.0f;
                                                                          add("stochastic+scale", o, true, false, false);
    o = omc_options_t{}; o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 9; o.scale_mode = OMC_SCALE_DYNAMIC_POW2;
    o.flags = OMC_FLAG_SATURATE;                                          add("stochastic+dynamic", o, true, true, false);
    return v;
}

enum Layout { L_1D = 0, L_CONTIG, L_STRIDED, L_TRANS, L_PAD, L_TRANS_PAD, L_COUNT };
static const char *kLayoutName[L_COUNT] = {"1-D", "2-D", "2-D ld>m", "transposed", "padded", "transposed+padded"};
enum Mode { M_OUT = 0, M_IN, M_ALIAS, M_COUNT };
static const char *kModeName[M_COUNT] = {"out of place", "in place", "dst == src"};

struct Shape {
    int64_t m, n;            /* source */
    int64_t lds;             /* out of place source ld; in place: ld before and after */
    int64_t dr, dc, ldd;     /* destination region and its ld (out of place) */
    bool tr, pad;
};

static Shape shape_for(Layout L, int64_t m, int64_t n, bool inplace) {
    Shape s{};
    s.m = m; s.n = n;
    s.tr = (L == L_TRANS || L == L_TRANS_PAD);
    s.pad = (L == L_PAD || L == L_TRANS_PAD);
    if (L == L_1D) { s.m = m * n; s.n = 1; }
    const int64_t rows = s.tr ? s.n : s.m, cols = s.tr ? s.m : s.n;
    s.dr = rows + (s.pad ? 3 : 0);
    s.dc = cols + (s.pad ? 2 : 0);
    s.lds = (L == L_STRIDED) ? s.m + 9 : s.m;
    s.ldd = (L == L_STRIDED) ? s.dr + 5 : s.dr;
    if (inplace) {
        /* one ld for both: large enough for the source and the result */
        s.lds = std::max(s.lds, s.dr);
        if (L == L_TRANS || L == L_PAD || L == L_TRANS_PAD) s.lds += 2;
        s.ldd = s.lds;
    }
    return s;
}

/* the backend under test */
struct Backend {
    const char *name;
    bool gpu;
};

struct Case {
    omc_dtype_t st, dt;
    const OptSet *os;
    Layout L;
    Mode M;
};

/* host reference into dst (dr x dc, ldd), generic code path */
static omc_status_t reference(const Case &c, const Shape &s, const Bytes &src, Bytes &dst, float *scale_out) {
    omc_options_t o = c.os->o;
    float dev_scale = 2.5f;
    if (c.os->device_scale) o.scale_ptr = &dev_scale;
    if (c.os->dynamic) o.scale_ptr = scale_out;
    if (s.tr) o.flags |= OMC_FLAG_TRANSPOSE;
    if (s.pad) { o.dst_rows = s.dr; o.dst_cols = s.dc; }
    omc_cpu_set_isa("generic");
    if (c.L == L_1D) return omc_convert(c.st, src.data(), c.dt, dst.data(), (size_t)s.m, &o);
    return omc_convert_2d(c.st, src.data(), s.lds, c.dt, dst.data(), s.ldd, s.m, s.n, &o);
}

/* the call under test; buf holds the source (lds) and, out of place, dst is
 * separate. Host or device memory as the backend requires. */
static omc_status_t run_case(const Backend &b, const Case &c, const Shape &s, void *srcp, void *dstp, float *scale_dev) {
    omc_options_t o = c.os->o;
    if (c.os->device_scale || c.os->dynamic) o.scale_ptr = scale_dev;
    if (s.tr) o.flags |= OMC_FLAG_TRANSPOSE;
    if (s.pad) { o.dst_rows = s.dr; o.dst_cols = s.dc; }
    const bool one_d = (c.L == L_1D);
    if (!b.gpu) {
        switch (c.M) {
        case M_OUT:   return one_d ? omc_convert(c.st, srcp, c.dt, dstp, (size_t)s.m, &o)
                                   : omc_convert_2d(c.st, srcp, s.lds, c.dt, dstp, s.ldd, s.m, s.n, &o);
        case M_IN:    return one_d ? omc_convert_inplace(c.st, c.dt, srcp, (size_t)s.m, &o)
                                   : omc_convert_inplace_2d(c.st, c.dt, srcp, s.lds, s.m, s.n, &o);
        default:      return one_d ? omc_convert(c.st, srcp, c.dt, srcp, (size_t)s.m, &o)
                                   : omc_convert_2d(c.st, srcp, s.lds, c.dt, srcp, s.lds, s.m, s.n, &o);
        }
    }
#if defined(OMC_TEST_GPU)
    switch (c.M) {
    case M_OUT:   return one_d ? omc_gpu_convert(c.st, srcp, c.dt, dstp, (size_t)s.m, &o, nullptr)
                               : omc_gpu_convert_2d(c.st, srcp, s.lds, c.dt, dstp, s.ldd, s.m, s.n, &o, nullptr);
    case M_IN:    return one_d ? omc_gpu_convert_inplace(c.st, c.dt, srcp, (size_t)s.m, &o, nullptr)
                               : omc_gpu_convert_inplace_2d(c.st, c.dt, srcp, s.lds, s.m, s.n, &o, nullptr);
    default:      return one_d ? omc_gpu_convert(c.st, srcp, c.dt, srcp, (size_t)s.m, &o, nullptr)
                               : omc_gpu_convert_2d(c.st, srcp, s.lds, c.dt, srcp, s.lds, s.m, s.n, &o, nullptr);
    }
#else
    return OMC_ERROR_NOT_SUPPORTED;
#endif
}

struct Tally { long calls = 0, converted = 0, unsupported = 0, mismatched = 0; };

/* off: byte offset of the buffers from a 256-byte boundary (8: aligned for
 * every element type, not for 16-byte vectors) */
static void one_case(const Backend &b, const char *isa, const Case &c, int64_t m, int64_t n, std::mt19937_64 &g,
                     Tally &t, size_t off = 0) {
    const bool inplace = (c.M != M_OUT);
    const Shape s = shape_for(c.L, m, n, inplace);
    const size_t ss = omc_dtype_size(c.st), sd = omc_dtype_size(c.dt);
    const int64_t src_elems = (s.n - 1) * s.lds + s.m;
    const int64_t dst_elems = (s.dc - 1) * s.ldd + s.dr;
    const size_t bytes = std::max((size_t)src_elems * ss, (size_t)dst_elems * sd) + 64;

    /* source with random gap bytes */
    Bytes src(bytes);
    for (auto &x : src) x = (unsigned char)g();
    for (int64_t j = 0; j < s.n; j++) fill_src(c.st, src.data() + (size_t)j * s.lds * ss, (size_t)s.m, g);

    /* reference, written with the ld of the call under test */
    Bytes ref(bytes, 0xA5);
    float ref_scale[3] = {0, 0, 0};
    const omc_status_t rs = reference(c, s, src, ref, ref_scale);

    float scale_host[3] = {2.5f, 0, 0};
    Bytes out(bytes, 0xA5);
    omc_status_t st;
    if (!b.gpu) {
        Bytes work(bytes + 256), dsto(bytes + 256);
        unsigned char *w0 = work.data() + ((256 - ((uintptr_t)work.data() & 255)) & 255) + off;
        unsigned char *d0 = dsto.data() + ((256 - ((uintptr_t)dsto.data() & 255)) & 255) + off;
        memcpy(w0, src.data(), bytes);
        memcpy(d0, out.data(), bytes);
        if (isa) omc_cpu_set_isa(isa);              /* reference() selected the generic path */
        st = run_case(b, c, s, w0, d0, scale_host);
        memcpy(out.data(), inplace ? w0 : d0, bytes);
    } else {
#if defined(OMC_TEST_GPU)
        unsigned char *d_src = nullptr, *d_dst = nullptr;
        float *d_scale = nullptr;
        gMalloc((void **)&d_src, bytes + off); gMalloc((void **)&d_dst, bytes + off);
        gMalloc((void **)&d_scale, 3 * sizeof(float));
        gMemcpy(d_src + off, src.data(), bytes, gH2D);
        gMemcpy(d_dst + off, out.data(), bytes, gH2D);
        gMemcpy(d_scale, scale_host, sizeof scale_host, gH2D);
        st = run_case(b, c, s, d_src + off, d_dst + off, d_scale);
        gSync();
        CHECK(gGetLastError() == gSuccess, "%s: device error", b.name);
        gMemcpy(out.data(), (inplace ? d_src : d_dst) + off, bytes, gD2H);
        gMemcpy(scale_host, d_scale, sizeof scale_host, gD2H);
        gFree(d_src); gFree(d_dst); gFree(d_scale);
#else
        st = OMC_ERROR_NOT_SUPPORTED;
#endif
    }
    t.calls++;
    if (st != rs) {
        t.mismatched++;
        CHECK(false, "%s %s: %s -> %s, %s, %s, offset %zu: status %s, reference %s", b.name, kModeName[c.M],
              omc_dtype_name(c.st), omc_dtype_name(c.dt), c.os->name, kLayoutName[c.L], off, omc_status_string(st),
              omc_status_string(rs));
        return;
    }
    if (rs != OMC_SUCCESS) { t.unsupported++; return; }
    t.converted++;
    /* compare the destination region (dr x dc, ld = ldd) */
    long bad = 0;
    int64_t bi = -1, bj = -1;
    for (int64_t j = 0; j < s.dc; j++)
        for (int64_t i = 0; i < s.dr; i++) {
            const size_t off = ((size_t)j * s.ldd + i) * sd;
            if (memcmp(out.data() + off, ref.data() + off, sd) &&
                !(is_nan_bits(c.dt, out.data() + off) && is_nan_bits(c.dt, ref.data() + off))) {
                if (!bad) { bi = i; bj = j; }
                bad++;
            }
        }
    if (c.os->dynamic && memcmp(scale_host, ref_scale, sizeof ref_scale)) bad++;
    if (bad) {
        t.mismatched++;
        CHECK(false, "%s %s: %s -> %s, %s, %s (m=%lld n=%lld ld=%lld offset %zu): %ld differences, first at (%lld, %lld)",
              b.name, kModeName[c.M], omc_dtype_name(c.st), omc_dtype_name(c.dt), c.os->name, kLayoutName[c.L],
              (long long)s.m, (long long)s.n, (long long)s.lds, off, bad, (long long)bi, (long long)bj);
    }
}

static void run_backend(const Backend &b, const char *isa) {
    std::mt19937_64 g(12345);
    const std::vector<OptSet> sets = option_sets();
    Tally tm[M_COUNT];
    for (int st = 0; st < OMC_NUM_DTYPES; st++)
        for (int dt = 0; dt < OMC_NUM_DTYPES; dt++)
            for (const OptSet &os : sets)
                for (int L = 0; L < L_COUNT; L++)
                    for (int M = 0; M < M_COUNT; M++) {
                        Case c{(omc_dtype_t)st, (omc_dtype_t)dt, &os, (Layout)L, (Mode)M};
                        /* small odd shapes everywhere; larger ones (several GPU chunks) without scaling */
                        one_case(b, isa, c, 37, 23, g, tm[M]);
                        if ((L == L_CONTIG || L == L_STRIDED || L == L_1D) && os.o.scale_mode == OMC_SCALE_NONE) {
                            one_case(b, isa, c, 3001, 29, g, tm[M]);
                            one_case(b, isa, c, 1003, 7, g, tm[M], 8);      /* not 16-byte aligned */
                        }
                    }
    for (int M = 0; M < M_COUNT; M++)
        printf("  %-8s %-13s %6ld calls: %6ld converted, %5ld not supported (as the reference), %ld mismatches\n",
               isa ? isa : b.name, kModeName[M], tm[M].calls, tm[M].converted, tm[M].unsupported, tm[M].mismatched);
}

int main() {
    printf("OmniCast %s: every pair x %zu option sets x %d layouts, out of place / in place / dst == src\n",
           omc_version_string(), option_sets().size(), (int)L_COUNT);
#if defined(OMC_TEST_GPU)
    Backend gb{omc_gpu_backend(), true};
    run_backend(gb, nullptr);
    omc_gpu_release_workspaces();
#else
    Backend hb{"host", false};
    const char *isas[] = {"generic", "avx2", "avx512"};
    for (const char *isa : isas) {
        if (omc_cpu_set_isa(isa) != 0) continue;
        run_backend(hb, isa);
    }
#endif
    printf(g_fail ? "FAILED (%d)\n" : "PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
