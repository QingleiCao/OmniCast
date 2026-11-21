/**
 * @file cpu_api.cpp
 * @brief Host API: ISA dispatch, options, 2-D layouts, in-place, threads.
 */
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <atomic>
#include <algorithm>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "cpu_kernels.hpp"
#include "../core/omc_common.hpp"

namespace omc {
namespace cpu {

void fill_table_generic(Table &t);
#if defined(OMC_HAVE_AVX2_TU)
void fill_table_avx2(Table &t);
#endif
#if defined(OMC_HAVE_AVX512_TU)
void fill_table_avx512(Table &t);
#endif

namespace {

Table g_generic, g_avx2, g_avx512;
std::atomic<const Table *> g_active{nullptr};
std::once_flag g_once;

bool cpu_has_avx2() {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}
bool cpu_has_avx512() {
#if (defined(__x86_64__) || defined(_M_X64)) && defined(__GNUC__)
    __builtin_cpu_init();
    return cpu_has_avx2() && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw")
        && __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
#else
    return false;
#endif
}

const Table *pick(const char *name) {
    if (!name) return nullptr;
    if (!strcmp(name, "generic") || !strcmp(name, "neon") || !strcmp(name, "scalar")) return &g_generic;
#if defined(OMC_HAVE_AVX2_TU)
    if (!strcmp(name, "avx2") && g_avx2.name && cpu_has_avx2()) return &g_avx2;
#endif
#if defined(OMC_HAVE_AVX512_TU)
    if (!strcmp(name, "avx512") && g_avx512.name && cpu_has_avx512()) return &g_avx512;
#endif
    return nullptr;
}

void init_tables() {
    memset(&g_generic, 0, sizeof(Table));
    memset(&g_avx2, 0, sizeof(Table));
    memset(&g_avx512, 0, sizeof(Table));
    fill_table_generic(g_generic);
    const Table *best = &g_generic;
#if defined(OMC_HAVE_AVX2_TU)
    fill_table_avx2(g_avx2);
    if (g_avx2.name && cpu_has_avx2()) best = &g_avx2;
#endif
#if defined(OMC_HAVE_AVX512_TU)
    fill_table_avx512(g_avx512);
    if (g_avx512.name && cpu_has_avx512()) best = &g_avx512;
#endif
    const Table *forced = pick(getenv("OMNICAST_CPU_ISA"));
    g_active.store(forced ? forced : best);
}

inline const Table &table() {
    const Table *t = g_active.load(std::memory_order_acquire);
    if (!t) {
        std::call_once(g_once, init_tables);
        t = g_active.load(std::memory_order_acquire);
    }
    return *t;
}

/* ------------------------------------------------------------------------ */

struct Plan {
    KFn fn;
    KParams p;
    bool memcpy_ok;      /* same type, nothing to do but copy */
    size_t si, so;
};

/** Validate options and pick the kernel. src/ld/m/n describe the source for
 *  OMC_SCALE_DYNAMIC_POW2. Host calls read OMC_SCALE_DEVICE's factor from
 *  host memory (scale_ptr[0]). */
omc_status_t make_plan(omc_dtype_t st, omc_dtype_t dt, const omc_options_t *o, Plan &pl,
                       const void *src, int64_t ld, int64_t m, int64_t n) {
    if ((unsigned)st >= OMC_NUM_DTYPES || (unsigned)dt >= OMC_NUM_DTYPES) return OMC_ERROR_INVALID_VALUE;
    static const omc_options_t zero = {};
    if (!o) o = &zero;
    const Table &t = table();
    pl.si = omc_dtype_size(st);
    pl.so = omc_dtype_size(dt);
    pl.p = KParams{};
    pl.p.eo.sat = (o->flags & OMC_FLAG_SATURATE) != 0;
    pl.p.eo.rne = (o->rounding == OMC_ROUND_NEAREST_EVEN);
    pl.p.seed   = o->seed;
    pl.memcpy_ok = false;
    if ((unsigned)o->rounding > (unsigned)OMC_ROUND_STOCHASTIC) return OMC_ERROR_INVALID_VALUE;
    if (o->rounding == OMC_ROUND_TOWARD_ZERO && omc_dtype_is_float(dt)) return OMC_ERROR_NOT_SUPPORTED;
    const bool sr = (o->rounding == OMC_ROUND_STOCHASTIC);
    if (sr && !t.sr[st][dt]) return OMC_ERROR_NOT_SUPPORTED;
    bool scaled = true;
    switch (o->scale_mode) {
    case OMC_SCALE_NONE:
        scaled = false;
        break;
    case OMC_SCALE_HOST:
        pl.p.scale = o->scale;
        break;
    case OMC_SCALE_DEVICE:
        if (!o->scale_ptr) return OMC_ERROR_INVALID_VALUE;
        pl.p.scale = o->scale_ptr[0];
        break;
    case OMC_SCALE_DYNAMIC_POW2: {
        if (!omc_dtype_is_float(st) || !o->scale_ptr) return OMC_ERROR_INVALID_VALUE;
        double amax = 0.0;
        if (src) omc_amax_2d(st, src, ld, m, n, &amax);
        const float target = (o->scale_max > 0.0f) ? o->scale_max : max_finite_of(dt);
        const float af = (float)amax;
        const float s = pow2_scale(af, target);
        o->scale_ptr[0] = af;
        o->scale_ptr[1] = s;
        o->scale_ptr[2] = 1.0f / s;
        pl.p.scale = s;
        break;
    }
    default:
        return OMC_ERROR_INVALID_VALUE;
    }
    if (sr) {
        pl.fn = scaled ? t.sr_scaled[st][dt] : t.sr[st][dt];
    } else if (scaled) {
        pl.fn = t.scaled[st][dt];
    } else {
        pl.fn = t.cvt[st][dt];
        pl.memcpy_ok = (st == dt) && !pl.p.eo.sat;
    }
    return pl.fn ? OMC_SUCCESS : OMC_ERROR_NOT_SUPPORTED;
}

inline int nthreads(const omc_options_t *o, size_t work) {
#if defined(_OPENMP)
    if (!o || o->num_threads <= 1) return 1;
    /* at least 32K elements per thread */
    size_t cap = work / 32768 + 1;
    return (int)std::min<size_t>((size_t)o->num_threads, cap);
#else
    (void)o; (void)work;
    return 1;
#endif
}

/* contiguous run of n elements whose first element is at position (i0, j) */
inline void run1d(const Plan &pl, const char *s, char *d, size_t n, uint32_t i0, uint32_t j) {
    if (pl.memcpy_ok) { memcpy(d, s, n * pl.si); return; }
    KParams p = pl.p;
    p.i0 = i0; p.j = j;
    pl.fn(s, d, n, p);
}

void run1d_threads(const Plan &pl, const char *s, char *d, size_t n, int nt) {
    if (nt <= 1) { run1d(pl, s, d, n, 0, 0); return; }
#if defined(_OPENMP)
    const size_t chunk = ((n + nt - 1) / nt + 63) & ~(size_t)63;
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (int t = 0; t < nt; t++) {
        const size_t b = (size_t)t * chunk;
        if (b < n) {
            const size_t e = std::min(n, b + chunk);
            run1d(pl, s + b * pl.si, d + b * pl.so, e - b, (uint32_t)b, 0);
        }
    }
#endif
}

/* Zero dst outside the rows x cols block it received, up to dr x dc */
void pad_zero(char *d, int64_t ldd, size_t so, int64_t rows, int64_t cols, int64_t dr, int64_t dc) {
    if (dr > rows)
        for (int64_t j = 0; j < std::min(cols, dc); j++)
            memset(d + ((size_t)j * ldd + rows) * so, 0, (size_t)(dr - rows) * so);
    for (int64_t j = cols; j < dc; j++)
        memset(d + (size_t)j * ldd * so, 0, (size_t)dr * so);
}

/* Scatter a TB x TB tile (column-major, leading dimension TB) transposed */
template <class T>
void scatter_t(const char *tile, int tb, int rows, int cols, char *d, int64_t ldd) {
    const T *t = (const T *)tile;
    T *o = (T *)d;
    for (int r = 0; r < rows; r++)         /* source row r -> dst column r */
        for (int c = 0; c < cols; c++)     /* source column c -> dst row c */
            o[(size_t)r * ldd + c] = t[(size_t)c * tb + r];
}

constexpr int TB = 64;

void transpose_block(const Plan &pl, const char *s, int64_t lds, char *d, int64_t ldd,
                     int64_t i0, int64_t j0, int rows, int cols) {
    alignas(64) char tile[TB * TB * 8];
    for (int c = 0; c < cols; c++)
        run1d(pl, s + ((size_t)(j0 + c) * lds + i0) * pl.si, tile + (size_t)c * TB * pl.so,
              (size_t)rows, (uint32_t)i0, (uint32_t)(j0 + c));
    char *dd = d + ((size_t)i0 * ldd + j0) * pl.so;
    switch (pl.so) {
    case 1: scatter_t<uint8_t>(tile, TB, rows, cols, dd, ldd); break;
    case 2: scatter_t<uint16_t>(tile, TB, rows, cols, dd, ldd); break;
    case 4: scatter_t<uint32_t>(tile, TB, rows, cols, dd, ldd); break;
    default: scatter_t<uint64_t>(tile, TB, rows, cols, dd, ldd); break;
    }
}

/* In place, one contiguous run.
 *
 * When the elements shrink, chunk [a, b) can be converted directly (no copy)
 * as soon as its destination bytes [a so, b so) end before its source bytes
 * [a si, b si) begin, i.e. b <= a si / so: then the destination only covers
 * input already converted, and source and destination do not overlap. So the
 * first H elements go through an L1-sized buffer, and the rest in chunks that
 * grow geometrically ([H, H si/so), ...). When the elements grow, the same
 * from the back: chunks [a, b) with a >= b si / so, then the first H through
 * the buffer, back to front. Same-size conversions use the buffer. */
void inplace_run(const Plan &pl, char *buf_src, char *buf_dst, size_t n, uint32_t i0, uint32_t j) {
    constexpr size_t B = 1024;
    alignas(64) char tmp[B * 8];
    const size_t si = pl.si, so = pl.so;
    auto via_buffer_forward = [&](size_t lo, size_t hi) {
        for (size_t k = lo; k < hi; k += B) {
            const size_t m = std::min(B, hi - k);
            run1d(pl, buf_src + k * si, tmp, m, i0 + (uint32_t)k, j);
            memcpy(buf_dst + k * so, tmp, m * so);
        }
    };
    if (so < si) {
        const size_t H = std::min(n, B);
        via_buffer_forward(0, H);
        size_t a = H;
        while (a < n) {
            const size_t b = std::min(n, a * si / so);
            run1d(pl, buf_src + a * si, buf_dst + a * so, b - a, i0 + (uint32_t)a, j);
            a = b;
        }
    } else if (so > si) {
        const size_t H = std::min(n, B);
        size_t b = n;
        while (b > H) {
            size_t a = (b * si + so - 1) / so;         /* ceil(b si / so) */
            if (a < H) a = H;
            run1d(pl, buf_src + a * si, buf_dst + a * so, b - a, i0 + (uint32_t)a, j);
            b = a;
        }
        size_t k = H;
        while (k > 0) {
            const size_t m = std::min(B, k);
            k -= m;
            run1d(pl, buf_src + k * si, tmp, m, i0 + (uint32_t)k, j);
            memcpy(buf_dst + k * so, tmp, m * so);
        }
    } else {
        via_buffer_forward(0, n);
    }
}

/** Destination geometry of a 2-D call */
struct Geom {
    bool tr;
    int64_t rows, cols, dr, dc;
};
inline omc_status_t geometry(const omc_options_t *o, int64_t m, int64_t n, Geom &g) {
    if (m < 0 || n < 0) return OMC_ERROR_INVALID_VALUE;
    g.tr = o && (o->flags & OMC_FLAG_TRANSPOSE);
    g.rows = g.tr ? n : m;
    g.cols = g.tr ? m : n;
    g.dr = (o && o->dst_rows > 0) ? o->dst_rows : g.rows;
    g.dc = (o && o->dst_cols > 0) ? o->dst_cols : g.cols;
    if (g.dr < g.rows || g.dc < g.cols) return OMC_ERROR_INVALID_VALUE;
    return OMC_SUCCESS;
}

/** In place, m-by-n: the source has leading dimension lds (src_type
 *  elements), the result ldd (dst_type elements). Without transposition or
 *  padding and with lds == ldd: a single pass (columns front to back when
 *  elements shrink, back to front when they grow). Otherwise the result goes
 *  through a temporary of dr x dc elements and is copied back. */
omc_status_t inplace_2d(omc_dtype_t st, omc_dtype_t dt, void *buf, int64_t lds, int64_t ldd,
                        int64_t m, int64_t n, const omc_options_t *opts) {
    Geom g;
    omc_status_t r = geometry(opts, m, n, g);
    if (r != OMC_SUCCESS) return r;
    if (g.dr == 0 || g.dc == 0) return OMC_SUCCESS;
    if (!buf || lds < std::max<int64_t>(1, m) || ldd < std::max<int64_t>(1, g.dr)) return OMC_ERROR_INVALID_VALUE;
    const bool padded = (g.dr != g.rows) || (g.dc != g.cols);
    if (g.tr || padded || lds != ldd) {
        const size_t so = omc_dtype_size(dt);
        if (so == 0) return OMC_ERROR_INVALID_VALUE;
        char *tmp = (char *)malloc((size_t)g.dr * (size_t)g.dc * so);
        if (!tmp) return OMC_ERROR_ALLOCATION;
        r = omc_convert_2d(st, buf, lds, dt, tmp, g.dr, m, n, opts);
        if (r == OMC_SUCCESS)
            for (int64_t j = 0; j < g.dc; j++)
                memcpy((char *)buf + (size_t)j * ldd * so, tmp + (size_t)j * g.dr * so, (size_t)g.dr * so);
        free(tmp);
        return r;
    }
    if (m == 0 || n == 0) return OMC_SUCCESS;
    const int64_t ld = lds;
    Plan pl;
    r = make_plan(st, dt, opts, pl, buf, ld, m, n);
    if (r != OMC_SUCCESS) return r;
    if (pl.memcpy_ok) return OMC_SUCCESS;
    char *b = (char *)buf;
    const bool sr = opts && opts->rounding == OMC_ROUND_STOCHASTIC;
    if (ld == m && !sr) {
        inplace_run(pl, b, b, (size_t)m * n, 0, 0);
    } else if (pl.so <= pl.si) {
        for (int64_t j = 0; j < n; j++)
            inplace_run(pl, b + (size_t)j * ld * pl.si, b + (size_t)j * ld * pl.so, (size_t)m, 0, (uint32_t)j);
    } else {
        for (int64_t j = n - 1; j >= 0; j--)
            inplace_run(pl, b + (size_t)j * ld * pl.si, b + (size_t)j * ld * pl.so, (size_t)m, 0, (uint32_t)j);
    }
    return OMC_SUCCESS;
}
} /* namespace */

const char *active_isa() { return table().name; }

int set_isa(const char *name) {
    table();
    const Table *t = pick(name);
    if (!t) return -1;
    g_active.store(t);
    return 0;
}

KFn generic_cvt(omc_dtype_t s, omc_dtype_t d) { table(); return g_generic.cvt[s][d]; }

} /* namespace cpu */
} /* namespace omc */

using namespace omc;
using namespace omc::cpu;

extern "C" {

const char *omc_cpu_isa(void) { return omc::cpu::active_isa(); }
int omc_cpu_set_isa(const char *name) { return omc::cpu::set_isa(name); }

omc_status_t omc_convert(omc_dtype_t st, const void *src, omc_dtype_t dt, void *dst,
                         size_t count, const omc_options_t *opts) {
    if (count == 0) return OMC_SUCCESS;
    if (!src || !dst) return OMC_ERROR_INVALID_VALUE;
    if (opts && (opts->flags & OMC_FLAG_TRANSPOSE)) return OMC_ERROR_INVALID_VALUE;
    if (src == dst) return omc_convert_inplace(st, dt, dst, count, opts);
    Plan pl;
    omc_status_t r = make_plan(st, dt, opts, pl, src, (int64_t)count, (int64_t)count, 1);
    if (r != OMC_SUCCESS) return r;
    run1d_threads(pl, (const char *)src, (char *)dst, count, nthreads(opts, count));
    return OMC_SUCCESS;
}

omc_status_t omc_convert_2d(omc_dtype_t st, const void *src, int64_t lds,
                            omc_dtype_t dt, void *dst, int64_t ldd,
                            int64_t m, int64_t n, const omc_options_t *opts) {
    Geom g;
    omc_status_t r = geometry(opts, m, n, g);
    if (r != OMC_SUCCESS) return r;
    if (g.dr == 0 || g.dc == 0) return OMC_SUCCESS;
    if (!dst || (m > 0 && n > 0 && !src)) return OMC_ERROR_INVALID_VALUE;
    if (lds < std::max<int64_t>(1, m) || ldd < std::max<int64_t>(1, g.dr)) return OMC_ERROR_INVALID_VALUE;
    if (src == dst) return inplace_2d(st, dt, dst, lds, ldd, m, n, opts);
    Plan pl;
    r = make_plan(st, dt, opts, pl, src, lds, m, n);
    if (r != OMC_SUCCESS) return r;
    const char *s = (const char *)src;
    char *d = (char *)dst;
    const bool sr = opts && opts->rounding == OMC_ROUND_STOCHASTIC;
    if (m > 0 && n > 0) {
        if (!g.tr) {
            if (!sr && lds == m && ldd == m) {
                run1d_threads(pl, s, d, (size_t)m * n, nthreads(opts, (size_t)m * n));
            } else {
                const int nt = nthreads(opts, (size_t)m * n);
                #pragma omp parallel for num_threads(nt) schedule(static) if (nt > 1)
                for (int64_t j = 0; j < n; j++)
                    run1d(pl, s + (size_t)j * lds * pl.si, d + (size_t)j * ldd * pl.so, (size_t)m, 0, (uint32_t)j);
            }
        } else {
            const int64_t nbj = (n + TB - 1) / TB;
            const int64_t nbi = (m + TB - 1) / TB;
            const int nt = nthreads(opts, (size_t)m * n);
            #pragma omp parallel for num_threads(nt) schedule(static) collapse(2) if (nt > 1)
            for (int64_t bj = 0; bj < nbj; bj++)
                for (int64_t bi = 0; bi < nbi; bi++) {
                    const int64_t i0 = bi * TB, j0 = bj * TB;
                    transpose_block(pl, s, lds, d, ldd, i0, j0,
                                    (int)std::min<int64_t>(TB, m - i0), (int)std::min<int64_t>(TB, n - j0));
                }
        }
    }
    pad_zero(d, ldd, pl.so, g.rows, g.cols, g.dr, g.dc);
    return OMC_SUCCESS;
}

omc_status_t omc_convert_inplace(omc_dtype_t st, omc_dtype_t dt, void *buf, size_t count,
                                 const omc_options_t *opts) {
    if (count == 0) return OMC_SUCCESS;
    if (!buf) return OMC_ERROR_INVALID_VALUE;
    if (opts && (opts->flags & OMC_FLAG_TRANSPOSE)) return OMC_ERROR_INVALID_VALUE;
    Plan pl;
    omc_status_t r = make_plan(st, dt, opts, pl, buf, (int64_t)count, (int64_t)count, 1);
    if (r != OMC_SUCCESS) return r;
    if (pl.memcpy_ok) return OMC_SUCCESS;
    inplace_run(pl, (char *)buf, (char *)buf, count, 0, 0);
    return OMC_SUCCESS;
}

omc_status_t omc_convert_inplace_2d(omc_dtype_t st, omc_dtype_t dt, void *buf, int64_t ld,
                                    int64_t m, int64_t n, const omc_options_t *opts) {
    return inplace_2d(st, dt, buf, ld, ld, m, n, opts);
}

omc_status_t omc_amax_2d(omc_dtype_t type, const void *src, int64_t ld, int64_t m, int64_t n, double *amax) {
    if ((unsigned)type >= OMC_NUM_DTYPES || !amax || m < 0 || n < 0 || (m > 0 && ld < m))
        return OMC_ERROR_INVALID_VALUE;
    *amax = 0.0;
    const AmaxFn f = table().amax[type];
    if (!f) return OMC_ERROR_NOT_SUPPORTED;
    if (m == 0 || n == 0) return OMC_SUCCESS;
    if (!src) return OMC_ERROR_INVALID_VALUE;
    const size_t sz = omc_dtype_size(type);
    double r = 0.0;
    if (ld == m) {
        r = f(src, (size_t)m * n);
    } else {
        for (int64_t j = 0; j < n; j++) r = std::max(r, f((const char *)src + (size_t)j * ld * sz, (size_t)m));
    }
    *amax = r;
    return OMC_SUCCESS;
}

omc_status_t omc_convert_scalar(omc_dtype_t st, const void *src, omc_dtype_t dt, void *dst,
                                const omc_options_t *opts) {
    return omc_convert(st, src, dt, dst, 1, opts);
}

/* Scalars */
uint16_t omc_fp32_to_fp16(float x)  { return (uint16_t)enc_f32<FmtF16>(x, false); }
uint16_t omc_fp64_to_fp16(double x) { return (uint16_t)enc_f32<FmtF16>(f64_to_f32_ro(x), false); }
float    omc_fp16_to_fp32(uint16_t h) { return dec_f32<FmtF16>(h); }
uint16_t omc_fp32_to_bf16(float x)  { return (uint16_t)enc_f32<FmtBF16>(x, false); }
uint16_t omc_fp64_to_bf16(double x) { return (uint16_t)enc_f32<FmtBF16>(f64_to_f32_ro(x), false); }
float    omc_bf16_to_fp32(uint16_t h) { return dec_f32<FmtBF16>(h); }
uint8_t  omc_fp32_to_e4m3(float x, int sat)  { return (uint8_t)enc_f32<FmtE4M3>(x, sat != 0); }
uint8_t  omc_fp64_to_e4m3(double x, int sat) { return (uint8_t)enc_f32<FmtE4M3>(f64_to_f32_ro(x), sat != 0); }
float    omc_e4m3_to_fp32(uint8_t h) { return dec_f32<FmtE4M3>(h); }
uint8_t  omc_fp32_to_e5m2(float x, int sat)  { return (uint8_t)enc_f32<FmtE5M2>(x, sat != 0); }
uint8_t  omc_fp64_to_e5m2(double x, int sat) { return (uint8_t)enc_f32<FmtE5M2>(f64_to_f32_ro(x), sat != 0); }
float    omc_e5m2_to_fp32(uint8_t h) { return dec_f32<FmtE5M2>(h); }

} /* extern "C" */
