/**
 * @file gpu_api.cu
 * @brief Public GPU API: argument checks, options, plain copies, and the
 * internal per-(device, stream) workspace used by the in-place and
 * dynamically scaled conversions.
 */
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <utility>
#include "gpu_rt.hpp"
#include "../core/omc_common.hpp"

namespace omc {
namespace gpu {

omc_status_t device_error(omc_status_t s, gpuError_t e, const char *where, ...) {
    char w[256];
    va_list ap;
    va_start(ap, where);
    vsnprintf(w, sizeof w, where, ap);
    va_end(ap);
    set_error_detail("%s: %s (%s)", w, gpuGetErrorName(e), gpuGetErrorString(e));
    return s;
}

namespace {

/** Called by every public entry point before its first runtime call. An error
 *  already pending on the thread (from an earlier runtime call whose error
 *  nobody read) would otherwise come back from check_launch as a failure of
 *  OmniCast's own launch. It is reported as what it is, OMC_ERROR_DEVICE with
 *  nothing launched, and left pending for the code that caused it.
 *  "Not ready" is not an error. HIP (ROCm 6) keeps the status of its last call
 *  as the last error, successes included: after an event or stream query of
 *  unfinished work, as a runtime system polls, it is hipErrorNotReady. CUDA
 *  does not record cudaErrorNotReady. */
omc_status_t check_pending() {
    const gpuError_t e = gpuPeekAtLastError();
    if (e == gpuSuccess || e == gpuErrorNotReady) return OMC_SUCCESS;
    set_error_detail("pending before the call, from an earlier runtime call (left pending): %s (%s)",
                     gpuGetErrorName(e), gpuGetErrorString(e));
    return OMC_ERROR_DEVICE;
}

/** A runtime call OmniCast checks itself: its error is reported (and read, so
 *  that it does not stay pending) */
omc_status_t rt_check(gpuError_t e, const char *call) {
    if (e == gpuSuccess) return OMC_SUCCESS;
    gpuGetLastError();
    return device_error(OMC_ERROR_DEVICE, e, "%s", call);
}

const LaunchFn g_launch[OMC_NUM_DTYPES] = {
    launch_src_fp64, launch_src_fp32, launch_src_fp16, launch_src_bf16, launch_src_e4m3, launch_src_e5m2,
    launch_src_int8, launch_src_int16, launch_src_int32, launch_src_int64,
    launch_src_uint8, launch_src_uint16, launch_src_uint32, launch_src_uint64};

struct Ws { void *ptr; size_t bytes; };
std::mutex g_mutex;
std::map<std::pair<int, void *>, Ws> g_ws;
std::map<int, int> g_sms;

int current_device() { int d = 0; gpuGetDevice(&d); return d; }

int num_sms(int dev) {
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = g_sms.find(dev);
    if (it != g_sms.end()) return it->second;
    int v = 0;
    gpuDeviceGetAttribute(&v, gpuDevAttrMultiProcessorCount, dev);
    g_sms[dev] = v;
    return v;
}

/** Zeroed workspace of at least `need` bytes for (device, stream). Growing it
 *  frees the old one (cudaFree waits for the work using it). */
omc_status_t internal_ws(size_t need, gpuStream_t stream, void **ptr, size_t *bytes) {
    const int dev = current_device();
    std::lock_guard<std::mutex> lk(g_mutex);
    Ws &w = g_ws[std::make_pair(dev, (void *)stream)];
    if (w.bytes < need) {
        if (w.ptr) gpuFree(w.ptr);
        w.ptr = nullptr; w.bytes = 0;
        size_t sz = need < ((size_t)1 << 16) ? ((size_t)1 << 16) : need;
        sz = (sz + ((size_t)1 << 16) - 1) & ~(((size_t)1 << 16) - 1);
        void *p = nullptr;
        gpuError_t e = gpuMalloc(&p, sz);
        if (e != gpuSuccess) {
            gpuGetLastError();
            return device_error(OMC_ERROR_ALLOCATION, e, OMC_RT "Malloc of %zu bytes (workspace)", sz);
        }
        e = gpuMemset(p, 0, sz);
        if (e != gpuSuccess) {
            gpuFree(p);
            gpuGetLastError();
            return device_error(OMC_ERROR_ALLOCATION, e, OMC_RT "Memset of the workspace");
        }
        w.ptr = p; w.bytes = sz;
    }
    *ptr = w.ptr; *bytes = w.bytes;
    return OMC_SUCCESS;
}

omc_status_t get_ws(const omc_options_t *o, size_t count, gpuStream_t s, LaunchArgs &a) {
    const size_t need = workspace_bytes_for(count);
    if (o && o->workspace) {
        if (o->workspace_bytes < need) return OMC_ERROR_INVALID_VALUE;
        a.ws = o->workspace; a.ws_bytes = o->workspace_bytes;
        return OMC_SUCCESS;
    }
    return internal_ws(need, s, &a.ws, &a.ws_bytes);
}

/** Whether k_cvtT_dyn may run on a device: OMNICAST_FUSED_DYNAMIC=1 and
 *  cooperative launch support. Off by default: alone it is 1.04-1.27x faster
 *  than k_amax + k_cvtT (H100), but a cooperative kernel starts only when all
 *  its blocks fit at once, so next to kernels of other streams it waits and
 *  holds the GPU (an FP8-heavy HiCMA-X factorization: 4 % slower). */
bool coop_launch(int dev) {
    static std::map<int, int> known;
    std::lock_guard<std::mutex> lk(g_mutex);
    auto it = known.find(dev);
    if (it != known.end()) return it->second != 0;
    int v = 0;
    const char *e = getenv("OMNICAST_FUSED_DYNAMIC");
    if (e && !strcmp(e, "1")) {
        if (gpuDeviceGetAttribute(&v, gpuDevAttrCooperativeLaunch, dev) != gpuSuccess) { gpuGetLastError(); v = 0; }
    }
    known[dev] = v;
    return v != 0;
}

/** Options -> launch arguments (mode, rounding, scale). Runs the amax pass
 *  first for OMC_SCALE_DYNAMIC_POW2, unless `fuse` (a transposed conversion)
 *  and coop_launch: then k_cvtT_dyn does both. */
omc_status_t prepare(omc_dtype_t st, omc_dtype_t dt, const void *src, int64_t lds, int64_t m, int64_t n,
                     const omc_options_t *o, gpuStream_t stream, LaunchArgs &a, bool fuse = false) {
    if ((unsigned)st >= OMC_NUM_DTYPES || (unsigned)dt >= OMC_NUM_DTYPES) return OMC_ERROR_INVALID_VALUE;
    static const omc_options_t zero = {};
    if (!o) o = &zero;
    a = LaunchArgs{};
    a.st = st; a.dt = dt; a.stream = stream;
    a.device = current_device();
    a.sat = (o->flags & OMC_FLAG_SATURATE) != 0;
    a.rne = (o->rounding == OMC_ROUND_NEAREST_EVEN);
    a.seed = o->seed;
    a.mode = MODE_PLAIN;
    a.scale = 1.0f;
    if ((unsigned)o->rounding > (unsigned)OMC_ROUND_STOCHASTIC) return OMC_ERROR_INVALID_VALUE;
    if (o->rounding == OMC_ROUND_TOWARD_ZERO && omc_dtype_is_float(dt)) return OMC_ERROR_NOT_SUPPORTED;
    const bool sr = (o->rounding == OMC_ROUND_STOCHASTIC);
    if (sr && !sr_supported(st, dt)) return OMC_ERROR_NOT_SUPPORTED;
    bool scaled = true;
    switch (o->scale_mode) {
    case OMC_SCALE_NONE:
        scaled = false;
        break;
    case OMC_SCALE_HOST:
        a.scale = o->scale;
        break;
    case OMC_SCALE_DEVICE:
        if (!o->scale_ptr) return OMC_ERROR_INVALID_VALUE;
        a.scale_ptr = o->scale_ptr; a.scale_index = 0;
        break;
    case OMC_SCALE_DYNAMIC_POW2: {
        if (!o->scale_ptr || !omc_dtype_is_float(st)) return OMC_ERROR_INVALID_VALUE;
        LaunchArgs am = LaunchArgs{};
        am.op = OP_AMAX; am.st = st; am.dt = dt; am.src = src; am.lds = lds; am.m = m; am.n = n;
        am.amax_out = o->scale_ptr; am.amax_scale = 1;
        am.scale_max = (o->scale_max > 0.0f) ? o->scale_max : max_finite_of(dt);
        am.stream = stream;
        omc_status_t r = get_ws(o, 0, stream, am);
        if (r != OMC_SUCCESS) return r;
        if (m > 0 && n > 0 && fuse && coop_launch(a.device)) {
            a.fused_amax = true;
            a.amax_out = am.amax_out; a.scale_max = am.scale_max;
            a.ws = am.ws; a.ws_bytes = am.ws_bytes;
        } else if (m > 0 && n > 0) {
            r = g_launch[st](am);
            if (r != OMC_SUCCESS) return r;
        } else {
            /* empty source: amax 0, scale 1 */
            const float one[3] = {0.0f, 1.0f, 1.0f};
            r = rt_check(gpuMemcpyAsync(o->scale_ptr, one, sizeof one, gpuMemcpyHostToDevice, stream),
                         OMC_RT "MemcpyAsync of the scale triple");
            if (r != OMC_SUCCESS) return r;
        }
        a.scale_ptr = o->scale_ptr; a.scale_index = 1;
        break;
    }
    default:
        return OMC_ERROR_INVALID_VALUE;
    }
    a.mode = sr ? (scaled ? MODE_SR_SCALED : MODE_SR) : (scaled ? MODE_SCALED : MODE_PLAIN);
    return OMC_SUCCESS;
}

/** Stream-ordered temporary (cudaMallocAsync / hipMallocAsync where the
 *  device has memory pools; otherwise a plain allocation released after the
 *  stream has finished with it). */
struct Temp {
    void *p = nullptr;
    bool pooled = false;
    gpuStream_t s = nullptr;
    omc_status_t alloc(size_t bytes, gpuStream_t stream) {
        s = stream;
        int pools = 0;
        gpuDeviceGetAttribute(&pools, gpuDevAttrMemoryPoolsSupported, current_device());
        if (pools && gpuMallocAsync(&p, bytes ? bytes : 1, stream) == gpuSuccess) { pooled = true; return OMC_SUCCESS; }
        gpuGetLastError();
        const gpuError_t e = gpuMalloc(&p, bytes ? bytes : 1);
        if (e != gpuSuccess) {
            gpuGetLastError();
            p = nullptr;
            return device_error(OMC_ERROR_ALLOCATION, e, OMC_RT "Malloc of %zu bytes (temporary)", bytes);
        }
        return OMC_SUCCESS;
    }
    void release() {
        if (!p) return;
        if (pooled) gpuFreeAsync(p, s);
        else { gpuStreamSynchronize(s); gpuFree(p); }
        p = nullptr;
    }
};

/** Destination geometry of a 2-D call */
struct Geom {
    bool tr;
    int64_t rows, cols, dr, dc;
};
omc_status_t geometry(const omc_options_t *o, int64_t m, int64_t n, Geom &g) {
    if (m < 0 || n < 0) return OMC_ERROR_INVALID_VALUE;
    g.tr = o && (o->flags & OMC_FLAG_TRANSPOSE);
    g.rows = g.tr ? n : m;
    g.cols = g.tr ? m : n;
    g.dr = (o && o->dst_rows > 0) ? o->dst_rows : g.rows;
    g.dc = (o && o->dst_cols > 0) ? o->dst_cols : g.cols;
    if (g.dr < g.rows || g.dc < g.cols) return OMC_ERROR_INVALID_VALUE;
    return OMC_SUCCESS;
}

bool plain_copy(const LaunchArgs &a) { return a.st == a.dt && a.mode == MODE_PLAIN && !a.sat; }

/** In place, m-by-n: source leading dimension lds (src_type elements), result
 *  ldd (dst_type elements). Without transposition or padding and with
 *  lds == ldd: one ordered single-pass kernel (k_inplace), or k_cvt when the
 *  element size does not change. Otherwise the result goes through a
 *  stream-ordered temporary of dr x dc elements and is copied back. */
omc_status_t inplace_2d(omc_dtype_t st, omc_dtype_t dt, void *buf, int64_t lds, int64_t ldd, int64_t m, int64_t n,
                        const omc_options_t *opts, gpuStream_t s) {
    Geom g;
    omc_status_t r = geometry(opts, m, n, g);
    if (r != OMC_SUCCESS) return r;
    if (g.dr == 0 || g.dc == 0) return OMC_SUCCESS;
    if (!buf || lds < (m > 1 ? m : 1) || ldd < (g.dr > 1 ? g.dr : 1)) return OMC_ERROR_INVALID_VALUE;
    r = check_pending();
    if (r != OMC_SUCCESS) return r;
    const bool padded = (g.dr != g.rows) || (g.dc != g.cols);
    if (g.tr || padded || lds != ldd) {
        const size_t so = omc_dtype_size(dt);
        if (so == 0) return OMC_ERROR_INVALID_VALUE;
        Temp t;
        r = t.alloc((size_t)g.dr * (size_t)g.dc * so, s);
        if (r != OMC_SUCCESS) return r;
        r = omc_gpu_convert_2d(st, buf, lds, dt, t.p, g.dr, m, n, opts, s);
        if (r == OMC_SUCCESS)
            r = rt_check(gpuMemcpy2DAsync(buf, (size_t)ldd * so, t.p, (size_t)g.dr * so, (size_t)g.dr * so,
                                          (size_t)g.dc, gpuMemcpyDeviceToDevice, s),
                         OMC_RT "Memcpy2DAsync of the result");
        t.release();
        return r;
    }
    if (m == 0 || n == 0) return OMC_SUCCESS;
    const int64_t ld = lds;
    LaunchArgs a;
    r = prepare(st, dt, buf, ld, m, n, opts, s, a);
    if (r != OMC_SUCCESS) return r;
    if (plain_copy(a)) return OMC_SUCCESS;
    a.src = buf; a.dst = buf;
    a.num_sms = num_sms(current_device());
    const bool pos = (a.mode == MODE_SR || a.mode == MODE_SR_SCALED);
    if (omc_dtype_size(st) == omc_dtype_size(dt)) {
        /* same element size: each thread reads then writes its own elements */
        a.op = OP_CVT; a.lds = a.ldd = ld; a.m = m; a.n = n; a.dr = m; a.dc = n;
        if (!pos && ld == m) { a.m = a.lds = a.ldd = a.dr = m * n; a.n = a.dc = 1; }
        return g_launch[st](a);
    }
    a.op = OP_INPLACE; a.lds = a.ldd = ld; a.m = m; a.n = n;
    r = get_ws(opts, (size_t)((n - 1) * ld + m), s, a);
    if (r != OMC_SUCCESS) return r;
    return g_launch[st](a);
}
} /* namespace */
} /* namespace gpu */
} /* namespace omc */

using namespace omc;
using namespace omc::gpu;

extern "C" {

const char *omc_gpu_backend(void) {
#if defined(OMC_BACKEND_HIP)
    return "hip";
#else
    return "cuda";
#endif
}

size_t omc_gpu_workspace_size(size_t count) { return workspace_bytes_for(count); }

void omc_gpu_release_workspaces(void) {
    std::lock_guard<std::mutex> lk(g_mutex);
    int cur = 0;
    gpuGetDevice(&cur);
    for (auto &kv : g_ws) {
        if (kv.second.ptr) { gpuSetDevice(kv.first.first); gpuFree(kv.second.ptr); }
    }
    g_ws.clear();
    gpuSetDevice(cur);
}

omc_status_t omc_gpu_convert(omc_dtype_t st, const void *src, omc_dtype_t dt, void *dst, size_t count,
                             const omc_options_t *opts, omc_stream_t stream) {
    if (count == 0) return OMC_SUCCESS;
    if (!src || !dst) return OMC_ERROR_INVALID_VALUE;
    if (opts && (opts->flags & OMC_FLAG_TRANSPOSE)) return OMC_ERROR_INVALID_VALUE;
    if (src == dst) return omc_gpu_convert_inplace(st, dt, dst, count, opts, stream);
    omc_status_t r = check_pending();
    if (r != OMC_SUCCESS) return r;
    LaunchArgs a;
    gpuStream_t s = (gpuStream_t)stream;
    r = prepare(st, dt, src, (int64_t)count, (int64_t)count, 1, opts, s, a);
    if (r != OMC_SUCCESS) return r;
    if (plain_copy(a))
        return rt_check(gpuMemcpyAsync(dst, src, count * omc_dtype_size(st), gpuMemcpyDeviceToDevice, s),
                        OMC_RT "MemcpyAsync");
    a.op = OP_CVT; a.src = src; a.dst = dst;
    a.m = a.lds = a.ldd = a.dr = (int64_t)count; a.n = a.dc = 1;
    a.num_sms = num_sms(current_device());
    return g_launch[st](a);
}

omc_status_t omc_gpu_convert_2d(omc_dtype_t st, const void *src, int64_t lds, omc_dtype_t dt, void *dst, int64_t ldd,
                                int64_t m, int64_t n, const omc_options_t *opts, omc_stream_t stream) {
    Geom g;
    omc_status_t r = geometry(opts, m, n, g);
    if (r != OMC_SUCCESS) return r;
    if (g.dr == 0 || g.dc == 0) return OMC_SUCCESS;
    if (!dst || (m > 0 && n > 0 && !src)) return OMC_ERROR_INVALID_VALUE;
    if (lds < (m > 1 ? m : 1) || ldd < (g.dr > 1 ? g.dr : 1)) return OMC_ERROR_INVALID_VALUE;
    gpuStream_t s = (gpuStream_t)stream;
    if (src == dst) return inplace_2d(st, dt, dst, lds, ldd, m, n, opts, s);
    r = check_pending();
    if (r != OMC_SUCCESS) return r;
    LaunchArgs a;
    r = prepare(st, dt, src, lds, m, n, opts, s, a, g.tr);
    if (r != OMC_SUCCESS) return r;
    const bool padded = (g.dr != g.rows) || (g.dc != g.cols);
    if (!g.tr && !padded && plain_copy(a) && m > 0 && n > 0) {
        const size_t es = omc_dtype_size(st);
        return rt_check(gpuMemcpy2DAsync(dst, (size_t)ldd * es, src, (size_t)lds * es, (size_t)m * es, (size_t)n,
                                         gpuMemcpyDeviceToDevice, s),
                        OMC_RT "Memcpy2DAsync");
    }
    a.src = src; a.dst = dst; a.lds = lds; a.ldd = ldd; a.m = m; a.n = n; a.dr = g.dr; a.dc = g.dc;
    a.num_sms = num_sms(current_device());
    if (g.tr) {
        a.op = OP_TRANSPOSE;
    } else {
        a.op = OP_CVT;
        const bool pos = (a.mode == MODE_SR || a.mode == MODE_SR_SCALED);
        if (!padded && !pos && lds == m && ldd == m) {   /* contiguous: one long column */
            a.m = a.lds = a.ldd = a.dr = m * n;
            a.n = a.dc = 1;
        }
    }
    return g_launch[st](a);
}

omc_status_t omc_gpu_convert_inplace(omc_dtype_t st, omc_dtype_t dt, void *buf, size_t count,
                                     const omc_options_t *opts, omc_stream_t stream) {
    if (count == 0) return OMC_SUCCESS;
    if (opts && (opts->flags & OMC_FLAG_TRANSPOSE)) return OMC_ERROR_INVALID_VALUE;
    return omc_gpu_convert_inplace_2d(st, dt, buf, (int64_t)count, (int64_t)count, 1, opts, stream);
}

omc_status_t omc_gpu_convert_inplace_2d(omc_dtype_t st, omc_dtype_t dt, void *buf, int64_t ld, int64_t m, int64_t n,
                                        const omc_options_t *opts, omc_stream_t stream) {
    return inplace_2d(st, dt, buf, ld, ld, m, n, opts, (gpuStream_t)stream);
}

omc_status_t omc_gpu_amax_2d(omc_dtype_t type, const void *src, int64_t ld, int64_t m, int64_t n, float *d_amax,
                             omc_stream_t stream) {
    if ((unsigned)type >= OMC_NUM_DTYPES || !d_amax || m < 0 || n < 0 || (m > 0 && ld < m)) return OMC_ERROR_INVALID_VALUE;
    if (!omc_dtype_is_float(type)) return OMC_ERROR_NOT_SUPPORTED;
    if (m > 0 && n > 0 && !src) return OMC_ERROR_INVALID_VALUE;
    omc_status_t r = check_pending();
    if (r != OMC_SUCCESS) return r;
    gpuStream_t s = (gpuStream_t)stream;
    if (m == 0 || n == 0) return rt_check(gpuMemsetAsync(d_amax, 0, sizeof(float), s), OMC_RT "MemsetAsync");
    LaunchArgs a = LaunchArgs{};
    a.op = OP_AMAX; a.st = type; a.src = src; a.lds = ld; a.m = m; a.n = n; a.amax_out = d_amax; a.stream = s;
    r = get_ws(nullptr, 0, s, a);
    if (r != OMC_SUCCESS) return r;
    return g_launch[type](a);
}

} /* extern "C" */
