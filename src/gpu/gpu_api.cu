/**
 * @file gpu_api.cu
 * @brief Public GPU API: argument checks, options, plain copies, and the
 * internal per-(device, stream) workspace used by the in-place and
 * dynamically scaled conversions.
 */
#include <map>
#include <mutex>
#include <utility>
#include "gpu_rt.hpp"
#include "../core/omc_common.hpp"

namespace omc {
namespace gpu {
namespace {

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
        if (gpuMalloc(&p, sz) != gpuSuccess) { gpuGetLastError(); return OMC_ERROR_ALLOCATION; }
        if (gpuMemset(p, 0, sz) != gpuSuccess) { gpuFree(p); gpuGetLastError(); return OMC_ERROR_ALLOCATION; }
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

/** Options -> launch arguments (mode, rounding, scale). Runs the amax pass
 *  first for OMC_SCALE_DYNAMIC_POW2. */
omc_status_t prepare(omc_dtype_t st, omc_dtype_t dt, const void *src, int64_t lds, int64_t m, int64_t n,
                     const omc_options_t *o, gpuStream_t stream, LaunchArgs &a) {
    if ((unsigned)st >= OMC_NUM_DTYPES || (unsigned)dt >= OMC_NUM_DTYPES) return OMC_ERROR_INVALID_VALUE;
    static const omc_options_t zero = {};
    if (!o) o = &zero;
    a = LaunchArgs{};
    a.st = st; a.dt = dt; a.stream = stream;
    a.sat = (o->flags & OMC_FLAG_SATURATE) != 0;
    a.rne = (o->rounding == OMC_ROUND_NEAREST_EVEN);
    a.seed = o->seed;
    a.mode = MODE_PLAIN;
    a.scale = 1.0f;
    if (o->rounding == OMC_ROUND_TOWARD_ZERO && omc_dtype_is_float(dt)) return OMC_ERROR_NOT_SUPPORTED;
    if (o->rounding == OMC_ROUND_STOCHASTIC) {
        if (o->scale_mode != OMC_SCALE_NONE) return OMC_ERROR_NOT_SUPPORTED;
        a.mode = MODE_SR;
        return OMC_SUCCESS;
    }
    switch (o->scale_mode) {
    case OMC_SCALE_NONE:
        return OMC_SUCCESS;
    case OMC_SCALE_HOST:
        a.mode = MODE_SCALED; a.scale = o->scale;
        return OMC_SUCCESS;
    case OMC_SCALE_DEVICE:
        if (!o->scale_ptr) return OMC_ERROR_INVALID_VALUE;
        a.mode = MODE_SCALED; a.scale_ptr = o->scale_ptr; a.scale_index = 0;
        return OMC_SUCCESS;
    case OMC_SCALE_DYNAMIC_POW2: {
        if (!o->scale_ptr || !omc_dtype_is_float(st) || !omc_dtype_is_float(dt)) return OMC_ERROR_INVALID_VALUE;
        LaunchArgs am = LaunchArgs{};
        am.op = OP_AMAX; am.st = st; am.dt = dt; am.src = src; am.lds = lds; am.m = m; am.n = n;
        am.amax_out = o->scale_ptr; am.amax_scale = 1;
        am.scale_max = (o->scale_max > 0.0f) ? o->scale_max : max_finite_of(dt);
        am.stream = stream;
        omc_status_t r = get_ws(o, 0, stream, am);
        if (r != OMC_SUCCESS) return r;
        if (m > 0 && n > 0) {
            r = g_launch[st](am);
            if (r != OMC_SUCCESS) return r;
        } else {
            /* empty source: amax 0, scale 1 */
            const float one[3] = {0.0f, 1.0f, 1.0f};
            if (gpuMemcpyAsync(o->scale_ptr, one, sizeof one, gpuMemcpyHostToDevice, stream) != gpuSuccess)
                return OMC_ERROR_DEVICE;
        }
        a.mode = MODE_SCALED; a.scale_ptr = o->scale_ptr; a.scale_index = 1;
        return OMC_SUCCESS;
    }
    }
    return OMC_ERROR_INVALID_VALUE;
}

bool plain_copy(const LaunchArgs &a) { return a.st == a.dt && a.mode == MODE_PLAIN && !a.sat; }

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
    LaunchArgs a;
    gpuStream_t s = (gpuStream_t)stream;
    omc_status_t r = prepare(st, dt, src, (int64_t)count, (int64_t)count, 1, opts, s, a);
    if (r != OMC_SUCCESS) return r;
    if (plain_copy(a)) {
        return gpuMemcpyAsync(dst, src, count * omc_dtype_size(st), gpuMemcpyDeviceToDevice, s) == gpuSuccess
                   ? OMC_SUCCESS : OMC_ERROR_DEVICE;
    }
    a.op = OP_CVT; a.src = src; a.dst = dst;
    a.m = a.lds = a.ldd = a.dr = (int64_t)count; a.n = a.dc = 1;
    a.num_sms = num_sms(current_device());
    return g_launch[st](a);
}

omc_status_t omc_gpu_convert_2d(omc_dtype_t st, const void *src, int64_t lds, omc_dtype_t dt, void *dst, int64_t ldd,
                                int64_t m, int64_t n, const omc_options_t *opts, omc_stream_t stream) {
    if (m < 0 || n < 0) return OMC_ERROR_INVALID_VALUE;
    const bool tr = opts && (opts->flags & OMC_FLAG_TRANSPOSE);
    const int64_t rows = tr ? n : m, cols = tr ? m : n;
    const int64_t dr = (opts && opts->dst_rows > 0) ? opts->dst_rows : rows;
    const int64_t dc = (opts && opts->dst_cols > 0) ? opts->dst_cols : cols;
    if (dr < rows || dc < cols) return OMC_ERROR_INVALID_VALUE;
    if (dr == 0 || dc == 0) return OMC_SUCCESS;
    if (!dst || (m > 0 && n > 0 && !src)) return OMC_ERROR_INVALID_VALUE;
    if (lds < (m > 1 ? m : 1) || ldd < (dr > 1 ? dr : 1)) return OMC_ERROR_INVALID_VALUE;
    LaunchArgs a;
    gpuStream_t s = (gpuStream_t)stream;
    omc_status_t r = prepare(st, dt, src, lds, m, n, opts, s, a);
    if (r != OMC_SUCCESS) return r;
    const bool padded = (dr != rows) || (dc != cols);
    if (!tr && !padded && plain_copy(a) && m > 0 && n > 0) {
        const size_t es = omc_dtype_size(st);
        return gpuMemcpy2DAsync(dst, (size_t)ldd * es, src, (size_t)lds * es, (size_t)m * es, (size_t)n,
                                gpuMemcpyDeviceToDevice, s) == gpuSuccess ? OMC_SUCCESS : OMC_ERROR_DEVICE;
    }
    a.src = src; a.dst = dst; a.lds = lds; a.ldd = ldd; a.m = m; a.n = n; a.dr = dr; a.dc = dc;
    a.num_sms = num_sms(current_device());
    if (tr) {
        a.op = OP_TRANSPOSE;
    } else {
        a.op = OP_CVT;
        if (!padded && a.mode != MODE_SR && lds == m && ldd == m) {   /* contiguous: one long column */
            a.m = a.lds = a.ldd = a.dr = m * n;
            a.n = a.dc = 1;
        }
    }
    return g_launch[st](a);
}

omc_status_t omc_gpu_convert_inplace(omc_dtype_t st, omc_dtype_t dt, void *buf, size_t count,
                                     const omc_options_t *opts, omc_stream_t stream) {
    if (count == 0) return OMC_SUCCESS;
    return omc_gpu_convert_inplace_2d(st, dt, buf, (int64_t)count, (int64_t)count, 1, opts, stream);
}

omc_status_t omc_gpu_convert_inplace_2d(omc_dtype_t st, omc_dtype_t dt, void *buf, int64_t ld, int64_t m, int64_t n,
                                        const omc_options_t *opts, omc_stream_t stream) {
    if (m < 0 || n < 0) return OMC_ERROR_INVALID_VALUE;
    if (m == 0 || n == 0) return OMC_SUCCESS;
    if (!buf || ld < m) return OMC_ERROR_INVALID_VALUE;
    if (opts && ((opts->flags & OMC_FLAG_TRANSPOSE) || opts->dst_rows > 0 || opts->dst_cols > 0))
        return OMC_ERROR_INVALID_VALUE;
    LaunchArgs a;
    gpuStream_t s = (gpuStream_t)stream;
    omc_status_t r = prepare(st, dt, buf, ld, m, n, opts, s, a);
    if (r != OMC_SUCCESS) return r;
    if (plain_copy(a)) return OMC_SUCCESS;
    a.src = buf; a.dst = buf;
    a.num_sms = num_sms(current_device());
    if (omc_dtype_size(st) == omc_dtype_size(dt)) {
        /* same element size: each thread reads then writes its own elements */
        a.op = OP_CVT; a.lds = a.ldd = ld; a.m = m; a.n = n; a.dr = m; a.dc = n;
        if (a.mode != MODE_SR && ld == m) { a.m = a.lds = a.ldd = a.dr = m * n; a.n = a.dc = 1; }
        return g_launch[st](a);
    }
    if (ld != m && n > 1) return OMC_ERROR_NOT_SUPPORTED;
    a.op = OP_INPLACE; a.m = m; a.n = n;
    r = get_ws(opts, (size_t)(m * n), s, a);
    if (r != OMC_SUCCESS) return r;
    return g_launch[st](a);
}

omc_status_t omc_gpu_amax_2d(omc_dtype_t type, const void *src, int64_t ld, int64_t m, int64_t n, float *d_amax,
                             omc_stream_t stream) {
    if ((unsigned)type >= OMC_NUM_DTYPES || !d_amax || m < 0 || n < 0 || (m > 0 && ld < m)) return OMC_ERROR_INVALID_VALUE;
    if (!omc_dtype_is_float(type)) return OMC_ERROR_NOT_SUPPORTED;
    gpuStream_t s = (gpuStream_t)stream;
    if (m == 0 || n == 0)
        return gpuMemsetAsync(d_amax, 0, sizeof(float), s) == gpuSuccess ? OMC_SUCCESS : OMC_ERROR_DEVICE;
    if (!src) return OMC_ERROR_INVALID_VALUE;
    LaunchArgs a = LaunchArgs{};
    a.op = OP_AMAX; a.st = type; a.src = src; a.lds = ld; a.m = m; a.n = n; a.amax_out = d_amax; a.stream = s;
    omc_status_t r = get_ws(nullptr, 0, s, a);
    if (r != OMC_SUCCESS) return r;
    return g_launch[type](a);
}

} /* extern "C" */
