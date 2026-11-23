/**
 * @file prof_gpu.cu
 * @brief One GPU conversion, for profilers (Nsight Compute, Nsight Systems,
 * rocprofv3): warm-up, then `--reps` calls between cudaProfilerStart/Stop
 * (hipProfilerStart/Stop), timed with events.
 *
 * omnicast_prof_gpu --src fp64 --dst fp32 [--m 8192] [--n 8192] [--ld L]
 *                   [--inplace] [--sat] [--sr] [--rne] [--scale s] [--dyn]
 *                   [--trans] [--pad rows cols] [--ldd L] [--copy] [--reps 5]
 * --copy times a device copy of the same bytes instead (the roofline).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "omnicast/omnicast.h"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#define gMalloc hipMalloc
#define gFree hipFree
#define gMemcpy hipMemcpy
#define gMemcpyAsync hipMemcpyAsync
#define gMemset hipMemset
#define gH2D hipMemcpyHostToDevice
#define gD2D hipMemcpyDeviceToDevice
#define gSync hipDeviceSynchronize
#define gEvent_t hipEvent_t
#define gEventCreate hipEventCreate
#define gEventRecord hipEventRecord
#define gEventSynchronize hipEventSynchronize
#define gEventElapsedTime hipEventElapsedTime
#define gProfilerStart hipProfilerStart
#define gProfilerStop hipProfilerStop
#else
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#define gMalloc cudaMalloc
#define gFree cudaFree
#define gMemcpy cudaMemcpy
#define gMemcpyAsync cudaMemcpyAsync
#define gMemset cudaMemset
#define gH2D cudaMemcpyHostToDevice
#define gD2D cudaMemcpyDeviceToDevice
#define gSync cudaDeviceSynchronize
#define gEvent_t cudaEvent_t
#define gEventCreate cudaEventCreate
#define gEventRecord cudaEventRecord
#define gEventSynchronize cudaEventSynchronize
#define gEventElapsedTime cudaEventElapsedTime
#define gProfilerStart cudaProfilerStart
#define gProfilerStop cudaProfilerStop
#endif

int main(int argc, char **argv) {
    omc_dtype_t st = OMC_FP64, dt = OMC_FP32;
    long long m = 8192, n = 8192, ld = 0, ldd = 0, pr = 0, pc = 0;
    int reps = 5, inplace = 0, copy = 0;
    omc_options_t o = {};
    for (int a = 1; a < argc; a++) {
        const char *s = argv[a];
        auto next = [&]() -> const char * { return (a + 1 < argc) ? argv[++a] : "0"; };
        if (!strcmp(s, "--src")) st = omc_dtype_from_name(next());
        else if (!strcmp(s, "--dst")) dt = omc_dtype_from_name(next());
        else if (!strcmp(s, "--m")) m = atoll(next());
        else if (!strcmp(s, "--n")) n = atoll(next());
        else if (!strcmp(s, "--ld")) ld = atoll(next());
        else if (!strcmp(s, "--ldd")) ldd = atoll(next());
        else if (!strcmp(s, "--reps")) reps = atoi(next());
        else if (!strcmp(s, "--inplace")) inplace = 1;
        else if (!strcmp(s, "--copy")) copy = 1;
        else if (!strcmp(s, "--sat")) o.flags |= OMC_FLAG_SATURATE;
        else if (!strcmp(s, "--trans")) o.flags |= OMC_FLAG_TRANSPOSE;
        else if (!strcmp(s, "--sr")) { o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 12345; }
        else if (!strcmp(s, "--rne")) o.rounding = OMC_ROUND_NEAREST_EVEN;
        else if (!strcmp(s, "--scale")) { o.scale_mode = OMC_SCALE_HOST; o.scale = (float)atof(next()); }
        else if (!strcmp(s, "--dyn")) o.scale_mode = OMC_SCALE_DYNAMIC_POW2;
        else if (!strcmp(s, "--pad")) { pr = atoll(next()); pc = atoll(next()); }
        else { fprintf(stderr, "unknown option %s\n", s); return 2; }
    }
    if (st >= OMC_NUM_DTYPES || dt >= OMC_NUM_DTYPES) { fprintf(stderr, "bad type\n"); return 2; }
    if (!ld) ld = m;
    const bool tr = (o.flags & OMC_FLAG_TRANSPOSE) != 0;
    const long long rows = pr ? pr : (tr ? n : m), cols = pc ? pc : (tr ? m : n);
    if (pr) { o.dst_rows = pr; o.dst_cols = pc; }
    if (!ldd) ldd = inplace ? ld : rows;
    const size_t si = omc_dtype_size(st), so = omc_dtype_size(dt);
    const size_t src_bytes = (size_t)((n - 1) * ld + m) * si;
    const size_t dst_bytes = (size_t)((cols - 1) * ldd + rows) * so;
    const size_t big = (src_bytes > dst_bytes ? src_bytes : dst_bytes) + 256;
    void *A, *B, *S;
    gMalloc(&A, big); gMalloc(&B, big); gMalloc(&S, 64);
    o.scale_ptr = (float *)S;
    {   /* source values: FP64 in +-64 converted (saturated) to the source type */
        const size_t cnt = (size_t)((n - 1) * ld + m);
        std::vector<double> h(cnt);
        for (size_t k = 0; k < cnt; k++) h[k] = ((double)((k * 2654435761u) % 1000003) - 500000.0) / 7777.0;
        void *T; gMalloc(&T, cnt * 8);
        gMemcpy(T, h.data(), cnt * 8, gH2D);
        omc_options_t sat = {}; sat.flags = OMC_FLAG_SATURATE;
        omc_gpu_convert(OMC_FP64, T, st, A, cnt, &sat, nullptr);
        gSync(); gFree(T);
    }
    void *base = nullptr;
    if (inplace) { gMalloc(&base, src_bytes + 256); gMemcpy(base, A, src_bytes, gD2D); }
    auto call = [&]() -> omc_status_t {
        if (copy) return gMemcpyAsync(B, A, (si * (size_t)m * n + so * (size_t)rows * cols) / 2, gD2D, 0) == 0
                             ? OMC_SUCCESS : OMC_ERROR_DEVICE;
        if (inplace) {
            /* restore the source (not timed separately: the copy is subtracted below) */
            return omc_gpu_convert_inplace_2d(st, dt, A, ld, m, n, &o, nullptr);
        }
        return omc_gpu_convert_2d(st, A, ld, dt, B, ldd, m, n, &o, nullptr);
    };
    gEvent_t e0, e1;
    gEventCreate(&e0); gEventCreate(&e1);
    /* warm-up: 0.3 s (clocks ramp up); in place restores the source each time */
    omc_status_t r = OMC_SUCCESS;
    gEventRecord(e0, 0);
    for (float ms = 0.0f; ms < 300.0f;) {
        for (int w = 0; w < 10; w++) {
            if (inplace) gMemcpyAsync(A, base, src_bytes, gD2D, 0);
            r = call();
            if (r != OMC_SUCCESS) { fprintf(stderr, "omnicast: %s\n", omc_status_string(r)); return 1; }
        }
        gEventRecord(e1, 0); gEventSynchronize(e1); gEventElapsedTime(&ms, e0, e1);
    }
    float best = 1e30f;
    gProfilerStart();
    for (int k = 0; k < reps; k++) {
        if (inplace) gMemcpyAsync(A, base, src_bytes, gD2D, 0);
        gEventRecord(e0, 0);
        call();
        gEventRecord(e1, 0);
        gEventSynchronize(e1);
        float ms; gEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    gProfilerStop();
    gSync();
    const double bytes = (double)si * m * n + (double)so * rows * cols;
    printf("%s %s>%s m=%lld n=%lld ld=%lld%s%s%s%s: %.2f us, %.1f GB/s (read + write once)\n", copy ? "copy" : (inplace ? "inplace" : "out"),
           omc_dtype_name(st), omc_dtype_name(dt), m, n, ld, (o.rounding == OMC_ROUND_STOCHASTIC) ? " sr" : "",
           tr ? " trans" : "", pr ? " pad" : "", (o.scale_mode == OMC_SCALE_DYNAMIC_POW2) ? " dyn" : "", best * 1e3,
           bytes / (best * 1e-3) * 1e-9);
    gFree(A); gFree(B); gFree(S);
    if (base) gFree(base);
    omc_gpu_release_workspaces();
    return 0;
}
