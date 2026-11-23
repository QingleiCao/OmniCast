/**
 * @file bench_modes_gpu.cu
 * @brief Every type pair on the GPU, out of place and in place, against a
 * device copy of the same bytes (bytes read + bytes written once).
 *
 * For each nb (default 4096): nb x nb matrices, out of place (ld = nb), in
 * place (ld = nb) and in place on a strided matrix (ld = nb + 64). In-place
 * calls are repeated on the same buffer (the kernels do not depend on the
 * values). Prints one line per pair: microseconds and % of the copy time.
 * Usage: omnicast_bench_modes_gpu [--sr] [nb ...]
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <functional>
#include <algorithm>
#include "omnicast/omnicast.h"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gMalloc hipMalloc
#define gFree hipFree
#define gMemcpyAsync hipMemcpyAsync
#define gMemset hipMemset
#define gD2D hipMemcpyDeviceToDevice
#define gSync hipDeviceSynchronize
#define gEvent_t hipEvent_t
#define gEventCreate hipEventCreate
#define gEventRecord hipEventRecord
#define gEventSynchronize hipEventSynchronize
#define gEventElapsedTime hipEventElapsedTime
#define gGetDeviceProperties hipGetDeviceProperties
typedef hipDeviceProp_t gProp;
#else
#include <cuda_runtime.h>
#define gMalloc cudaMalloc
#define gFree cudaFree
#define gMemcpyAsync cudaMemcpyAsync
#define gMemset cudaMemset
#define gD2D cudaMemcpyDeviceToDevice
#define gSync cudaDeviceSynchronize
#define gEvent_t cudaEvent_t
#define gEventCreate cudaEventCreate
#define gEventRecord cudaEventRecord
#define gEventSynchronize cudaEventSynchronize
#define gEventElapsedTime cudaEventElapsedTime
#define gGetDeviceProperties cudaGetDeviceProperties
typedef cudaDeviceProp gProp;
#endif

/* best of 3 averages over >= 20 ms, microseconds per call */
static double time_us(const std::function<void()> &f) {
    gEvent_t a, b;
    gEventCreate(&a); gEventCreate(&b);
    for (int w = 0; w < 3; w++) f();
    gSync();
    int iters = 1;
    for (;;) {
        gEventRecord(a, 0);
        for (int k = 0; k < iters; k++) f();
        gEventRecord(b, 0);
        gEventSynchronize(b);
        float ms; gEventElapsedTime(&ms, a, b);
        if (ms > 20.0f || iters >= (1 << 14)) {
            double best = ms * 1000.0 / iters;
            for (int r = 0; r < 2; r++) {
                gEventRecord(a, 0);
                for (int k = 0; k < iters; k++) f();
                gEventRecord(b, 0);
                gEventSynchronize(b);
                gEventElapsedTime(&ms, a, b);
                best = std::min(best, (double)ms * 1000.0 / iters);
            }
            return best;
        }
        iters *= 2;
    }
}

int main(int argc, char **argv) {
    std::vector<long> nbs;
    bool sr = false;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--sr")) sr = true;
        else nbs.push_back(atol(argv[a]));
    }
    if (nbs.empty()) nbs = {4096};
    gProp p;
    gGetDeviceProperties(&p, 0);
    printf("# OmniCast %s, %s, device %s%s\n", omc_version_string(), omc_gpu_backend(), p.name,
           sr ? ", stochastic rounding" : "");
    printf("%-6s %-6s %6s %9s %9s %9s %9s %7s %7s %7s\n", "src", "dst", "nb", "copy(us)", "out(us)", "in(us)",
           "inS(us)", "out%", "in%", "inS%");
    omc_options_t o = {};
    o.flags = OMC_FLAG_SATURATE;
    if (sr) { o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 7; }
    for (long nb : nbs) {
        const long lds = nb + 64;
        const size_t big = (size_t)lds * nb * 8 + 4096;
        void *A, *B;
        if (gMalloc(&A, big) != 0 || gMalloc(&B, big) != 0) { fprintf(stderr, "allocation failed\n"); return 1; }
        gMemset(A, 0, big); gMemset(B, 0, big);
        /* warm-up for 0.5 s */
        {
            gEvent_t w0, w1; gEventCreate(&w0); gEventCreate(&w1);
            gEventRecord(w0, 0);
            for (float ms = 0.0f; ms < 500.0f;) {
                for (int w = 0; w < 20; w++) omc_gpu_convert_2d(OMC_FP64, A, nb, OMC_FP32, B, nb, nb, nb, nullptr, nullptr);
                gEventRecord(w1, 0); gEventSynchronize(w1); gEventElapsedTime(&ms, w0, w1);
            }
        }
        std::vector<double> copy_t(9 * 9, 0.0);
        for (int st = 0; st < OMC_NUM_DTYPES; st++)
            for (int dt = 0; dt < OMC_NUM_DTYPES; dt++) {
                const omc_dtype_t S = (omc_dtype_t)st, D = (omc_dtype_t)dt;
                if (sr) {
                    omc_options_t t = o;
                    if (omc_gpu_convert_2d(S, A, nb, D, B, nb, 1, 1, &t, nullptr) != OMC_SUCCESS) continue;
                }
                const size_t si = omc_dtype_size(S), so = omc_dtype_size(D);
                const size_t bytes = (size_t)nb * nb * (si + so);
                double &tc = copy_t[si * 9 + so];
                if (tc == 0.0) tc = time_us([&] { gMemcpyAsync(B, A, bytes / 2, gD2D, 0); });
                const double to = time_us([&] { omc_gpu_convert_2d(S, A, nb, D, B, nb, nb, nb, &o, nullptr); });
                const double ti = time_us([&] { omc_gpu_convert_inplace_2d(S, D, A, nb, nb, nb, &o, nullptr); });
                const double ts = time_us([&] { omc_gpu_convert_inplace_2d(S, D, A, lds, nb, nb, &o, nullptr); });
                printf("%-6s %-6s %6ld %9.1f %9.1f %9.1f %9.1f %6.0f%% %6.0f%% %6.0f%%\n", omc_dtype_name(S), omc_dtype_name(D),
                       nb, tc, to, ti, ts, 100.0 * tc / to, 100.0 * tc / ti, 100.0 * tc / ts);
                fflush(stdout);
            }
        gFree(A); gFree(B);
        omc_gpu_release_workspaces();
    }
    return 0;
}
