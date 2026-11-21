/**
 * @file bench_gpu.cu
 * @brief GPU conversion time: OmniCast against the HiCMA-X 0.2 kernels and
 * against a device-to-device copy that moves the same bytes.
 *
 * Square tiles nb x nb (the HiCMA-X case) and long vectors. GB/s counts bytes
 * read + bytes written once (the minimum any conversion must move).
 * Usage: omnicast_bench_gpu [nb ...]   (default 512 1024 2048 4096 8192)
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include "omnicast/omnicast.h"
#include "legacy_hicma_gpu.cuh"

#if defined(__HIPCC__)
#define gMalloc hipMalloc
#define gFree hipFree
#define gMemcpy hipMemcpy
#define gMemcpyAsync hipMemcpyAsync
#define gMemset hipMemset
#define gH2D hipMemcpyHostToDevice
#define gD2H hipMemcpyDeviceToHost
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
#define gMalloc cudaMalloc
#define gFree cudaFree
#define gMemcpy cudaMemcpy
#define gMemcpyAsync cudaMemcpyAsync
#define gMemset cudaMemset
#define gH2D cudaMemcpyHostToDevice
#define gD2H cudaMemcpyDeviceToHost
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

/* average time of one call in microseconds */
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
        if (ms > 30.0f || iters >= (1 << 16)) {
            /* repeat 3 times, keep the best */
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

struct Case {
    std::string name;
    size_t si, so;
    std::function<void()> omc, legacy;
    std::function<long()> diff;   /* compare the two outputs once */
};

int main(int argc, char **argv) {
    std::vector<int> nbs;
    for (int a = 1; a < argc; a++) nbs.push_back(atoi(argv[a]));
    if (nbs.empty()) nbs = {512, 1024, 2048, 4096, 8192};
    gProp p;
    gGetDeviceProperties(&p, 0);
    printf("# OmniCast %s GPU bench, backend %s, device %s\n", omc_version_string(), omc_gpu_backend(), p.name);
    printf("%-14s %6s %10s %9s %7s %10s %8s %8s\n", "case", "nb", "omc(us)", "GB/s", "%copy", "legacy(us)", "speedup", "diffs");
    for (int nb : nbs) {
        const size_t n = (size_t)nb * nb;
        void *A, *B, *C, *S;
        gMalloc(&A, n * 16 + 4096); gMalloc(&B, n * 16 + 4096); gMalloc(&C, n * 16 + 4096); gMalloc(&S, 64);
        std::vector<double> h(n);
        for (size_t k = 0; k < n; k++) h[k] = ((double)((k * 2654435761u) % 1000003) - 500000.0) / 7777.0;
        const omc_options_t none = {};
        omc_options_t sat = {}; sat.flags = OMC_FLAG_SATURATE;
        omc_options_t sr = {}; sr.rounding = OMC_ROUND_STOCHASTIC; sr.seed = 12345;
        auto load_d = [&] { gMemcpy(A, h.data(), n * 8, gH2D); };
        auto load = [&](omc_dtype_t t) { load_d(); if (t != OMC_FP64) { omc_gpu_convert(OMC_FP64, A, t, C, n, &sat, nullptr); gMemcpy(A, C, n * omc_dtype_size(t), gD2D); } };
        auto bytes_diff = [&](size_t bytes) -> long {
            std::vector<unsigned char> x(bytes), y(bytes);
            gMemcpy(x.data(), B, bytes, gD2H); gMemcpy(y.data(), C, bytes, gD2H);
            long d = 0; for (size_t k = 0; k < bytes; k++) d += (x[k] != y[k]);
            return d;
        };
        auto run_case = [&](const char *nm, omc_dtype_t st, omc_dtype_t dt, std::function<void()> omcf,
                            std::function<void()> leg, size_t out_bytes, int inplace) {
            const size_t si = omc_dtype_size(st), so = omc_dtype_size(dt);
            load(st);
            long diffs = -1;
            if (leg) {
                if (inplace) {
                    gMemcpy(B, A, n * si, gD2D); gMemcpy(C, A, n * si, gD2D);
                    void *keepA = A;
                    A = B; omcf(); A = C; leg(); A = keepA;     /* outputs in place in B and C */
                    gSync();
                    diffs = bytes_diff(out_bytes);
                    load(st);
                } else {
                    omcf(); gSync();
                    gMemcpy(C, B, out_bytes, gD2D);      /* keep the OmniCast result in C */
                    leg(); gSync();                       /* legacy writes B */
                    diffs = bytes_diff(out_bytes);
                }
            }
            double t = time_us(omcf);
            double tl = leg ? time_us(leg) : 0.0;
            const double bytes = (double)n * (si + so);
            double tc = time_us([&] { gMemcpyAsync(C, A, (size_t)(bytes / 2), gD2D, 0); });
            if (leg)
                printf("%-14s %6d %10.2f %9.1f %6.0f%% %10.2f %7.2fx %8ld\n", nm, nb, t, bytes / t * 1e-3, 100.0 * tc / t, tl, tl / t, diffs);
            else
                printf("%-14s %6d %10.2f %9.1f %6.0f%% %10s %8s %8s\n", nm, nb, t, bytes / t * 1e-3, 100.0 * tc / t, "-", "-", "-");
            fflush(stdout);
        };
#define OMC2D(st, dt, opt) [&] { omc_gpu_convert_2d(st, A, nb, dt, B, nb, nb, nb, opt, nullptr); }
        /* warm-up for 0.5 s: GPU clocks ramp up before the first measured case */
        {
            gEvent_t w0, w1; gEventCreate(&w0); gEventCreate(&w1);
            gEventRecord(w0, 0);
            for (float ms = 0.0f; ms < 500.0f; ) {
                for (int w = 0; w < 50; w++) omc_gpu_convert_2d(OMC_FP64, A, nb, OMC_FP32, B, nb, nb, nb, &none, nullptr);
                gEventRecord(w1, 0); gEventSynchronize(w1); gEventElapsedTime(&ms, w0, w1);
            }
        }
        run_case("fp64>fp32", OMC_FP64, OMC_FP32, OMC2D(OMC_FP64, OMC_FP32, &none), [&] { l_double2float(nb, nb, (double *)A, nb, (float *)B, nb, 0); }, n * 4, 0);
        run_case("fp32>fp64", OMC_FP32, OMC_FP64, OMC2D(OMC_FP32, OMC_FP64, &none), [&] { l_float2double(nb, nb, (float *)A, nb, (double *)B, nb, 0); }, n * 8, 0);
        run_case("fp32>fp16", OMC_FP32, OMC_FP16, OMC2D(OMC_FP32, OMC_FP16, &none), [&] { l_float2half(nb, nb, (float *)A, nb, B, nb, 0); }, n * 2, 0);
        run_case("fp16>fp32", OMC_FP16, OMC_FP32, OMC2D(OMC_FP16, OMC_FP32, &none), [&] { l_half2float(nb, nb, A, nb, (float *)B, nb, 0); }, n * 4, 0);
        run_case("fp64>fp16", OMC_FP64, OMC_FP16, OMC2D(OMC_FP64, OMC_FP16, &none), [&] { l_double2half(nb, nb, (double *)A, nb, B, nb, 0); }, n * 2, 0);
        run_case("fp16>fp64", OMC_FP16, OMC_FP64, OMC2D(OMC_FP16, OMC_FP64, &none), [&] { l_half2double(nb, nb, A, nb, (double *)B, nb, 0); }, n * 8, 0);
#if !defined(__HIPCC__)
        run_case("fp64>bf16", OMC_FP64, OMC_BF16, OMC2D(OMC_FP64, OMC_BF16, &none), [&] { l_double2bf(nb, nb, (double *)A, nb, B, nb, 0); }, n * 2, 0);
        run_case("fp32>bf16", OMC_FP32, OMC_BF16, OMC2D(OMC_FP32, OMC_BF16, &none), [&] { l_float2bf(nb, nb, (float *)A, nb, B, nb, 0); }, n * 2, 0);
        run_case("bf16>fp32", OMC_BF16, OMC_FP32, OMC2D(OMC_BF16, OMC_FP32, &none), [&] { l_bf2float(nb, nb, A, nb, (float *)B, nb, 0); }, n * 4, 0);
#else
        run_case("fp64>bf16", OMC_FP64, OMC_BF16, OMC2D(OMC_FP64, OMC_BF16, &none), nullptr, n * 2, 0);
        run_case("fp32>bf16", OMC_FP32, OMC_BF16, OMC2D(OMC_FP32, OMC_BF16, &none), nullptr, n * 2, 0);
#endif
        run_case("SR fp64>fp32", OMC_FP64, OMC_FP32, OMC2D(OMC_FP64, OMC_FP32, &sr), [&] { l_d2s_round(nb, nb, (double *)A, nb, (float *)B, nb, 12345, 0); }, n * 4, 0);
        run_case("SR fp64>fp16", OMC_FP64, OMC_FP16, OMC2D(OMC_FP64, OMC_FP16, &sr), [&] { l_d2h_round(nb, nb, (double *)A, nb, B, nb, 12345, 0); }, n * 2, 0);
        run_case("SR fp32>fp16", OMC_FP32, OMC_FP16, OMC2D(OMC_FP32, OMC_FP16, &sr), [&] { l_s2h_round(nb, nb, (float *)A, nb, B, nb, 12345, 0); }, n * 2, 0);
        /* FP8 for cublasLt: transposed, zero-padded to a multiple of 16, saturated */
        {
            const int pd = (nb + 15) / 16 * 16 + 16;
            omc_options_t t8 = {}; t8.flags = OMC_FLAG_SATURATE | OMC_FLAG_TRANSPOSE; t8.dst_rows = pd; t8.dst_cols = pd;
            omc_options_t d8 = t8; d8.scale_mode = OMC_SCALE_DYNAMIC_POW2; d8.scale_ptr = (float *)S;
#if !defined(__HIPCC__)
            run_case("T fp64>e4m3", OMC_FP64, OMC_FP8_E4M3, [&] { omc_gpu_convert_2d(OMC_FP64, A, nb, OMC_FP8_E4M3, B, pd, nb, nb, &t8, nullptr); },
                     [&] { l_fp8_padded<double>(nb, nb, (double *)A, nb, B, pd, pd, pd, 0); }, (size_t)pd * pd, 0);
            run_case("T fp32>e4m3", OMC_FP32, OMC_FP8_E4M3, [&] { omc_gpu_convert_2d(OMC_FP32, A, nb, OMC_FP8_E4M3, B, pd, nb, nb, &t8, nullptr); },
                     [&] { l_fp8_padded<float>(nb, nb, (float *)A, nb, B, pd, pd, pd, 0); }, (size_t)pd * pd, 0);
            run_case("T fp16>e4m3", OMC_FP16, OMC_FP8_E4M3, [&] { omc_gpu_convert_2d(OMC_FP16, A, nb, OMC_FP8_E4M3, B, pd, nb, nb, &t8, nullptr); },
                     [&] { l_fp8_padded<__half>(nb, nb, (__half *)A, nb, B, pd, pd, pd, 0); }, (size_t)pd * pd, 0);
            run_case("TS fp64>e4m3", OMC_FP64, OMC_FP8_E4M3, [&] { omc_gpu_convert_2d(OMC_FP64, A, nb, OMC_FP8_E4M3, B, pd, nb, nb, &d8, nullptr); },
                     [&] { l_fp8_scaled<double>(nb, nb, (double *)A, nb, B, pd, pd, pd, (float *)S + 4, 0); }, (size_t)pd * pd, 0);
            run_case("TS fp32>e4m3", OMC_FP32, OMC_FP8_E4M3, [&] { omc_gpu_convert_2d(OMC_FP32, A, nb, OMC_FP8_E4M3, B, pd, nb, nb, &d8, nullptr); },
                     [&] { l_fp8_scaled<float>(nb, nb, (float *)A, nb, B, pd, pd, pd, (float *)S + 4, 0); }, (size_t)pd * pd, 0);
#else
            run_case("T fp64>e4m3", OMC_FP64, OMC_FP8_E4M3, [&] { omc_gpu_convert_2d(OMC_FP64, A, nb, OMC_FP8_E4M3, B, pd, nb, nb, &t8, nullptr); }, nullptr, 0, 0);
            run_case("TS fp32>e4m3", OMC_FP32, OMC_FP8_E4M3, [&] { omc_gpu_convert_2d(OMC_FP32, A, nb, OMC_FP8_E4M3, B, pd, nb, nb, &d8, nullptr); }, nullptr, 0, 0);
#endif
        }
        /* in place: HiCMA-X converts into a workspace and copies back */
        run_case("IP fp32>fp16", OMC_FP32, OMC_FP16, [&] { omc_gpu_convert_inplace(OMC_FP32, OMC_FP16, A, n, nullptr, nullptr); },
                 [&] { l_float2half(nb, nb, (float *)A, nb, (char *)A + n * 8, nb, 0); l_memcpy<__half>(nb, nb, (char *)A + n * 8, A, 0); }, n * 2, 1);
        run_case("IP fp16>fp32", OMC_FP16, OMC_FP32, [&] { omc_gpu_convert_inplace(OMC_FP16, OMC_FP32, A, n, nullptr, nullptr); },
                 [&] { l_half2float(nb, nb, A, nb, (float *)((char *)A + n * 4), nb, 0); l_memcpy<float>(nb, nb, (char *)A + n * 4, A, 0); }, n * 4, 1);
        run_case("IP fp64>fp32", OMC_FP64, OMC_FP32, [&] { omc_gpu_convert_inplace(OMC_FP64, OMC_FP32, A, n, nullptr, nullptr); },
                 [&] { l_double2float(nb, nb, (double *)A, nb, (float *)((char *)A + n * 8), nb, 0); l_memcpy<float>(nb, nb, (char *)A + n * 8, A, 0); }, n * 4, 1);
        run_case("IP fp32>fp64", OMC_FP32, OMC_FP64, [&] { omc_gpu_convert_inplace(OMC_FP32, OMC_FP64, A, n, nullptr, nullptr); },
                 [&] { l_float2double(nb, nb, (float *)A, nb, (double *)((char *)A + n * 8), nb, 0); l_memcpy<double>(nb, nb, (char *)A + n * 8, A, 0); }, n * 8, 1);
        run_case("IP fp64>fp16", OMC_FP64, OMC_FP16, [&] { omc_gpu_convert_inplace(OMC_FP64, OMC_FP16, A, n, nullptr, nullptr); },
                 [&] { l_double2half(nb, nb, (double *)A, nb, (char *)A + n * 8, nb, 0); l_memcpy<__half>(nb, nb, (char *)A + n * 8, A, 0); }, n * 2, 1);
        gFree(A); gFree(B); gFree(C); gFree(S);
        omc_gpu_release_workspaces();
    }
    return 0;
}
