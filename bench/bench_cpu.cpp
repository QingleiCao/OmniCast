/**
 * @file bench_cpu.cpp
 * @brief Host conversion throughput: OmniCast (each CPU code path) against
 * memcpy of the same traffic and the HiCMA-X 0.2 loops.
 *
 * Reports GB/s counted as bytes read + bytes written, best of several runs.
 * Usage: omnicast_bench_cpu [elements ...] [--threads N] [--isa NAME]
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include "omnicast/omnicast.h"

extern "C" {
void legacy_convert_d2s_binary_CPU(float *, double *, int, int);
void legacy_convert_s2d_binary_CPU(double *, float *, int, int);
void legacy_convert_d2s_unary_CPU(double *, int, int);
void legacy_convert_s2d_unary_CPU(float *, int, int);
void legacy_convert_s2i8_binary_CPU(int8_t *, float *, int, int, int);
void legacy_convert_8i2s_binary_CPU(float *, int8_t *, int, int, int);
void legacy_convert_d2i_binary_CPU(int *, double *, int, int, int);
void legacy_convert_i2d_binary_CPU(double *, int *, int, int, int);
void legacy_double2float_round_CPU(int, int, const double *, int, float *, int, uint64_t);
}

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

/* best time of 7 runs, each repeated to last >= 20 ms */
static double timeit(const std::function<void()> &f) {
    f();
    int reps = 1;
    for (;;) { double t0 = now(); for (int r = 0; r < reps; r++) f(); double t = now() - t0; if (t > 0.02) break; reps *= 2; }
    double best = 1e30;
    for (int k = 0; k < 7; k++) { double t0 = now(); for (int r = 0; r < reps; r++) f(); best = std::min(best, (now() - t0) / reps); }
    return best;
}

static void fill(void *p, omc_dtype_t t, size_t n) {
    std::vector<double> v(n);
    for (size_t k = 0; k < n; k++) v[k] = ((double)((k * 2654435761u) % 100003) - 50000.0) / 3.7;
    omc_convert(OMC_FP64, v.data(), t, p, n, nullptr);
}

int main(int argc, char **argv) {
    std::vector<size_t> sizes;
    int threads = 1;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--threads") && a + 1 < argc) threads = atoi(argv[++a]);
        else if (!strcmp(argv[a], "--isa") && a + 1 < argc) omc_cpu_set_isa(argv[++a]);
        else sizes.push_back((size_t)atoll(argv[a]));
    }
    if (sizes.empty()) sizes = {16384, 262144, 4194304, 33554432};
    printf("# OmniCast %s CPU bench, path %s, threads %d\n", omc_version_string(), omc_cpu_isa(), threads);
    struct P { omc_dtype_t s, d; int sr; };
    const P pairs[] = {
        {OMC_FP64, OMC_FP32, 0}, {OMC_FP32, OMC_FP64, 0}, {OMC_FP32, OMC_FP16, 0}, {OMC_FP16, OMC_FP32, 0},
        {OMC_FP64, OMC_FP16, 0}, {OMC_FP16, OMC_FP64, 0}, {OMC_FP32, OMC_BF16, 0}, {OMC_BF16, OMC_FP32, 0},
        {OMC_FP64, OMC_BF16, 0}, {OMC_FP32, OMC_FP8_E4M3, 0}, {OMC_FP8_E4M3, OMC_FP32, 0},
        {OMC_FP64, OMC_FP8_E4M3, 0}, {OMC_FP16, OMC_FP8_E4M3, 0}, {OMC_FP32, OMC_FP8_E5M2, 0}, {OMC_FP8_E5M2, OMC_FP16, 0},
        {OMC_FP32, OMC_INT8, 0}, {OMC_INT8, OMC_FP32, 0}, {OMC_FP64, OMC_INT32, 0}, {OMC_INT32, OMC_FP64, 0},
        {OMC_FP64, OMC_FP32, 1}, {OMC_FP64, OMC_FP16, 1}, {OMC_FP32, OMC_FP16, 1},
    };
    printf("%-12s %10s %12s %10s %12s %12s %8s\n", "pair", "elements", "omc GB/s", "ns/elem", "memcpy GB/s", "legacy GB/s", "speedup");
    for (size_t n : sizes) {
        std::vector<char> A(n * 8 + 64), B(n * 8 + 64);
        for (const P &p : pairs) {
            const size_t ss = omc_dtype_size(p.s), ds = omc_dtype_size(p.d);
            fill(A.data(), p.s, n);
            omc_options_t o = {};
            o.num_threads = threads;
            if (p.sr) { o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 1; }
            const double bytes = (double)n * (ss + ds);
            double t = timeit([&] {
                if (p.sr) omc_convert_2d(p.s, A.data(), (int64_t)n, p.d, B.data(), (int64_t)n, (int64_t)n, 1, &o);
                else      omc_convert(p.s, A.data(), p.d, B.data(), n, &o);
            });
            /* memcpy of the same number of bytes (half read, half written) */
            double tm = timeit([&] { memcpy(B.data(), A.data(), (size_t)(bytes / 2)); });
            double tl = 0;
            const int nn = (int)n;
            if (!p.sr && p.s == OMC_FP64 && p.d == OMC_FP32) tl = timeit([&] { legacy_convert_d2s_binary_CPU((float *)B.data(), (double *)A.data(), nn, 1); });
            if (!p.sr && p.s == OMC_FP32 && p.d == OMC_FP64) tl = timeit([&] { legacy_convert_s2d_binary_CPU((double *)B.data(), (float *)A.data(), nn, 1); });
            if (!p.sr && p.s == OMC_FP32 && p.d == OMC_INT8) tl = timeit([&] { legacy_convert_s2i8_binary_CPU((int8_t *)B.data(), (float *)A.data(), nn, 1, nn); });
            if (!p.sr && p.s == OMC_INT8 && p.d == OMC_FP32) tl = timeit([&] { legacy_convert_8i2s_binary_CPU((float *)B.data(), (int8_t *)A.data(), nn, 1, nn); });
            if (!p.sr && p.s == OMC_FP64 && p.d == OMC_INT32) tl = timeit([&] { legacy_convert_d2i_binary_CPU((int *)B.data(), (double *)A.data(), nn, 1, nn); });
            if (!p.sr && p.s == OMC_INT32 && p.d == OMC_FP64) tl = timeit([&] { legacy_convert_i2d_binary_CPU((double *)B.data(), (int *)A.data(), nn, 1, nn); });
            if (p.sr && p.s == OMC_FP64 && p.d == OMC_FP32) tl = timeit([&] { legacy_double2float_round_CPU(nn, 1, (double *)A.data(), nn, (float *)B.data(), nn, 1); });
            char name[32];
            snprintf(name, sizeof name, "%s%s>%s", p.sr ? "SR:" : "", omc_dtype_name(p.s), omc_dtype_name(p.d));
            if (tl > 0)
                printf("%-12s %10zu %12.2f %10.3f %12.2f %12.2f %7.2fx\n", name, n, bytes / t * 1e-9, t / n * 1e9, bytes / tm * 1e-9, bytes / tl * 1e-9, tl / t);
            else
                printf("%-12s %10zu %12.2f %10.3f %12.2f %12s %8s\n", name, n, bytes / t * 1e-9, t / n * 1e9, bytes / tm * 1e-9, "-", "-");
        }
        /* in place, with the HiCMA-X unary loops as baseline */
        {
            fill(A.data(), OMC_FP64, n);
            double t = timeit([&] { omc_convert_inplace(OMC_FP64, OMC_FP32, A.data(), n, nullptr); fill(A.data(), OMC_FP64, 0); });
            double tl = timeit([&] { legacy_convert_d2s_unary_CPU((double *)A.data(), (int)n, 1); });
            printf("%-12s %10zu %12.2f %10.3f %12s %12.2f %7.2fx\n", "IP:fp64>fp32", n, n * 12.0 / t * 1e-9, t / n * 1e9, "-", n * 12.0 / tl * 1e-9, tl / t);
            t = timeit([&] { omc_convert_inplace(OMC_FP32, OMC_FP64, A.data(), n, nullptr); });
            tl = timeit([&] { legacy_convert_s2d_unary_CPU((float *)A.data(), (int)n, 1); });
            printf("%-12s %10zu %12.2f %10.3f %12s %12.2f %7.2fx\n", "IP:fp32>fp64", n, n * 12.0 / t * 1e-9, t / n * 1e9, "-", n * 12.0 / tl * 1e-9, tl / t);
        }
    }
    return 0;
}
