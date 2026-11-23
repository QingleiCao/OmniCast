/**
 * @file bench_modes_cpu.cpp
 * @brief Every type pair on the host, out of place and in place, against
 * memcpy of the same bytes (bytes read + bytes written once), one thread.
 *
 * Usage: omnicast_bench_modes_cpu [--isa avx2] [--sr] [n ...]   (default 262144 33554432)
 * In-place calls are repeated on the same buffer (the kernels do not depend
 * on the values).
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <functional>
#include <algorithm>
#include "omnicast/omnicast.h"

/* best of 5 runs of >= 20 ms each, microseconds per call */
static double time_us(const std::function<void()> &f) {
    f();
    long iters = 1;
    for (;;) {
        const auto t0 = std::chrono::steady_clock::now();
        for (long k = 0; k < iters; k++) f();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (ms > 20.0 || iters >= (1L << 20)) {
            double best = ms * 1000.0 / iters;
            for (int r = 0; r < 4; r++) {
                const auto t1 = std::chrono::steady_clock::now();
                for (long k = 0; k < iters; k++) f();
                best = std::min(best, std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / iters);
            }
            return best;
        }
        iters *= 2;
    }
}

int main(int argc, char **argv) {
    std::vector<size_t> ns;
    bool sr = false;
    for (int a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--isa") && a + 1 < argc) { if (omc_cpu_set_isa(argv[++a])) { fprintf(stderr, "ISA not available\n"); return 2; } }
        else if (!strcmp(argv[a], "--sr")) sr = true;
        else ns.push_back((size_t)atoll(argv[a]));
    }
    if (ns.empty()) ns = {262144, 33554432};
    printf("# OmniCast %s host, %s%s, one thread\n", omc_version_string(), omc_cpu_isa(), sr ? ", stochastic rounding" : "");
    printf("%-6s %-6s %9s %9s %9s %9s %7s %7s\n", "src", "dst", "n", "copy(us)", "out(us)", "in(us)", "out%", "in%");
    omc_options_t o = {};
    o.flags = OMC_FLAG_SATURATE;
    if (sr) { o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 7; }
    for (size_t n : ns) {
        std::vector<unsigned char> A(n * 8 + 128, 0), B(n * 8 + 128, 0);
        unsigned char *a = A.data() + ((64 - ((uintptr_t)A.data() & 63)) & 63);
        unsigned char *b = B.data() + ((64 - ((uintptr_t)B.data() & 63)) & 63);
        for (size_t k = 0; k < n * 8; k++) a[k] = (unsigned char)(k * 2654435761u >> 13);
        std::vector<double> copy_t(81, 0.0);
        for (int st = 0; st < OMC_NUM_DTYPES; st++)
            for (int dt = 0; dt < OMC_NUM_DTYPES; dt++) {
                const omc_dtype_t S = (omc_dtype_t)st, D = (omc_dtype_t)dt;
                unsigned char one[16];
                if (omc_convert(S, a, D, one, 1, &o) != OMC_SUCCESS) continue;
                const size_t si = omc_dtype_size(S), so = omc_dtype_size(D);
                const size_t bytes = n * (si + so);
                double &tc = copy_t[si * 9 + so];
                if (tc == 0.0) tc = time_us([&] { memcpy(b, a, bytes / 2); });
                const double to = time_us([&] { omc_convert(S, a, D, b, n, &o); });
                const double ti = time_us([&] { omc_convert_inplace(S, D, a, n, &o); });
                printf("%-6s %-6s %9zu %9.1f %9.1f %9.1f %6.0f%% %6.0f%%\n", omc_dtype_name(S), omc_dtype_name(D), n, tc, to, ti,
                       100.0 * tc / to, 100.0 * tc / ti);
                fflush(stdout);
            }
    }
    return 0;
}
