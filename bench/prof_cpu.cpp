/**
 * @file prof_cpu.cpp
 * @brief One host conversion repeated, for `perf stat` / `perf record`:
 *
 * omnicast_prof_cpu --src fp64 --dst fp32 [--n 262144] [--inplace] [--sat]
 *                   [--sr] [--rne] [--scale s] [--isa avx2] [--reps R]
 *                   [--threads T] [--copy]
 * Prints the time per call and GB/s (bytes read + written once).
 */
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "omnicast/omnicast.h"

int main(int argc, char **argv) {
    omc_dtype_t st = OMC_FP64, dt = OMC_FP32;
    size_t n = 262144;
    long reps = 0;
    int inplace = 0, copy = 0;
    omc_options_t o = {};
    for (int a = 1; a < argc; a++) {
        const char *s = argv[a];
        auto next = [&]() -> const char * { return (a + 1 < argc) ? argv[++a] : "0"; };
        if (!strcmp(s, "--src")) st = omc_dtype_from_name(next());
        else if (!strcmp(s, "--dst")) dt = omc_dtype_from_name(next());
        else if (!strcmp(s, "--n")) n = (size_t)atoll(next());
        else if (!strcmp(s, "--reps")) reps = atol(next());
        else if (!strcmp(s, "--inplace")) inplace = 1;
        else if (!strcmp(s, "--copy")) copy = 1;
        else if (!strcmp(s, "--sat")) o.flags |= OMC_FLAG_SATURATE;
        else if (!strcmp(s, "--sr")) { o.rounding = OMC_ROUND_STOCHASTIC; o.seed = 12345; }
        else if (!strcmp(s, "--rne")) o.rounding = OMC_ROUND_NEAREST_EVEN;
        else if (!strcmp(s, "--scale")) { o.scale_mode = OMC_SCALE_HOST; o.scale = (float)atof(next()); }
        else if (!strcmp(s, "--isa")) { if (omc_cpu_set_isa(next())) { fprintf(stderr, "ISA not available\n"); return 2; } }
        else if (!strcmp(s, "--threads")) o.num_threads = atoi(next());
        else { fprintf(stderr, "unknown option %s\n", s); return 2; }
    }
    const size_t si = omc_dtype_size(st), so = omc_dtype_size(dt);
    const size_t big = n * (si > so ? si : so) + 64;
    std::vector<unsigned char> A(big + 64), B(big + 64), base(big + 64);
    unsigned char *a = A.data() + ((64 - ((uintptr_t)A.data() & 63)) & 63);
    unsigned char *b = B.data() + ((64 - ((uintptr_t)B.data() & 63)) & 63);
    {
        std::vector<double> h(n);
        for (size_t k = 0; k < n; k++) h[k] = ((double)((k * 2654435761u) % 1000003) - 500000.0) / 7777.0;
        omc_options_t sat = {}; sat.flags = OMC_FLAG_SATURATE;
        omc_convert(OMC_FP64, h.data(), st, a, n, &sat);
        memcpy(base.data(), a, n * si);
    }
    const double bytes = (double)n * (si + so);
    if (reps <= 0) reps = (long)(4e10 / bytes) + 1;       /* about 2-4 s */
    auto call = [&] {
        if (copy) { memcpy(b, a, (size_t)(bytes / 2)); return; }
        if (inplace) { omc_convert_inplace(st, dt, a, n, &o); return; }
        omc_convert(st, a, dt, b, n, &o);
    };
    call();
    double best = 1e30;
    const auto t0 = std::chrono::steady_clock::now();
    for (long r = 0; r < reps; r++) {
        if (inplace) memcpy(a, base.data(), n * si);    /* restore (also counted: see --copy) */
        const auto c0 = std::chrono::steady_clock::now();
        call();
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - c0).count();
        if (us < best) best = us;
    }
    const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%s %s %s>%s n=%zu%s: best %.2f us (%.2f GB/s, %.3f ns/element), %ld calls in %.2f s\n", omc_cpu_isa(),
           copy ? "copy" : (inplace ? "inplace" : "out"), omc_dtype_name(st), omc_dtype_name(dt), n,
           o.rounding == OMC_ROUND_STOCHASTIC ? " sr" : "", best, bytes / (best * 1e-6) * 1e-9, best * 1e3 / (double)n,
           reps, total);
    return 0;
}
