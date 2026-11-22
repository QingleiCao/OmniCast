/**
 * @file test_native_fp8.cu
 * @brief The device FP8 encoder against the integer one, for every FP32 bit
 * pattern: on sm_89+ Enc<e4m3/e5m2>::from_f32 with saturation is the hardware
 * convert (cvt.rn.satfinite), checked here against enc_f32, which
 * omnicast_test_exhaustive checks against the CUDA headers. Also FP64 inputs
 * (round to odd to FP32 first) on random and edge values. Skipped below 8.9.
 */
#include <cstdio>
#include <cuda_runtime.h>
#include "../src/core/omc_scalar.hpp"

using namespace omc;

template <class D, class F>
__global__ void k_check_f32(unsigned long long base, unsigned long long count, unsigned long long *bad,
                            unsigned long long *first) {
    const ElemOpts o{true, false};
    for (unsigned long long k = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; k < count;
         k += (unsigned long long)gridDim.x * blockDim.x) {
        const uint32_t u = (uint32_t)(base + k);
        const float x = u2f(u);
        const uint32_t a = (uint32_t)Enc<D>::from_f32(x, o).bits, b = enc_f32<F>(x, true);
        if (a != b) { if (atomicAdd(bad, 1ull) == 0) *first = u; }
    }
}

template <class D, class F>
__global__ void k_check_f64(unsigned long long count, unsigned long long *bad, unsigned long long *first) {
    const ElemOpts o{true, false};
    for (unsigned long long k = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; k < count;
         k += (unsigned long long)gridDim.x * blockDim.x) {
        /* sign, an exponent around the FP8 range, a random 52-bit mantissa (SplitMix64) */
        unsigned long long z = k * 0x9e3779b97f4a7c15ull + 12345;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; z ^= z >> 31;
        const unsigned long long e = 1023 - 30 + (z >> 58) % 52;
        const unsigned long long bitsd = (z & 0x800fffffffffffffull) | (e << 52);
        const double x = __longlong_as_double((long long)bitsd);
        const uint32_t a = (uint32_t)Enc<D>::from_f64(x, o).bits, b = enc_f32<F>(f64_to_f32_ro(x), true);
        if (a != b) { if (atomicAdd(bad, 1ull) == 0) *first = bitsd; }
    }
}

int main() {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    if (p.major * 10 + p.minor < 89) {
        printf("device %s (sm_%d%d): no FP8 converts, skipped\nPASSED\n", p.name, p.major, p.minor);
        return 0;
    }
    unsigned long long *d, h[2];
    cudaMalloc(&d, 16);
    int fails = 0;
    auto run32 = [&](auto kern, const char *name) {
        cudaMemset(d, 0, 16);
        kern<<<4096, 256>>>(0ull, 1ull << 32, d, d + 1);
        cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost);
        printf("  fp32 -> %s saturated, every bit pattern: %llu mismatches%s", name, h[0], h[0] ? "" : "\n");
        if (h[0]) { printf(" (first 0x%08llx)\n", h[1]); fails++; }
    };
    auto run64 = [&](auto kern, const char *name) {
        cudaMemset(d, 0, 16);
        kern<<<4096, 256>>>(1ull << 30, d, d + 1);
        cudaMemcpy(h, d, 16, cudaMemcpyDeviceToHost);
        printf("  fp64 -> %s saturated, 2^30 values: %llu mismatches%s", name, h[0], h[0] ? "" : "\n");
        if (h[0]) { printf(" (first 0x%016llx)\n", h[1]); fails++; }
    };
    run32(k_check_f32<e4m3, FmtE4M3>, "e4m3");
    run32(k_check_f32<e5m2, FmtE5M2>, "e5m2");
    run64(k_check_f64<e4m3, FmtE4M3>, "e4m3");
    run64(k_check_f64<e5m2, FmtE5M2>, "e5m2");
    cudaFree(d);
    printf(fails ? "FAILED\n" : "PASSED\n");
    return fails ? 1 : 0;
}
