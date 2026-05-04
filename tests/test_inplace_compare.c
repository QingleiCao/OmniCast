#include "omnicast/c_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(const char* msg) {
    fprintf(stderr, "test_inplace_compare_c failed: %s\n", msg);
    exit(1);
}

static double benchmark_ms(void (*fn)(void*), void* ctx, int iterations) {
    const clock_t start = clock();
    for (int i = 0; i < iterations; ++i) {
        fn(ctx);
    }
    const clock_t end = clock();
    return 1000.0 * (double)(end - start) / (double)CLOCKS_PER_SEC / (double)iterations;
}

typedef struct compare_ctx {
    size_t count;
    const double* fp64_in;
    const float* fp32_in;
    const omnicast_fp16_t* fp16_in;
    unsigned char* target_buf;  // max-size destination buffer (double sized)
    unsigned char* out_buf;     // out-of-place conversion buffer
    size_t capacity_bytes;
} compare_ctx;

static void oop_memcpy_64_to_32(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    omnicast_convert_fp64_to_fp32(c->fp64_in, (float*)c->out_buf, c->count);
    memcpy(c->target_buf, c->out_buf, c->count * sizeof(float));
}
static void inplace_64_to_32(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    memcpy(c->target_buf, c->fp64_in, c->count * sizeof(double));
    (void)omnicast_convert_inplace(c->target_buf, c->count, OMNICAST_DTYPE_FP64, OMNICAST_DTYPE_FP32, c->capacity_bytes);
}

static void oop_memcpy_64_to_16(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    omnicast_convert_fp64_to_fp16(c->fp64_in, (omnicast_fp16_t*)c->out_buf, c->count);
    memcpy(c->target_buf, c->out_buf, c->count * sizeof(omnicast_fp16_t));
}
static void inplace_64_to_16(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    memcpy(c->target_buf, c->fp64_in, c->count * sizeof(double));
    (void)omnicast_convert_inplace(c->target_buf, c->count, OMNICAST_DTYPE_FP64, OMNICAST_DTYPE_FP16, c->capacity_bytes);
}

static void oop_memcpy_32_to_64(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    omnicast_convert_fp32_to_fp64(c->fp32_in, (double*)c->out_buf, c->count);
    memcpy(c->target_buf, c->out_buf, c->count * sizeof(double));
}
static void inplace_32_to_64(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    memcpy(c->target_buf, c->fp32_in, c->count * sizeof(float));
    (void)omnicast_convert_inplace(c->target_buf, c->count, OMNICAST_DTYPE_FP32, OMNICAST_DTYPE_FP64, c->capacity_bytes);
}

static void oop_memcpy_32_to_16(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    omnicast_convert_fp32_to_fp16(c->fp32_in, (omnicast_fp16_t*)c->out_buf, c->count);
    memcpy(c->target_buf, c->out_buf, c->count * sizeof(omnicast_fp16_t));
}
static void inplace_32_to_16(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    memcpy(c->target_buf, c->fp32_in, c->count * sizeof(float));
    (void)omnicast_convert_inplace(c->target_buf, c->count, OMNICAST_DTYPE_FP32, OMNICAST_DTYPE_FP16, c->capacity_bytes);
}

static void oop_memcpy_16_to_32(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    omnicast_convert_fp16_to_fp32(c->fp16_in, (float*)c->out_buf, c->count);
    memcpy(c->target_buf, c->out_buf, c->count * sizeof(float));
}
static void inplace_16_to_32(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    memcpy(c->target_buf, c->fp16_in, c->count * sizeof(omnicast_fp16_t));
    (void)omnicast_convert_inplace(c->target_buf, c->count, OMNICAST_DTYPE_FP16, OMNICAST_DTYPE_FP32, c->capacity_bytes);
}

static void oop_memcpy_16_to_64(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    omnicast_convert_fp16_to_fp64(c->fp16_in, (double*)c->out_buf, c->count);
    memcpy(c->target_buf, c->out_buf, c->count * sizeof(double));
}
static void inplace_16_to_64(void* p) {
    compare_ctx* c = (compare_ctx*)p;
    memcpy(c->target_buf, c->fp16_in, c->count * sizeof(omnicast_fp16_t));
    (void)omnicast_convert_inplace(c->target_buf, c->count, OMNICAST_DTYPE_FP16, OMNICAST_DTYPE_FP64, c->capacity_bytes);
}

static void print_compare(const char* label, double oop_ms, double inplace_ms) {
    const double speedup = inplace_ms > 0.0 ? oop_ms / inplace_ms : 0.0;
    printf("%-12s oop+memcpy: %-10.4f ms/iteration inplace: %-10.4f ms/iteration speedup: %.2fx\n",
           label, oop_ms, inplace_ms, speedup);
}

int main(int argc, char** argv) {
    size_t count = 1u << 20u;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--size") == 0) {
            if (i + 1 >= argc) {
                fail("missing value for --size");
            }
            const unsigned long long parsed = strtoull(argv[++i], NULL, 10);
            if (parsed == 0ULL) {
                fail("invalid --size value");
            }
            count = (size_t)parsed;
        } else {
            fprintf(stderr, "usage: test_inplace_compare_c [--size N]\n");
            return 2;
        }
    }

    const int iterations = 30;
    const size_t cap_bytes = count * sizeof(double);

    double* fp64_in = (double*)malloc(count * sizeof(double));
    float* fp32_in = (float*)malloc(count * sizeof(float));
    omnicast_fp16_t* fp16_in = (omnicast_fp16_t*)malloc(count * sizeof(omnicast_fp16_t));
    unsigned char* target_buf = (unsigned char*)malloc(cap_bytes);
    unsigned char* out_buf = (unsigned char*)malloc(cap_bytes);

    if (!fp64_in || !fp32_in || !fp16_in || !target_buf || !out_buf) {
        fail("allocation failed");
    }

    srand(42);
    for (size_t i = 0; i < count; ++i) {
        const double v = ((double)(rand() % 200000) / 100.0) - 1000.0;
        fp64_in[i] = v;
        fp32_in[i] = (float)v;
    }
    omnicast_convert_fp32_to_fp16(fp32_in, fp16_in, count);

    if (omnicast_convert_inplace(target_buf, count, OMNICAST_DTYPE_FP16, OMNICAST_DTYPE_FP64,
                                 count * sizeof(omnicast_fp16_t)) != OMNICAST_INPLACE_INSUFFICIENT_CAPACITY) {
        fail("inplace capacity check failed");
    }

    compare_ctx ctx = {count, fp64_in, fp32_in, fp16_in, target_buf, out_buf, cap_bytes};

    const double t_oop_64_32 = benchmark_ms(oop_memcpy_64_to_32, &ctx, iterations);
    const double t_inp_64_32 = benchmark_ms(inplace_64_to_32, &ctx, iterations);

    const double t_oop_64_16 = benchmark_ms(oop_memcpy_64_to_16, &ctx, iterations);
    const double t_inp_64_16 = benchmark_ms(inplace_64_to_16, &ctx, iterations);

    const double t_oop_32_64 = benchmark_ms(oop_memcpy_32_to_64, &ctx, iterations);
    const double t_inp_32_64 = benchmark_ms(inplace_32_to_64, &ctx, iterations);

    const double t_oop_32_16 = benchmark_ms(oop_memcpy_32_to_16, &ctx, iterations);
    const double t_inp_32_16 = benchmark_ms(inplace_32_to_16, &ctx, iterations);

    const double t_oop_16_32 = benchmark_ms(oop_memcpy_16_to_32, &ctx, iterations);
    const double t_inp_16_32 = benchmark_ms(inplace_16_to_32, &ctx, iterations);

    const double t_oop_16_64 = benchmark_ms(oop_memcpy_16_to_64, &ctx, iterations);
    const double t_inp_16_64 = benchmark_ms(inplace_16_to_64, &ctx, iterations);

    printf("OmniCast in-place comparison C (%zu elements, avg over %d iterations)\n", count, iterations);
    printf("Active backend    : %s (SIMD available: %s)\n", omnicast_simd_backend(),
           omnicast_simd_available() ? "yes" : "no");
    print_compare("FP64->FP32", t_oop_64_32, t_inp_64_32);
    print_compare("FP64->FP16", t_oop_64_16, t_inp_64_16);
    print_compare("FP32->FP64", t_oop_32_64, t_inp_32_64);
    print_compare("FP32->FP16", t_oop_32_16, t_inp_32_16);
    print_compare("FP16->FP32", t_oop_16_32, t_inp_16_32);
    print_compare("FP16->FP64", t_oop_16_64, t_inp_16_64);

    free(out_buf);
    free(target_buf);
    free(fp16_in);
    free(fp32_in);
    free(fp64_in);
    return 0;
}
