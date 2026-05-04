#include "omnicast/c_api.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void check(int condition, const char* message) {
    if (!condition) {
        fprintf(stderr, "C test failed: %s\n", message);
        exit(1);
    }
}

static double benchmark_ms(void (*fn)(void*), void* ctx, int iterations) {
    const clock_t start = clock();
    for (int i = 0; i < iterations; ++i) {
        fn(ctx);
    }
    const clock_t end = clock();
    const double elapsed = (double)(end - start) / (double)CLOCKS_PER_SEC;
    return (elapsed * 1000.0) / (double)iterations;
}

static void print_result_with_speedup(const char* label, double scalar_ms, double auto_ms) {
    const double speedup = auto_ms > 0.0 ? scalar_ms / auto_ms : 0.0;
    printf("%-12s scalar: %-10.4f ms/iteration auto: %-10.4f ms/iteration speedup: %.2fx\n",
           label, scalar_ms, auto_ms, speedup);
}

static double rel_f_norm_fp64_vs_fp32(const double* ref, const float* out, size_t count) {
    double diff_accum = 0.0;
    double ref_accum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = (double)out[i] - ref[i];
        diff_accum += diff * diff;
        ref_accum += ref[i] * ref[i];
    }
    return ref_accum > 0.0 ? sqrt(diff_accum) / sqrt(ref_accum) : 0.0;
}

static double rel_f_norm_fp64_vs_fp16(const double* ref, const omnicast_fp16_t* out, size_t count) {
    double diff_accum = 0.0;
    double ref_accum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = omnicast_fp16_to_fp64(out[i]) - ref[i];
        diff_accum += diff * diff;
        ref_accum += ref[i] * ref[i];
    }
    return ref_accum > 0.0 ? sqrt(diff_accum) / sqrt(ref_accum) : 0.0;
}

static double rel_f_norm_fp64_vs_fp64(const double* ref, const double* out, size_t count) {
    double diff_accum = 0.0;
    double ref_accum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = out[i] - ref[i];
        diff_accum += diff * diff;
        ref_accum += ref[i] * ref[i];
    }
    return ref_accum > 0.0 ? sqrt(diff_accum) / sqrt(ref_accum) : 0.0;
}

static void print_f_norm(const char* label, double error) {
    printf("%-12s rel F-norm (||A-A'||/||A||): %.6e\n", label, error);
}

typedef struct bench_ctx {
    const double* fp64_in;
    const float* fp32_in;
    const omnicast_fp16_t* fp16_in;
    float* fp32_out;
    double* fp64_out;
    omnicast_fp16_t* fp16_out;
    size_t count;
    unsigned char* inplace;
    size_t inplace_bytes;
} bench_ctx;

static void run_64_to_32_scalar(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp64_to_fp32_scalar(c->fp64_in, c->fp32_out, c->count);
}
static void run_64_to_32_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp64_to_fp32(c->fp64_in, c->fp32_out, c->count);
}
static void run_64_to_16_scalar(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp64_to_fp16_scalar(c->fp64_in, c->fp16_out, c->count);
}
static void run_64_to_16_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp64_to_fp16(c->fp64_in, c->fp16_out, c->count);
}
static void run_32_to_64_scalar(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp32_to_fp64_scalar(c->fp32_in, c->fp64_out, c->count);
}
static void run_32_to_64_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp32_to_fp64(c->fp32_in, c->fp64_out, c->count);
}
static void run_32_to_16_scalar(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp32_to_fp16_scalar(c->fp32_in, c->fp16_out, c->count);
}
static void run_32_to_16_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp32_to_fp16(c->fp32_in, c->fp16_out, c->count);
}
static void run_16_to_32_scalar(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp16_to_fp32_scalar(c->fp16_in, c->fp32_out, c->count);
}
static void run_16_to_32_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp16_to_fp32(c->fp16_in, c->fp32_out, c->count);
}
static void run_16_to_64_scalar(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp16_to_fp64_scalar(c->fp16_in, c->fp64_out, c->count);
}
static void run_16_to_64_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    omnicast_convert_fp16_to_fp64(c->fp16_in, c->fp64_out, c->count);
}

static void run_inplace_64_to_32_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    memcpy(c->inplace, c->fp64_in, c->count * sizeof(double));
    (void)omnicast_convert_inplace(
        c->inplace, c->count, OMNICAST_DTYPE_FP64, OMNICAST_DTYPE_FP32, c->inplace_bytes);
}

static void run_inplace_64_to_16_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    memcpy(c->inplace, c->fp64_in, c->count * sizeof(double));
    (void)omnicast_convert_inplace(
        c->inplace, c->count, OMNICAST_DTYPE_FP64, OMNICAST_DTYPE_FP16, c->inplace_bytes);
}

static void run_inplace_32_to_64_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    memcpy(c->inplace, c->fp32_in, c->count * sizeof(float));
    (void)omnicast_convert_inplace(
        c->inplace, c->count, OMNICAST_DTYPE_FP32, OMNICAST_DTYPE_FP64, c->inplace_bytes);
}

static void run_inplace_32_to_16_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    memcpy(c->inplace, c->fp32_in, c->count * sizeof(float));
    (void)omnicast_convert_inplace(
        c->inplace, c->count, OMNICAST_DTYPE_FP32, OMNICAST_DTYPE_FP16, c->inplace_bytes);
}

static void run_inplace_16_to_32_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    memcpy(c->inplace, c->fp16_in, c->count * sizeof(omnicast_fp16_t));
    (void)omnicast_convert_inplace(
        c->inplace, c->count, OMNICAST_DTYPE_FP16, OMNICAST_DTYPE_FP32, c->inplace_bytes);
}

static void run_inplace_16_to_64_auto(void* p) {
    bench_ctx* c = (bench_ctx*)p;
    memcpy(c->inplace, c->fp16_in, c->count * sizeof(omnicast_fp16_t));
    (void)omnicast_convert_inplace(
        c->inplace, c->count, OMNICAST_DTYPE_FP16, OMNICAST_DTYPE_FP64, c->inplace_bytes);
}

int main(int argc, char** argv) {
    int check_error = 0;
    int run_inplace = 0;
    size_t count = 1u << 20u;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--check") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --check (use 0 or 1)\n");
                return 2;
            }
            const int value = atoi(argv[++i]);
            if (value != 0 && value != 1) {
                fprintf(stderr, "invalid --check value (use 0 or 1)\n");
                return 2;
            }
            check_error = value;
        } else if (strcmp(argv[i], "--inplace") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --inplace (use 0 or 1)\n");
                return 2;
            }
            const int value = atoi(argv[++i]);
            if (value != 0 && value != 1) {
                fprintf(stderr, "invalid --inplace value (use 0 or 1)\n");
                return 2;
            }
            run_inplace = value;
        } else if (strcmp(argv[i], "--size") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for --size\n");
                return 2;
            }
            const unsigned long long parsed = strtoull(argv[++i], NULL, 10);
            if (parsed == 0ULL) {
                fprintf(stderr, "invalid --size value (must be > 0)\n");
                return 2;
            }
            count = (size_t)parsed;
        } else {
            fprintf(stderr, "unknown argument: %s\n", argv[i]);
            fprintf(stderr, "usage: test_casts_c [--check 0|1] [--inplace 0|1] [--size N]\n");
            return 2;
        }
    }
    const int iterations = 30;

    double* fp64_in = (double*)malloc(count * sizeof(double));
    float* fp32_in = (float*)malloc(count * sizeof(float));
    omnicast_fp16_t* fp16_in = (omnicast_fp16_t*)malloc(count * sizeof(omnicast_fp16_t));
    float* fp32_out = (float*)malloc(count * sizeof(float));
    double* fp64_out = (double*)malloc(count * sizeof(double));
    omnicast_fp16_t* fp16_out = (omnicast_fp16_t*)malloc(count * sizeof(omnicast_fp16_t));
    unsigned char* inplace = (unsigned char*)malloc(count * sizeof(double));

    check(fp64_in && fp32_in && fp16_in && fp32_out && fp64_out && fp16_out && inplace, "allocation failed");

    srand(42);
    for (size_t i = 0; i < count; ++i) {
        const double v = ((double)(rand() % 200000) / 100.0) - 1000.0;
        fp64_in[i] = v;
        fp32_in[i] = (float)v;
    }
    omnicast_convert_fp32_to_fp16(fp32_in, fp16_in, count);

    bench_ctx ctx = {
        fp64_in, fp32_in, fp16_in, fp32_out, fp64_out, fp16_out, count, inplace, count * sizeof(double)};

    double t_64_to_32_scalar = 0.0, t_64_to_32_auto = 0.0;
    double t_64_to_16_scalar = 0.0, t_64_to_16_auto = 0.0;
    double t_32_to_64_scalar = 0.0, t_32_to_64_auto = 0.0;
    double t_32_to_16_scalar = 0.0, t_32_to_16_auto = 0.0;
    double t_16_to_32_scalar = 0.0, t_16_to_32_auto = 0.0;
    double t_16_to_64_scalar = 0.0, t_16_to_64_auto = 0.0;
    double t_inplace_64_to_32 = 0.0, t_inplace_64_to_16 = 0.0;
    double t_inplace_32_to_64 = 0.0, t_inplace_32_to_16 = 0.0;
    double t_inplace_16_to_32 = 0.0, t_inplace_16_to_64 = 0.0;

    {
        const float ref32 = omnicast_fp64_to_fp32(123.5);
        const omnicast_fp16_t ref16 = omnicast_fp64_to_fp16(123.5);
        const double back64 = omnicast_fp16_to_fp64(ref16);
        check(fabs((double)ref32 - 123.5) < 1e-6, "basic fp64->fp32");
        check(fabs(back64 - 123.5) < 0.1, "basic fp16->fp64");
    }

    if (run_inplace) {
        memcpy(inplace, fp64_in, count * sizeof(double));
        check(omnicast_convert_inplace(
                  inplace, count, OMNICAST_DTYPE_FP64, OMNICAST_DTYPE_FP32, count * sizeof(double)) ==
                  OMNICAST_INPLACE_SUCCESS,
              "inplace fp64->fp32");
        check(fabs((double)((float*)inplace)[17] - (double)((float)fp64_in[17])) < 1e-5,
              "inplace fp64->fp32 value");

        memcpy(inplace, fp16_in, count * sizeof(omnicast_fp16_t));
        check(omnicast_convert_inplace(
                  inplace, count, OMNICAST_DTYPE_FP16, OMNICAST_DTYPE_FP64, count * sizeof(double)) ==
                  OMNICAST_INPLACE_SUCCESS,
              "inplace fp16->fp64");
        check(fabs(((double*)inplace)[17] - omnicast_fp16_to_fp64(fp16_in[17])) < 1e-12,
              "inplace fp16->fp64 value");
        check(omnicast_convert_inplace(
                  inplace, count, OMNICAST_DTYPE_FP16, OMNICAST_DTYPE_FP64, count * sizeof(omnicast_fp16_t)) ==
                  OMNICAST_INPLACE_INSUFFICIENT_CAPACITY,
              "inplace insufficient capacity");

        t_inplace_64_to_32 = benchmark_ms(run_inplace_64_to_32_auto, &ctx, iterations);
        t_inplace_64_to_16 = benchmark_ms(run_inplace_64_to_16_auto, &ctx, iterations);
        t_inplace_32_to_64 = benchmark_ms(run_inplace_32_to_64_auto, &ctx, iterations);
        t_inplace_32_to_16 = benchmark_ms(run_inplace_32_to_16_auto, &ctx, iterations);
        t_inplace_16_to_32 = benchmark_ms(run_inplace_16_to_32_auto, &ctx, iterations);
        t_inplace_16_to_64 = benchmark_ms(run_inplace_16_to_64_auto, &ctx, iterations);

        printf("OmniCast CPU conversion benchmark (%zu elements, avg over %d iterations)\n",
               count, iterations);
        printf("Active backend    : %s (SIMD available: %s)\n", omnicast_simd_backend(),
               omnicast_simd_available() ? "yes" : "no");
        printf("%-12s auto: %-10.4f ms/iteration\n", "InP64->P32", t_inplace_64_to_32);
        printf("%-12s auto: %-10.4f ms/iteration\n", "InP64->P16", t_inplace_64_to_16);
        printf("%-12s auto: %-10.4f ms/iteration\n", "InP32->P64", t_inplace_32_to_64);
        printf("%-12s auto: %-10.4f ms/iteration\n", "InP32->P16", t_inplace_32_to_16);
        printf("%-12s auto: %-10.4f ms/iteration\n", "InP16->P32", t_inplace_16_to_32);
        printf("%-12s auto: %-10.4f ms/iteration\n", "InP16->P64", t_inplace_16_to_64);

        if (check_error) {
            memcpy(inplace, fp64_in, count * sizeof(double));
            (void)omnicast_convert_inplace(inplace, count, OMNICAST_DTYPE_FP64, OMNICAST_DTYPE_FP32, count * sizeof(double));
            const double e_64_to_32 = rel_f_norm_fp64_vs_fp32(fp64_in, (float*)inplace, count);

            memcpy(inplace, fp64_in, count * sizeof(double));
            (void)omnicast_convert_inplace(inplace, count, OMNICAST_DTYPE_FP64, OMNICAST_DTYPE_FP16, count * sizeof(double));
            const double e_64_to_16 = rel_f_norm_fp64_vs_fp16(fp64_in, (omnicast_fp16_t*)inplace, count);

            memcpy(inplace, fp32_in, count * sizeof(float));
            (void)omnicast_convert_inplace(inplace, count, OMNICAST_DTYPE_FP32, OMNICAST_DTYPE_FP64, count * sizeof(double));
            const double e_32_to_64 = rel_f_norm_fp64_vs_fp64(fp64_in, (double*)inplace, count);

            memcpy(inplace, fp32_in, count * sizeof(float));
            (void)omnicast_convert_inplace(inplace, count, OMNICAST_DTYPE_FP32, OMNICAST_DTYPE_FP16, count * sizeof(double));
            const double e_32_to_16 = rel_f_norm_fp64_vs_fp16(fp64_in, (omnicast_fp16_t*)inplace, count);

            memcpy(inplace, fp16_in, count * sizeof(omnicast_fp16_t));
            (void)omnicast_convert_inplace(inplace, count, OMNICAST_DTYPE_FP16, OMNICAST_DTYPE_FP32, count * sizeof(double));
            const double e_16_to_32 = rel_f_norm_fp64_vs_fp32(fp64_in, (float*)inplace, count);

            memcpy(inplace, fp16_in, count * sizeof(omnicast_fp16_t));
            (void)omnicast_convert_inplace(inplace, count, OMNICAST_DTYPE_FP16, OMNICAST_DTYPE_FP64, count * sizeof(double));
            const double e_16_to_64 = rel_f_norm_fp64_vs_fp64(fp64_in, (double*)inplace, count);

            print_f_norm("FP64->FP32", e_64_to_32);
            print_f_norm("FP64->FP16", e_64_to_16);
            print_f_norm("FP32->FP64", e_32_to_64);
            print_f_norm("FP32->FP16", e_32_to_16);
            print_f_norm("FP16->FP32", e_16_to_32);
            print_f_norm("FP16->FP64", e_16_to_64);
        } else {
            printf("Error check      : disabled (use --check 1)\n");
        }
    } else {
        t_64_to_32_scalar = benchmark_ms(run_64_to_32_scalar, &ctx, iterations);
        t_64_to_32_auto = benchmark_ms(run_64_to_32_auto, &ctx, iterations);
        t_64_to_16_scalar = benchmark_ms(run_64_to_16_scalar, &ctx, iterations);
        t_64_to_16_auto = benchmark_ms(run_64_to_16_auto, &ctx, iterations);
        t_32_to_64_scalar = benchmark_ms(run_32_to_64_scalar, &ctx, iterations);
        t_32_to_64_auto = benchmark_ms(run_32_to_64_auto, &ctx, iterations);
        t_32_to_16_scalar = benchmark_ms(run_32_to_16_scalar, &ctx, iterations);
        t_32_to_16_auto = benchmark_ms(run_32_to_16_auto, &ctx, iterations);
        t_16_to_32_scalar = benchmark_ms(run_16_to_32_scalar, &ctx, iterations);
        t_16_to_32_auto = benchmark_ms(run_16_to_32_auto, &ctx, iterations);
        t_16_to_64_scalar = benchmark_ms(run_16_to_64_scalar, &ctx, iterations);
        t_16_to_64_auto = benchmark_ms(run_16_to_64_auto, &ctx, iterations);

        printf("OmniCast CPU conversion benchmark (%zu elements, avg over %d iterations)\n",
               count, iterations);
        printf("Active backend    : %s (SIMD available: %s)\n", omnicast_simd_backend(),
               omnicast_simd_available() ? "yes" : "no");
        print_result_with_speedup("FP64->FP32", t_64_to_32_scalar, t_64_to_32_auto);
        print_result_with_speedup("FP64->FP16", t_64_to_16_scalar, t_64_to_16_auto);
        print_result_with_speedup("FP32->FP64", t_32_to_64_scalar, t_32_to_64_auto);
        print_result_with_speedup("FP32->FP16", t_32_to_16_scalar, t_32_to_16_auto);
        print_result_with_speedup("FP16->FP32", t_16_to_32_scalar, t_16_to_32_auto);
        print_result_with_speedup("FP16->FP64", t_16_to_64_scalar, t_16_to_64_auto);
        printf("Inplace check    : disabled in this mode (use --inplace 1)\n");

        if (check_error) {
            omnicast_convert_fp64_to_fp32(fp64_in, fp32_out, count);
            const double e_64_to_32 = rel_f_norm_fp64_vs_fp32(fp64_in, fp32_out, count);
            omnicast_convert_fp64_to_fp16(fp64_in, fp16_out, count);
            const double e_64_to_16 = rel_f_norm_fp64_vs_fp16(fp64_in, fp16_out, count);
            omnicast_convert_fp32_to_fp64(fp32_in, fp64_out, count);
            const double e_32_to_64 = rel_f_norm_fp64_vs_fp64(fp64_in, fp64_out, count);
            omnicast_convert_fp32_to_fp16(fp32_in, fp16_out, count);
            const double e_32_to_16 = rel_f_norm_fp64_vs_fp16(fp64_in, fp16_out, count);
            omnicast_convert_fp16_to_fp32(fp16_in, fp32_out, count);
            const double e_16_to_32 = rel_f_norm_fp64_vs_fp32(fp64_in, fp32_out, count);
            omnicast_convert_fp16_to_fp64(fp16_in, fp64_out, count);
            const double e_16_to_64 = rel_f_norm_fp64_vs_fp64(fp64_in, fp64_out, count);
            print_f_norm("FP64->FP32", e_64_to_32);
            print_f_norm("FP64->FP16", e_64_to_16);
            print_f_norm("FP32->FP64", e_32_to_64);
            print_f_norm("FP32->FP16", e_32_to_16);
            print_f_norm("FP16->FP32", e_16_to_32);
            print_f_norm("FP16->FP64", e_16_to_64);
        } else {
            printf("Error check      : disabled (use --check 1)\n");
        }
    }

    free(inplace);
    free(fp16_out);
    free(fp64_out);
    free(fp32_out);
    free(fp16_in);
    free(fp32_in);
    free(fp64_in);
    return 0;
}
