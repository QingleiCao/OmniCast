/* HiCMA-X 0.2 CPU conversion loops (hicma_parsec_datatype_convert.c,
 * hicma_parsec_stochastic_rounding.c), verbatim, compiled like HiCMA-X
 * (C, -O3, no -march) as the baseline of the CPU benchmark.
 *
 * HiCMA-X code: Copyright (c) 2023-2025 SLU, MIT, Nvidia; 2018-2025 KAUST;
 * 2018-2023 The University of Tennessee and The University of Tennessee Research
 * Foundation. BSD 3-Clause; see the section "Code from HiCMA-X" in LICENSE. */
#include <stdint.h>
#include <string.h>
#include <math.h>

void legacy_convert_d2s_binary_CPU(float *_target, double *_source, int mb, int nb) {
    float *target = (float *)_target; double *source = (double *)_source;
    for (int j = 0; j < nb; j++) for (int i = 0; i < mb; i++) target[j*mb+i] = (float)source[j*mb+i];
}
void legacy_convert_s2d_binary_CPU(double *_target, float *_source, int mb, int nb) {
    double *target = (double *)_target; float *source = (float *)_source;
    for (int j = 0; j < nb; j++) for (int i = 0; i < mb; i++) target[j*mb+i] = (double)source[j*mb+i];
}
void legacy_convert_d2s_unary_CPU(double *data, int mb, int nb) {
    float *data_s = (float *)data;
    for (int j = 0; j < nb; j++) for (int i = 0; i < mb; i++) data_s[j*mb+i] = (float)data[j*mb+i];
}
void legacy_convert_s2d_unary_CPU(float *data, int mb, int nb) {
    double *data_d = (double *)data;
    for (int j = nb-1; j >= 0; j--) for (int i = mb-1; i >= 0; i--) data_d[j*mb+i] = (double)data[j*mb+i];
}
void legacy_convert_s2i8_binary_CPU(int8_t *target, float *source, int mb, int nb, int lda) {
    for (int j = 0; j < nb; j++) for (int i = 0; i < mb; i++) target[j*lda+i] = (int8_t)source[j*lda+i];
}
void legacy_convert_8i2s_binary_CPU(float *target, int8_t *source, int mb, int nb, int lda) {
    for (int j = nb-1; j >= 0; j--) for (int i = mb-1; i >= 0; i--) target[j*lda+i] = (float)source[j*lda+i];
}
void legacy_convert_d2i_binary_CPU(int *target, double *source, int mb, int nb, int lda) {
    for (int j = 0; j < nb; j++) for (int i = 0; i < mb; i++) target[j*lda+i] = (int)source[j*lda+i];
}
void legacy_convert_i2d_binary_CPU(double *target, int *source, int mb, int nb, int lda) {
    for (int j = nb-1; j >= 0; j--) for (int i = mb-1; i >= 0; i--) target[j*lda+i] = (double)source[j*lda+i];
}

static inline uint64_t sm64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}
static inline float u01(int i, int j, uint64_t seed, double value) {
    uint64_t bits, pos = (uint64_t)(uint32_t)i | ((uint64_t)(uint32_t)j << 32);
    memcpy(&bits, &value, sizeof(bits));
    uint64_t rnd = sm64(bits ^ sm64(pos ^ seed));
    return (float)(rnd >> 40) * (1.0f / 16777216.0f);
}
static inline float sr32(double value, float random) {
    float nearest = (float)value;
    if (!isfinite(value) || !isfinite(nearest)) return nearest;
    double a = fabs(value); float n = fabsf(nearest); float lower, upper;
    if (a > (double)n) { lower = n; upper = nextafterf(n, INFINITY); }
    else if (a < (double)n) { lower = nextafterf(n, 0.0f); upper = n; }
    else return nearest;
    if (isinf(upper)) return nearest;
    double p_lower = ((double)upper - a) / ((double)upper - (double)lower);
    float r = (random < p_lower) ? lower : upper;
    return (value < 0.0) ? -r : r;
}
void legacy_double2float_round_CPU(int nrows, int ncols, const double *D, int ldd, float *F, int ldf, uint64_t seed) {
    for (int j = 0; j < ncols; ++j) for (int i = 0; i < nrows; ++i) {
        double value = D[j * ldd + i];
        F[j * ldf + i] = sr32(value, u01(i, j, seed, value));
    }
}
