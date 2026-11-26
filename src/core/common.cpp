/** @file common.cpp  @brief Type and status queries, error details. */
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include "omnicast/omnicast.h"
#include "omc_common.hpp"

namespace {
struct Info { const char *name; size_t size; int is_float; float maxf; };
const Info kInfo[OMC_NUM_DTYPES] = {
    {"fp64", 8, 1, 3.4028234663852886e38f},
    {"fp32", 4, 1, 3.4028234663852886e38f},
    {"fp16", 2, 1, 65504.0f},
    {"bf16", 2, 1, 3.3895313892515355e38f},
    {"e4m3", 1, 1, 448.0f},
    {"e5m2", 1, 1, 57344.0f},
    /* integers: the largest float not above the largest value (dynamic scaling target) */
    {"int8", 1, 0, 127.0f},  {"int16", 2, 0, 32767.0f},  {"int32", 4, 0, 2147483520.0f},
    {"int64", 8, 0, 9223371487098961920.0f},
    {"uint8", 1, 0, 255.0f}, {"uint16", 2, 0, 65535.0f}, {"uint32", 4, 0, 4294967040.0f},
    {"uint64", 8, 0, 18446742974197923840.0f},
};
struct Alias { const char *name; omc_dtype_t t; };
const Alias kAlias[] = {
    {"double", OMC_FP64}, {"d", OMC_FP64}, {"float64", OMC_FP64},
    {"float", OMC_FP32}, {"single", OMC_FP32}, {"s", OMC_FP32}, {"float32", OMC_FP32},
    {"half", OMC_FP16}, {"h", OMC_FP16}, {"float16", OMC_FP16},
    {"bfloat16", OMC_BF16},
    {"fp8", OMC_FP8_E4M3}, {"fp8_e4m3", OMC_FP8_E4M3}, {"e4m3fn", OMC_FP8_E4M3},
    {"fp8_e5m2", OMC_FP8_E5M2},
    {"i8", OMC_INT8}, {"i16", OMC_INT16}, {"int", OMC_INT32}, {"i32", OMC_INT32}, {"i", OMC_INT32},
    {"i64", OMC_INT64}, {"u8", OMC_UINT8}, {"u16", OMC_UINT16}, {"uint", OMC_UINT32},
    {"u32", OMC_UINT32}, {"u64", OMC_UINT64},
};
}

namespace omc {
float max_finite_of(omc_dtype_t t) { return ((unsigned)t < OMC_NUM_DTYPES) ? kInfo[t].maxf : 0.0f; }
bool sr_supported(omc_dtype_t s, omc_dtype_t d) {
    static const int rank[OMC_NUM_DTYPES] = {4, 3, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0};
    if ((unsigned)s >= OMC_NUM_DTYPES || (unsigned)d >= OMC_NUM_DTYPES) return false;
    return rank[d] > 0 && rank[d] < rank[s];
}

/* omc_last_error_detail: one per thread, zero-initialized (""), no constructor */
static thread_local char t_error_detail[512];

void set_error_detail(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(t_error_detail, sizeof t_error_detail, fmt, ap);
    va_end(ap);
}
}

extern "C" {

size_t omc_dtype_size(omc_dtype_t t) { return ((unsigned)t < OMC_NUM_DTYPES) ? kInfo[t].size : 0; }
const char *omc_dtype_name(omc_dtype_t t) { return ((unsigned)t < OMC_NUM_DTYPES) ? kInfo[t].name : "invalid"; }
int omc_dtype_is_float(omc_dtype_t t) { return ((unsigned)t < OMC_NUM_DTYPES) ? kInfo[t].is_float : 0; }

omc_dtype_t omc_dtype_from_name(const char *name) {
    if (!name) return OMC_NUM_DTYPES;
    for (int t = 0; t < OMC_NUM_DTYPES; t++)
        if (!strcasecmp(name, kInfo[t].name)) return (omc_dtype_t)t;
    for (const Alias &a : kAlias)
        if (!strcasecmp(name, a.name)) return a.t;
    return OMC_NUM_DTYPES;
}

const char *omc_status_string(omc_status_t s) {
    switch (s) {
    case OMC_SUCCESS:             return "success";
    case OMC_ERROR_INVALID_VALUE: return "invalid value";
    case OMC_ERROR_NOT_SUPPORTED: return "not supported";
    case OMC_ERROR_DEVICE:        return "device runtime error";
    case OMC_ERROR_ALLOCATION:    return "allocation failed";
    }
    return "unknown status";
}

const char *omc_last_error_detail(void) { return omc::t_error_detail; }

/* OMC_VERSION_STRING comes from project(OmniCast VERSION ...) in CMakeLists.txt */
const char *omc_version_string(void) { return OMC_VERSION_STRING; }

} /* extern "C" */
