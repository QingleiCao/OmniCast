/** @file omc_common.hpp  @brief Internal helpers shared by the backends. */
#ifndef OMC_COMMON_HPP
#define OMC_COMMON_HPP
#include "omnicast/omnicast.h"

namespace omc {
/** Largest finite value of a float type (FLT_MAX for FP64); for integers the
 *  largest float not above the largest value (the dynamic scaling target) */
float max_finite_of(omc_dtype_t t);
/** Pairs with stochastic rounding: float source, float target of lower
 *  precision (FP64 > FP32 > FP16, BF16 > FP8); mirrors sr_ok<D, S> */
bool sr_supported(omc_dtype_t src, omc_dtype_t dst);
}
#endif
