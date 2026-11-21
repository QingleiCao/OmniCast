/** @file omc_common.hpp  @brief Internal helpers shared by the backends. */
#ifndef OMC_COMMON_HPP
#define OMC_COMMON_HPP
#include "omnicast/omnicast.h"

namespace omc {
/** Largest finite value of a float type (FLT_MAX for FP64), 0 for integers */
float max_finite_of(omc_dtype_t t);
}
#endif
