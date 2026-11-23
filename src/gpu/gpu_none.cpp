/** @file gpu_none.cpp  @brief GPU entry points of a CPU-only build. */
#include "omnicast/omnicast.h"

extern "C" {
const char *omc_gpu_backend(void) { return "none"; }
omc_status_t omc_gpu_convert(omc_dtype_t, const void *, omc_dtype_t, void *, size_t,
                             const omc_options_t *, omc_stream_t) { return OMC_ERROR_NOT_SUPPORTED; }
omc_status_t omc_gpu_convert_2d(omc_dtype_t, const void *, int64_t, omc_dtype_t, void *, int64_t,
                                int64_t, int64_t, const omc_options_t *, omc_stream_t) { return OMC_ERROR_NOT_SUPPORTED; }
omc_status_t omc_gpu_convert_inplace(omc_dtype_t, omc_dtype_t, void *, size_t,
                                     const omc_options_t *, omc_stream_t) { return OMC_ERROR_NOT_SUPPORTED; }
omc_status_t omc_gpu_convert_inplace_2d(omc_dtype_t, omc_dtype_t, void *, int64_t, int64_t, int64_t,
                                        const omc_options_t *, omc_stream_t) { return OMC_ERROR_NOT_SUPPORTED; }
omc_status_t omc_gpu_amax_2d(omc_dtype_t, const void *, int64_t, int64_t, int64_t, float *,
                             omc_stream_t) { return OMC_ERROR_NOT_SUPPORTED; }
size_t omc_gpu_workspace_size(size_t) { return 0; }
void omc_gpu_release_workspaces(void) {}
}
