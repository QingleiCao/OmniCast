#include "omnicast/gpu_cast.hpp"

#if !OMNICAST_HAS_CUDA

namespace omnicast {

bool cuda_available() {
    return false;
}

const char* cuda_backend() {
    return "cuda-unavailable";
}

int cuda_device_major() {
    return 0;
}

int cuda_device_minor() {
    return 0;
}

GpuStatus convert_device(
    const void* input,
    GpuDType src_type,
    void* output,
    GpuDType dst_type,
    std::size_t count,
    cuda_stream_handle_t stream
) {
    (void)src_type;
    (void)dst_type;
    (void)stream;
    if (count > 0 && (input == nullptr || output == nullptr)) {
        return GpuStatus::null_pointer;
    }
    return GpuStatus::cuda_unavailable;
}

}  // namespace omnicast

#endif
