cmake_minimum_required(VERSION 3.16)

project(OmniCast
    VERSION 0.1.0
    DESCRIPTION "Universal data-type casting/conversion across all precisions, CPUs, and GPUs."
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(OMNICAST_ENABLE_CPU_SIMD "Enable CPU SIMD acceleration (AVX/F16C where supported)" ON)
option(OMNICAST_ENABLE_CUDA "Enable NVIDIA CUDA backend" ON)
set(OMNICAST_INTERNAL_AUTO_NATIVE ON CACHE INTERNAL "Auto-enable host-native compile tuning")

set(OMNICAST_HAS_CUDA OFF)
if(OMNICAST_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit QUIET)
        if(NOT CUDAToolkit_FOUND)
            message(WARNING "CUDA compiler found but CUDAToolkit package missing; building CPU-only.")
            set(OMNICAST_HAS_CUDA OFF)
        else()
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
                # Datacenter-focused defaults (Ampere/Ada/Hopper).
                set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90 CACHE STRING "CUDA arch list" FORCE)
            endif()
            set(OMNICAST_HAS_CUDA ON)
        endif()
    else()
        message(WARNING "OMNICAST_ENABLE_CUDA=ON but no CUDA compiler found; building CPU-only.")
    endif()
endif()

add_subdirectory(src)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
