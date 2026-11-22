cmake_minimum_required(VERSION 3.18)
project(OmniCast VERSION 2.0.0 LANGUAGES C CXX)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(_omc_top ON)
else()
    set(_omc_top OFF)
endif()

option(OMNICAST_BUILD_TESTS      "Build the OmniCast tests"                       ${_omc_top})
option(OMNICAST_BUILD_BENCHMARKS "Build the OmniCast benchmarks"                  ${_omc_top})
option(OMNICAST_INSTALL          "Install the library, headers and CMake package" ${_omc_top})
option(OMNICAST_BUILD_SHARED     "Build a shared library (default: static, PIC)"  OFF)
option(OMNICAST_OPENMP           "Multi-threaded host conversions with OpenMP"    ON)
set(OMNICAST_GPU_BACKEND "AUTO" CACHE STRING "GPU backend: AUTO, CUDA, HIP or NONE")
set_property(CACHE OMNICAST_GPU_BACKEND PROPERTY STRINGS AUTO CUDA HIP NONE)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES AND _omc_top)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# GPU backend
# ---------------------------------------------------------------------------
include(CheckLanguage)
string(TOUPPER "${OMNICAST_GPU_BACKEND}" _omc_gpu)
if(_omc_gpu STREQUAL "AUTO")
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(_omc_gpu CUDA)
    else()
        check_language(HIP)
        if(CMAKE_HIP_COMPILER)
            set(_omc_gpu HIP)
        else()
            set(_omc_gpu NONE)
        endif()
    endif()
endif()
if(_omc_gpu STREQUAL "CUDA")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
        set(CMAKE_CUDA_ARCHITECTURES "70;80;90" CACHE STRING "CUDA architectures")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
elseif(_omc_gpu STREQUAL "HIP")
    enable_language(HIP)
elseif(NOT _omc_gpu STREQUAL "NONE")
    message(FATAL_ERROR "OMNICAST_GPU_BACKEND must be AUTO, CUDA, HIP or NONE")
endif()
set(OMNICAST_GPU ${_omc_gpu} CACHE INTERNAL "GPU backend in use")
message(STATUS "OmniCast ${PROJECT_VERSION}: GPU backend ${_omc_gpu}")

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
set(_omc_src
    src/core/common.cpp
    src/cpu/cpu_api.cpp
    src/cpu/cpu_isa_generic.cpp)

include(CheckCXXCompilerFlag)
set(_omc_defs)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|x64)$")
    check_cxx_compiler_flag("-mavx2 -mf16c -mfma" OMC_CXX_HAS_AVX2)
    check_cxx_compiler_flag("-mavx512f -mavx512bw -mavx512vl -mavx512dq" OMC_CXX_HAS_AVX512)
    if(OMC_CXX_HAS_AVX2)
        list(APPEND _omc_src src/cpu/cpu_isa_avx2.cpp)
        set_source_files_properties(src/cpu/cpu_isa_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx2;-mf16c;-mfma")
        list(APPEND _omc_defs OMC_HAVE_AVX2_TU)
    endif()
    if(OMC_CXX_HAS_AVX512)
        list(APPEND _omc_src src/cpu/cpu_isa_avx512.cpp)
        set_source_files_properties(src/cpu/cpu_isa_avx512.cpp PROPERTIES
            COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx2;-mf16c;-mfma;-mprefer-vector-width=512")
        list(APPEND _omc_defs OMC_HAVE_AVX512_TU)
    endif()
endif()

if(_omc_gpu STREQUAL "NONE")
    list(APPEND _omc_src src/gpu/gpu_none.cpp)
else()
    # one translation unit per source type, so the kernels build in parallel
    set(_omc_gpu_src src/gpu/gpu_api.cu)
    set(OMC_INST_DIR ${PROJECT_SOURCE_DIR}/src/gpu)
    foreach(_pair "fp64;double" "fp32;float" "fp16;::omc::f16" "bf16;::omc::bf16" "e4m3;::omc::e4m3"
                  "e5m2;::omc::e5m2" "int8;int8_t" "int16;int16_t" "int32;int32_t" "int64;int64_t"
                  "uint8;uint8_t" "uint16;uint16_t" "uint32;uint32_t" "uint64;uint64_t")
        list(GET _pair 0 OMC_INST_NAME)
        list(GET _pair 1 OMC_INST_TYPE)
        configure_file(src/gpu/gpu_inst.cu.in ${PROJECT_BINARY_DIR}/gpu_inst/gpu_inst_${OMC_INST_NAME}.cu @ONLY)
        list(APPEND _omc_gpu_src ${PROJECT_BINARY_DIR}/gpu_inst/gpu_inst_${OMC_INST_NAME}.cu)
    endforeach()
    if(_omc_gpu STREQUAL "HIP")
        set_source_files_properties(${_omc_gpu_src} PROPERTIES LANGUAGE HIP)
    endif()
    list(APPEND _omc_src ${_omc_gpu_src})
endif()

if(OMNICAST_BUILD_SHARED)
    add_library(omnicast SHARED ${_omc_src})
else()
    add_library(omnicast STATIC ${_omc_src})
endif()
add_library(OmniCast::omnicast ALIAS omnicast)
set_target_properties(omnicast PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON
    HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON
    VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
target_include_directories(omnicast PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_definitions(omnicast PRIVATE ${_omc_defs}
    $<$<STREQUAL:${_omc_gpu},CUDA>:OMC_BACKEND_CUDA>
    $<$<STREQUAL:${_omc_gpu},HIP>:OMC_BACKEND_HIP>)
# Conversions depend on IEEE semantics (NaN, Inf, subnormals): never
# fast-math. -fno-trapping-math only gives up floating-point exception flags,
# which OmniCast does not report; without it GCC does not vectorize the
# branch-free selects of the FP8 and integer conversions. The kernels are
# compiled with full optimization whatever the build type.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(omnicast PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -fno-fast-math -fno-math-errno -fno-trapping-math>)
endif()
# No floating-point contraction on the device: a product followed by a
# conversion must round twice, as on the host (gfx906 otherwise fuses an FP32
# multiply and the FP16 conversion into one v_mad_mix, rounding once).
if(_omc_gpu STREQUAL "CUDA")
    target_compile_options(omnicast PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --ftz=false --prec-div=true --fmad=false>)
    target_link_libraries(omnicast PRIVATE CUDA::cudart)
elseif(_omc_gpu STREQUAL "HIP")
    target_compile_options(omnicast PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-O3 -fno-fast-math -ffp-contract=off>)
endif()

if(OMNICAST_OPENMP)
    find_package(OpenMP COMPONENTS CXX)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(omnicast PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# ---------------------------------------------------------------------------
# Tests and benchmarks
# ---------------------------------------------------------------------------
if(OMNICAST_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
if(OMNICAST_BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
if(OMNICAST_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)
    install(TARGETS omnicast EXPORT OmniCastTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    install(FILES include/omnicast/omnicast.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/omnicast)
    install(EXPORT OmniCastTargets NAMESPACE OmniCast::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OmniCast)
    configure_package_config_file(cmake/OmniCastConfig.cmake.in
        ${PROJECT_BINARY_DIR}/OmniCastConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OmniCast)
    write_basic_package_version_file(${PROJECT_BINARY_DIR}/OmniCastConfigVersion.cmake
        COMPATIBILITY SameMajorVersion)
    install(FILES ${PROJECT_BINARY_DIR}/OmniCastConfig.cmake ${PROJECT_BINARY_DIR}/OmniCastConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/OmniCast)
endif()
