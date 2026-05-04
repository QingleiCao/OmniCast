add_library(omnicast_cpu
    c_api.cpp
    cpu_cast.cpp
    gpu_cast_stub.cpp
)

target_include_directories(omnicast_cpu
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(omnicast_cpu PUBLIC cxx_std_17)

if(OMNICAST_ENABLE_CPU_SIMD)
    target_compile_definitions(omnicast_cpu PUBLIC OMNICAST_ENABLE_CPU_SIMD=1)
else()
    target_compile_definitions(omnicast_cpu PUBLIC OMNICAST_ENABLE_CPU_SIMD=0)
endif()

if(OMNICAST_HAS_CUDA)
    target_sources(omnicast_cpu PRIVATE gpu_cast_cuda.cu)
    set_source_files_properties(gpu_cast_cuda.cu PROPERTIES LANGUAGE CUDA)
    set_target_properties(omnicast_cpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_compile_definitions(omnicast_cpu PUBLIC OMNICAST_HAS_CUDA=1)
    target_link_libraries(omnicast_cpu PRIVATE CUDA::cudart)
    target_compile_options(omnicast_cpu PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
else()
    target_compile_definitions(omnicast_cpu PUBLIC OMNICAST_HAS_CUDA=0)
endif()

if(OMNICAST_INTERNAL_AUTO_NATIVE)
    include(CheckCXXCompilerFlag)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        check_cxx_compiler_flag("-march=native" OMNICAST_HAS_MARCH_NATIVE)
        check_cxx_compiler_flag("-mtune=native" OMNICAST_HAS_MTUNE_NATIVE)

        if(OMNICAST_HAS_MARCH_NATIVE)
            target_compile_options(omnicast_cpu PRIVATE $<$<NOT:$<CONFIG:Debug>>:-march=native>)
        endif()
        if(OMNICAST_HAS_MTUNE_NATIVE)
            target_compile_options(omnicast_cpu PRIVATE $<$<NOT:$<CONFIG:Debug>>:-mtune=native>)
        endif()
    endif()
endif()
