add_executable(test_casts_cpp
    test_casts.cpp
)

add_executable(test_casts_c
    test_casts.c
)

add_executable(test_inplace_compare_c
    test_inplace_compare.c
)

if(OMNICAST_HAS_CUDA)
    add_executable(test_gpu_casts_cpp
        test_gpu_casts.cpp
    )
endif()

target_link_libraries(test_casts_cpp
    PRIVATE
        omnicast_cpu
)

target_link_libraries(test_casts_c
    PRIVATE
        omnicast_cpu
)

target_link_libraries(test_inplace_compare_c
    PRIVATE
        omnicast_cpu
)

if(OMNICAST_HAS_CUDA)
    target_link_libraries(test_gpu_casts_cpp
        PRIVATE
            omnicast_cpu
            CUDA::cudart
    )
    target_compile_features(test_gpu_casts_cpp PRIVATE cxx_std_17)
endif()

target_compile_features(test_casts_cpp PRIVATE cxx_std_17)

if(OMNICAST_INTERNAL_AUTO_NATIVE)
    include(CheckCXXCompilerFlag)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        check_cxx_compiler_flag("-march=native" OMNICAST_BENCH_HAS_MARCH_NATIVE)
        check_cxx_compiler_flag("-mtune=native" OMNICAST_BENCH_HAS_MTUNE_NATIVE)

        if(OMNICAST_BENCH_HAS_MARCH_NATIVE)
            target_compile_options(test_casts_cpp PRIVATE $<$<NOT:$<CONFIG:Debug>>:-march=native>)
        endif()
        if(OMNICAST_BENCH_HAS_MTUNE_NATIVE)
            target_compile_options(test_casts_cpp PRIVATE $<$<NOT:$<CONFIG:Debug>>:-mtune=native>)
        endif()
    endif()
endif()

add_test(NAME test_casts_cpp COMMAND test_casts_cpp)
add_test(NAME test_casts_c COMMAND test_casts_c)
add_test(NAME test_inplace_compare_c COMMAND test_inplace_compare_c)
if(OMNICAST_HAS_CUDA)
    add_test(NAME test_gpu_casts_cpp COMMAND test_gpu_casts_cpp)
endif()
