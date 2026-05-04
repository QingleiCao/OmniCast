#include "omnicast/cpu_cast.hpp"
#include "omnicast/gpu_cast.hpp"

#if OMNICAST_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace {

template <typename RefFn, typename OutFn>
double relative_f_norm_error(std::size_t count, RefFn&& ref_value, OutFn&& out_value) {
    double diff_accum = 0.0;
    double ref_accum = 0.0;
    for (std::size_t i = 0; i < count; ++i) {
        const double ref = ref_value(i);
        const double diff = out_value(i) - ref;
        diff_accum += diff * diff;
        ref_accum += ref * ref;
    }
    const double denom = std::sqrt(ref_accum);
    return (denom == 0.0) ? 0.0 : (std::sqrt(diff_accum) / denom);
}

void print_metric(const char* label, double ms, double err) {
    std::cout << std::left << std::setw(18) << label << " time: " << std::fixed << std::setprecision(4) << ms
              << " ms/iteration  rel F-norm: " << std::scientific << std::setprecision(6) << err << '\n';
}

const char* dtype_name(omnicast::GpuDType t) {
    switch (t) {
        case omnicast::GpuDType::fp16:
            return "FP16";
        case omnicast::GpuDType::fp32:
            return "FP32";
        case omnicast::GpuDType::fp64:
            return "FP64";
        case omnicast::GpuDType::fp8_e4m3:
            return "FP8_E4M3";
        case omnicast::GpuDType::fp8_e5m2:
            return "FP8_E5M2";
        default:
            return "UNKNOWN";
    }
}

std::size_t dtype_bytes(omnicast::GpuDType t) {
    switch (t) {
        case omnicast::GpuDType::fp64:
            return sizeof(double);
        case omnicast::GpuDType::fp32:
            return sizeof(float);
        case omnicast::GpuDType::fp16:
            return sizeof(std::uint16_t);
        case omnicast::GpuDType::fp8_e4m3:
        case omnicast::GpuDType::fp8_e5m2:
            return sizeof(std::uint8_t);
        default:
            return 0;
    }
}

struct ConversionStat {
    std::string label;
    double ms_per_iter = 0.0;
    double rel_f_norm = std::numeric_limits<double>::quiet_NaN();
    std::size_t src_bytes = 0;
    std::size_t dst_bytes = 0;
    omnicast::GpuDType src_type = omnicast::GpuDType::fp32;
    omnicast::GpuDType dst_type = omnicast::GpuDType::fp32;
};

void require_or_abort(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "validation failed: " << msg << '\n';
        std::abort();
    }
}

// Host-side surrogate FP8 quantizer used for deterministic remap references.
float decode_fp8_e4m3_host(std::uint8_t x) {
    return static_cast<float>(static_cast<std::int8_t>(x)) * (1.0f / 16.0f);
}

float decode_fp8_e5m2_host(std::uint8_t x) {
    return static_cast<float>(static_cast<std::int8_t>(x)) * (1.0f / 8.0f);
}

std::uint8_t encode_fp8_e4m3_host(float value) {
    float clamped = std::fmin(std::fmax(value, -8.0f), 7.9375f);
    int q = static_cast<int>(std::lrint(clamped * 16.0f));
    q = (q < -128) ? -128 : q;
    q = (q > 127) ? 127 : q;
    return static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
}

std::uint8_t encode_fp8_e5m2_host(float value) {
    float clamped = std::fmin(std::fmax(value, -16.0f), 15.875f);
    int q = static_cast<int>(std::lrint(clamped * 8.0f));
    q = (q < -128) ? -128 : q;
    q = (q > 127) ? 127 : q;
    return static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
}

// Treat NaN payload differences as equivalent in reference-vs-variant checks.
bool fp32_bits_equal_or_both_nan(float a, float b) {
    if (std::isnan(a) && std::isnan(b)) {
        return true;
    }
    std::uint32_t ua = 0;
    std::uint32_t ub = 0;
    std::memcpy(&ua, &a, sizeof(float));
    std::memcpy(&ub, &b, sizeof(float));
    return ua == ub;
}

double gelem_per_s(std::size_t count, double ms_per_iter) {
    if (ms_per_iter <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(count) / (ms_per_iter * 1.0e6);
}

double gb_per_s(std::size_t count, std::size_t src_bytes, std::size_t dst_bytes, double ms_per_iter) {
    if (ms_per_iter <= 0.0) {
        return 0.0;
    }
    const double moved = static_cast<double>(count) * static_cast<double>(src_bytes + dst_bytes);
    return moved / (ms_per_iter * 1.0e6);
}

#if OMNICAST_HAS_CUDA
void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << '\n';
        std::abort();
    }
}

void check_status(omnicast::GpuStatus status, const char* msg) {
    if (status != omnicast::GpuStatus::success) {
        std::cerr << msg << " failed with status " << static_cast<int>(status) << '\n';
        std::abort();
    }
}

double benchmark_kernel_ms(const std::function<void()>& fn, int iters) {
    for (int i = 0; i < 5; ++i) {
        fn();
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize warmup");

    cudaEvent_t start_evt{};
    cudaEvent_t stop_evt{};
    check_cuda(cudaEventCreate(&start_evt), "cudaEventCreate start");
    check_cuda(cudaEventCreate(&stop_evt), "cudaEventCreate stop");
    check_cuda(cudaEventRecord(start_evt, nullptr), "cudaEventRecord start");
    for (int i = 0; i < iters; ++i) {
        fn();
    }
    check_cuda(cudaEventRecord(stop_evt, nullptr), "cudaEventRecord stop");
    check_cuda(cudaEventSynchronize(stop_evt), "cudaEventSynchronize stop");
    float total_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&total_ms, start_evt, stop_evt), "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(start_evt), "cudaEventDestroy start");
    check_cuda(cudaEventDestroy(stop_evt), "cudaEventDestroy stop");
    return static_cast<double>(total_ms) / static_cast<double>(iters);
}

#endif

}  // namespace

int main(int argc, char** argv) {
    bool check_error = false;
    std::size_t count = 1U << 20U;
    int iterations = 20;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--check") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --check (use 0 or 1)\n";
                return 2;
            }
            const int value = std::atoi(argv[++i]);
            if (value != 0 && value != 1) {
                std::cerr << "invalid --check value (use 0 or 1)\n";
                return 2;
            }
            check_error = (value == 1);
        } else if (std::strcmp(argv[i], "--size") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --size\n";
                return 2;
            }
            const unsigned long long parsed = std::strtoull(argv[++i], nullptr, 10);
            if (parsed == 0ULL) {
                std::cerr << "invalid --size value (must be > 0)\n";
                return 2;
            }
            count = static_cast<std::size_t>(parsed);
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --iters\n";
                return 2;
            }
            const int parsed = std::atoi(argv[++i]);
            if (parsed <= 0) {
                std::cerr << "invalid --iters value (must be > 0)\n";
                return 2;
            }
            iterations = parsed;
        } else {
            std::cerr << "usage: test_gpu_casts_cpp [--check 0|1] [--size N] [--iters N]\n";
            return 2;
        }
    }

    std::cout << "OmniCast GPU benchmark (" << count << " elements, " << iterations << " iterations)\n";
    std::cout << "CUDA backend: " << omnicast::cuda_backend() << '\n';
    if (!omnicast::cuda_available()) {
        std::cout << "CUDA unavailable; skipping GPU checks.\n";
        return 0;
    }

    const char* expect_sm_env = std::getenv("OMNICAST_EXPECT_SM");
    if (expect_sm_env != nullptr && expect_sm_env[0] != '\0') {
        const int expected_sm = std::atoi(expect_sm_env);
        const int expected_major = expected_sm / 10;
        const int expected_minor = expected_sm % 10;
        const int actual_major = omnicast::cuda_device_major();
        const int actual_minor = omnicast::cuda_device_minor();
        if (actual_major != expected_major || actual_minor != expected_minor) {
            std::cout << "Skipping GPU checks: preset expects SM" << expected_sm << " but detected SM"
                      << actual_major << actual_minor << ".\n";
            return 0;
        }
    }

#if !OMNICAST_HAS_CUDA
    std::cout << "Built without CUDA support.\n";
    return 0;
#else
    constexpr int kTypeCount = 5;
    const omnicast::GpuDType types[kTypeCount] = {
        omnicast::GpuDType::fp64,
        omnicast::GpuDType::fp32,
        omnicast::GpuDType::fp16,
        omnicast::GpuDType::fp8_e4m3,
        omnicast::GpuDType::fp8_e5m2,
    };

    auto type_idx = [&](omnicast::GpuDType t) -> int {
        for (int i = 0; i < kTypeCount; ++i) {
            if (types[i] == t) {
                return i;
            }
        }
        return -1;
    };

    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    std::vector<double> h64(count);
    for (double& v : h64) {
        v = dist(rng);
    }
    std::vector<float> h32(count);
    for (std::size_t i = 0; i < count; ++i) {
        h32[i] = static_cast<float>(h64[i]);
    }
    std::vector<omnicast::fp16_t> h16(count);
    omnicast::convert_fp32_to_fp16(h32.data(), h16.data(), count);

    std::vector<double> ref64(count, 0.0);
    std::vector<double> out64(count, 0.0);
    std::vector<float> out32(count, 0.0f);
    std::vector<omnicast::fp16_t> out16(count, 0);

    void* dev_src[kTypeCount] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    void* dev_dst[kTypeCount] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    double* dev_tmp64 = nullptr;

    for (int i = 0; i < kTypeCount; ++i) {
        check_cuda(cudaMalloc(&dev_src[i], count * dtype_bytes(types[i])), "cudaMalloc dev_src");
        check_cuda(cudaMalloc(&dev_dst[i], count * dtype_bytes(types[i])), "cudaMalloc dev_dst");
    }
    check_cuda(cudaMalloc(&dev_tmp64, count * sizeof(double)), "cudaMalloc dev_tmp64");

    check_cuda(
        cudaMemcpy(dev_src[type_idx(omnicast::GpuDType::fp64)], h64.data(), count * sizeof(double), cudaMemcpyHostToDevice),
        "copy h64"
    );
    check_cuda(
        cudaMemcpy(dev_src[type_idx(omnicast::GpuDType::fp32)], h32.data(), count * sizeof(float), cudaMemcpyHostToDevice),
        "copy h32"
    );
    check_cuda(
        cudaMemcpy(dev_src[type_idx(omnicast::GpuDType::fp16)], h16.data(), count * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
        "copy h16"
    );

    check_status(
        omnicast::convert_device(
            dev_src[type_idx(omnicast::GpuDType::fp64)],
            omnicast::GpuDType::fp64,
            dev_src[type_idx(omnicast::GpuDType::fp8_e4m3)],
            omnicast::GpuDType::fp8_e4m3,
            count,
            nullptr
        ),
        "init fp8 e4m3"
    );
    check_status(
        omnicast::convert_device(
            dev_src[type_idx(omnicast::GpuDType::fp64)],
            omnicast::GpuDType::fp64,
            dev_src[type_idx(omnicast::GpuDType::fp8_e5m2)],
            omnicast::GpuDType::fp8_e5m2,
            count,
            nullptr
        ),
        "init fp8 e5m2"
    );
    check_cuda(cudaDeviceSynchronize(), "sync after fp8 init");

    // Validate FP8 format remap correctness for all 256 storage values.
    {
        std::vector<std::uint8_t> h_all(256);
        std::iota(h_all.begin(), h_all.end(), static_cast<std::uint8_t>(0));
        std::vector<std::uint8_t> h_conv(256, 0);
        std::uint8_t* d_all_in = nullptr;
        std::uint8_t* d_all_out = nullptr;
        check_cuda(cudaMalloc(&d_all_in, h_all.size()), "cudaMalloc d_all_in");
        check_cuda(cudaMalloc(&d_all_out, h_all.size()), "cudaMalloc d_all_out");

        check_cuda(cudaMemcpy(d_all_in, h_all.data(), h_all.size(), cudaMemcpyHostToDevice), "copy all e4");
        check_status(
            omnicast::convert_device(
                d_all_in,
                omnicast::GpuDType::fp8_e4m3,
                d_all_out,
                omnicast::GpuDType::fp8_e5m2,
                h_all.size(),
                nullptr
            ),
            "validate e4->e5"
        );
        check_cuda(cudaMemcpy(h_conv.data(), d_all_out, h_conv.size(), cudaMemcpyDeviceToHost), "copy conv e4->e5");
        for (int i = 0; i < 256; ++i) {
            const std::uint8_t expected =
                encode_fp8_e5m2_host(decode_fp8_e4m3_host(static_cast<std::uint8_t>(i)));
            require_or_abort(h_conv[static_cast<std::size_t>(i)] == expected, "e4->e5 LUT mismatch");
        }

        check_cuda(cudaMemcpy(d_all_in, h_all.data(), h_all.size(), cudaMemcpyHostToDevice), "copy all e5");
        check_status(
            omnicast::convert_device(
                d_all_in,
                omnicast::GpuDType::fp8_e5m2,
                d_all_out,
                omnicast::GpuDType::fp8_e4m3,
                h_all.size(),
                nullptr
            ),
            "validate e5->e4"
        );
        check_cuda(cudaMemcpy(h_conv.data(), d_all_out, h_conv.size(), cudaMemcpyDeviceToHost), "copy conv e5->e4");
        for (int i = 0; i < 256; ++i) {
            const std::uint8_t expected =
                encode_fp8_e4m3_host(decode_fp8_e5m2_host(static_cast<std::uint8_t>(i)));
            require_or_abort(h_conv[static_cast<std::size_t>(i)] == expected, "e5->e4 LUT mismatch");
        }
        check_cuda(cudaFree(d_all_in), "free d_all_in");
        check_cuda(cudaFree(d_all_out), "free d_all_out");
    }

    // Validate finite/saturation behavior for FP8 outputs on extreme FP64 inputs.
    {
        const std::vector<double> edge_in = {
            -1.0e12, -128.0, -32.0, -16.0, -8.0, -1.0, 0.0, 1.0, 8.0, 16.0, 32.0, 1.0e12,
            std::numeric_limits<double>::quiet_NaN()
        };
        std::vector<std::uint8_t> edge_out(edge_in.size(), 0);
        std::vector<double> edge_back(edge_in.size(), 0.0);
        double* d_edge_in = nullptr;
        std::uint8_t* d_edge_mid = nullptr;
        double* d_edge_back = nullptr;
        check_cuda(cudaMalloc(&d_edge_in, edge_in.size() * sizeof(double)), "cudaMalloc d_edge_in");
        check_cuda(cudaMalloc(&d_edge_mid, edge_in.size() * sizeof(std::uint8_t)), "cudaMalloc d_edge_mid");
        check_cuda(cudaMalloc(&d_edge_back, edge_in.size() * sizeof(double)), "cudaMalloc d_edge_back");
        check_cuda(
            cudaMemcpy(d_edge_in, edge_in.data(), edge_in.size() * sizeof(double), cudaMemcpyHostToDevice),
            "copy d_edge_in"
        );

        check_status(
            omnicast::convert_device(
                d_edge_in,
                omnicast::GpuDType::fp64,
                d_edge_mid,
                omnicast::GpuDType::fp8_e4m3,
                edge_in.size(),
                nullptr
            ),
            "edge fp64->e4"
        );
        check_status(
            omnicast::convert_device(
                d_edge_mid,
                omnicast::GpuDType::fp8_e4m3,
                d_edge_back,
                omnicast::GpuDType::fp64,
                edge_in.size(),
                nullptr
            ),
            "edge e4->fp64"
        );
        check_cuda(cudaMemcpy(edge_back.data(), d_edge_back, edge_back.size() * sizeof(double), cudaMemcpyDeviceToHost),
                   "copy edge e4 back");
        for (std::size_t i = 0; i < edge_back.size(); ++i) {
            require_or_abort(std::isfinite(edge_back[i]), "e4 decoded value must be finite");
            require_or_abort(edge_back[i] >= -8.0001 && edge_back[i] <= 7.9376, "e4 saturation bound");
        }

        check_status(
            omnicast::convert_device(
                d_edge_in,
                omnicast::GpuDType::fp64,
                d_edge_mid,
                omnicast::GpuDType::fp8_e5m2,
                edge_in.size(),
                nullptr
            ),
            "edge fp64->e5"
        );
        check_status(
            omnicast::convert_device(
                d_edge_mid,
                omnicast::GpuDType::fp8_e5m2,
                d_edge_back,
                omnicast::GpuDType::fp64,
                edge_in.size(),
                nullptr
            ),
            "edge e5->fp64"
        );
        check_cuda(cudaMemcpy(edge_back.data(), d_edge_back, edge_back.size() * sizeof(double), cudaMemcpyDeviceToHost),
                   "copy edge e5 back");
        for (std::size_t i = 0; i < edge_back.size(); ++i) {
            require_or_abort(std::isfinite(edge_back[i]), "e5 decoded value must be finite");
            require_or_abort(edge_back[i] >= -16.0001 && edge_back[i] <= 15.8751, "e5 saturation bound");
        }

        check_cuda(cudaFree(d_edge_in), "free d_edge_in");
        check_cuda(cudaFree(d_edge_mid), "free d_edge_mid");
        check_cuda(cudaFree(d_edge_back), "free d_edge_back");
    }

    // Validate bitwise identity between scalar and LUT-based FP8 remap on large random arrays.
    {
        const std::size_t n = std::max<std::size_t>(count, 1U << 20U);
        std::vector<std::uint8_t> h_e4(n);
        std::vector<std::uint8_t> h_e5(n);
        for (std::size_t i = 0; i < n; ++i) {
            h_e4[i] = static_cast<std::uint8_t>(rng() & 0xFFU);
            h_e5[i] = static_cast<std::uint8_t>(rng() & 0xFFU);
        }
        std::vector<std::uint8_t> out_scalar(n, 0);
        std::vector<std::uint8_t> out_lut(n, 0);
        std::uint8_t* d_in = nullptr;
        std::uint8_t* d_out = nullptr;
        check_cuda(cudaMalloc(&d_in, n), "cudaMalloc fp8 remap d_in");
        check_cuda(cudaMalloc(&d_out, n), "cudaMalloc fp8 remap d_out");

        // E4 -> E5: scalar reference
        check_cuda(cudaMemcpy(d_in, h_e4.data(), n, cudaMemcpyHostToDevice), "copy random e4");
        setenv("OMNICAST_FP8_REMAP_IMPL", "scalar", 1);
        check_status(
            omnicast::convert_device(
                d_in,
                omnicast::GpuDType::fp8_e4m3,
                d_out,
                omnicast::GpuDType::fp8_e5m2,
                n,
                nullptr
            ),
            "random e4->e5 scalar"
        );
        check_cuda(cudaMemcpy(out_scalar.data(), d_out, n, cudaMemcpyDeviceToHost), "copy random e4->e5 scalar");

        const char* remap_impls[] = {"lut8_const", "lut32_u32", "lut32_u64", "lut32_u128", "lut16", "lutx16",
                                     "lut8_shared"};
        for (const char* impl : remap_impls) {
            setenv("OMNICAST_FP8_REMAP_IMPL", impl, 1);
            check_status(
                omnicast::convert_device(
                    d_in,
                    omnicast::GpuDType::fp8_e4m3,
                    d_out,
                    omnicast::GpuDType::fp8_e5m2,
                    n,
                    nullptr
                ),
                "random e4->e5 lut impl"
            );
            check_cuda(cudaMemcpy(out_lut.data(), d_out, n, cudaMemcpyDeviceToHost), "copy random e4->e5 lut impl");
            require_or_abort(out_scalar == out_lut, "random e4->e5 scalar vs lut impl mismatch");
        }

        // E5 -> E4: scalar reference
        check_cuda(cudaMemcpy(d_in, h_e5.data(), n, cudaMemcpyHostToDevice), "copy random e5");
        setenv("OMNICAST_FP8_REMAP_IMPL", "scalar", 1);
        check_status(
            omnicast::convert_device(
                d_in,
                omnicast::GpuDType::fp8_e5m2,
                d_out,
                omnicast::GpuDType::fp8_e4m3,
                n,
                nullptr
            ),
            "random e5->e4 scalar"
        );
        check_cuda(cudaMemcpy(out_scalar.data(), d_out, n, cudaMemcpyDeviceToHost), "copy random e5->e4 scalar");

        for (const char* impl : remap_impls) {
            setenv("OMNICAST_FP8_REMAP_IMPL", impl, 1);
            check_status(
                omnicast::convert_device(
                    d_in,
                    omnicast::GpuDType::fp8_e5m2,
                    d_out,
                    omnicast::GpuDType::fp8_e4m3,
                    n,
                    nullptr
                ),
                "random e5->e4 lut impl"
            );
            check_cuda(cudaMemcpy(out_lut.data(), d_out, n, cudaMemcpyDeviceToHost), "copy random e5->e4 lut impl");
            require_or_abort(out_scalar == out_lut, "random e5->e4 scalar vs lut impl mismatch");
        }

        setenv("OMNICAST_FP8_REMAP_IMPL", "auto", 1);
        check_cuda(cudaFree(d_in), "free fp8 remap d_in");
        check_cuda(cudaFree(d_out), "free fp8 remap d_out");
    }

    // Validate FP16<->FP8 implementation variants against scalar bitwise reference.
    {
        // FP8 -> FP16 exhaustive (all 256 values), both formats.
        std::vector<std::uint8_t> h_all(256);
        std::iota(h_all.begin(), h_all.end(), static_cast<std::uint8_t>(0));
        std::vector<std::uint16_t> out_scalar16(256, 0);
        std::vector<std::uint16_t> out_variant16(256, 0);
        std::uint8_t* d_fp8_in = nullptr;
        std::uint16_t* d_fp16_out = nullptr;
        check_cuda(cudaMalloc(&d_fp8_in, h_all.size()), "cudaMalloc d_fp8_in");
        check_cuda(cudaMalloc(&d_fp16_out, h_all.size() * sizeof(std::uint16_t)), "cudaMalloc d_fp16_out");
        check_cuda(cudaMemcpy(d_fp8_in, h_all.data(), h_all.size(), cudaMemcpyHostToDevice), "copy fp8 exhaustive");

        const char* fp8_to_fp16_impls[] = {"packed", "packed_wide", "lut8_to_half", "native", "auto"};
        for (omnicast::GpuDType src_type : {omnicast::GpuDType::fp8_e4m3, omnicast::GpuDType::fp8_e5m2}) {
            setenv("OMNICAST_FP16_FP8_IMPL", "scalar", 1);
            check_status(
                omnicast::convert_device(d_fp8_in, src_type, d_fp16_out, omnicast::GpuDType::fp16, h_all.size(), nullptr),
                "fp8->fp16 scalar exhaustive"
            );
            check_cuda(
                cudaMemcpy(out_scalar16.data(), d_fp16_out, out_scalar16.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
                "copy fp8->fp16 scalar exhaustive"
            );
            for (const char* impl : fp8_to_fp16_impls) {
                setenv("OMNICAST_FP16_FP8_IMPL", impl, 1);
                check_status(
                    omnicast::convert_device(
                        d_fp8_in, src_type, d_fp16_out, omnicast::GpuDType::fp16, h_all.size(), nullptr
                    ),
                    "fp8->fp16 variant exhaustive"
                );
                check_cuda(
                    cudaMemcpy(out_variant16.data(), d_fp16_out, out_variant16.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost),
                    "copy fp8->fp16 variant exhaustive"
                );
                require_or_abort(out_variant16 == out_scalar16, "fp8->fp16 variant mismatch vs scalar");
            }
        }
        check_cuda(cudaFree(d_fp8_in), "free d_fp8_in");
        check_cuda(cudaFree(d_fp16_out), "free d_fp16_out");

        // FP16 -> FP8 exhaustive (all 65536 bit patterns), both FP8 formats.
        std::vector<std::uint16_t> h_all16(1U << 16U);
        for (std::size_t i = 0; i < h_all16.size(); ++i) {
            h_all16[i] = static_cast<std::uint16_t>(i);
        }
        std::vector<std::uint8_t> out_scalar8(h_all16.size(), 0);
        std::vector<std::uint8_t> out_variant8(h_all16.size(), 0);
        std::uint16_t* d_fp16_in = nullptr;
        std::uint8_t* d_fp8_out = nullptr;
        check_cuda(cudaMalloc(&d_fp16_in, h_all16.size() * sizeof(std::uint16_t)), "cudaMalloc d_fp16_in");
        check_cuda(cudaMalloc(&d_fp8_out, h_all16.size()), "cudaMalloc d_fp8_out");
        check_cuda(
            cudaMemcpy(d_fp16_in, h_all16.data(), h_all16.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice),
            "copy fp16 exhaustive"
        );
        const char* fp16_to_fp8_impls[] = {"packed", "packed_wide", "lut16_to_fp8", "native", "auto"};
        for (omnicast::GpuDType dst_type : {omnicast::GpuDType::fp8_e4m3, omnicast::GpuDType::fp8_e5m2}) {
            setenv("OMNICAST_FP16_FP8_IMPL", "scalar", 1);
            check_status(
                omnicast::convert_device(d_fp16_in, omnicast::GpuDType::fp16, d_fp8_out, dst_type, h_all16.size(), nullptr),
                "fp16->fp8 scalar exhaustive"
            );
            check_cuda(cudaMemcpy(out_scalar8.data(), d_fp8_out, out_scalar8.size(), cudaMemcpyDeviceToHost),
                       "copy fp16->fp8 scalar exhaustive");
            for (const char* impl : fp16_to_fp8_impls) {
                setenv("OMNICAST_FP16_FP8_IMPL", impl, 1);
                check_status(
                    omnicast::convert_device(d_fp16_in, omnicast::GpuDType::fp16, d_fp8_out, dst_type, h_all16.size(), nullptr),
                    "fp16->fp8 variant exhaustive"
                );
                check_cuda(cudaMemcpy(out_variant8.data(), d_fp8_out, out_variant8.size(), cudaMemcpyDeviceToHost),
                           "copy fp16->fp8 variant exhaustive");
                require_or_abort(out_variant8 == out_scalar8, "fp16->fp8 variant mismatch vs scalar");
            }
        }

        // Targeted edge-pattern smoke checks.
        const std::uint16_t edge_vals[] = {
            0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0x7FFF
        };
        check_cuda(
            cudaMemcpy(d_fp16_in, edge_vals, sizeof(edge_vals), cudaMemcpyHostToDevice),
            "copy fp16 edge values"
        );
        setenv("OMNICAST_FP16_FP8_IMPL", "auto", 1);
        check_status(
            omnicast::convert_device(
                d_fp16_in, omnicast::GpuDType::fp16, d_fp8_out, omnicast::GpuDType::fp8_e4m3,
                sizeof(edge_vals) / sizeof(edge_vals[0]), nullptr
            ),
            "fp16 edge auto e4"
        );
        check_status(
            omnicast::convert_device(
                d_fp16_in, omnicast::GpuDType::fp16, d_fp8_out, omnicast::GpuDType::fp8_e5m2,
                sizeof(edge_vals) / sizeof(edge_vals[0]), nullptr
            ),
            "fp16 edge auto e5"
        );

        check_cuda(cudaFree(d_fp16_in), "free d_fp16_in");
        check_cuda(cudaFree(d_fp8_out), "free d_fp8_out");
        setenv("OMNICAST_FP16_FP8_IMPL", "auto", 1);
    }

    // Validate FP32<->FP8 implementation variants against scalar reference.
    {
        constexpr std::size_t kAllFp8 = 256;
        constexpr std::size_t kRandomCount = 1U << 16U;
        const char* fp8_to_fp32_modes[] = {"native", "packed", "packed_wide", "lut8_to_fp32", "auto"};
        const char* fp32_to_fp8_modes[] = {"native", "packed", "packed_wide", "auto"};

        float* d_fp32_in = nullptr;
        float* d_fp32_out = nullptr;
        std::uint8_t* d_fp8_in = nullptr;
        std::uint8_t* d_fp8_out = nullptr;
        check_cuda(cudaMalloc(&d_fp32_in, kRandomCount * sizeof(float)), "alloc d_fp32_in");
        check_cuda(cudaMalloc(&d_fp32_out, kRandomCount * sizeof(float)), "alloc d_fp32_out");
        check_cuda(cudaMalloc(&d_fp8_in, kRandomCount * sizeof(std::uint8_t)), "alloc d_fp8_in");
        check_cuda(cudaMalloc(&d_fp8_out, kRandomCount * sizeof(std::uint8_t)), "alloc d_fp8_out");

        // FP8 -> FP32 exhaustive: all FP8 bit patterns, both formats.
        std::vector<std::uint8_t> h_all_fp8(kAllFp8);
        for (std::size_t i = 0; i < kAllFp8; ++i) {
            h_all_fp8[i] = static_cast<std::uint8_t>(i);
        }
        std::vector<float> out_scalar32(kAllFp8);
        std::vector<float> out_variant32(kAllFp8);
        check_cuda(cudaMemcpy(d_fp8_in, h_all_fp8.data(), kAllFp8 * sizeof(std::uint8_t), cudaMemcpyHostToDevice),
                   "copy fp8 exhaustive for fp32");
        for (omnicast::GpuDType src_t : {omnicast::GpuDType::fp8_e4m3, omnicast::GpuDType::fp8_e5m2}) {
            setenv("OMNICAST_FP32_FP8_IMPL", "scalar", 1);
            check_status(
                omnicast::convert_device(d_fp8_in, src_t, d_fp32_out, omnicast::GpuDType::fp32, kAllFp8, nullptr),
                "fp8->fp32 scalar exhaustive"
            );
            check_cuda(cudaMemcpy(out_scalar32.data(), d_fp32_out, kAllFp8 * sizeof(float), cudaMemcpyDeviceToHost),
                       "copy fp8->fp32 scalar exhaustive");
            for (const char* mode : fp8_to_fp32_modes) {
                setenv("OMNICAST_FP32_FP8_IMPL", mode, 1);
                check_status(
                    omnicast::convert_device(d_fp8_in, src_t, d_fp32_out, omnicast::GpuDType::fp32, kAllFp8, nullptr),
                    "fp8->fp32 variant exhaustive"
                );
                check_cuda(cudaMemcpy(out_variant32.data(), d_fp32_out, kAllFp8 * sizeof(float), cudaMemcpyDeviceToHost),
                           "copy fp8->fp32 variant exhaustive");
                for (std::size_t i = 0; i < kAllFp8; ++i) {
                    require_or_abort(
                        fp32_bits_equal_or_both_nan(out_variant32[i], out_scalar32[i]),
                        "fp8->fp32 exhaustive mismatch vs scalar"
                    );
                }
            }
        }

        // FP32 -> FP8 boundary and random tests: compare bytes against scalar, both formats.
        std::vector<float> h_fp32_boundary = {
            0.0f, -0.0f, std::numeric_limits<float>::denorm_min(), -std::numeric_limits<float>::denorm_min(),
            std::numeric_limits<float>::min(), -std::numeric_limits<float>::min(), 1.0e-5f, -1.0e-5f,
            1.0f, -1.0f, 8.0f, -8.0f, 16.0f, -16.0f,
            std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
            std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(), -std::numeric_limits<float>::quiet_NaN()
        };
        std::vector<float> h_fp32_random(kRandomCount);
        std::mt19937 rng_fp32(2026);
        std::uniform_real_distribution<float> dist_fp32(-1.0e5f, 1.0e5f);
        for (float& v : h_fp32_random) {
            v = dist_fp32(rng_fp32);
        }
        // Inject special values into random stream so saturation/NaN/Inf behavior is always exercised.
        if (kRandomCount >= 8) {
            h_fp32_random[0] = std::numeric_limits<float>::infinity();
            h_fp32_random[1] = -std::numeric_limits<float>::infinity();
            h_fp32_random[2] = std::numeric_limits<float>::quiet_NaN();
            h_fp32_random[3] = -std::numeric_limits<float>::quiet_NaN();
            h_fp32_random[4] = std::numeric_limits<float>::max();
            h_fp32_random[5] = -std::numeric_limits<float>::max();
            h_fp32_random[6] = std::numeric_limits<float>::denorm_min();
            h_fp32_random[7] = -std::numeric_limits<float>::denorm_min();
        }
        std::vector<std::uint8_t> out_scalar8(kRandomCount);
        std::vector<std::uint8_t> out_variant8(kRandomCount);
        const auto check_fp32_to_fp8_dataset = [&](const std::vector<float>& src_values, const char* tag) {
            const std::size_t n = src_values.size();
            check_cuda(cudaMemcpy(d_fp32_in, src_values.data(), n * sizeof(float), cudaMemcpyHostToDevice),
                       "copy fp32 test dataset");
            for (omnicast::GpuDType dst_t : {omnicast::GpuDType::fp8_e4m3, omnicast::GpuDType::fp8_e5m2}) {
                setenv("OMNICAST_FP32_FP8_IMPL", "scalar", 1);
                check_status(
                    omnicast::convert_device(d_fp32_in, omnicast::GpuDType::fp32, d_fp8_out, dst_t, n, nullptr),
                    "fp32->fp8 scalar test"
                );
                check_cuda(cudaMemcpy(out_scalar8.data(), d_fp8_out, n * sizeof(std::uint8_t), cudaMemcpyDeviceToHost),
                           "copy fp32->fp8 scalar test");
                for (const char* mode : fp32_to_fp8_modes) {
                    setenv("OMNICAST_FP32_FP8_IMPL", mode, 1);
                    check_status(
                        omnicast::convert_device(d_fp32_in, omnicast::GpuDType::fp32, d_fp8_out, dst_t, n, nullptr),
                        "fp32->fp8 variant test"
                    );
                    check_cuda(cudaMemcpy(out_variant8.data(), d_fp8_out, n * sizeof(std::uint8_t), cudaMemcpyDeviceToHost),
                               "copy fp32->fp8 variant test");
                    require_or_abort(
                        std::equal(out_variant8.begin(), out_variant8.begin() + n, out_scalar8.begin()),
                        tag
                    );
                }
            }
        };
        check_fp32_to_fp8_dataset(h_fp32_boundary, "fp32->fp8 boundary mismatch vs scalar");
        check_fp32_to_fp8_dataset(h_fp32_random, "fp32->fp8 random mismatch vs scalar");

        check_cuda(cudaFree(d_fp32_in), "free d_fp32_in");
        check_cuda(cudaFree(d_fp32_out), "free d_fp32_out");
        check_cuda(cudaFree(d_fp8_in), "free d_fp8_in");
        check_cuda(cudaFree(d_fp8_out), "free d_fp8_out");
        setenv("OMNICAST_FP32_FP8_IMPL", "auto", 1);
    }

    auto materialize_reference = [&](omnicast::GpuDType src_type) {
        if (src_type == omnicast::GpuDType::fp64) {
            std::memcpy(ref64.data(), h64.data(), count * sizeof(double));
            return;
        }
        if (src_type == omnicast::GpuDType::fp32) {
            for (std::size_t i = 0; i < count; ++i) {
                ref64[i] = static_cast<double>(h32[i]);
            }
            return;
        }
        if (src_type == omnicast::GpuDType::fp16) {
            for (std::size_t i = 0; i < count; ++i) {
                ref64[i] = omnicast::fp16_to_fp64(h16[i]);
            }
            return;
        }
        check_status(
            omnicast::convert_device(
                dev_src[type_idx(src_type)],
                src_type,
                dev_tmp64,
                omnicast::GpuDType::fp64,
                count,
                nullptr
            ),
            "materialize fp8 reference"
        );
        check_cuda(cudaMemcpy(out64.data(), dev_tmp64, count * sizeof(double), cudaMemcpyDeviceToHost), "copy ref64");
        std::memcpy(ref64.data(), out64.data(), count * sizeof(double));
    };

    auto materialize_output_as_fp64 = [&](omnicast::GpuDType dst_type) {
        if (dst_type == omnicast::GpuDType::fp64) {
            check_cuda(
                cudaMemcpy(
                    out64.data(),
                    dev_dst[type_idx(dst_type)],
                    count * sizeof(double),
                    cudaMemcpyDeviceToHost
                ),
                "copy output fp64"
            );
            return;
        }
        if (dst_type == omnicast::GpuDType::fp32) {
            check_cuda(
                cudaMemcpy(
                    out32.data(),
                    dev_dst[type_idx(dst_type)],
                    count * sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "copy output fp32"
            );
            for (std::size_t i = 0; i < count; ++i) {
                out64[i] = static_cast<double>(out32[i]);
            }
            return;
        }
        if (dst_type == omnicast::GpuDType::fp16) {
            check_cuda(
                cudaMemcpy(
                    out16.data(),
                    dev_dst[type_idx(dst_type)],
                    count * sizeof(std::uint16_t),
                    cudaMemcpyDeviceToHost
                ),
                "copy output fp16"
            );
            for (std::size_t i = 0; i < count; ++i) {
                out64[i] = omnicast::fp16_to_fp64(out16[i]);
            }
            return;
        }
        check_status(
            omnicast::convert_device(
                dev_dst[type_idx(dst_type)],
                dst_type,
                dev_tmp64,
                omnicast::GpuDType::fp64,
                count,
                nullptr
            ),
            "materialize fp8 output"
        );
        check_cuda(cudaMemcpy(out64.data(), dev_tmp64, count * sizeof(double), cudaMemcpyDeviceToHost), "copy fp8->fp64");
    };

    std::cout << "All directed conversions:\n";
    std::vector<ConversionStat> stats;
    stats.reserve(static_cast<std::size_t>(kTypeCount * (kTypeCount - 1)));
    for (int si = 0; si < kTypeCount; ++si) {
        for (int di = 0; di < kTypeCount; ++di) {
            if (si == di) {
                continue;
            }
            const omnicast::GpuDType src_t = types[si];
            const omnicast::GpuDType dst_t = types[di];
            const std::size_t src_b = dtype_bytes(src_t);
            const std::size_t dst_b = dtype_bytes(dst_t);

            const double t_case = benchmark_kernel_ms([&]() {
                check_status(omnicast::convert_device(dev_src[si], src_t, dev_dst[di], dst_t, count, nullptr), "benchmark case");
            }, iterations);

            std::string label = std::string(dtype_name(src_t)) + "->" + dtype_name(dst_t);
            if (check_error) {
                materialize_reference(src_t);
                check_status(omnicast::convert_device(dev_src[si], src_t, dev_dst[di], dst_t, count, nullptr), "final case");
                materialize_output_as_fp64(dst_t);
                const double err = relative_f_norm_error(
                    count,
                    [&](std::size_t i) { return ref64[i]; },
                    [&](std::size_t i) { return out64[i]; }
                );
                print_metric(label.c_str(), t_case, err);
                stats.push_back(ConversionStat{label, t_case, err, src_b, dst_b, src_t, dst_t});
            } else {
                std::cout << std::left << std::setw(18) << label << " time: " << std::fixed << std::setprecision(4)
                          << t_case << " ms/iteration  rel F-norm: disabled (--check 1)\n";
                stats.push_back(ConversionStat{
                    label, t_case, std::numeric_limits<double>::quiet_NaN(), src_b, dst_b, src_t, dst_t
                });
            }
        }
    }

    // Nsight Compute suggestions for deeper kernel analysis:
    // dram__bytes.sum
    // dram__throughput.avg.pct_of_peak_sustained_elapsed
    // l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
    // l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
    // smsp__sass_thread_inst_executed_op_conversion.sum
    // smsp__sass_thread_inst_executed_op_memory.sum
    double best_observed_gbps = 0.0;
    for (const auto& s : stats) {
        best_observed_gbps = std::max(best_observed_gbps, gb_per_s(count, s.src_bytes, s.dst_bytes, s.ms_per_iter));
    }

    std::cout << "\nThroughput summary (effective):\n";
    std::cout << "  Observed memory roof proxy (max effective GB/s): "
              << std::fixed << std::setprecision(2) << best_observed_gbps << "\n";
    std::cout << "  For byte/narrow-type conversion paths, the FP64-derived memory roof can overestimate the\n";
    std::cout << "  achievable ceiling. Dedicated packed-copy or copy-shape baselines are reported below and\n";
    std::cout << "  should be used for FP8 remap, FP16<->FP8, and FP32<->FP8 efficiency.\n";
    for (const auto& s : stats) {
        const std::size_t bytes_per_elem = s.src_bytes + s.dst_bytes;
        const double expected_mem_bound_gelem = (bytes_per_elem == 0)
                                                    ? 0.0
                                                    : (best_observed_gbps / static_cast<double>(bytes_per_elem));
        const double ratio = (expected_mem_bound_gelem > 0.0)
                                 ? (gelem_per_s(count, s.ms_per_iter) / expected_mem_bound_gelem)
                                 : 0.0;
        std::cout << std::left << std::setw(18) << s.label
                  << " GElem/s: " << std::setw(8) << std::fixed << std::setprecision(2)
                  << gelem_per_s(count, s.ms_per_iter)
                  << " GB/s: " << std::setw(8) << std::fixed << std::setprecision(2)
                  << gb_per_s(count, s.src_bytes, s.dst_bytes, s.ms_per_iter)
                  << " B/elem: " << bytes_per_elem
                  << " mem-bound GElem/s: " << std::setw(8) << std::fixed << std::setprecision(2)
                  << expected_mem_bound_gelem
                  << " ratio: " << std::setw(6) << std::fixed << std::setprecision(2)
                  << ratio
                  << '\n';
    }

    std::sort(stats.begin(), stats.end(), [](const ConversionStat& a, const ConversionStat& b) {
        return a.ms_per_iter < b.ms_per_iter;
    });
    const std::size_t top_k = std::min<std::size_t>(3, stats.size());
    std::cout << "\nFastest conversions:\n";
    for (std::size_t i = 0; i < top_k; ++i) {
        const auto& s = stats[i];
        std::cout << "  " << s.label << " (" << std::fixed << std::setprecision(4) << s.ms_per_iter << " ms/iteration)\n";
    }
    std::cout << "Slowest conversions:\n";
    for (std::size_t i = 0; i < top_k; ++i) {
        const auto& s = stats[stats.size() - 1 - i];
        std::cout << "  " << s.label << " (" << std::fixed << std::setprecision(4) << s.ms_per_iter << " ms/iteration)\n";
    }

    // Dedicated FP8<->FP8 remap benchmark variants.
    const int fp8e4_idx = type_idx(omnicast::GpuDType::fp8_e4m3);
    const int fp8e5_idx = type_idx(omnicast::GpuDType::fp8_e5m2);
    const auto run_fp8_remap_variant = [&](const char* mode, bool e5_to_e4) -> double {
        setenv("OMNICAST_FP8_REMAP_IMPL", mode, 1);
        if (!e5_to_e4) {
            return benchmark_kernel_ms([&]() {
                check_status(
                    omnicast::convert_device(
                        dev_src[fp8e4_idx],
                        omnicast::GpuDType::fp8_e4m3,
                        dev_dst[fp8e5_idx],
                        omnicast::GpuDType::fp8_e5m2,
                        count,
                        nullptr
                    ),
                    "fp8 remap e4->e5 variant"
                );
            }, iterations);
        }
        return benchmark_kernel_ms([&]() {
            check_status(
                omnicast::convert_device(
                    dev_src[fp8e5_idx],
                    omnicast::GpuDType::fp8_e5m2,
                    dev_dst[fp8e4_idx],
                    omnicast::GpuDType::fp8_e4m3,
                    count,
                    nullptr
                ),
                "fp8 remap e5->e4 variant"
            );
        }, iterations);
    };

    std::cout << "\nFP8 remap variants benchmark (B/elem=2, ratio vs packed-copy peak):\n";
    const char* remap_modes[] = {
        "copy32", "copy64", "copy128", "scalar", "lut32_u32", "lut32_u64", "lut32_u128",
        "lut16", "lut8_const", "lutx16", "lut8_shared", "auto"
    };
    const double copy64_e4_to_e5 = run_fp8_remap_variant("copy64", false);
    const double copy64_e5_to_e4 = run_fp8_remap_variant("copy64", true);
    const double copy128_e4_to_e5 = run_fp8_remap_variant("copy128", false);
    const double copy128_e5_to_e4 = run_fp8_remap_variant("copy128", true);
    const double copy64_ref_gelem =
        0.5 * (gelem_per_s(count, copy64_e4_to_e5) + gelem_per_s(count, copy64_e5_to_e4));
    const double copy128_ref_gelem =
        0.5 * (gelem_per_s(count, copy128_e4_to_e5) + gelem_per_s(count, copy128_e5_to_e4));
    const char* remap_peak_name = (copy128_ref_gelem >= copy64_ref_gelem) ? "copy128" : "copy64";
    const double remap_peak_gelem = std::max(copy64_ref_gelem, copy128_ref_gelem);
    std::cout << "  packed-copy baseline: " << remap_peak_name << " at " << std::fixed << std::setprecision(2)
              << remap_peak_gelem << " GElem/s\n";
    std::cout << "  legend: baseline=copy32/copy64/copy128 packed-copy; reference=scalar;\n";
    std::cout << "  default=auto (lut32_u32); optional=lut32_u64/lut32_u128 wider LUT32 variants;\n";
    std::cout << "  experimental=lut16/lut8_const/lutx16/lut8_shared (lut8_shared currently slow).\n";
    for (const char* mode : remap_modes) {
        const double t_e4_to_e5 = run_fp8_remap_variant(mode, false);
        const double t_e5_to_e4 = run_fp8_remap_variant(mode, true);
        const double g_e4_to_e5 = gelem_per_s(count, t_e4_to_e5);
        const double g_e5_to_e4 = gelem_per_s(count, t_e5_to_e4);
        const double g_avg = 0.5 * (g_e4_to_e5 + g_e5_to_e4);
        const double ratio = (remap_peak_gelem > 0.0) ? (g_avg / remap_peak_gelem) : 0.0;
        std::cout << "  mode=" << std::setw(6) << mode
                  << " E4->E5: " << std::setw(8) << std::fixed << std::setprecision(4) << t_e4_to_e5 << " ms, "
                  << std::setw(8) << std::fixed << std::setprecision(2) << g_e4_to_e5 << " GElem/s"
                  << " | E5->E4: " << std::setw(8) << std::fixed << std::setprecision(4) << t_e5_to_e4 << " ms, "
                  << std::setw(8) << std::fixed << std::setprecision(2) << g_e5_to_e4 << " GElem/s"
                  << " | avg eff vs " << remap_peak_name << ": " << std::setw(6) << std::fixed << std::setprecision(2)
                  << ratio
                  << '\n';
    }
    std::cout << "\nFP8 remap rows in main table (efficiency vs packed-copy baseline):\n";
    for (const auto& s : stats) {
        const bool is_fp8_to_fp8 =
            (s.src_type == omnicast::GpuDType::fp8_e4m3 || s.src_type == omnicast::GpuDType::fp8_e5m2) &&
            (s.dst_type == omnicast::GpuDType::fp8_e4m3 || s.dst_type == omnicast::GpuDType::fp8_e5m2) &&
            (s.src_type != s.dst_type);
        if (!is_fp8_to_fp8) {
            continue;
        }
        const double g = gelem_per_s(count, s.ms_per_iter);
        const double eff = (remap_peak_gelem > 0.0) ? (g / remap_peak_gelem) : 0.0;
        std::cout << "  " << std::left << std::setw(18) << s.label
                  << " GElem/s: " << std::setw(8) << std::fixed << std::setprecision(2) << g
                  << " remap baseline: " << remap_peak_name << " (" << std::setw(8) << std::fixed
                  << std::setprecision(2) << remap_peak_gelem << " GElem/s)"
                  << " remap efficiency: " << std::setw(6) << std::fixed << std::setprecision(2) << eff
                  << '\n';
    }
    setenv("OMNICAST_FP8_REMAP_IMPL", "auto", 1);

    // Dedicated FP16<->FP8 benchmark variants.
    const int fp16_idx = type_idx(omnicast::GpuDType::fp16);
    const int fp32_idx = type_idx(omnicast::GpuDType::fp32);
    const auto run_fp16_fp8_variant = [&](const char* mode, bool fp8_to_fp16, bool e5) -> double {
        setenv("OMNICAST_FP16_FP8_IMPL", mode, 1);
        if (!fp8_to_fp16) {
            return benchmark_kernel_ms([&]() {
                check_status(
                    omnicast::convert_device(
                        dev_src[fp16_idx],
                        omnicast::GpuDType::fp16,
                        dev_dst[e5 ? fp8e5_idx : fp8e4_idx],
                        e5 ? omnicast::GpuDType::fp8_e5m2 : omnicast::GpuDType::fp8_e4m3,
                        count,
                        nullptr
                    ),
                    "fp16->fp8 variant"
                );
            }, iterations);
        }
        return benchmark_kernel_ms([&]() {
            check_status(
                omnicast::convert_device(
                    dev_src[e5 ? fp8e5_idx : fp8e4_idx],
                    e5 ? omnicast::GpuDType::fp8_e5m2 : omnicast::GpuDType::fp8_e4m3,
                    dev_dst[fp16_idx],
                    omnicast::GpuDType::fp16,
                    count,
                    nullptr
                ),
                "fp8->fp16 variant"
            );
        }, iterations);
    };
    const auto run_fp16_fp8_copyshape = [&](bool fp8_to_fp16) -> double {
        setenv("OMNICAST_FP16_FP8_IMPL", "copyshape", 1);
        if (!fp8_to_fp16) {
            return benchmark_kernel_ms([&]() {
                check_status(
                    omnicast::convert_device(
                        dev_src[fp16_idx],
                        omnicast::GpuDType::fp16,
                        dev_dst[fp8e4_idx],
                        omnicast::GpuDType::fp8_e4m3,
                        count,
                        nullptr
                    ),
                    "fp16->fp8 copyshape"
                );
            }, iterations);
        }
        return benchmark_kernel_ms([&]() {
            check_status(
                omnicast::convert_device(
                    dev_src[fp8e4_idx],
                    omnicast::GpuDType::fp8_e4m3,
                    dev_dst[fp16_idx],
                    omnicast::GpuDType::fp16,
                    count,
                    nullptr
                ),
                "fp8->fp16 copyshape"
            );
        }, iterations);
    };

    const double fp16_to_fp8_copy_ms = run_fp16_fp8_copyshape(false);
    const double fp8_to_fp16_copy_ms = run_fp16_fp8_copyshape(true);
    const double fp16_to_fp8_copy_gelem = gelem_per_s(count, fp16_to_fp8_copy_ms);
    const double fp8_to_fp16_copy_gelem = gelem_per_s(count, fp8_to_fp16_copy_ms);
    const double fp16fp8_copy_baseline_gelem = 0.5 * (fp16_to_fp8_copy_gelem + fp8_to_fp16_copy_gelem);

    std::cout << "\nFP16<->FP8 variants benchmark (B/elem=3, ratio vs copy-shape baseline):\n";
    std::cout << "  fp16->fp8 copy baseline: " << std::fixed << std::setprecision(2) << fp16_to_fp8_copy_gelem
              << " GElem/s, " << gb_per_s(count, 2, 1, fp16_to_fp8_copy_ms) << " GB/s\n";
    std::cout << "  fp8->fp16 copy baseline: " << std::fixed << std::setprecision(2) << fp8_to_fp16_copy_gelem
              << " GElem/s, " << gb_per_s(count, 1, 2, fp8_to_fp16_copy_ms) << " GB/s\n";
    std::cout << "  combined copy-shape baseline: " << std::fixed << std::setprecision(2) << fp16fp8_copy_baseline_gelem
              << " GElem/s\n";
    std::cout << "  legend: baseline=copyshape (B/elem=3); reference=scalar;\n";
    std::cout << "  recommended=native/packed; optional=packed_wide;\n";
    std::cout << "  experimental=lut8_to_half/lut16_to_fp8; default=auto (best validated path).\n";

    const char* fp16_fp8_modes[] = {
        "copyshape", "scalar", "native", "packed", "packed_wide", "lut8_to_half", "lut16_to_fp8", "auto"
    };
    for (const char* mode : fp16_fp8_modes) {
        const double t_16_to_e4 = run_fp16_fp8_variant(mode, false, false);
        const double t_16_to_e5 = run_fp16_fp8_variant(mode, false, true);
        const double t_e4_to_16 = run_fp16_fp8_variant(mode, true, false);
        const double t_e5_to_16 = run_fp16_fp8_variant(mode, true, true);
        const double g_16_to_e4 = gelem_per_s(count, t_16_to_e4);
        const double g_16_to_e5 = gelem_per_s(count, t_16_to_e5);
        const double g_e4_to_16 = gelem_per_s(count, t_e4_to_16);
        const double g_e5_to_16 = gelem_per_s(count, t_e5_to_16);
        const double gb_16_to_e4 = gb_per_s(count, 2, 1, t_16_to_e4);
        const double gb_16_to_e5 = gb_per_s(count, 2, 1, t_16_to_e5);
        const double gb_e4_to_16 = gb_per_s(count, 1, 2, t_e4_to_16);
        const double gb_e5_to_16 = gb_per_s(count, 1, 2, t_e5_to_16);
        const double g_avg = 0.25 * (g_16_to_e4 + g_16_to_e5 + g_e4_to_16 + g_e5_to_16);
        const double eff = (fp16fp8_copy_baseline_gelem > 0.0) ? (g_avg / fp16fp8_copy_baseline_gelem) : 0.0;
        std::cout << "  mode=" << std::setw(10) << mode
                  << " 16->E4: " << std::setw(7) << std::fixed << std::setprecision(2) << g_16_to_e4 << " GElem/s ("
                  << std::setw(7) << gb_16_to_e4 << " GB/s)"
                  << " 16->E5: " << std::setw(7) << g_16_to_e5 << " GElem/s (" << std::setw(7) << gb_16_to_e5
                  << " GB/s)"
                  << " E4->16: " << std::setw(7) << g_e4_to_16 << " GElem/s (" << std::setw(7) << gb_e4_to_16
                  << " GB/s)"
                  << " E5->16: " << std::setw(7) << g_e5_to_16 << " GElem/s (" << std::setw(7) << gb_e5_to_16
                  << " GB/s)"
                  << " B/elem: 3"
                  << " | avg eff vs baseline: " << std::setw(6) << std::fixed << std::setprecision(2) << eff << '\n';
    }
    setenv("OMNICAST_FP16_FP8_IMPL", "auto", 1);

    // Dedicated FP32<->FP8 benchmark variants.
    const auto run_fp32_fp8_variant = [&](const char* mode, bool fp8_to_fp32, bool e5) -> double {
        setenv("OMNICAST_FP32_FP8_IMPL", mode, 1);
        if (!fp8_to_fp32) {
            return benchmark_kernel_ms([&]() {
                check_status(
                    omnicast::convert_device(
                        dev_src[fp32_idx],
                        omnicast::GpuDType::fp32,
                        dev_dst[e5 ? fp8e5_idx : fp8e4_idx],
                        e5 ? omnicast::GpuDType::fp8_e5m2 : omnicast::GpuDType::fp8_e4m3,
                        count,
                        nullptr
                    ),
                    "fp32->fp8 variant"
                );
            }, iterations);
        }
        return benchmark_kernel_ms([&]() {
            check_status(
                omnicast::convert_device(
                    dev_src[e5 ? fp8e5_idx : fp8e4_idx],
                    e5 ? omnicast::GpuDType::fp8_e5m2 : omnicast::GpuDType::fp8_e4m3,
                    dev_dst[fp32_idx],
                    omnicast::GpuDType::fp32,
                    count,
                    nullptr
                ),
                "fp8->fp32 variant"
            );
        }, iterations);
    };
    const auto run_fp32_fp8_copyshape = [&](bool fp8_to_fp32) -> double {
        setenv("OMNICAST_FP32_FP8_IMPL", "copyshape", 1);
        if (!fp8_to_fp32) {
            return benchmark_kernel_ms([&]() {
                check_status(
                    omnicast::convert_device(
                        dev_src[fp32_idx],
                        omnicast::GpuDType::fp32,
                        dev_dst[fp8e4_idx],
                        omnicast::GpuDType::fp8_e4m3,
                        count,
                        nullptr
                    ),
                    "fp32->fp8 copyshape"
                );
            }, iterations);
        }
        return benchmark_kernel_ms([&]() {
            check_status(
                omnicast::convert_device(
                    dev_src[fp8e4_idx],
                    omnicast::GpuDType::fp8_e4m3,
                    dev_dst[fp32_idx],
                    omnicast::GpuDType::fp32,
                    count,
                    nullptr
                ),
                "fp8->fp32 copyshape"
            );
        }, iterations);
    };
    const double fp32_to_fp8_copy_ms = run_fp32_fp8_copyshape(false);
    const double fp8_to_fp32_copy_ms = run_fp32_fp8_copyshape(true);
    const double fp32_to_fp8_copy_gelem = gelem_per_s(count, fp32_to_fp8_copy_ms);
    const double fp8_to_fp32_copy_gelem = gelem_per_s(count, fp8_to_fp32_copy_ms);
    const double fp32fp8_copy_baseline_gelem = 0.5 * (fp32_to_fp8_copy_gelem + fp8_to_fp32_copy_gelem);
    std::cout << "\nFP32<->FP8 variants benchmark (B/elem=5, ratio vs copy-shape baseline):\n";
    std::cout << "  fp32->fp8 copy baseline: " << std::fixed << std::setprecision(2) << fp32_to_fp8_copy_gelem
              << " GElem/s, " << gb_per_s(count, 4, 1, fp32_to_fp8_copy_ms) << " GB/s\n";
    std::cout << "  fp8->fp32 copy baseline: " << std::fixed << std::setprecision(2) << fp8_to_fp32_copy_gelem
              << " GElem/s, " << gb_per_s(count, 1, 4, fp8_to_fp32_copy_ms) << " GB/s\n";
    std::cout << "  combined copy-shape baseline: " << std::fixed << std::setprecision(2) << fp32fp8_copy_baseline_gelem
              << " GElem/s\n";
    std::cout << "  legend: baseline=copyshape (B/elem=5); reference=scalar;\n";
    std::cout << "  recommended=native/packed; optional=packed_wide;\n";
    std::cout << "  experimental=lut8_to_fp32; default=auto (best validated path).\n";
    const char* fp32_fp8_modes[] = {"copyshape", "scalar", "native", "packed", "packed_wide", "lut8_to_fp32", "auto"};
    for (const char* mode : fp32_fp8_modes) {
        const double t_32_to_e4 = run_fp32_fp8_variant(mode, false, false);
        const double t_32_to_e5 = run_fp32_fp8_variant(mode, false, true);
        const double t_e4_to_32 = run_fp32_fp8_variant(mode, true, false);
        const double t_e5_to_32 = run_fp32_fp8_variant(mode, true, true);
        const double g_32_to_e4 = gelem_per_s(count, t_32_to_e4);
        const double g_32_to_e5 = gelem_per_s(count, t_32_to_e5);
        const double g_e4_to_32 = gelem_per_s(count, t_e4_to_32);
        const double g_e5_to_32 = gelem_per_s(count, t_e5_to_32);
        const double g_avg = 0.25 * (g_32_to_e4 + g_32_to_e5 + g_e4_to_32 + g_e5_to_32);
        const double eff = (fp32fp8_copy_baseline_gelem > 0.0) ? (g_avg / fp32fp8_copy_baseline_gelem) : 0.0;
        std::cout << "  mode=" << std::setw(10) << mode
                  << " 32->E4: " << std::setw(7) << std::fixed << std::setprecision(2) << g_32_to_e4
                  << " 32->E5: " << std::setw(7) << g_32_to_e5
                  << " E4->32: " << std::setw(7) << g_e4_to_32
                  << " E5->32: " << std::setw(7) << g_e5_to_32
                  << " B/elem: 5"
                  << " | avg eff vs baseline: " << std::setw(6) << std::fixed << std::setprecision(2) << eff << '\n';
    }
    setenv("OMNICAST_FP32_FP8_IMPL", "auto", 1);

    const auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        omnicast::convert_fp32_to_fp16(h32.data(), out16.data(), count);
    }
    const auto cpu_end = std::chrono::high_resolution_clock::now();
    const double t_cpu_32_to_16 =
        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end - cpu_start).count()) /
        1.0e6 / static_cast<double>(iterations);

    const double t_gpu_32_to_16 = benchmark_kernel_ms([&]() {
        check_status(
            omnicast::convert_device(
                dev_src[fp32_idx],
                omnicast::GpuDType::fp32,
                dev_dst[fp16_idx],
                omnicast::GpuDType::fp16,
                count,
                nullptr
            ),
            "gpu speedup case"
        );
    }, iterations);

    std::cout << std::left << std::setw(18) << "GPU FP32->FP16" << " time: " << std::fixed << std::setprecision(4)
              << t_gpu_32_to_16 << " ms/iteration\n";
    std::cout << std::left << std::setw(18) << "CPU FP32->FP16" << " time: " << std::fixed << std::setprecision(4)
              << t_cpu_32_to_16 << " ms/iteration\n";
    std::cout << "GPU speedup (32->16): " << std::setprecision(2) << (t_cpu_32_to_16 / t_gpu_32_to_16) << "x\n";

    for (int i = 0; i < kTypeCount; ++i) {
        check_cuda(cudaFree(dev_src[i]), "free dev_src");
        check_cuda(cudaFree(dev_dst[i]), "free dev_dst");
    }
    check_cuda(cudaFree(dev_tmp64), "free dev_tmp64");
    return 0;
#endif
}
