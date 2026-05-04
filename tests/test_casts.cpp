#include "omnicast/cpu_cast.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

template <typename Fn>
double benchmark_ms(Fn&& fn, int iterations) {
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        fn();
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    return static_cast<double>(total_ns) / 1.0e6 / static_cast<double>(iterations);
}

void print_result_with_speedup(const char* label, double scalar_ms, double auto_ms) {
    const double speedup = (auto_ms > 0.0) ? (scalar_ms / auto_ms) : 0.0;
    std::cout << std::left << std::setw(12) << label
              << " scalar: " << std::setw(10) << std::fixed << std::setprecision(4) << scalar_ms << " ms/iteration"
              << " auto: " << std::setw(10) << std::fixed << std::setprecision(4) << auto_ms << " ms/iteration"
              << " speedup: " << std::setprecision(2) << speedup << "x\n";
}

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
    if (denom == 0.0) {
        return 0.0;
    }
    return std::sqrt(diff_accum) / denom;
}

void print_f_norm(const char* label, double value) {
    std::cout << std::left << std::setw(12) << label
              << " rel F-norm (||A-A'||/||A||): " << std::scientific << std::setprecision(6)
              << value << '\n';
}

void check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "check failed: " << message << '\n';
        std::abort();
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool check_error = false;
    bool run_inplace = false;
    std::size_t count = 1U << 20U;

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
        } else if (std::strcmp(argv[i], "--inplace") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for --inplace (use 0 or 1)\n";
                return 2;
            }
            const int value = std::atoi(argv[++i]);
            if (value != 0 && value != 1) {
                std::cerr << "invalid --inplace value (use 0 or 1)\n";
                return 2;
            }
            run_inplace = (value == 1);
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
        } else {
            std::cerr << "unknown argument: " << argv[i] << '\n';
            std::cerr << "usage: test_casts_cpp [--check 0|1] [--inplace 0|1] [--size N]\n";
            return 2;
        }
    }

    constexpr int iterations = 30;

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist64(-1000.0, 1000.0);

    std::vector<double> fp64_in(count);
    for (double& x : fp64_in) {
        x = dist64(rng);
    }

    std::vector<float> fp32_in(count);
    for (std::size_t i = 0; i < count; ++i) {
        fp32_in[i] = static_cast<float>(fp64_in[i]);
    }

    std::vector<omnicast::fp16_t> fp16_in(count);
    omnicast::convert_fp32_to_fp16(fp32_in.data(), fp16_in.data(), count);

    std::vector<float> fp32_out(count, 0.0F);
    std::vector<double> fp64_out(count, 0.0);
    std::vector<omnicast::fp16_t> fp16_out(count, 0);
    std::vector<unsigned char> inplace_buf(count * sizeof(double), 0);
    auto* as_f64 = reinterpret_cast<double*>(inplace_buf.data());
    auto* as_f32 = reinterpret_cast<float*>(inplace_buf.data());
    auto* as_f16 = reinterpret_cast<omnicast::fp16_t*>(inplace_buf.data());

    const auto benchmark_inplace = [&](const void* src, std::size_t src_bytes,
                                       omnicast::DType src_t, omnicast::DType dst_t) {
        return benchmark_ms([&]() {
            std::memcpy(inplace_buf.data(), src, src_bytes);
            (void)omnicast::convert_inplace(inplace_buf.data(), count, src_t, dst_t, inplace_buf.size());
        }, iterations);
    };

    double t_64_to_32_scalar = 0.0;
    double t_64_to_32_auto = 0.0;
    double t_64_to_16_scalar = 0.0;
    double t_64_to_16_auto = 0.0;
    double t_32_to_64_scalar = 0.0;
    double t_32_to_64_auto = 0.0;
    double t_32_to_16_scalar = 0.0;
    double t_32_to_16_auto = 0.0;
    double t_16_to_32_scalar = 0.0;
    double t_16_to_32_auto = 0.0;
    double t_16_to_64_scalar = 0.0;
    double t_16_to_64_auto = 0.0;

    double t_inplace_64_to_32 = 0.0;
    double t_inplace_64_to_16 = 0.0;
    double t_inplace_32_to_64 = 0.0;
    double t_inplace_32_to_16 = 0.0;
    double t_inplace_16_to_32 = 0.0;
    double t_inplace_16_to_64 = 0.0;

    // Basic correctness checks so timing doesn't hide broken kernels.
    const float ref_32 = omnicast::fp64_to_fp32(123.5);
    const omnicast::fp16_t ref_16 = omnicast::fp64_to_fp16(123.5);
    const double back_64 = omnicast::fp16_to_fp64(ref_16);
    check(std::fabs(ref_32 - 123.5F) < 1e-6F, "scalar conversion check");
    check(std::fabs(back_64 - 123.5) < 0.1, "scalar conversion round-trip check");

    std::cout << "OmniCast CPU conversion benchmark (" << count
              << " elements, avg over " << iterations << " iterations)\n";
    std::cout << "Active backend    : " << omnicast::simd_backend()
              << " (SIMD available: " << (omnicast::simd_available() ? "yes" : "no") << ")\n";

    if (run_inplace) {
        // In-place correctness across all six conversions.
        std::memcpy(as_f64, fp64_in.data(), count * sizeof(double));
        check(omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp64,
                                        omnicast::DType::fp32, inplace_buf.size()) ==
                  omnicast::InplaceStatus::success,
              "inplace fp64->fp32");
        check(std::fabs(as_f32[17] - static_cast<float>(fp64_in[17])) < 1e-5F, "inplace fp64->fp32 value");

        std::memcpy(as_f64, fp64_in.data(), count * sizeof(double));
        check(omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp64,
                                        omnicast::DType::fp16, inplace_buf.size()) ==
                  omnicast::InplaceStatus::success,
              "inplace fp64->fp16");

        std::memcpy(as_f32, fp32_in.data(), count * sizeof(float));
        check(omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp32,
                                        omnicast::DType::fp64, inplace_buf.size()) ==
                  omnicast::InplaceStatus::success,
              "inplace fp32->fp64");

        std::memcpy(as_f32, fp32_in.data(), count * sizeof(float));
        check(omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp32,
                                        omnicast::DType::fp16, inplace_buf.size()) ==
                  omnicast::InplaceStatus::success,
              "inplace fp32->fp16");

        std::memcpy(as_f16, fp16_in.data(), count * sizeof(omnicast::fp16_t));
        check(omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp16,
                                        omnicast::DType::fp32, inplace_buf.size()) ==
                  omnicast::InplaceStatus::success,
              "inplace fp16->fp32");

        std::memcpy(as_f16, fp16_in.data(), count * sizeof(omnicast::fp16_t));
        check(omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp16,
                                        omnicast::DType::fp64, inplace_buf.size()) ==
                  omnicast::InplaceStatus::success,
              "inplace fp16->fp64");

        check(omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp16,
                                        omnicast::DType::fp64, count * sizeof(omnicast::fp16_t)) ==
                  omnicast::InplaceStatus::insufficient_capacity,
              "inplace insufficient capacity");

        t_inplace_64_to_32 = benchmark_inplace(fp64_in.data(), count * sizeof(double), omnicast::DType::fp64, omnicast::DType::fp32);
        t_inplace_64_to_16 = benchmark_inplace(fp64_in.data(), count * sizeof(double), omnicast::DType::fp64, omnicast::DType::fp16);
        t_inplace_32_to_64 = benchmark_inplace(fp32_in.data(), count * sizeof(float), omnicast::DType::fp32, omnicast::DType::fp64);
        t_inplace_32_to_16 = benchmark_inplace(fp32_in.data(), count * sizeof(float), omnicast::DType::fp32, omnicast::DType::fp16);
        t_inplace_16_to_32 = benchmark_inplace(fp16_in.data(), count * sizeof(omnicast::fp16_t), omnicast::DType::fp16, omnicast::DType::fp32);
        t_inplace_16_to_64 = benchmark_inplace(fp16_in.data(), count * sizeof(omnicast::fp16_t), omnicast::DType::fp16, omnicast::DType::fp64);

        std::cout << std::left << std::setw(12) << "InP64->P32"
                  << " auto: " << std::setw(10) << std::fixed << std::setprecision(4) << t_inplace_64_to_32 << " ms/iteration\n";
        std::cout << std::left << std::setw(12) << "InP64->P16"
                  << " auto: " << std::setw(10) << std::fixed << std::setprecision(4) << t_inplace_64_to_16 << " ms/iteration\n";
        std::cout << std::left << std::setw(12) << "InP32->P64"
                  << " auto: " << std::setw(10) << std::fixed << std::setprecision(4) << t_inplace_32_to_64 << " ms/iteration\n";
        std::cout << std::left << std::setw(12) << "InP32->P16"
                  << " auto: " << std::setw(10) << std::fixed << std::setprecision(4) << t_inplace_32_to_16 << " ms/iteration\n";
        std::cout << std::left << std::setw(12) << "InP16->P32"
                  << " auto: " << std::setw(10) << std::fixed << std::setprecision(4) << t_inplace_16_to_32 << " ms/iteration\n";
        std::cout << std::left << std::setw(12) << "InP16->P64"
                  << " auto: " << std::setw(10) << std::fixed << std::setprecision(4) << t_inplace_16_to_64 << " ms/iteration\n";

        if (check_error) {
            std::memcpy(as_f64, fp64_in.data(), count * sizeof(double));
            (void)omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp64, omnicast::DType::fp32, inplace_buf.size());
            const double e_64_to_32 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return static_cast<double>(as_f32[i]); });

            std::memcpy(as_f64, fp64_in.data(), count * sizeof(double));
            (void)omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp64, omnicast::DType::fp16, inplace_buf.size());
            const double e_64_to_16 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return omnicast::fp16_to_fp64(as_f16[i]); });

            std::memcpy(as_f32, fp32_in.data(), count * sizeof(float));
            (void)omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp32, omnicast::DType::fp64, inplace_buf.size());
            const double e_32_to_64 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return as_f64[i]; });

            std::memcpy(as_f32, fp32_in.data(), count * sizeof(float));
            (void)omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp32, omnicast::DType::fp16, inplace_buf.size());
            const double e_32_to_16 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return omnicast::fp16_to_fp64(as_f16[i]); });

            std::memcpy(as_f16, fp16_in.data(), count * sizeof(omnicast::fp16_t));
            (void)omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp16, omnicast::DType::fp32, inplace_buf.size());
            const double e_16_to_32 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return static_cast<double>(as_f32[i]); });

            std::memcpy(as_f16, fp16_in.data(), count * sizeof(omnicast::fp16_t));
            (void)omnicast::convert_inplace(inplace_buf.data(), count, omnicast::DType::fp16, omnicast::DType::fp64, inplace_buf.size());
            const double e_16_to_64 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return as_f64[i]; });

            print_f_norm("FP64->FP32", e_64_to_32);
            print_f_norm("FP64->FP16", e_64_to_16);
            print_f_norm("FP32->FP64", e_32_to_64);
            print_f_norm("FP32->FP16", e_32_to_16);
            print_f_norm("FP16->FP32", e_16_to_32);
            print_f_norm("FP16->FP64", e_16_to_64);
        } else {
            std::cout << "Error check      : disabled (use --check 1)\n";
        }
    } else {
        t_64_to_32_scalar = benchmark_ms([&]() { omnicast::convert_fp64_to_fp32_scalar(fp64_in.data(), fp32_out.data(), count); }, iterations);
        t_64_to_32_auto = benchmark_ms([&]() { omnicast::convert_fp64_to_fp32(fp64_in.data(), fp32_out.data(), count); }, iterations);
        t_64_to_16_scalar = benchmark_ms([&]() { omnicast::convert_fp64_to_fp16_scalar(fp64_in.data(), fp16_out.data(), count); }, iterations);
        t_64_to_16_auto = benchmark_ms([&]() { omnicast::convert_fp64_to_fp16(fp64_in.data(), fp16_out.data(), count); }, iterations);
        t_32_to_64_scalar = benchmark_ms([&]() { omnicast::convert_fp32_to_fp64_scalar(fp32_in.data(), fp64_out.data(), count); }, iterations);
        t_32_to_64_auto = benchmark_ms([&]() { omnicast::convert_fp32_to_fp64(fp32_in.data(), fp64_out.data(), count); }, iterations);
        t_32_to_16_scalar = benchmark_ms([&]() { omnicast::convert_fp32_to_fp16_scalar(fp32_in.data(), fp16_out.data(), count); }, iterations);
        t_32_to_16_auto = benchmark_ms([&]() { omnicast::convert_fp32_to_fp16(fp32_in.data(), fp16_out.data(), count); }, iterations);
        t_16_to_32_scalar = benchmark_ms([&]() { omnicast::convert_fp16_to_fp32_scalar(fp16_in.data(), fp32_out.data(), count); }, iterations);
        t_16_to_32_auto = benchmark_ms([&]() { omnicast::convert_fp16_to_fp32(fp16_in.data(), fp32_out.data(), count); }, iterations);
        t_16_to_64_scalar = benchmark_ms([&]() { omnicast::convert_fp16_to_fp64_scalar(fp16_in.data(), fp64_out.data(), count); }, iterations);
        t_16_to_64_auto = benchmark_ms([&]() { omnicast::convert_fp16_to_fp64(fp16_in.data(), fp64_out.data(), count); }, iterations);

        print_result_with_speedup("FP64->FP32", t_64_to_32_scalar, t_64_to_32_auto);
        print_result_with_speedup("FP64->FP16", t_64_to_16_scalar, t_64_to_16_auto);
        print_result_with_speedup("FP32->FP64", t_32_to_64_scalar, t_32_to_64_auto);
        print_result_with_speedup("FP32->FP16", t_32_to_16_scalar, t_32_to_16_auto);
        print_result_with_speedup("FP16->FP32", t_16_to_32_scalar, t_16_to_32_auto);
        print_result_with_speedup("FP16->FP64", t_16_to_64_scalar, t_16_to_64_auto);
        std::cout << "Inplace check    : disabled in this mode (use --inplace 1)\n";

        if (check_error) {
            omnicast::convert_fp64_to_fp32(fp64_in.data(), fp32_out.data(), count);
            const double e_64_to_32 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return static_cast<double>(fp32_out[i]); });
            omnicast::convert_fp64_to_fp16(fp64_in.data(), fp16_out.data(), count);
            const double e_64_to_16 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return omnicast::fp16_to_fp64(fp16_out[i]); });
            omnicast::convert_fp32_to_fp64(fp32_in.data(), fp64_out.data(), count);
            const double e_32_to_64 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return fp64_out[i]; });
            omnicast::convert_fp32_to_fp16(fp32_in.data(), fp16_out.data(), count);
            const double e_32_to_16 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return omnicast::fp16_to_fp64(fp16_out[i]); });
            omnicast::convert_fp16_to_fp32(fp16_in.data(), fp32_out.data(), count);
            const double e_16_to_32 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return static_cast<double>(fp32_out[i]); });
            omnicast::convert_fp16_to_fp64(fp16_in.data(), fp64_out.data(), count);
            const double e_16_to_64 = relative_f_norm_error(count, [&](std::size_t i) { return fp64_in[i]; },
                                                             [&](std::size_t i) { return fp64_out[i]; });
            print_f_norm("FP64->FP32", e_64_to_32);
            print_f_norm("FP64->FP16", e_64_to_16);
            print_f_norm("FP32->FP64", e_32_to_64);
            print_f_norm("FP32->FP16", e_32_to_16);
            print_f_norm("FP16->FP32", e_16_to_32);
            print_f_norm("FP16->FP64", e_16_to_64);
        } else {
            std::cout << "Error check      : disabled (use --check 1)\n";
        }
    }

    return 0;
}
