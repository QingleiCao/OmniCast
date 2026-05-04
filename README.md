# OmniCast
Universal data-type casting/conversion across all precisions, CPUs, and GPUs.

## CPU (initial)

Current CPU implementation supports:

- FP64 -> FP32
- FP64 -> FP16
- FP32 -> FP64
- FP32 -> FP16
- FP16 -> FP32
- FP16 -> FP64

## Build and run

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
./build/tests/test_casts_cpp
```

The library target is `omnicast_cpu` and is built from `src`.

Builds are automatically tuned for the host CPU (`-march=native` / `-mtune=native` when supported), so install/build on each target machine for best performance.

For heterogeneous clusters, build on each node type (or per CPU family image) so each deployment gets the right native tuning and instruction-set compatibility.

### SIMD toggle

CPU SIMD dispatch is enabled by default. On Intel/AMD x86 CPUs, runtime selection prefers the highest available backend automatically:

- AVX512 + F16C
- AVX2 + F16C
- AVX + F16C
- scalar fallback

```bash
cmake -S . -B build -DOMNICAST_ENABLE_CPU_SIMD=OFF
```

You can force scalar at runtime without rebuilding:

```bash
OMNICAST_FORCE_SCALAR=1 ./build/tests/test_casts_cpp
```

You can request SIMD at runtime for explicit testing:

```bash
OMNICAST_FORCE_SIMD=1 ./build/tests/test_casts_cpp
```

If both `OMNICAST_FORCE_SCALAR` and `OMNICAST_FORCE_SIMD` are set, scalar takes priority.

### In-place conversion

Use `convert_inplace(...)` for single-buffer conversion with safety checks:

- shrinking (`FP64->FP32`, `FP64->FP16`, `FP32->FP16`) runs forward in-place
- widening (`FP16->FP32`, `FP16->FP64`, `FP32->FP64`) runs backward in-place
- returns `insufficient_capacity` if destination byte size exceeds provided capacity

### C test

A pure C test target is included:

```bash
ctest --test-dir build -R test_casts_c --output-on-failure
./build/tests/test_casts_c
```

An additional C benchmark compares out-of-place+memcpy vs in-place:

```bash
ctest --test-dir build -R test_inplace_compare_c --output-on-failure
./build/tests/test_inplace_compare_c --size 2097152
```

Enable relative FP64 error checking in either benchmark with:

```bash
./build/tests/test_casts_cpp --check 1
./build/tests/test_casts_c --check 1
```

Toggle in-place checks/benchmark with:

```bash
./build/tests/test_casts_cpp --inplace 0
./build/tests/test_casts_c --inplace 1
```

When `--inplace 1`, tests run in-place-only mode and benchmark all six conversion directions in-place.

Control benchmark data size (number of elements) with:

```bash
./build/tests/test_casts_cpp --size 2097152
./build/tests/test_casts_c --size 2097152
```

## NVIDIA CUDA backend

CUDA device-pointer casting is now integrated and optional at build time. CPU APIs remain unchanged.
GPU in-place conversion support has been removed; use distinct input/output device buffers for CUDA conversions.

### Build with/without CUDA

```bash
# Default: tries CUDA, falls back to CPU-only if toolkit/compiler is missing.
cmake -S . -B build

# Force CPU-only build.
cmake -S . -B build -DOMNICAST_ENABLE_CUDA=OFF

# Override default datacenter architectures (defaults: 80;86;89;90).
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES="80;90"
```

### Reproducible GPU presets (auto-detect runtime tuning still applies)

```bash
# A100 (SM80)
cmake --preset a100
cmake --build --preset a100
ctest --preset a100 -R test_gpu_casts_cpp

# H100 (SM90)
cmake --preset h100
cmake --build --preset h100
ctest --preset h100 -R test_gpu_casts_cpp

# B100 workflow (default preset is toolchain-safe)
cmake --preset b100
cmake --build --preset b100
ctest --preset b100 -R test_gpu_casts_cpp

# When your CUDA toolkit supports Blackwell SM100:
cmake --preset b100 -DCMAKE_CUDA_ARCHITECTURES=100
```

If a preset is run on a different GPU architecture than its target SM, `test_gpu_casts_cpp` now auto-skips to avoid false failures.

### Supported GPU types

- FP64 (`GpuDType::fp64`)
- FP32 (`GpuDType::fp32`)
- FP16 (`GpuDType::fp16`)
- FP8 E4M3 (`GpuDType::fp8_e4m3`)
- FP8 E5M2 (`GpuDType::fp8_e5m2`)

FP8 paths are architecture-aware with runtime auto-detection:

- A100 (SM80): `a100-fp8-emulated`
- H100 (SM90): `h100-fp8-native`
- B100-class (SM100+): `b100-fp8-native`
- other GPUs: `cuda-fp8-native` or `cuda-fp8-emulated`

### C++ device API example

```cpp
#include "omnicast/gpu_cast.hpp"

// input/output are device pointers, stream is optional (nullptr for default stream).
omnicast::GpuStatus st = omnicast::convert_device(
    input_dev,
    omnicast::GpuDType::fp64,
    output_dev,
    omnicast::GpuDType::fp8_e4m3,
    count,
    stream
);
```

### C device API example

```c
#include "omnicast/c_api.h"

omnicast_gpu_status st = omnicast_convert_device(
    input_dev,
    OMNICAST_GPU_DTYPE_FP32,
    output_dev,
    OMNICAST_GPU_DTYPE_FP16,
    count,
    (omnicast_cuda_stream)stream
);
```

### GPU tests and benchmarking

```bash
ctest --test-dir build -R test_gpu_casts_cpp --output-on-failure
./build/tests/test_gpu_casts_cpp --check 1 --size 1048576 --iters 50
```

`test_gpu_casts_cpp` includes:

- correctness checks with relative Frobenius norm `||A-A'||/||A||`
- timing output in `ms/iteration`
- CPU-vs-GPU A/B on FP32->FP16
- FP8 E4M3 vs E5M2 error/performance comparison
- configurable iteration count via `--iters N`

### Performance guidance

- Device APIs require device pointers; no hidden host-device copies are performed.
- Use streams to overlap conversions with other GPU work.
- For end-to-end transfer + kernel workflows, use pinned host memory for better H2D/D2H throughput.
- Batch larger buffers to amortize launch overhead.
- FP8 LUT remap mode can be overridden for A/B testing: `OMNICAST_FP8_LUT_MODE=x4|x16|auto` (default `auto`).
- Legend conventions used in benchmark sections:
  - `baseline`: denominator used for efficiency (`avg_eff = observed / baseline`)
  - `reference`: correctness-oriented path used as semantic reference
  - `recommended`: preferred high-performance paths for production
  - `optional`: useful benchmark variants, but not preferred defaults
  - `experimental`: available for study, not recommended as defaults
  - `default`: `auto` policy used when no explicit implementation is forced
- FP8 remap backend can be forced for benchmarking:
  `OMNICAST_FP8_REMAP_IMPL=scalar|copy32|copy64|copy128|lut8_const|lut8_shared|lut32_u32|lut32_u64|lut32_u128|lut16|lutx16|auto`.
  Uses the legend conventions above with baseline=`copy32|copy64|copy128`, reference=`scalar`,
  default=`auto` (`lut32_u32`), optional=`lut32_u64|lut32_u128`,
  experimental=`lut16|lut8_const|lutx16|lut8_shared` (`lut8_shared` currently slower).
- FP16<->FP8 implementation can be selected with:
  `OMNICAST_FP16_FP8_IMPL=scalar|native|packed|packed_wide|lut8_to_half|lut16_to_fp8|copyshape|auto`.
  Uses the legend conventions above with baseline=`copyshape` (bytes-per-element=3), reference=`scalar`,
  recommended=`native|packed`, optional=`packed_wide`, experimental=`lut8_to_half|lut16_to_fp8`,
  default=`auto` (best validated path).
  `auto` runs a one-time per-device microprobe and selects between `native|packed`.
  In current A100 measurements, this reaches roughly 94-96% of the measured bytes-per-element=3 copy-shape baseline.
  Set `OMNICAST_FP16_FP8_AUTOTUNE=0` to disable the microprobe and use static defaults.
- FP32<->FP8 implementation can be selected with:
  `OMNICAST_FP32_FP8_IMPL=scalar|native|packed|packed_wide|lut8_to_fp32|copyshape|auto`.
  Uses the legend conventions above with baseline=`copyshape` (bytes-per-element=5), reference=`scalar`,
  recommended=`native|packed`, optional=`packed_wide`, experimental=`lut8_to_fp32`,
  default=`auto` (best validated path).
  `auto` runs a one-time per-device microprobe and selects between `native|packed`.
  In current A100 measurements, this reaches roughly 95% of the measured bytes-per-element=5 copy-shape baseline.
  Set `OMNICAST_FP32_FP8_AUTOTUNE=0` to disable this microprobe.
- For byte/narrow-type paths, FP64-derived memory-roof ratios can be misleading; use the dedicated FP8 remap packed-copy and FP16/FP32<->FP8 copy-shape baselines.
- Current benchmark conclusion:
  FP8<->FP8 `lut32_u32` is about 70-72% of the packed-copy baseline, while FP16<->FP8 and FP32<->FP8 `auto` are already near realistic streaming ceilings (~94-96% and ~95% respectively).

### Environment examples

```bash
OMNICAST_FP8_REMAP_IMPL=auto ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP8_REMAP_IMPL=lut32_u32 ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP8_REMAP_IMPL=lut32_u64 ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP8_REMAP_IMPL=lut32_u128 ./build/tests/test_gpu_casts_cpp --check 1

OMNICAST_FP16_FP8_IMPL=auto ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP16_FP8_IMPL=native ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP16_FP8_IMPL=packed ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP16_FP8_IMPL=packed_wide ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP16_FP8_IMPL=lut8_to_half ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP16_FP8_IMPL=lut16_to_fp8 ./build/tests/test_gpu_casts_cpp --check 1

OMNICAST_FP32_FP8_IMPL=auto ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP32_FP8_IMPL=native ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP32_FP8_IMPL=packed ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP32_FP8_IMPL=packed_wide ./build/tests/test_gpu_casts_cpp --check 1
OMNICAST_FP32_FP8_IMPL=lut8_to_fp32 ./build/tests/test_gpu_casts_cpp --check 1
```
