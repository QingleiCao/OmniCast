# OmniCast

Fast, exact data-type conversion on CPUs and GPUs (NVIDIA CUDA, AMD HIP), behind
one C API.

| | |
|---|---|
| Types | FP64, FP32, FP16, BF16, FP8 E4M3 / E5M2 (OCP), signed and unsigned 8/16/32/64-bit integers — every pair |
| Layouts | contiguous arrays; column-major matrices with leading dimensions; transposed output; zero-padded output |
| Modes | every conversion with every option out of place **and in place** (single pass, also for strided matrices on the GPU), on the CPU, CUDA and HIP |
| Rounding | nearest even (bit-identical to the CUDA conversion intrinsics); stochastic rounding (reproducible, independent of the thread decomposition); truncation for integers |
| Extras | saturation, scaling by a host or device factor, dynamic power-of-two scaling from the amax (the FP8 recipe, also into integers), stochastic rounding of scaled values, amax |
| CPU | AVX-512, AVX2+F16C, NEON, generic — chosen at run time |
| GPU | CUDA (sm_70 and later) and HIP (gfx9 and later) from one source; asynchronous on a stream |

Every conversion gives the same bits on every backend (NaN payloads aside):
the scalar definition in `src/core/omc_scalar.hpp` is checked exhaustively
against NVIDIA's reference conversions (all 2^32 FP32 inputs into FP16, BF16,
E4M3 and E5M2, both saturation modes), and the SIMD and GPU kernels are checked
against it for every type pair.

## API

```c
#include <omnicast/omnicast.h>

/* host */
omc_convert(OMC_FP64, x, OMC_FP16, h, n, NULL);                     /* 1-D */
omc_convert_2d(OMC_FP32, A, lda, OMC_BF16, B, ldb, m, n, NULL);     /* matrix */
omc_convert_inplace(OMC_FP64, OMC_FP32, buf, n, NULL);              /* in place */
omc_convert(OMC_FP64, buf, OMC_FP32, buf, n, NULL);                  /* the same: dst == src */

/* GPU: device pointers, asynchronous on the stream */
omc_gpu_convert_2d(OMC_FP64, dA, lda, OMC_FP16, dB, ldb, m, n, NULL, stream);
omc_gpu_convert_inplace(OMC_FP32, OMC_FP16, dC, n, NULL, stream);   /* one pass, no copy */
omc_gpu_convert_inplace_2d(OMC_FP64, OMC_FP16, dT, ld, m, n, NULL, stream);  /* any ld >= m */

/* options: a zero-initialized struct (or NULL) means the defaults */
omc_options_t o = { .rounding = OMC_ROUND_STOCHASTIC, .seed = 42 };
omc_gpu_convert_2d(OMC_FP64, dA, lda, OMC_FP32, dS, lds, m, n, &o, stream);

/* FP8 operand of a cuBLASLt TN matmul: transposed, padded, scaled by a power of two */
omc_options_t f8 = { .flags = OMC_FLAG_SATURATE | OMC_FLAG_TRANSPOSE,
                     .dst_rows = 2048, .dst_cols = 2048,
                     .scale_mode = OMC_SCALE_DYNAMIC_POW2, .scale_ptr = d_scale /* {amax, s, 1/s} */ };
omc_gpu_convert_2d(OMC_FP32, dA, lda, OMC_FP8_E4M3, dA8, 2048, m, n, &f8, stream);
```

All functions return an `omc_status_t`. The full reference is in
[`include/omnicast/omnicast.h`](include/omnicast/omnicast.h).

### Semantics

- float → narrower float: round to nearest even. FP64 sources are rounded
  once, directly to the target (no double rounding through FP32).
- Overflow gives Inf (FP32, FP16, BF16, E5M2) or NaN (E4M3, which has no Inf);
  `OMC_FLAG_SATURATE` gives the largest finite value instead (also for ±Inf).
- float → integer: truncation (C cast) by default, nearest even with
  `OMC_ROUND_NEAREST_EVEN`; always saturating; NaN → 0.
- Stochastic rounding (`OMC_ROUND_STOCHASTIC`, float targets narrower than the
  source): element (i, j) rounds down with probability (up − x)/(up − down),
  with a uniform number hashed (SplitMix64) from (i, j), the seed and the bits of
  x. Results are reproducible and identical on the CPU, CUDA and HIP. FP64 → FP32
  and FP64/FP32 → FP16 are bit-compatible with HiCMA-X 0.2.
- Scaling multiplies in the source's precision (FP64 for FP64 and 32/64-bit
  integer sources, FP32 otherwise) and rounds the product before the
  conversion, on every backend.
- In place: the `_inplace` functions, or `dst == src`. Without transposition
  or padding it is a single pass with no allocation; transposed or padded
  results go through a temporary and a copy back.

## Build

```bash
cmake -S . -B build -DOMNICAST_GPU_BACKEND=AUTO      # AUTO, CUDA, HIP or NONE
cmake --build build -j
ctest --test-dir build --output-on-failure          # -L exhaustive for the 2^32 sweep (CUDA)
```

Options: `CMAKE_CUDA_ARCHITECTURES` / `CMAKE_HIP_ARCHITECTURES`,
`OMNICAST_BUILD_SHARED` (default static, position independent),
`OMNICAST_OPENMP` (host threads for `num_threads > 1`),
`OMNICAST_BUILD_TESTS`, `OMNICAST_BUILD_BENCHMARKS`, `OMNICAST_INSTALL`.
Installed, `find_package(OmniCast)` gives `OmniCast::omnicast`. As a
subdirectory, link `omnicast`. `OMNICAST_CPU_ISA=avx2` (or `omc_cpu_set_isa`)
forces a CPU code path. `OMNICAST_STREAM_BYTES` sets the destination size
above which host conversions use streaming stores (0: never).
`OMNICAST_FUSED_DYNAMIC=1` makes transposed GPU conversions with
`OMC_SCALE_DYNAMIC_POW2` run as one cooperative kernel. That is faster when
nothing else runs on the GPU, and slower beside kernels of other streams
(`docs/REPORT.md`, section 5).

## Design

- **One scalar definition** (`src/core/omc_scalar.hpp`, `__host__ __device__`):
  small formats are encoded from FP32 by integer rounding of the bit pattern
  (normals) and one FP32 addition of a power of two (subnormals); FP64 goes
  through FP32 rounded to odd, which makes the second rounding exact. Written
  branch-free so that compilers vectorize it.
- **CPU**: the generic kernels are compiled once per instruction set
  (baseline, AVX2+F16C, AVX-512) with the same source, and the hot pairs
  (FP64↔FP32, FP32↔FP16, FP64↔FP16, FP64→BF16/FP8) use the hardware converts
  (`vcvtps2ph`, `vcvtph2ps`, `vcvt[t]pd2ps` with embedded rounding). The table
  is picked at start-up from CPUID. In place converts blocks through an
  L1-sized buffer, front to back or back to front.
- **GPU**: one source compiled by nvcc or hipcc, one translation unit per
  source type. `k_cvt` gives each thread vectors of 16 bytes of the wider type
  in striped rounds, so both sides are fully coalesced; transposition goes
  through a 64×64 shared tile with 4-byte packed stores; in place is a single
  pass where blocks take chunks in address order and wait (on flags in a small
  per-stream workspace) only for the chunks their output overlaps.
- **Tests**: exhaustive scalar checks against the CUDA host reference
  conversions; every pair on every CPU code path; every pair on the GPU against
  the host; 2-D, transposed, padded, in place (including 4 concurrent streams),
  stochastic rounding against the HiCMA-X implementation, scaling.

`docs/REPORT.md` is the full report (design, profiling, performance);
`docs/DESIGN.md` has the design details and `docs/PERFORMANCE.md` the
measurement tables.
