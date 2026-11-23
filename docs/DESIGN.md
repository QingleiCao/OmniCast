# OmniCast 2 design

## Goals

1. Every conversion an HPC mixed-precision code needs — FP64, FP32, FP16,
   BF16, FP8 and integers, any pair — on the CPU and on NVIDIA and AMD GPUs.
2. At the memory roofline: a conversion reads `n * sizeof(src)` and writes
   `n * sizeof(dst)` bytes and should take as long as a copy of that traffic.
3. Exact and identical everywhere: the same bits from the CPU, CUDA and HIP
   paths, so that a result does not depend on where a tile was converted.
4. The layouts linear-algebra codes use (leading dimensions, transposed and
   padded operands for cuBLASLt, in place) without extra passes.
5. Every conversion with every option out of place and in place, on every
   backend: nothing a caller can do on one backend fails on another.

OmniCast 1 had separate C and C++ APIs per backend, FP16/FP32/FP64 on the
CPU only, FP8 only on the GPU with a dozen experimental code paths selected by
environment variables, no strides, no stochastic rounding, and no in place on
the GPU. Version 2 is a rewrite.

## API

One C header, `omnicast/omnicast.h`, used from C, C++, CUDA and HIP code.

- **Types** are an enum (`OMC_FP64`, ..., `OMC_UINT64`); buffers are `void *`.
  The library has one entry per operation, not one per type pair: 14 × 14
  pairs times the layouts would not be an API.
- **Operations**: `omc_convert` (1-D), `omc_convert_2d` (column-major, leading
  dimensions, optional transposition and zero padding), `omc_convert_inplace`
  and `omc_convert_inplace_2d`, `omc_amax_2d`; the same with `omc_gpu_` taking
  device pointers and a stream; scalar helpers (`omc_fp32_to_fp16`, ...).
- **Options** are a struct whose all-zero value is the default, passed by
  pointer (NULL for the defaults). New options extend the struct without
  changing signatures, and C callers write
  `&(omc_options_t){ .rounding = OMC_ROUND_STOCHASTIC, .seed = s }` inline.
- **Errors** are returned (`omc_status_t`), never printed: a library inside a
  runtime must not abort or write to stderr on its own.
- **In place and out of place**: every type pair with every option (rounding,
  saturation, stochastic rounding, the three scale modes and their
  combinations, transposition, padding) runs out of place and in place, on
  the host and on the GPU. In place is either the `_inplace` functions or
  `dst == src` in `omc_convert` / `omc_convert_2d` (source and destination
  must be identical or disjoint). In place without transposition or padding
  is a single pass with no allocation (on the GPU for any leading dimension);
  with transposition or padding the result is built in a temporary
  (`malloc`, or `cudaMallocAsync` / `hipMallocAsync` on the stream) and
  copied back. `tests/test_modes` checks all 15288 combinations of each mode
  on every backend.
- **GPU calls** are asynchronous on the given stream and run on the current
  device, like cuBLAS. The in-place and dynamically scaled conversions need a
  small zeroed workspace; OmniCast keeps one per (device, stream) unless the
  caller passes its own (`omc_gpu_workspace_size`).
- **Thread safety**: all calls may be made concurrently; the only shared state
  is the CPU dispatch table (set once) and the workspace cache (mutex).

## Numerics

The scalar definition of every conversion is `src/core/omc_scalar.hpp`,
compiled as host and device code. The fast paths must match it bit for bit.

**Small formats from FP32** (FP16, BF16, E4M3, E5M2 are one template on the
exponent and mantissa widths and on "finite-only"):

- normal results: rebias the exponent and round the mantissa to nearest even
  on the integer bit pattern (`(a + rebias + (half_ulp - 1) + odd) >> shift`);
- subnormal results: add a power of two whose FP32 ulp is the subnormal step;
  the FPU rounds at the right position (round-to-nearest-even mode);
- overflow: anything that rounds above the largest finite code becomes Inf /
  NaN (E4M3) or, saturated, the largest finite value;
- NaN: the canonical quiet NaN of the format.

Checked against NVIDIA's reference conversions (`__float2half_rn`,
`__float2bfloat16_rn`, `__nv_cvt_float_to_fp8`, host versions) on all 2^32
FP32 inputs, in both saturation modes: no mismatch.

**From FP64** the value is first rounded to FP32 *to odd* (truncate, set the
last bit when inexact). FP32 keeps at least p + 2 bits of every target, so the
second rounding gives the correctly rounded result of the FP64 value: one
rounding, not two. On CUDA this is `cvt.rz.f32.f64` plus an inexact test; on
AVX-512 an embedded-rounding `vcvtpd2ps`; elsewhere a branch-free fix-up of
the nearest FP32. Checked against an exact neighbour search and `__double2half`.

**Stochastic rounding** follows HiCMA-X 0.2 so that results stay comparable:
element (i, j) of a tile draws u24 = SplitMix64(bits(x) ^ SplitMix64(pos ^
seed)) >> 40, a 24-bit uniform integer, and rounds a = |x| down to lo, the
representable value below it, when u24 · 2^-24 < (up − a) / (up − lo).

Since up − lo is a power of two (the step), t = (a − lo) / step is an exact
binary fraction and the test is u24 + 2^24 t < 2^24, i.e. (u24 being an
integer) u24 + floor(2^24 t) < 2^24: round up exactly when u24 + t24 carries
into bit 24, t24 being the top 24 bits of t. lo and the bits of t are fields
of the bit pattern of a, shifted by an amount that depends only on the
exponent (more for targets' subnormals). So the integer core (`sr_code`) has
no floating-point operation, no division and no branch, and gives the same
bits as the floating-point form: checked on 4M random values for each of the
13 pairs and on every 61st FP32 pattern (tests/legacy_sr_ref.h keeps the old
form). FP64 → FP32 everywhere and FP64 → FP16 on GPUs keep the
floating-point form (exact reciprocal of the step instead of a division),
faster there. FP64 → FP32, FP64 → FP16 and FP32 → FP16 are tested bit for
bit against a copy of the HiCMA-X code. Stochastic rounding combines with
every scale mode (it rounds src · scale).

**Integers**: float → integer truncates (C) or rounds to nearest even, clamps
to the range, NaN → 0; integer → integer saturates; 64-bit integers go through
FP64 rounded to odd so that a second rounding to a narrow float is exact.

**Saturation** replaces overflow and ±Inf by the largest finite value of the
target, for every float target including FP64.

**Scaling** multiplies in the source's precision and rounds the product
before the conversion, on every backend. Device code is built without
floating-point contraction (CUDA `--fmad=false`, HIP `-ffp-contract=off`),
and an empty `asm` keeps the product: on gfx906 the AMDGPU backend otherwise
fuses an FP32 multiply with the FP16 conversion that follows into one
`v_mad_mix` (one rounding instead of two), even with contraction off.
Dynamic scaling (`OMC_SCALE_DYNAMIC_POW2`) takes float sources and any target
(for integers the largest value of the type is the default target).

## CPU backend

- `cpu_kernels.hpp` holds the kernel table and generic kernels: one loop per
  (source, destination) pair over the scalar element function, with the
  option flags hoisted into template parameters so that each loop body is
  constant. The header is compiled three times (baseline, `-mavx2 -mf16c
  -mfma`, AVX-512 F/BW/VL/DQ with 512-bit vectors) in different namespaces;
  the dispatcher picks the table from CPUID at first use (`OMNICAST_CPU_ISA`
  forces one).
- GCC 11 vectorized none of the FP8 and integer loops at first: with the
  default `-ftrapping-math` it refuses to if-convert selects that could raise
  floating-point exceptions on the unused side. OmniCast does not report FP
  exceptions, so the kernels are built with `-fno-trapping-math` (values are
  unchanged; `-ffast-math` stays off). FP8 encoding went from 1.5 to 0.33
  ns per element, stochastic rounding from 15.4 to 2.0.
- ISA files then replace the pairs that have hardware converts compilers do
  not generate from integer code: `vcvtps2ph`/`vcvtph2ps` (FP32↔FP16),
  `vcvtpd2ps`/`vcvtps2pd`, FP64 → FP32 rounded to odd before FP16/BF16/FP8,
  masked tails on AVX-512. On AArch64 the baseline file uses NEON `fcvtn`,
  `fcvtl` and `fcvtxn` (round to odd), untested on hardware so far.
- The AVX-512 table reuses the AVX2 float → integer kernels (clamp,
  truncating convert, saturating packs), faster than the generic AVX-512
  loops (FP32 → INT8 0.17 against 0.22 ns per element on Skylake-SP).
- FP8 on AVX2 (round 3): x · 2^k goes to FP16 by `vcvtps2ph` with truncation,
  and the last bit is set when that was inexact (`vcvtph2ps` and a compare).
  That is FP16 rounded to odd. With k = −8 for E4M3 (bias 7 against 15) and
  k = 0 for E5M2, the FP16 bit pattern lines up with the FP8 one, subnormals
  on subnormals. Round to nearest even on the 16-bit pattern, with overflow
  and NaN by compares, then finishes the job: about 2 operations per element
  against 20. Rounding to odd first makes the two roundings exact (11 ≥ 4 + 2
  significand bits). FP64 sources are rounded to odd to FP32 with the MXCSR
  rounding mode set to toward zero, plus the sticky bit, which takes 5
  operations per 4 elements against 12. The result is 1.0–2.3× faster on
  Skylake-SP, Broadwell and Zen 2. FP32 → E4M3 saturated keeps the integer
  encoder (as fast), and so does BF16. The AVX-512 table keeps its generic
  encoders, which are faster there.
- Large in-place conversions that grow (FP16/BF16/FP32 → wider) stream the
  segments they write, as out of place does above the same threshold. The
  lines written were read long before, as the input of higher elements.
- The 64-bit variable shifts of the integer stochastic rounding vectorize
  only with the exponent arithmetic in 64-bit lanes: GCC 11 narrows the count
  of `sig & ((1 << k) - 1)` to 32 bits and then gives up on the loop, so the
  remainder is computed as `sig - ((sig >> k) << k)`.
- 2-D calls loop over columns (contiguous matrices collapse to one run);
  transposition converts 64 × 64 blocks through a buffer; in place converts
  blocks of 1024 elements through an L1 buffer, front to back when elements
  shrink and back to front when they grow, so a block never overwrites input
  not yet read (column by column for 2-D, which is safe for any ld ≥ m).
- `num_threads > 1` splits the work with OpenMP (off when embedded in HiCMA-X,
  whose conversions already run inside PaRSEC tasks).

## GPU backend

One source (`gpu_kernels.cuh`) compiled by nvcc or hipcc; `gpu_rt.hpp` maps
the runtime calls. One translation unit per source type (generated by CMake)
instantiates the kernels for all 14 destinations, so the ~900 kernel
instantiations compile in parallel (35 s per architecture on 32 cores).

- **`k_cvt`** (1-D, 2-D, padded, scaled, stochastic). Each thread converts
  R vectors of VW elements, VW = 16 bytes of the wider type; in round r
  thread t takes vector r·blockDim + t, so every warp-level load and store is
  one contiguous range on both the narrow and the wide side. R = 4, except
  2 for stochastic rounding from 4- and 2-byte sources (compute bound: fewer
  registers, more resident warps); more rounds for widening conversions were
  measured slower. Contiguous matrices collapse to one column. Misaligned
  pointers or leading dimensions take a scalar path in the same kernel.
- **1-byte sources** (FP8, INT8, UINT8): each block computes the 256 possible
  results in shared memory, and an element is one lookup. FP8 → FP8 went
  from 43 to about 90 % of a copy on the H100 (they were compute bound,
  about 40 integer operations for 2 bytes moved).
- Profiling note: Nsight showed FP64 → FP32 at 9.4 µs where FP64 → FP16 took
  5.1 µs for the same reads. The SASS had two `STG.E` per vector instead of one
  `STG.E.64`: the scalar fallback path stores to the same addresses, and NVVM
  sank both into one store with the fallback's 4-byte alignment. The 8- and
  16-byte stores are now inline PTX (`st.global.v2/v4.u32`); the final kernel
  takes 5.5 µs (nb = 1024, A100).
- **`k_cvtT`** (transposed, for the cuBLASLt TN layout of FP8 GEMMs): 64 × 64
  tiles through shared memory with an odd-word row pitch (no bank conflicts),
  coalesced reads along the source columns and 4-byte packed stores along the
  destination columns (four FP8 or two FP16 values per thread). HiCMA-X wrote
  one byte per thread at a stride of ld. Each thread issues its 16 loads
  before converting any (they were issued 4 at a time): FP32 and FP16
  sources were latency bound (45-68 % of DRAM bandwidth on the H100).
- **`k_inplace`**, single-pass in place. A buffer that shrinks (FP32 → FP16)
  or grows cannot be converted by independent blocks: the output of element e
  overlaps the input of element e·size_dst/size_src, owned by another block.
  Blocks take chunks in address order from an atomic ticket (back to front
  when growing), load and convert their chunk into registers, publish a flag,
  then wait for the flags of the chunks their output overlaps — which always
  have earlier tickets, so they are already running and the wait cannot
  deadlock — and store. The last block resets the flags, so the workspace is
  zero for the next call. A matrix with ld > m is the linear range
  [0, (n − 1)·ld + m) whose elements with (e mod ld) ≥ m are gaps, neither
  read nor written: the ordering argument is unchanged. A flag is published
  without a fence (the block has written nothing and its loads are consumed),
  and polled with device-coherent loads instead of atomics; with this the
  kernel got 4-17 % faster on the H100. Buffers that are not 16-byte aligned
  take an element-by-element path. HiCMA-X converted into a scratch tile and
  copied back: 20 instead of 12 bytes per element for FP64 → FP32 and two
  launches.
- **`k_amax`** reduces with 16-byte loads, 4 columns in flight per thread,
  and writes per-block partials. The last block finishes the reduction and,
  for `OMC_SCALE_DYNAMIC_POW2`, writes {amax, s, 1/s}; the conversion kernel
  then reads s from device memory. That is two kernels and no memset
  (HiCMA-X: memset, amax, scale, convert). One column in flight per thread
  was too few on the Radeon VII (with four, the whole convert is 1.9× faster
  at 1024²).
- **`k_cvtT_dyn`** (opt-in, `OMNICAST_FUSED_DYNAMIC=1`): `k_amax` and
  `k_cvtT` in one cooperative kernel for transposed conversions with a
  dynamic scale. Each block reduces its tiles, then a grid barrier (an arrival
  and a departure counter in the workspace, left zero), then converts the
  same tiles in reverse order, so the last ones read come from L2. Alone it is
  1.03–1.33× faster on the H100 and 1.1–2.1× on the Radeon VII. It is not the
  default because a cooperative launch waits until all its blocks fit on the
  GPU and then holds it: beside GEMMs on other streams, HiCMA-X was 2 %
  slower.
- Conversion code on the device uses the hardware converts where they match
  the definition (`cvt.rn.f16.f32`, `cvt.f32.f16`, `cvt.rz.f32.f64`). On
  sm_89+, saturated FP8 targets use `cvt.rn.satfinite.{e4m3,e5m2}x2.f32`
  (round 3). It agrees with the integer encoder except for NaN, where it
  returns one canonical code, so a NaN input keeps the encoder's code and
  sign. FP64 sources go through FP32 rounded to odd first, as on the host.
  `omnicast_native_fp8` checks all 2^32 FP32 inputs on the GPU.

- Floating-point environment: the kernels assume round to nearest and IEEE
  subnormals. Under flush-to-zero / denormals-are-zero (MXCSR bits 15 and 6)
  `vcvtpd2ps` flushes subnormal results and the FP32 additions of the
  small-format encoders read subnormal inputs as zero; under another rounding
  mode the AVX2 converts round differently. Every host entry point therefore
  runs under `FpEnvGuard` (`cpu_api.cpp`), which installs the default MXCSR
  (AArch64: clears FPCR.FZ/FZ16/RMode/DN/AHP and the trap enables) for the
  call and restores the caller's, keeping the status flags. OpenMP threads
  have their own control register, so each thread body takes a guard too. The
  cost is one `stmxcsr` per call (and per column or 64×64 block in threaded
  2-D calls) when the environment is already the default. Found with
  HiCMA-X, which sets FTZ/DAZ on its CPU threads (subnormal FP32 tiles cost
  microcode assists in MKL's SGEMM): without the guard, 803 checks of
  `omnicast_cpu` and 288 of `omnicast_modes` changed under FTZ/DAZ.

## Testing

| Test | What |
|---|---|
| `omnicast_exhaustive` | all 2^32 FP32 inputs → FP16/BF16/E4M3/E5M2, both saturation modes, against the CUDA headers; all codes → FP32; FP64 edge cases and 4M random values against an exact reference |
| `omnicast_cpu` | every pair × 3 option sets × lengths 1..4099 (tails, misalignment) on every CPU path against the scalar core; stochastic rounding against HiCMA-X 0.2; 720 2-D/transposed/padded cases; in place for every pair (1-D and 2-D with ld > m); scaling, amax, threads, errors; 210 cases (14 pairs and option sets × 5 layouts × each CPU path), amax and the scalar functions computed again with FTZ, DAZ and rounding toward zero set in the caller and in the OpenMP threads, bit for bit; the SIMD FP8 encoders on every 47th FP32 pattern (`OMNICAST_TEST_FP8_STRIDE=1`: all), every FP16/BF16 pattern and FP64 edge values |
| `omnicast_gpu` | every pair × 2 option sets × aligned/misaligned against the host; 896 2-D/transposed/padded/stochastic cases; in place for every pair up to 10^6 elements, 4 concurrent streams × 20 calls; dynamic and device scaling; 144 fused dynamic-scale transposed cases (float sources, nearest and stochastic, NaN/infinity, 1 to many tiles per block) and 4 concurrent streams × 8 calls with a different scale each |
| `omnicast_native_fp8` | CUDA, sm_89+: the hardware FP8 encode against the integer encoder on all 2^32 FP32 inputs (E4M3, E5M2) and 2^30 FP64 values |
| `omnicast_modes`, `omnicast_modes_gpu` | every pair × 9 option sets (default, saturate, nearest, host scale, device scale, dynamic scale, stochastic, stochastic + scale, stochastic + dynamic) × 6 layouts (1-D, 2-D, ld > m, transposed, padded, both) × {out of place, in place, dst == src}, small odd and multi-chunk sizes, aligned and not: 15288 calls per mode, against the generic host conversion, on every host code path and on the GPU |

## Limitations and next steps

- Packed 4-bit formats (FP4 E2M1, INT4) and microscaling (MX block scales) are
  not implemented yet; the enum and the transposed kernel leave room for them.
- The native FP8 converts of gfx942/950 are not used (no hardware here); those
  of sm_89+ are, for saturated targets (round 3).
- Stochastic rounding is bound by its random numbers: two SplitMix64 hashes
  (four 64-bit multiplies) per element, kept for compatibility with
  HiCMA-X 0.2. From 2-byte sources to FP8 that is 25-40 % of a copy on GPUs.
  A counter-based generator producing several numbers per call would be a
  new rounding definition (an option, not a replacement).
- Dynamic scaling reads the source twice (amax, then the conversion). The
  fused kernel (`k_cvtT_dyn`) reads the second time mostly from L2, but it
  stays opt-in because of its cooperative launch.
- On small tiles (2048², L2-resident) in place is slower than out of place
  (median 1.1x on the H100): the ordered single pass costs a ticket, a flag
  and a barrier per chunk. Persistent blocks taking tickets in a loop measured
  0.94–1.02× (round 3) and were not kept.
- On the Radeon VII, the transposed convert of 8192² FP64 tiles takes 4.3×
  the plain one (2.1× at 4096²). The fused kernel, with the same tile code
  over a persistent grid, takes half of it, so the launch geometry of
  `k_cvtT` is the suspect there.
- The NEON code path compiles in the generic file but has not run on hardware.
