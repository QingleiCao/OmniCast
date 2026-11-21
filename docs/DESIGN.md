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
element (i, j) of a tile draws u = SplitMix64(bits(x) ^ SplitMix64(pos ^ seed))
>> 40 as a 24-bit uniform number, and rounds |x| down to the representable
value below it with probability (up − |x|) / (up − down). Two rewrites keep
the bits while making it fast: the code is branch-free (so it vectorizes on
the CPU), and the division by up − down, a power of two, is a product by its
exact reciprocal built from the exponent bits. FP64 → FP32, FP64 → FP16 and
FP32 → FP16 are tested bit for bit against a copy of the HiCMA-X code.

**Integers**: float → integer truncates (C) or rounds to nearest even, clamps
to the range, NaN → 0; integer → integer saturates; 64-bit integers go through
FP64 rounded to odd so that a second rounding to a narrow float is exact.

**Saturation** replaces overflow and ±Inf by the largest finite value of the
target, for every float target including FP64.

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
  `kRounds` vectors of VW elements, VW = 16 bytes of the wider type; in round r
  thread t takes vector r·blockDim + t, so every warp-level load and store is
  one contiguous range on both the narrow and the wide side. Contiguous
  matrices collapse to one column. Misaligned pointers or leading dimensions
  take a scalar path in the same kernel.
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
  one byte per thread at a stride of ld.
- **`k_inplace`**, single-pass in place. A buffer that shrinks (FP32 → FP16)
  or grows cannot be converted by independent blocks: the output of element e
  overlaps the input of element e·size_dst/size_src, owned by another block.
  Blocks take chunks in address order from an atomic ticket (back to front
  when growing), load and convert their chunk into registers, publish a flag,
  then wait for the flags of the chunks their output overlaps — which always
  have earlier tickets, so they are already running and the wait cannot
  deadlock — and store. The last block resets the flags, so the workspace is
  zero for the next call. HiCMA-X converted into a scratch tile and copied
  back: 20 instead of 12 bytes per element for FP64 → FP32 and two launches.
- **`k_amax`** reduces with 16-byte loads, writes per-block partials, and the
  last block finishes the reduction and, for `OMC_SCALE_DYNAMIC_POW2`, writes
  {amax, s, 1/s}; the conversion kernel then reads s from device memory. Two
  kernels, no memset (HiCMA-X: memset, amax, scale, convert).
- Conversion code on the device uses the hardware converts where they match
  the definition (`cvt.rn.f16.f32`, `cvt.f32.f16`, `cvt.rz.f32.f64`).

## Testing

| Test | What |
|---|---|
| `omnicast_exhaustive` | all 2^32 FP32 inputs → FP16/BF16/E4M3/E5M2, both saturation modes, against the CUDA headers; all codes → FP32; FP64 edge cases and 4M random values against an exact reference |
| `omnicast_cpu` | every pair × 3 option sets × lengths 1..4099 (tails, misalignment) on every CPU path against the scalar core; stochastic rounding against HiCMA-X 0.2; 720 2-D/transposed/padded cases; in place for every pair (1-D and 2-D with ld > m); scaling, amax, threads, errors |
| `omnicast_gpu` | every pair × 2 option sets × aligned/misaligned against the host; 896 2-D/transposed/padded/stochastic cases; in place for every pair up to 10^6 elements, 4 concurrent streams × 20 calls; dynamic and device scaling |

## Limitations and next steps

- Packed 4-bit formats (FP4 E2M1, INT4) and microscaling (MX block scales) are
  not implemented yet; the enum and the transposed kernel leave room for them.
- The native FP8 converts of sm_89+ (`cvt.rn.satfinite.e4m3x2.f32`) and of
  gfx942/950 are not used yet; the software path matches them bit for bit.
- GPU in place with a size change needs a contiguous buffer (ld == m).
- Float → small integer on the CPU is about half the speed of a plain C cast,
  the price of saturation and NaN handling.
- AVX2 FP8 encoding is compute bound (about 20 integer operations per element).
- The NEON code path compiles in the generic file but has not run on hardware.
