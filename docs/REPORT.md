# OmniCast 2: design, profiling and performance

This report covers OmniCast 2 after its second round of work: every
conversion in place and out of place on every backend, a profiling campaign
on NVIDIA and AMD GPUs and on three x86 CPUs, the optimizations it led to,
and the measurements. `DESIGN.md` is the design reference, `PERFORMANCE.md`
the tables of the HiCMA-X comparison, and `results/` the raw outputs; every
number below comes from a file there or under `results/profiling/`.

Platforms: H100 PCIe (hexane), V100-SXM2-32GB and Xeon E5-2698 v4 Broadwell
(leconte), Radeon VII gfx906 and EPYC 7302 Zen 2 (histamine2, histamine3), A100-PCIe-40GB
and Xeon Gold 6140 Skylake-SP (methane). CUDA 12.6, ROCm 6.4.3, GCC 11.5.
The A100 was taken by another user's job (39 GB, 100 % utilization) during
this round, so its numbers are from the first round (2026-11-21) and its
Nsight Compute sweep describes the code before this round.

## Summary

- **Coverage.** All 196 type pairs × 9 option sets × 6 layouts were checked out
  of place, in place, and with `dst == src`, on the host (generic, AVX2,
  AVX-512) and on CUDA and HIP. That is 15288 calls per mode on each backend,
  and every one gave the status and the bits of the reference. Section 1.
- **Large GPU tiles run at copy speed.** At 8192², the median pair reaches
  94–101 % of a device copy of the same bytes out of place and 90–97 % in
  place, on the H100, the V100 and the Radeon VII. In place costs a median
  2–5 % more time than out of place.
- **Against the HiCMA-X 0.2 kernels, 8192² tiles, same output bytes** (H100,
  V100, Radeon VII; the HiCMA-X HIP code had no FP8 kernels):

  | conversion | speedup |
  |---|---|
  | plain | 1.13–1.31x |
  | stochastic rounding | 1.27–1.38x |
  | transposed FP8 operands | 4.8–5.1x |
  | in place | 2.0–2.5x |

- **Gains from this round's profiling:**

  | change | speedup |
  |---|---|
  | stochastic rounding on the integer grid of the target | FP32 → BF16/FP8 and FP16/BF16 → FP8 1.3–2.2x on GPUs (FP32 → FP16 1.0–1.3x); up to 12x in cycles on the host |
  | 256-entry tables for 1-byte sources | FP8 → FP8 2.3x |
  | transposed FP8 operands from FP32/FP16 | 1.1–1.3x on NVIDIA |
  | GPU in place | 1.04–1.18x |
  | one vector per thread on the Radeon VII | plain conversions 1.0–1.3x |
  | streaming stores for large host conversions | Broadwell 1.5–1.8x, Zen 2 1.2–1.3x on widening pairs |
  | AVX2 float → integer kernels in the AVX-512 table | 1.5x in cycles |

- **A bug.** On AMD, scaled FP32 → FP16 conversions rounded 1 ulp differently
  from the host in tie cases (`v_mad_mix` fusion). It is fixed, and HiCMA-X is
  not affected because it scales by powers of two. Section 2.4.
- **HiCMA-X with the updated submodule:** CTest 54/54 on the V100 and 30/30
  in the CPU-only build. All 39 accuracy figures are identical to the build
  without OmniCast.
- **Left:** stochastic rounding from 2- and 4-byte sources is bound by its two
  hashes per element (22–75 % of a copy). Dynamic scaling reads the source
  twice. In place on small H100 tiles pays for its ordering. Section 5.

## 1. Every conversion, in place and out of place

| | host | GPU (CUDA, HIP) |
|---|---|---|
| out of place, 1-D | `omc_convert` | `omc_gpu_convert` |
| out of place, matrix (ld, transposed, padded) | `omc_convert_2d` | `omc_gpu_convert_2d` |
| in place, 1-D | `omc_convert_inplace`, or `omc_convert(.., dst = src)` | `omc_gpu_convert_inplace`, or `dst = src` |
| in place, matrix | `omc_convert_inplace_2d`, or `omc_convert_2d(.., dst = src)` | `omc_gpu_convert_inplace_2d`, or `dst = src` |

Every type pair (14 × 14) takes every option in every mode:
round to nearest even or truncation, saturation, stochastic rounding, scaling
by a host value, a device (or, on the host, a host) pointer or a dynamic power
of two, stochastic rounding of a scaled value, transposition and zero padding.
The only refusals are the ones of the definitions: stochastic rounding needs a
float target narrower than the source (13 pairs), truncation of a float
target is not offered, and dynamic scaling needs a float source.

What changed in this round to get there:

| was missing | now |
|---|---|
| GPU in place of a matrix with ld > m when the element size changes (`OMC_ERROR_NOT_SUPPORTED`) | the ordered single-pass kernel runs on the linear range and skips the gaps |
| GPU in place of a buffer that is not 16-byte aligned (`OMC_ERROR_INVALID_VALUE`) | element-by-element path in the same kernel |
| in place with transposition or padding (host and GPU) | result built in a temporary (`malloc`, `cudaMallocAsync` / `hipMallocAsync` on the stream) and copied back |
| scaling of integer pairs on the GPU (the host had it) | every pair |
| `OMC_SCALE_DEVICE` on the host | reads a host pointer |
| dynamic scaling into integers | target = largest value of the type |
| stochastic rounding together with scaling | rounds src × scale |
| `dst == src` | in place, for the 1-D and 2-D calls |

In place without transposition or padding is a single pass with no
allocation on both backends. On the host, blocks of 1024 elements go through
an L1 buffer, then geometrically growing chunks are converted directly, front
to back when the elements shrink and back to front when they grow (column by
column for matrices, which is safe for any ld ≥ m). On the GPU, blocks take
chunks in address order from an atomic ticket, load and convert, publish a
flag, wait for the flags of the chunks their output overlaps (earlier tickets:
no deadlock) and store (section 2.3).

`tests/test_modes` checks this: every pair × 9 option sets × 6 layouts
(1-D, matrix, ld > m, transposed, padded, both) × {out of place, in place,
dst == src}, at small odd sizes, at sizes that span many GPU chunks, and on
buffers that are not 16-byte aligned. That is 15288 calls per mode; each must
return the status of the generic host conversion and, when it converts, its
bits (NaN payloads aside):

| backend | out of place | in place | dst == src |
|---|---|---|---|
| host generic, AVX2, AVX-512 (Skylake-SP); AVX2, generic (Broadwell); AVX2 (Zen 2) | 10224 converted, 5064 refused as defined, 0 differences | same | same |
| CUDA: H100, V100 (A100: the first build of this round) | same | same | same |
| HIP: Radeon VII | same | same | same |

The first HIP run of this test found a bug in the code that HiCMA-X already
used (section 2.4).

## 2. Design summary

One scalar definition of every conversion (`src/core/omc_scalar.hpp`,
host and device): small formats from FP32 by integer rounding of the bit
pattern (normals) and a power-of-two addition (subnormals), checked against
NVIDIA's reference conversions on all 2^32 FP32 inputs; FP64 sources through
FP32 rounded to odd, so that the second rounding is exact. The CPU kernels are
the scalar functions compiled three times (baseline, AVX2+F16C, AVX-512) in
separate namespaces, with hand-written intrinsics for the pairs that have
hardware converts; the table is chosen from CPUID. The GPU kernels are one
source compiled by nvcc or hipcc. `DESIGN.md` has the details; this section
describes what is new.

### 2.1 Stochastic rounding on the integer grid

HiCMA-X 0.2 rounds a = |x| down to lo (the representable value below a) when
u < (up − a) / (up − lo), with u = u24 · 2^-24 from two SplitMix64 hashes of
the position, the seed and the bits of x. OmniCast keeps these random numbers
and probabilities for compatibility. Since up − lo is a power of two, the
step, t = (a − lo) / step is an exact binary fraction, and

    u < 1 − t   ⇔   u24 + 2^24 · t < 2^24   ⇔   u24 + floor(2^24 · t) < 2^24

because u24 is an integer. So the result is lo + carry(u24 + t24), where t24 is
the top 24 bits of t. lo and the bits of t are fields of the bit pattern of a,
shifted by an amount that depends only on the exponent (more when the result
is subnormal in the target), and lo + 1 steps into the next binade by itself.
The integer core (`sr_code`) has no floating-point operation, no division
(the FP8 and BF16 path had a real FP64 division and branches) and no branch.
`tests/legacy_sr_ref.h` keeps the floating-point form; the two agree on 4M
random values for each of the 13 pairs (all exponents, target subnormals, the
overflow region, NaN, Inf, signed zeros) and on every 61st FP32 bit pattern,
in both saturation modes.

The floating-point form stays where it is faster: FP64 → FP32 on every
backend (one FP64 subtraction and product per element against 64-bit shifts
and selects) and FP64 → FP16 on GPUs (FP64 arithmetic is cheap there, 64-bit
integer shifts are not). On x86 the 64-bit variable shifts vectorize only if
the exponent arithmetic is done in 64-bit lanes: GCC 11 narrows the count of
`sig & ((1 << k) − 1)` to 32 bits and then gives up on the loop
(`vector/vector shift with unusable type`), so the remainder is computed as
`sig − ((sig >> k) << k)`.

### 2.2 GPU kernels

| kernel | what | rounds / loads per thread |
|---|---|---|
| `k_cvt` | 1-D and matrices: each thread converts R vectors of 16 bytes of the wider type; in round r thread t takes vector r·blockDim + t, so every warp access is one contiguous range on both sides | CUDA: R = 4; HIP (gfx906): R = 1 (fewer VGPRs, more waves). Stochastic rounding: 2 from 4- and 2-byte sources (compute bound), from FP64 4 (CUDA) or 2 (HIP) |
| `k_cvtT` | transposed (the cuBLASLt TN layout of FP8 operands): 64 × 64 shared tile with an odd-word pitch, 4-byte packed stores | 16 loads issued before any conversion (was 4) |
| `k_inplace` | ordered single pass, 1-D or matrix with ld ≥ m (section 2.3) | chunks of 4 rounds on both backends (bigger chunks: fewer tickets and flags) |
| `k_amax` | 16-byte loads, per-block partials, last block finishes and writes {amax, s, 1/s} for dynamic scaling | — |
| 1-byte sources | each block computes the 256 results of an FP8/INT8/UINT8 source in shared memory; an element is one lookup | — |

The round counts are compile-time macros (`OMC_ROUNDS_NARROW`,
`OMC_ROUNDS_WIDEN_MAX`, `OMC_ROUNDS_SR`, `OMC_ROUNDS_SR64`,
`OMC_ROUNDS_INPLACE`, `OMC_T_LOADS`) with per-backend defaults; section 3.3
has the measurements that set them.

### 2.3 GPU in place (`k_inplace`)

A buffer that shrinks or grows cannot be converted by independent blocks: the
output of element e overlaps the input of element e · size_dst / size_src,
which another block owns. Each block takes a chunk from an atomic ticket (in
address order when shrinking, from the end when growing), loads and converts
its chunk into registers, publishes a flag, waits for the flags of the input
chunks its output overlaps, and stores. Those chunks have earlier tickets, so
their blocks are already resident and the wait cannot deadlock. The last
block resets the flags (a done counter), so the workspace (a small per-device,
per-stream allocation, or one passed by the caller) is zero for the next call.

A matrix with ld > m is the linear range [0, (n − 1) · ld + m) whose elements
with (e mod ld) ≥ m are gaps, neither read nor written: the chunks, overlaps
and tickets are those of the contiguous case. The row of the first element of
each round is computed once per thread (one division), and later rounds step
by the round stride. When the element size changes, the rows m..ld − 1 of the
result are undefined (they can hold source bytes), on both backends.

### 2.4 One definition on every backend: the scaled FP16 bug

Scaling multiplies in the source's precision and rounds the product before
the conversion (`cvt(round(src × scale))`). On the Radeon VII the new coverage
test found 1-ulp differences in about 0.1 % of random FP32 → FP16 scaled
conversions. The cases were ties of the FP32 product. The AMDGPU backend
fuses an FP32 multiply and the FP16 conversion that follows into one
`v_mad_mix` instruction, which rounds once, from the exact product. It did
this even with `-ffp-contract=off`. Only scales that are not powers of two
are affected: HiCMA-X's HIP build scales FP32 → FP16 operands by powers of
two (`hicma_fp16_operand_scale`), whose products are exact, so its results
did not change. An empty `asm` on the product keeps it rounded
(`scaled_product` in `omc_scalar.hpp`), and device code is now built without
floating-point contraction (CUDA `--fmad=false`, HIP `-ffp-contract=off`).

## 3. Profiling

### 3.1 Method

- **NVIDIA**: Nsight Compute 2025.3 with `--clock-control none` on one kernel
  launch per case, after 0.3 s of warm-up (`bench/prof_gpu.cu` brackets the
  measured calls with `cudaProfilerStart/Stop`). The metrics are duration,
  DRAM bytes against the minimum (read + write once), DRAM and SM throughput
  (% of peak), issue-slot use, achieved occupancy, registers, thread
  instructions per element, L2 hit rate and the dominant stall reasons. There
  are 36 cases (`results/profiling/ncu_sweep.sh`) at 8192² and 2048²: on the
  A100 (code before this round), on the H100 (before and after) and on the
  V100 (after). Summaries are in `results/profiling/ncu_*.txt` and the raw
  CSV in `ncu_*.csv.gz`.
- **AMD**: rocprofv3 on the Radeon VII, before and after: FETCH_SIZE,
  WRITE_SIZE, MemUnitBusy and VALUBusy, one counter per pass (gfx906 cannot
  collect these derived counters together), plus the kernels' VGPR counts
  (`results/profiling/rocprof_sweep.sh`).
- **CPU**: `perf stat` cycles and instructions per element of one conversion
  repeated on 256K elements, one thread (`bench/prof_cpu.cpp`,
  `results/profiling/perf_sweep.sh`), and GCC's vectorizer reports
  (`-fopt-info-vec`, `-fdump-tree-vect-details`).
- **Roofline**: a device-to-device copy of the same bytes, which reaches 88 %
  of the DRAM peak on the A100, 91 % on the V100 and the H100, and 81 % on
  the Radeon VII. On the host the reference is glibc `memcpy`, which uses
  non-temporal stores for large copies.
- **Variants**: the round counts are compile-time macros. Variants were timed
  against the code before this round on the V100, the H100 and the Radeon VII,
  at 8192² and 2048², in interleaved runs, best of 7
  (`results/profiling/variants_*.txt`).

### 3.2 What limited each kernel before this round

H100, 8192² (A100 in `ncu_a100_nb8192.txt`, same picture):

| kernel class | example | DRAM % | issue % | occupancy | regs | instr./element | limiter |
|---|---|---|---|---|---|---|---|
| plain, narrowing | FP64 → FP32 | 93 | 14 | 40 % | 60 | 21 | DRAM (= copy) |
| plain, widening | FP32 → FP64 | 87 | 14 | 42 % | 54 | 23 | DRAM (write-heavy, as a copy) |
| stochastic, FP64 source | FP64 → FP32 | 88 | 61 | 33 % | 72 | 97 | DRAM, close to issue |
| stochastic, FP32 source | FP32 → FP16 | 51 | 68 | 24 % | 89 | 96 | issue slots, registers |
| stochastic, FP32 → FP8 | FP32 → E4M3 | 25 | 60 | 24 % | 101 | 146 | FP64 division + branches ("wait" stalls) |
| transposed, 4-byte source | FP32 → E4M3ᵀ | 68 | 50 | 98 % | 32 | 44 | memory latency: 4 loads of 128 B per warp in flight |
| transposed, 2-byte source | FP16 → E4M3ᵀ | 45 | 58 | 97 % | 32 | 45 | same, 64 B per warp load |
| amax (dynamic scale) | FP32 | 81 | 27 | 92 % | 31 | 14 | DRAM; the source is read twice in all |
| in place | FP64 → FP32 | 86 | 15 | 98 % | 32 | 25 | DRAM, barrier stalls next to memory |
| 1-byte → 1-byte (all-pairs benchmark) | E5M2 → E4M3 | 43 % of copy | | | | ≈ 40 / 2 bytes | integer pipes |

All kernels move within 5 % of the minimum traffic (DRAM/min 0.93–1.00):
there are no wasted bytes. The work was in the compute-bound and
latency-bound classes.

On the host (Skylake-SP, `perf_skylake_before_n262144.txt`), plain
conversions run at 0.2–0.5 ns per element at an IPC of 0.3–1.8, bound by the
caches. Stochastic rounding FP32 → E4M3 took 58 cycles per element on every
code path (122 instructions, scalar: the branches and the FP64 division did
not vectorize), and FP32 → INT8 was slower on the AVX-512 path (0.22 ns) than
on the AVX2 path (0.16 ns).

### 3.3 Optimizations and what they measured

| change | why (profile) | effect |
|---|---|---|
| stochastic rounding on the integer grid (section 2.1) | FP64 arithmetic, a division and branches per element | GPU, FP32 → BF16/FP8 and FP16/BF16 → FP8: 1.3–2.2x on the three GPUs (H100 FP16 → E4M3 1.67x, Radeon VII 1.98x); FP32 → FP16 1.0–1.3x; host: FP32 → E4M3 12.5x in cycles with AVX-512, 3.9–5.8x with AVX2 |
| 2 rounds per thread for stochastic rounding from 2- and 4-byte sources | 89–116 registers, 24 % occupancy | registers 89 → 62, occupancy 24 → 47 % (H100 FP32 → FP16); up to 1.08x on top of the integer core |
| 4 rounds kept for FP64 sources | integer form and 2 rounds measured 5–16 % slower there | FP64 → FP16 stochastic back to its old time |
| 16 loads in flight in `k_cvtT` | latency bound at 97 % occupancy | FP32 → E4M3ᵀ 1.13x (H100) / 1.19x (V100), FP16 → E4M3ᵀ 1.11x / 1.25x; DRAM 68 → 83 % |
| in place: no fence before the flag, flags polled by loads | barrier and fence stalls | H100 1.04–1.06x at 8192², 1.08–1.18x at 2048² |
| 256-entry table for 1-byte sources | compute bound, 2 bytes moved per element | FP8 → FP8 2.3x, FP8 → INT8 1.9x, FP8 → BF16 1.1–1.3x (H100) |
| AVX-512 table takes the AVX2 float → integer kernels | the generic AVX-512 loop was slower | FP32 → INT8 0.22 → 0.17 ns |
| streaming stores for large host conversions | read-for-ownership of the destination | Broadwell 1.45–1.83x, Zen 2 1.19–1.30x on widening pairs (section 4.5) |
| one vector per thread on gfx906 (plain conversions) | 36–48 VGPRs, few waves in flight | Radeon VII FP32 → FP16 1.27–1.29x, FP64 → FP32 1.03–1.13x, FP16 → FP32 1.01–1.12x; small tiles now 1.0–1.3x of the HiCMA-X kernels (were 0.9–1.0x) |
| 2 rounds for stochastic rounding from FP64 on gfx906 | ALU bound | 1.04–1.12x over 4 |

Measured and rejected:

- More rounds for widening conversions (4 × growth, up to 16): 2–12 %
  slower on both GPUs. The narrow side is read, so each round reads fewer
  bytes, but more rounds only add registers.
- 8 rounds for in place (bigger chunks, fewer tickets): 0.86–0.97x on the
  H100.
- 1 round for stochastic rounding: worse than 2 on most pairs (also on the
  Radeon VII, 0.93–0.96x).
- 1 or 2 rounds for plain conversions on NVIDIA: up to 1.1x on 1024² tiles
  of the H100 (4 µs kernels), no change at 2048² and 8192².
- 1 round for in place on the Radeon VII: 0.6–0.9x (more chunks, more
  tickets), so in place keeps chunks of 4 rounds on every backend.
- The integer stochastic core for FP64 sources: 0.84–0.92x on the GPUs for
  FP64 → FP16, and 0.8x on the host for FP64 → FP32 (64-bit shifts and
  selects against FP64 lanes).
- Streaming stores for narrowing pairs and 1-byte sources: 0.8–0.9x on Zen 2
  (the extra pass costs more than the read-for-ownership traffic it saves).

### 3.4 The same kernels after this round (H100, 8192²)

| kernel | before → after (µs) | DRAM % | issue % | occupancy | regs | instr./element | limiter now |
|---|---|---|---|---|---|---|---|
| stochastic FP32 → FP16 | 378 → 363 | 51 → 53 | 68 → 70 | 24 → 47 % | 89 → 62 | 96 → 94 | issue slots (the two hashes) |
| stochastic FP32 → E4M3 | 642 → 369 | 25 → 44 | 60 → 71 | 24 → 47 % | 101 → 64 | 146 → 99 | issue slots (the two hashes) |
| FP32 → E4M3ᵀ | 240 → 194 | 68 → 83 | 50 → 68 | 98 → 49 % | 32 → 64 | 44 → 48 | DRAM |
| FP16 → E4M3ᵀ | 208 → 172 | 45 → 55 | 58 → 79 | 97 → 48 % | 32 → 64 | 45 → 49 | issue slots (FP16 → E4M3 encoding) |
| in place FP64 → FP32 | 453 → 432 | 86 → 90 | 15 → 15 | 98 % | 32 | 25 | DRAM |
| in place FP32 → FP64 | 495 → 468 | 78 → 82 | 15 → 16 | 74 % | 40 | 28 | DRAM |
| E4M3 → FP32 (table) | 183 → 183 | 84 | 30 → 24 | 32 → 42 % | 70 → 52 | 20 → 16 | DRAM |

The stochastic kernels from 2- and 4-byte sources are now limited by the
random numbers: two SplitMix64 hashes per element, each with two 64-bit
multiplies (several 32-bit IMADs each) and 64-bit shifts and xors. The
rounding itself is now a handful of integer operations.

### 3.5 Radeon VII

rocprofv3 on gfx906, 8192², one counter per pass
(`results/profiling/rocprof_radeonvii_{before,final}_nb8192.txt`):

| kernel | before → after (µs) | bytes / minimum | memory unit busy | vector ALU busy | VGPRs |
|---|---|---|---|---|---|
| FP64 → FP32 | 965 → 932 | 1.00 | 97 → 83 % | 29 → 54 % | 36 → 16 |
| FP32 → FP16 | 598 → 465 | 1.00 | 76 → 83 % | 30 → 60 % | 40 → 16 |
| FP16 → FP32 | 493 → 483 | 1.00 | 71 → 95 % | 35 → 56 % | 44 → 16 |
| FP32 → E4M3 | 546 → 416 | 1.00 | 32 → 57 % | 50 → 86 % | 48 → 24 |
| E5M2 → E4M3 (table) | 641 → 159 | 0.99 | 8 → 33 % | 52 → 86 % | 140 → 28 |
| stochastic FP32 → E4M3 | 1965 → 1108 | 1.00 | 11 → 19 % | 93 → 97 % | 72 → 44 |
| stochastic FP64 → FP32 | 1626 → 1566 | 1.00 | 32 → 33 % | 81 → 90 % | 64 → 48 |
| FP32 → E4M3ᵀ | 590 → 603 | 1.07 | 49 → 54 % | 57 → 70 % | 36 → 68 |
| in place FP64 → FP32 | 1008 → 1007 | 1.00 | 77 % | 28 % | 32 |

As on NVIDIA, every kernel moves the minimum bytes. On gfx906 the plain
conversions were bound by the waves in flight: 4 vectors per thread took 36–48
VGPRs. One vector per thread brings this to 12–24 VGPRs, which gives more
waves and 1.0–1.3x. The E5M2 → E4M3 path used 140 VGPRs; the 256-entry table
cut it to 28 and made the kernel 4x faster. Stochastic rounding is bound by
the vector ALU (90–97 % busy): FP64 runs at 1/4 rate on gfx906, and 64-bit
integer multiplies are emulated. The transposed kernel gains nothing from 16
loads in flight there: it ends up at 0.98–1.0x with twice the VGPRs.

### 3.6 Host

`perf stat` on Skylake-SP, one thread, 256K elements, cycles per element
(clock independent; `results/profiling/perf_skylake_{before,after}_n262144.txt`):

| conversion | AVX-512 before → after | AVX2 before → after | instructions per element (AVX-512) |
|---|---|---|---|
| stochastic FP32 → E4M3 | 58.5 → 4.7 (12.5x) | 62.6 → 11.3 (5.6x) | 122 → 8.5 (now vectorized) |
| stochastic FP32 → FP16 | 8.8 → 5.5 (1.6x) | 22.1 → 12.5 (1.8x) | 15.9 → 8.7 |
| stochastic FP64 → FP16 | 9.2 → 7.0 (1.3x) | 23.3 → 20.4 (1.1x) | 16.1 → 12.5 |
| stochastic FP64 → FP32 | 6.8 → 6.9 | 16.6 → 17.3 | 8.0 (unchanged form) |
| FP32 → INT8 | 0.82 → 0.54 (1.5x) | 0.59 → 0.55 | 1.10 → 1.04 |
| plain FP64 ↔ FP32, FP32 ↔ FP16, in place | unchanged (1.2–1.6 / 0.6–0.7 / 2–3 cycles) | unchanged | |

Plain conversions run at an IPC of 0.3–0.5: they wait on the caches and on
memory, not on instructions. Stochastic rounding FP64 → FP32 is 8 AVX-512
instructions per element at an IPC of 1.2. Its four 64-bit multiplies per
element (`vpmullq`: 3 µops, 15-cycle latency on Skylake-SP) form one
dependency chain per element. On AVX2 there is no 64-bit multiply at all, and
each one is emulated with 32-bit multiplies (30 instructions per element).

On large arrays the host is bound by memory traffic. With plain stores a
conversion moves read + read-for-ownership + write: 20 bytes per FP32 → FP64
element instead of 12. Broadwell and Zen 2 reach 22 and 31 GB/s with
`memcpy` (non-temporal stores) on one thread, but 10 and 15 GB/s for
FP32 → FP64. Converting through a 16 KB buffer that is then streamed to the
destination (`_mm_stream_si128`, only above 16 MB and only when the element
keeps its size or grows) recovers most of it (section 4.5). Skylake-SP's
single-thread bandwidth is lower (13 GB/s for `memcpy`), its conversions
were already close to it, and streaming stores slow it down (FP32 → FP64
10.5 → 7.9 GB/s), so they are off there.

## 4. Performance

All GPU numbers are kernel times measured with events, best of 3 or 7, after
0.5 s of warm-up, with a 64 MB to 1.5 GB working set. "% of copy" is the time
of a device-to-device copy of the same bytes (read + write once) divided by
the conversion time. Above 100 % the conversion beats the copy: it reads and
writes one buffer, or L2 holds part of a 2048² tile.

### 4.1 Every pair against the memory roofline

Median over the pairs (10th percentile, minimum). In place on a strided matrix: ld = nb + 64.

| GPU | nb | pairs | out of place | in place | in place, ld > m | in place / out of place time (median, worst) |
|---|---|---|---|---|---|---|
| H100-PCIe | 2048 | 196 | 77% (61%, 45%) | 65% (40%, 33%) | 64% (32%, 24%) | 1.11, 1.88 |
| H100-PCIe | 8192 | 196 | 95% (90%, 79%) | 90% (82%, 71%) | 89% (79%, 65%) | 1.05, 1.17 |
| V100-SXM2-32GB | 2048 | 196 | 92% (86%, 75%) | 90% (86%, 81%) | 86% (83%, 67%) | 1.01, 1.13 |
| V100-SXM2-32GB | 8192 | 196 | 94% (89%, 84%) | 90% (86%, 83%) | 90% (85%, 79%) | 1.03, 1.13 |
| Radeon VII (gfx906) | 2048 | 196 | 94% (85%, 77%) | 78% (70%, 49%) | 75% (63%, 49%) | 1.16, 2.01 |
| Radeon VII (gfx906) | 8192 | 196 | 101% (93%, 78%) | 97% (90%, 80%) | 96% (86%, 72%) | 1.02, 1.16 |

- On large tiles a conversion runs at the speed of a copy. The median pair
  reaches 94–101 % of a copy out of place and 90–97 % in place on the three
  GPUs, and in place costs a median 2–5 % more time than out of place.
- The slowest pairs are the ones that do the most arithmetic per byte: 2-byte
  sources (FP16, BF16, INT16) into FP8, at 79–82 % on the H100. At 8192² no
  pair is below 78 % of a copy out of place, or below 65 % in place on a
  strided matrix.
- On 2048² tiles (32 MB in FP64, near the L2 size) the fixed costs show:
  launch, and for in place the tickets and flags. In place takes a median
  1.11x the out-of-place time on the H100 and 1.16x on the Radeon VII, where
  out of place got the one-vector kernel and in place kept its chunks of 4
  rounds. On the V100 it takes 1.01x.

H100, nb = 8192, % of copy, out of place / in place (rows: source, columns:
target):

| | fp64 | fp32 | fp16 | bf16 | e4m3 | e5m2 | int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **fp64** | 100/98 | 102/98 | 101/93 | 96/93 | 97/88 | 97/88 | 96/91 | 93/94 | 102/99 | 100/98 | 96/91 | 93/94 | 102/98 | 100/98 |
| **fp32** | 95/90 | 99/98 | 103/91 | 103/90 | 100/86 | 100/86 | 97/86 | 103/91 | 100/98 | 95/90 | 97/86 | 103/90 | 100/98 | 95/90 |
| **fp16** | 93/89 | 95/89 | 99/96 | 98/95 | 89/81 | 82/71 | 93/83 | 99/96 | 95/89 | 93/88 | 93/83 | 99/96 | 95/87 | 93/89 |
| **bf16** | 93/89 | 95/89 | 94/93 | 92/91 | 82/81 | 79/79 | 89/83 | 94/93 | 95/89 | 93/88 | 89/83 | 93/93 | 95/87 | 93/89 |
| **e4m3** | 93/92 | 94/85 | 90/82 | 89/82 | 99/97 | 99/97 | 99/97 | 90/82 | 94/89 | 93/82 | 99/97 | 89/82 | 94/89 | 93/82 |
| **e5m2** | 93/92 | 94/85 | 90/82 | 89/82 | 99/97 | 99/97 | 99/97 | 90/82 | 94/89 | 93/82 | 99/97 | 89/82 | 94/89 | 93/82 |
| **int8** | 93/92 | 94/85 | 90/82 | 89/82 | 99/97 | 99/97 | 99/97 | 90/82 | 94/89 | 93/82 | 99/97 | 90/82 | 94/90 | 93/82 |
| **int16** | 93/88 | 95/89 | 93/92 | 91/90 | 81/81 | 80/80 | 87/83 | 94/93 | 95/90 | 93/85 | 88/83 | 93/93 | 95/90 | 93/89 |
| **int32** | 95/90 | 99/98 | 102/92 | 101/93 | 98/84 | 98/84 | 97/90 | 103/94 | 99/98 | 95/87 | 97/90 | 103/94 | 99/98 | 95/90 |
| **int64** | 100/98 | 102/98 | 99/93 | 94/93 | 95/87 | 95/87 | 96/91 | 99/94 | 102/99 | 100/98 | 100/91 | 97/94 | 102/99 | 100/98 |
| **uint8** | 93/92 | 94/85 | 90/82 | 89/82 | 99/97 | 99/97 | 99/97 | 90/82 | 94/89 | 93/82 | 99/97 | 90/82 | 94/89 | 93/82 |
| **uint16** | 93/89 | 95/89 | 93/92 | 91/90 | 81/81 | 80/81 | 90/84 | 94/93 | 95/90 | 93/86 | 90/84 | 94/93 | 95/90 | 93/86 |
| **uint32** | 95/90 | 99/98 | 102/92 | 101/93 | 99/84 | 98/84 | 94/90 | 103/94 | 99/98 | 95/87 | 94/90 | 103/94 | 99/98 | 95/87 |
| **uint64** | 100/98 | 102/93 | 100/93 | 95/93 | 98/87 | 98/87 | 100/91 | 97/94 | 102/99 | 100/98 | 100/91 | 97/94 | 102/99 | 100/98 |

### 4.2 Stochastic rounding, every pair

| pair | H100-PCIe nb=2048 | H100-PCIe nb=8192 | V100-SXM2-32GB nb=2048 | V100-SXM2-32GB nb=8192 | Radeon VII (gfx906) nb=2048 | Radeon VII (gfx906) nb=8192 |
|---|---|---|---|---|---|---|
| fp64>fp32 | 73/74 | 94/81 | 92/86 | 99/91 | 59/64 | 64/66 |
| fp64>fp16 | 65/59 | 86/71 | 86/82 | 91/85 | 51/45 | 55/53 |
| fp64>bf16 | 52/51 | 62/63 | 81/79 | 86/82 | 48/44 | 51/53 |
| fp64>e4m3 | 43/36 | 66/56 | 85/76 | 90/79 | 44/40 | 47/48 |
| fp64>e5m2 | 42/36 | 63/55 | 85/76 | 90/79 | 44/41 | 47/48 |
| fp32>fp16 | 24/20 | 57/48 | 70/65 | 74/67 | 42/43 | 44/45 |
| fp32>bf16 | 23/19 | 53/46 | 71/62 | 75/64 | 42/41 | 44/48 |
| fp32>e4m3 | 21/18 | 46/41 | 60/56 | 61/56 | 36/37 | 37/38 |
| fp32>e5m2 | 21/18 | 43/41 | 60/56 | 61/56 | 36/37 | 37/38 |
| fp16>e4m3 | 17/15 | 30/26 | 37/35 | 37/33 | 23/24 | 22/23 |
| fp16>e5m2 | 16/14 | 27/26 | 37/35 | 37/33 | 23/24 | 22/23 |
| bf16>e4m3 | 16/13 | 28/26 | 37/35 | 37/33 | 23/24 | 22/23 |
| bf16>e5m2 | 16/14 | 28/26 | 37/35 | 37/33 | 23/24 | 22/23 |

From FP64 to FP32 and FP16 the kernels are close to memory bound on NVIDIA
(86–99 % of a copy at 8192²). From 2-byte sources to FP8 they run at
22–37 % of a copy on every GPU, bound by the two hashes per element
(section 5). On the Radeon VII every pair is ALU bound (FP64 at 1/4 rate,
emulated 64-bit multiplies).

### 4.3 This round, kernel by kernel

| case | H100-PCIe nb=2048 | H100-PCIe nb=8192 | Radeon VII (gfx906) nb=2048 | Radeon VII (gfx906) nb=8192 | V100-SXM2-32GB nb=2048 | V100-SXM2-32GB nb=8192 |
|---|---|---|---|---|---|---|
| copy_d2s | 24.8 → 24.6 (1.01x) | 441.0 → 441.1 (1.00x) | 65.9 → 66.1 (1.00x) | 976.6 → 977.9 (1.00x) | 65.5 → 65.5 (1.00x) | 984.1 → 984.1 (1.00x) |
| copy_s2d | 24.7 → 24.7 (1.00x) | 440.0 → 440.6 (1.00x) | 66.6 → 66.2 (1.00x) | 978.2 → 975.5 (1.00x) | 65.5 → 66.6 (0.98x) | 984.1 → 984.1 (1.00x) |
| fp64>fp32 | 34.2 → 29.7 (1.15x) | 429.5 → 427.1 (1.01x) | 75.5 → 66.2 (1.14x) | 961.9 → 936.6 (1.03x) | 66.6 → 66.6 (1.00x) | 982.0 → 981.0 (1.00x) |
| fp32>fp16 | 8.9 → 8.9 (1.01x) | 217.1 → 216.4 (1.00x) | 47.5 → 37.0 (1.29x) | 598.7 → 470.1 (1.27x) | 36.9 → 35.8 (1.03x) | 496.6 → 496.6 (1.00x) |
| fp32>fp64 | 30.3 → 30.2 (1.00x) | 460.0 → 460.3 (1.00x) | 70.7 → 68.0 (1.04x) | 994.1 → 966.2 (1.03x) | 68.6 → 68.6 (1.00x) | 1046.5 → 1047.5 (1.00x) |
| fp16>fp32 | 9.9 → 10.0 (0.99x) | 233.8 → 233.4 (1.00x) | 42.2 → 37.8 (1.12x) | 493.0 → 486.9 (1.01x) | 36.9 → 36.9 (1.00x) | 526.3 → 523.3 (1.01x) |
| fp16>fp64 | 22.7 → 22.4 (1.01x) | 392.6 → 392.0 (1.00x) | 62.9 → 59.2 (1.06x) | 851.5 → 830.9 (1.02x) | 58.4 → 59.4 (0.98x) | 919.5 → 922.6 (1.00x) |
| bf16>fp32 | 10.5 → 10.6 (0.99x) | 233.8 → 233.5 (1.00x) | 42.6 → 37.9 (1.12x) | 492.0 → 487.0 (1.01x) | 36.9 → 36.9 (1.00x) | 526.3 → 526.3 (1.00x) |
| e4m3>fp32 | 10.3 → 10.8 (0.96x) | 198.9 → 198.9 (1.00x) | 38.1 → 34.2 (1.11x) | 422.9 → 416.2 (1.02x) | 31.7 → 32.8 (0.97x) | 455.7 → 458.8 (0.99x) |
| int8>fp32 | 10.6 → 10.8 (0.98x) | 195.8 → 197.4 (0.99x) | 35.5 → 34.1 (1.04x) | 427.4 → 418.9 (1.02x) | 31.7 → 31.7 (1.00x) | 457.7 → 455.7 (1.00x) |
| SR_fp64>fp32 | 35.1 → 35.4 (0.99x) | 502.6 → 498.8 (1.01x) | 123.2 → 109.6 (1.12x) | 1612.5 → 1571.4 (1.03x) | 70.7 → 70.7 (1.00x) | 989.2 → 989.2 (1.00x) |
| SR_fp64>fp16 | 34.8 → 33.7 (1.03x) | 493.0 → 492.8 (1.00x) | 111.2 → 106.6 (1.04x) | 1488.3 → 1512.0 (0.98x) | 64.5 → 63.5 (1.02x) | 906.2 → 906.2 (1.00x) |
| SR_fp32>fp16 | 27.3 → 26.7 (1.02x) | 432.8 → 416.8 (1.04x) | 105.1 → 80.8 (1.30x) | 1388.5 → 1110.4 (1.25x) | 54.3 → 50.2 (1.08x) | 686.1 → 697.3 (0.98x) |
| SR_fp32>bf16 | 38.2 → 26.6 (1.44x) | 571.0 → 409.4 (1.39x) | 125.4 → 80.5 (1.56x) | 1703.8 → 1102.7 (1.55x) | 70.7 → 49.1 (1.44x) | 922.6 → 694.3 (1.33x) |
| SR_fp32>e4m3 | 44.7 → 27.4 (1.63x) | 650.1 → 413.3 (1.57x) | 142.7 → 80.5 (1.77x) | 1952.5 → 1102.2 (1.77x) | 77.8 → 49.1 (1.58x) | 1059.8 → 705.5 (1.50x) |
| SR_fp16>e4m3 | 49.0 → 26.8 (1.83x) | 614.8 → 368.9 (1.67x) | 176.0 → 80.6 (2.18x) | 2146.9 → 1084.8 (1.98x) | 82.9 → 50.2 (1.65x) | 1037.3 → 665.6 (1.56x) |
| SR_fp64>e4m3 | 49.1 → 33.5 (1.47x) | 758.8 → 520.4 (1.46x) | 153.3 → 109.9 (1.39x) | 2203.2 → 1574.1 (1.40x) | 76.8 → 59.4 (1.29x) | 1071.1 → 832.5 (1.29x) |
| T_fp64>e4m3 | 22.4 → 26.1 (0.86x) | 364.6 → 350.8 (1.04x) | 63.4 → 67.5 (0.94x) | 908.5 → 920.5 (0.99x) | 60.4 → 60.4 (1.00x) | 941.1 → 905.2 (1.04x) |
| T_fp32>e4m3 | 16.5 → 15.5 (1.06x) | 253.9 → 225.3 (1.13x) | 46.2 → 46.9 (0.99x) | 596.5 → 599.0 (1.00x) | 44.0 → 35.8 (1.23x) | 572.4 → 480.3 (1.19x) |
| T_fp16>e4m3 | 16.6 → 16.1 (1.03x) | 219.7 → 197.5 (1.11x) | 43.8 → 42.9 (1.02x) | 508.3 → 502.1 (1.01x) | 36.9 → 28.7 (1.29x) | 439.3 → 350.2 (1.25x) |
| T_fp64>fp32 | 31.9 → 31.6 (1.01x) | 442.7 → 440.2 (1.01x) | 167.4 → 150.7 (1.11x) | 10971.0 → 11227.7 (0.98x) | 71.7 → 67.6 (1.06x) | 1030.1 → 1031.2 (1.00x) |
| TS_fp32>e4m3 | 27.6 → 27.1 (1.02x) | 420.4 → 393.7 (1.07x) | 105.6 → 100.5 (1.05x) | 916.6 → 847.7 (1.08x) | 71.7 → 65.5 (1.09x) | 900.1 → 818.2 (1.10x) |
| TS_fp64>e4m3 | 38.2 → 37.6 (1.02x) | 678.7 → 667.4 (1.02x) | 132.5 → 132.8 (1.00x) | 1546.7 → 1531.8 (1.01x) | 105.5 → 108.5 (0.97x) | 1563.7 → 1532.9 (1.02x) |
| IP_fp64>fp32 | 31.0 → 28.3 (1.10x) | 471.6 → 447.6 (1.05x) | 85.4 → 85.0 (1.01x) | 1011.0 → 1009.6 (1.00x) | 76.8 → 75.8 (1.01x) | 1060.9 → 1051.7 (1.01x) |
| IP_fp32>fp64 | 28.4 → 25.0 (1.14x) | 501.6 → 476.7 (1.05x) | 76.3 → 76.2 (1.00x) | 976.0 → 976.5 (1.00x) | 68.6 → 67.6 (1.02x) | 1085.4 → 1068.0 (1.02x) |
| IP_fp32>fp16 | 15.2 → 13.5 (1.13x) | 262.5 → 247.4 (1.06x) | 49.3 → 49.1 (1.00x) | 498.9 → 499.5 (1.00x) | 41.0 → 41.0 (1.00x) | 533.5 → 529.4 (1.01x) |
| IP_fp16>fp32 | 17.0 → 14.4 (1.18x) | 252.6 → 240.4 (1.05x) | 46.6 → 46.7 (1.00x) | 495.2 → 495.7 (1.00x) | 36.9 → 34.8 (1.06x) | 547.8 → 539.6 (1.02x) |
| IP_fp64>fp16 | 29.6 → 26.5 (1.12x) | 416.3 → 395.4 (1.05x) | 71.8 → 71.5 (1.00x) | 823.8 → 821.1 (1.00x) | 67.6 → 65.5 (1.03x) | 917.5 → 914.4 (1.00x) |
| IPS_fp64>fp32 | 32.4 → 29.9 (1.08x) | 472.9 → 454.4 (1.04x) | 85.0 → 85.3 (1.00x) | 1013.4 → 1012.0 (1.00x) | 76.8 → 76.8 (1.00x) | 1061.9 → 1052.7 (1.01x) |

### 4.4 Against the HiCMA-X 0.2 kernels

`bench/bench_gpu.cu` runs, for every case, the HiCMA-X 0.2 kernel it
replaces (copied verbatim) and compares the outputs byte for byte. Every case
gave identical bytes on every GPU. Geometric-mean speedups, min–max in
brackets, are below. The A100 table is from the first round, because the A100
was not available.

**A100-PCIe-40GB** — geometric-mean speedup over the HiCMA-X 0.2 kernels (min–max); last column: median % of a device copy of the same bytes at nb=8192

| category | nb=1024 | nb=2048 | nb=4096 | nb=8192 | % of copy |
|---|---|---|---|---|---|
| plain (FP64/FP32/FP16/BF16) | 1.06x (0.96–1.21) | 1.30x (0.93–2.13) | 1.09x (0.95–1.20) | 1.10x (0.95–1.24) | 97% |
| stochastic rounding | 1.06x (0.94–1.20) | 1.18x (1.13–1.25) | 1.26x (1.16–1.39) | 1.29x (1.18–1.45) | 76% |
| FP8 transposed | 3.96x (3.79–4.06) | 4.82x (3.05–6.16) | 4.38x (3.50–5.18) | 4.63x (3.63–5.53) | 72% |
| FP8 transposed + dynamic scale | 2.52x (2.41–2.63) | 3.30x (2.58–4.21) | 2.97x (2.45–3.60) | 3.15x (2.56–3.88) | 50% |
| in place | 1.42x (1.36–1.50) | 2.21x (1.71–2.85) | 2.00x (1.62–2.47) | 2.06x (1.67–2.48) | 95% |

**H100-PCIe** — geometric-mean speedup over the HiCMA-X 0.2 kernels (min–max); last column: median % of a device copy of the same bytes at nb=8192

| category | nb=1024 | nb=2048 | nb=4096 | nb=8192 | % of copy |
|---|---|---|---|---|---|
| plain (FP64/FP32/FP16/BF16) | 1.10x (0.96–1.27) | 1.40x (1.09–2.08) | 1.26x (1.08–1.48) | 1.31x (1.10–1.57) | 96% |
| stochastic rounding | 1.08x (1.01–1.17) | 1.22x (1.19–1.26) | 1.31x (1.26–1.42) | 1.29x (1.17–1.48) | 72% |
| FP8 transposed | 3.96x (3.74–4.11) | 5.47x (4.23–6.24) | 5.14x (3.72–6.41) | 5.09x (3.67–6.34) | 82% |
| FP8 transposed + dynamic scale | 2.29x (2.19–2.40) | 3.51x (3.00–4.11) | 3.12x (2.51–3.88) | 3.17x (2.52–3.98) | 49% |
| in place | 1.44x (1.34–1.59) | 2.78x (2.42–3.55) | 2.33x (1.87–2.98) | 2.34x (1.88–2.95) | 91% |

**V100-SXM2-32GB** — geometric-mean speedup over the HiCMA-X 0.2 kernels (min–max); last column: median % of a device copy of the same bytes at nb=8192

| category | nb=1024 | nb=2048 | nb=4096 | nb=8192 | % of copy |
|---|---|---|---|---|---|
| plain (FP64/FP32/FP16/BF16) | 1.17x (0.97–1.47) | 1.07x (0.99–1.19) | 1.12x (0.98–1.35) | 1.13x (0.98–1.34) | 94% |
| stochastic rounding | 1.03x (1.00–1.08) | 1.18x (1.16–1.20) | 1.25x (1.15–1.38) | 1.27x (1.16–1.41) | 91% |
| FP8 transposed | 4.49x (2.51–6.07) | 4.49x (2.91–6.37) | 4.58x (2.75–7.17) | 4.78x (2.81–7.40) | 81% |
| FP8 transposed + dynamic scale | 2.69x (2.08–3.49) | 2.60x (2.11–3.21) | 2.66x (2.08–3.40) | 2.77x (2.11–3.63) | 50% |
| in place | 2.04x (1.44–2.96) | 1.85x (1.49–2.27) | 1.97x (1.55–2.48) | 1.99x (1.62–2.46) | 92% |

**Radeon VII (gfx906)** — geometric-mean speedup over the HiCMA-X 0.2 kernels (min–max); last column: median % of a device copy of the same bytes at nb=8192

| category | nb=1024 | nb=2048 | nb=4096 | nb=8192 | % of copy |
|---|---|---|---|---|---|
| plain (FP64/FP32/FP16/BF16) | 1.08x (0.88–1.38) | 1.15x (0.97–1.48) | 1.17x (0.97–1.54) | 1.24x (0.99–1.57) | 100% |
| stochastic rounding | 1.15x (1.03–1.35) | 1.30x (1.15–1.57) | 1.36x (1.20–1.65) | 1.38x (1.23–1.67) | 54% |
| FP8 transposed | – | – | – | – | 79% |
| FP8 transposed + dynamic scale | – | – | – | – | 47% |
| in place | 1.85x (1.20–3.06) | 1.83x (1.40–2.26) | 2.20x (1.67–2.79) | 2.45x (1.87–2.98) | 98% |

On the Radeon VII, small tiles of the plain conversions were 0.9–1.0x of the
HiCMA-X kernels with 4 vectors per thread. One vector per thread (now the
HIP default, section 3.3) brings the geometric mean to 1.08x at 1024² and
1.15x at 2048²; a few pairs stay at 0.88–0.97x there.

### 4.5 Host

Median over the pairs (10th percentile, minimum).

| CPU | path | elements | pairs | out of place | in place |
|---|---|---|---|---|---|
| Xeon Gold 6140 (Skylake-SP) | avx2 | 262144 | 196 | 31% (5%, 1%) | 31% (5%, 1%) |
| Xeon Gold 6140 (Skylake-SP) | avx2 | 33554432 | 196 | 97% (29%, 10%) | 100% (29%, 10%) |
| Xeon Gold 6140 (Skylake-SP) | avx512 | 262144 | 196 | 63% (10%, 2%) | 76% (11%, 2%) |
| Xeon Gold 6140 (Skylake-SP) | avx512 | 33554432 | 196 | 97% (46%, 10%) | 102% (46%, 11%) |
| Xeon E5-2698 v4 (Broadwell) | avx2 | 262144 | 196 | 42% (9%, 4%) | 42% (8%, 4%) |
| Xeon E5-2698 v4 (Broadwell) | avx2 | 33554432 | 196 | 36% (12%, 4%) | 42% (12%, 4%) |
| Xeon E5-2698 v4 (Broadwell) | generic | 262144 | 196 | 28% (7%, 3%) | 28% (7%, 3%) |
| Xeon E5-2698 v4 (Broadwell) | generic | 33554432 | 196 | 32% (8%, 3%) | 33% (8%, 3%) |
| EPYC 7302 (Zen 2) | avx2 | 262144 | 196 | 16% (4%, 1%) | 16% (3%, 1%) |
| EPYC 7302 (Zen 2) | avx2 | 33554432 | 196 | 37% (10%, 4%) | 39% (10%, 4%) |

| case | Xeon E5-2698 v4 (Broadwell) avx2 | Xeon Gold 6140 (Skylake-SP) avx512 | Xeon Gold 6140 (Skylake-SP) avx2 | EPYC 7302 (Zen 2) avx2 |
|---|---|---|---|---|
| fp64>fp32 --sr | 5.43 → 5.43 (1.00x) | 1.84 → 1.84 (1.00x) | 4.72 → 4.39 (1.07x) | 4.29 → 4.30 (1.00x) |
| fp64>fp16 --sr | 8.36 → 7.07 (1.18x) | 2.55 → 1.92 (1.32x) | 6.23 → 5.61 (1.11x) | 6.34 → 5.13 (1.24x) |
| fp32>fp16 --sr | 8.08 → 4.42 (1.83x) | 2.41 → 1.50 (1.61x) | 6.24 → 3.41 (1.83x) | 6.03 → 3.20 (1.88x) |
| fp32>bf16 --sr | 11.73 → 3.27 (3.59x) | 12.03 → 1.28 (9.40x) | 12.64 → 2.47 (5.11x) | 11.05 → 2.54 (4.35x) |
| fp32>e4m3 --sr --sat | 15.57 → 4.03 (3.86x) | 16.63 → 1.31 (12.71x) | 16.82 → 2.89 (5.83x) | 13.59 → 2.88 (4.71x) |
| fp64>e4m3 --sr --sat | 19.07 → 7.62 (2.50x) | 19.52 → 2.28 (8.56x) | 19.23 → 5.99 (3.21x) | 19.58 → 5.84 (3.35x) |
| fp32>int8 | 0.18 → 0.17 (1.02x) | 0.22 → 0.16 (1.39x) | 0.16 → 0.15 (1.03x) | 0.09 → 0.09 (1.00x) |
| fp64>int32 | 0.44 → 0.43 (1.00x) | 0.44 → 0.45 (0.98x) | 0.46 → 0.46 (0.99x) | 0.20 → 0.20 (1.00x) |
| fp64>fp32 | 0.39 → 0.39 (1.01x) | 0.44 → 0.44 (1.00x) | 0.44 → 0.44 (0.99x) | 0.17 → 0.17 (1.02x) |
| fp32>fp64 | 0.47 → 0.47 (1.00x) | 0.44 → 0.44 (1.01x) | 0.45 → 0.44 (1.02x) | 0.20 → 0.20 (0.99x) |
| fp64>fp16 | 0.65 → 0.65 (1.00x) | 0.36 → 0.37 (0.99x) | 0.60 → 0.60 (1.00x) | 0.65 → 0.65 (1.00x) |
| fp32>e4m3 --sat | 0.46 → 0.47 (0.96x) | 0.21 → 0.22 (0.97x) | 0.44 → 0.43 (1.00x) | 0.38 → 0.38 (1.00x) |
| fp64>fp32 --inplace | 0.38 → 0.38 (1.01x) | 0.36 → 0.34 (1.05x) | 0.35 → 0.35 (1.02x) | 0.16 → 0.16 (1.01x) |
| fp32>fp64 --inplace | 0.45 → 0.46 (0.99x) | 0.35 → 0.36 (0.98x) | 0.35 → 0.35 (1.00x) | 0.20 → 0.20 (1.00x) |

One core, 32768K elements (DRAM): GB/s counting bytes read + written; in brackets the speedup over the HiCMA-X 0.2 loop where HiCMA-X had one (it had no x86 FP16/BF16/FP8 conversion).

| pair | Skylake-SP AVX-512 | Skylake-SP AVX2 path | Zen 2 AVX2 | Broadwell AVX2 |
|---|---|---|---|---|
| fp64>fp32 | 12.9 (1.03x) | 13.3 (1.01x) | 21.7 (1.07x) | 11.8 (0.98x) |
| fp32>fp64 | 10.0 (0.86x) | 11.7 (0.93x) | 18.8 (1.13x) | 16.5 (1.65x) |
| fp32>fp16 | 10.4 | 13.5 | 21.4 | 11.7 |
| fp16>fp32 | 10.0 | 11.3 | 20.0 | 18.1 |
| fp64>fp16 | 11.7 | 10.7 | 13.1 | 10.3 |
| fp32>bf16 | 12.5 | 11.8 | 16.1 | 10.2 |
| fp64>bf16 | 10.7 | 8.3 | 10.6 | 8.0 |
| fp32>e4m3 | 10.4 | 6.2 | 7.5 | 5.7 |
| e4m3>fp32 | 11.7 | 11.3 | 11.6 | 7.8 |
| fp64>e4m3 | 6.0 | 5.5 | 6.6 | 5.3 |
| fp32>int8 | 11.7 (0.90x) | 11.7 (0.90x) | 17.6 (0.99x) | 11.9 (1.00x) |
| int8>fp32 | 9.6 (1.27x) | 10.9 (1.35x) | 15.4 (1.64x) | 9.7 (1.07x) |
| fp64>int32 | 12.4 (0.95x) | 12.6 (0.96x) | 18.6 (0.92x) | 12.0 (1.01x) |
| SR:fp64>fp32 | 5.3 (8.52x) | 2.4 (3.68x) | 2.8 (4.75x) | 2.1 (3.20x) |
| SR:fp64>fp16 | 4.5 | 1.7 | 1.9 | 1.4 |
| IP:fp64>fp32 | 14.1 (1.04x) | 13.6 (1.01x) | 21.6 (1.07x) | 12.0 (0.99x) |
| IP:fp32>fp64 | 10.3 (0.85x) | 12.0 (0.98x) | 16.8 (1.34x) | 10.4 (0.99x) |

On cache-resident sizes `memcpy` is very fast and the "% of memcpy" is low
for every conversion with real arithmetic. The DRAM rows are the useful ones.
There, in place often beats out of place, because a shrinking conversion
writes lines it has just read (no read-for-ownership miss). The FP64 ↔ FP32
tiles HiCMA-X converts on the CPU are 2.3–8x faster than MKL's
`LAPACKE_dlag2s` / `slag2d` (first round,
`cpu_lapacke_lag2_vs_omc_skylake.txt`).

### 4.6 HiCMA-X end to end

With the updated OmniCast, HiCMA-X's CTest suite passes and all 39 accuracy
figures are identical to the baseline built without OmniCast. In the first
round's A/B runs:

- The conversion kernels took 26 % less GPU time in the A100 Cholesky (Nsight
  Systems, N = 32768), for a factorization 0–1.6 % faster.
- On the CPU, factorizations with stochastic rounding were 18–19 % faster and
  the other kinds unchanged (10 interleaved runs).

This round speeds up the conversions HiCMA-X uses for FP8 operands (kinds
8/9) and its in-place tile conversions. Its most frequent stochastic
rounding, from FP64, was already memory bound and is unchanged. End-to-end
factorization times were not measured again in this round. HiCMA-X's own
documentation has the details (`docs/omnicast/README.md`).

## 5. Remaining headroom

| where | now | what limits it | what could be done |
|---|---|---|---|
| stochastic rounding from 2- and 4-byte sources | 27–57 % of a copy (H100), 37–75 % (V100) | issue slots: two SplitMix64 hashes per element | the hashes define the HiCMA-X 0.2 results; a counter-based generator giving several numbers per call would be a new, optional rounding definition |
| stochastic rounding on the Radeon VII | 22–66 % of a copy | 64-bit integer multiplies and FP64 (1/4 rate on gfx906) | same |
| dynamic scaling | about 50 % of a copy | the source is read twice (amax, then the conversion) | a single pass when the tile fits on chip (registers and shared memory, a few tens of MB on the H100), with persistent blocks and a grid barrier |
| in place on small GPU tiles | median 1.11x (H100) and 1.16x (Radeon VII) the out-of-place time at 2048² | ticket, flag and barrier per chunk, few waves | persistent blocks over the chunks; host-tracked epochs instead of the reset pass |
| FP16 → FP8 transposed | 55 % of DRAM (H100) | issue slots (the encoding) | the 256-entry table does not apply (65536 values); the native FP8 converts of sm_89+ and gfx942 (`cvt.rn.satfinite.e4m3x2.f16x2`) for the saturating case |
| host FP8 encoding with AVX2 | 0.4–1.0 ns per element | about 20 integer operations per element | F16C round-toward-zero plus sticky bit (FP16 rounded to odd) then FP16 → FP8 on 16-bit lanes: half the operations, still exact |
| host in place on large arrays | plain stores | read-for-ownership traffic | streaming stores as out of place (the destination of a chunk was read long before) |
| ARM | NEON path written, not run | no hardware here | run the tests on an AArch64 machine |

## 6. Reproducing

```bash
cmake -S . -B build -DOMNICAST_GPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES="70;80;90" -DOMNICAST_BUILD_BENCHMARKS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure                       # omnicast_cpu, _modes, _gpu, _modes_gpu, _exhaustive
build/bench/omnicast_bench_gpu 1024 2048 4096 8192               # against the HiCMA-X 0.2 kernels
build/bench/omnicast_bench_modes_gpu 2048 8192                   # every pair, out of place / in place / copy
build/bench/omnicast_bench_modes_gpu --sr 2048 8192              # stochastic rounding pairs
build/bench/omnicast_bench_modes_cpu --isa avx2 262144 33554432  # host, every pair
build/bench/omnicast_prof_gpu --src fp32 --dst e4m3 --sr --sat   # one case, for ncu / rocprofv3
results/profiling/ncu_sweep.sh build/bench/omnicast_prof_gpu out 8192
python3 results/make_report_tables.py                            # the tables of section 4
```

`results/profiling/variants_build.sh` and `variants_measure.sh` rebuild and
time the round-count variants of section 3.3.

