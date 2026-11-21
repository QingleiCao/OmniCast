# OmniCast 2 performance

All numbers measured on 2026-11-21 with the code of this branch. Raw outputs
are in [`../results`](../results): `gpu_*.txt` (`bench/bench_gpu.cu`),
`cpu_*.txt` (`bench/bench_cpu.cpp`), `test_*.txt`; the tables below are made by
`results/make_summaries.py`.

## Method

- **GPU** (`omnicast_bench_gpu`): nb × nb tiles (the HiCMA-X case), the source
  freshly converted from FP64 values in ±64. Each case runs 0.5 s of warm-up,
  then back-to-back calls timed with events until ≥ 30 ms, best of 3. Every
  case also runs the **HiCMA-X 0.2 kernel** it replaces (copied verbatim in
  `bench/legacy_hicma_gpu.cuh`: 32 × 32 thread blocks, one element per thread;
  FP8 written transposed one byte per thread; in place = convert into a
  scratch tile then copy back) and compares the two outputs byte for byte:
  **every case gave identical bytes on every GPU** (column `diffs`). "% of
  copy" divides the time of a device-to-device copy of the same traffic (bytes
  read + written) by the conversion time.
- **CPU** (`omnicast_bench_cpu`): one thread, best of 7 runs each ≥ 20 ms;
  GB/s counts bytes read + written. Baseline: the HiCMA-X 0.2 loops
  (`bench/legacy_hicma_cpu.c`, compiled like HiCMA-X: C, `-O3`, no `-march`).
- Machines: A100-PCIe-40GB + Xeon Gold 6140 (methane), H100-PCIe (hexane),
  V100-SXM2-32GB + Xeon E5-2698 v4 (leconte), Radeon VII (gfx906) + EPYC 7302
  (histamine2); CUDA 12.6, ROCm 6.4.3, GCC 11.5. The H100 numbers are from the
  build before the last change to the stochastic-rounding random number (an
  exact int32 conversion for AVX2); plain conversions are unaffected.

## GPU

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
| plain (FP64/FP32/FP16/BF16) | 1.11x (0.96–1.26) | 1.40x (1.09–2.05) | 1.26x (1.08–1.48) | 1.31x (1.10–1.57) | 96% |
| stochastic rounding | 1.02x (0.91–1.16) | 1.21x (1.18–1.24) | 1.25x (1.17–1.36) | 1.28x (1.18–1.43) | 74% |
| FP8 transposed | 3.28x (3.13–3.37) | 5.26x (4.15–5.94) | 4.74x (3.62–5.77) | 4.74x (3.56–5.80) | 75% |
| FP8 transposed + dynamic scale | 2.12x (2.03–2.22) | 3.45x (2.96–4.01) | 3.04x (2.49–3.71) | 3.04x (2.46–3.76) | 48% |
| in place | 1.30x (1.22–1.39) | 2.39x (2.02–3.20) | 2.32x (1.79–3.08) | 2.32x (1.79–2.99) | 90% |

**V100-SXM2-32GB** — geometric-mean speedup over the HiCMA-X 0.2 kernels (min–max); last column: median % of a device copy of the same bytes at nb=8192

| category | nb=1024 | nb=2048 | nb=4096 | nb=8192 | % of copy |
|---|---|---|---|---|---|
| plain (FP64/FP32/FP16/BF16) | 1.17x (0.97–1.45) | 1.07x (0.99–1.19) | 1.12x (0.98–1.35) | 1.13x (0.98–1.35) | 94% |
| stochastic rounding | 1.01x (0.96–1.08) | 1.15x (1.10–1.19) | 1.25x (1.15–1.37) | 1.28x (1.17–1.43) | 91% |
| FP8 transposed | 3.93x (2.15–5.35) | 3.77x (2.88–4.80) | 3.94x (2.45–5.87) | 4.10x (2.71–5.77) | 72% |
| FP8 transposed + dynamic scale | 2.58x (2.01–3.32) | 2.51x (2.13–2.96) | 2.52x (1.97–3.22) | 2.61x (2.07–3.30) | 47% |
| in place | 2.05x (1.53–2.85) | 1.87x (1.50–2.28) | 1.98x (1.57–2.48) | 2.01x (1.65–2.45) | 92% |

**Radeon VII (gfx906)** — geometric-mean speedup over the HiCMA-X 0.2 kernels (min–max); last column: median % of a device copy of the same bytes at nb=8192

| category | nb=1024 | nb=2048 | nb=4096 | nb=8192 | % of copy |
|---|---|---|---|---|---|
| plain (FP64/FP32/FP16/BF16) | 1.27x (1.08–1.51) | 1.15x (0.98–1.55) | 1.12x (0.97–1.51) | 1.18x (0.99–1.51) | 96% |
| stochastic rounding | 1.24x (1.15–1.29) | 1.23x (1.15–1.27) | 1.21x (1.12–1.27) | 1.21x (1.13–1.26) | 54% |
| FP8 transposed | – | – | – | – | 83% |
| FP8 transposed + dynamic scale | – | – | – | – | 45% |
| in place | 2.00x (1.22–2.92) | 2.45x (1.88–3.34) | 2.44x (1.87–3.16) | 2.51x (1.92–3.14) | 102% |

Notes:

- Plain conversions run at 94–97 % of a copy of the same bytes for large tiles
  on NVIDIA and 96 % on the Radeon VII. The HiCMA-X kernels were already close
  on large tiles of FP64 sources; the gains are largest for 2- and 4-byte
  sources (FP32 → FP16 2.1× at nb = 2048 on A100 and H100) and on the H100.
- FP8: the HiCMA-X converters wrote the transposed (cuBLASLt TN) operand one
  byte per thread at a stride of ld; OmniCast's 64 × 64 shared tile with packed
  4-byte stores is 3–6× faster. With the per-tile power-of-two scale, OmniCast
  needs two kernels (amax with a last-block finish, then the scaled
  conversion) against four operations, 2–4× faster.
- In place: one kernel that reads and writes each byte once (12 bytes per
  element for FP64 → FP32) instead of a conversion into a scratch tile plus a
  copy back (20 bytes): 1.6–3.3× on large tiles.
- Stochastic rounding is compute-heavy (two SplitMix64 hashes and a few FP64
  operations per element); the branch-free form and the exact reciprocal of the
  power-of-two step make it 1.2–1.45× faster for nb ≥ 2048, and it reaches
  74–91 % of a copy on the NVIDIA GPUs.
- At nb = 1024 (≤ 8 MB, L2-resident on A100/H100) launch latency dominates;
  OmniCast is 0.94–1.2× of HiCMA-X there for plain conversions and
  stochastic rounding, and still well ahead for FP8 and in place.

## CPU

One core, 256K elements (cache-resident): GB/s counting bytes read + written; in brackets the speedup over the HiCMA-X 0.2 loop where HiCMA-X had one (it had no x86 FP16/BF16/FP8 conversion).

| pair | Skylake-SP AVX-512 | Skylake-SP AVX2 path | Zen 2 AVX2 | Broadwell AVX2 |
|---|---|---|---|---|
| fp64>fp32 | 26.6 (1.00x) | 26.4 (1.00x) | 53.1 (1.39x) | 29.9 (1.06x) |
| fp32>fp64 | 25.9 (0.97x) | 26.2 (0.99x) | 33.9 (0.97x) | 24.7 (1.00x) |
| fp32>fp16 | 27.8 | 30.9 | 39.2 | 30.3 |
| fp16>fp32 | 29.4 | 29.1 | 37.8 | 25.1 |
| fp64>fp16 | 26.6 | 15.9 | 9.5 | 14.7 |
| fp32>bf16 | 27.9 | 19.9 | 15.9 | 18.0 |
| fp64>bf16 | 23.0 | 11.6 | 7.0 | 11.0 |
| fp32>e4m3 | 14.9 | 6.7 | 4.1 | 6.4 |
| e4m3>fp32 | 30.3 | 20.8 | 8.0 | 14.4 |
| fp64>e4m3 | 14.7 | 6.8 | 3.9 | 6.4 |
| fp32>int8 | 22.5 (0.71x) | 28.6 (0.91x) | 28.3 (1.12x) | 27.5 (0.97x) |
| int8>fp32 | 33.0 (3.61x) | 32.9 (3.59x) | 26.0 (4.03x) | 23.2 (2.27x) |
| fp64>int32 | 25.6 (0.96x) | 25.3 (0.96x) | 32.9 (0.89x) | 26.2 (0.92x) |
| SR:fp64>fp32 | 6.3 (9.34x) | 2.5 (3.73x) | 1.7 (3.68x) | 2.1 (3.23x) |
| SR:fp64>fp16 | 3.9 | 1.5 | 0.9 | 1.2 |
| IP:fp64>fp32 | 35.2 (1.01x) | 35.0 (1.02x) | 36.5 (0.98x) | 30.9 (1.05x) |
| IP:fp32>fp64 | 33.4 (1.19x) | 32.7 (1.18x) | 30.9 (1.70x) | 25.4 (1.09x) |

One core, 32768K elements (DRAM): GB/s counting bytes read + written; in brackets the speedup over the HiCMA-X 0.2 loop where HiCMA-X had one (it had no x86 FP16/BF16/FP8 conversion).

| pair | Skylake-SP AVX-512 | Skylake-SP AVX2 path | Zen 2 AVX2 | Broadwell AVX2 |
|---|---|---|---|---|
| fp64>fp32 | 13.4 (1.04x) | 13.0 (1.01x) | 10.7 (0.98x) | 11.5 (0.99x) |
| fp32>fp64 | 10.5 (0.86x) | 11.5 (0.93x) | 8.4 (0.95x) | 10.0 (1.01x) |
| fp32>fp16 | 10.6 | 13.2 | 10.7 | 11.4 |
| fp16>fp32 | 10.9 | 11.1 | 9.4 | 10.0 |
| fp64>fp16 | 12.8 | 10.5 | 8.0 | 10.3 |
| fp32>bf16 | 12.5 | 11.4 | 10.0 | 10.0 |
| fp64>bf16 | 10.6 | 8.1 | 5.9 | 7.9 |
| fp32>e4m3 | 10.3 | 6.0 | 5.5 | 5.7 |
| e4m3>fp32 | 11.6 | 11.1 | 8.2 | 7.7 |
| fp64>e4m3 | 8.4 | 5.3 | 4.3 | 5.2 |
| fp32>int8 | 12.3 (0.95x) | 12.2 (0.93x) | 11.2 (1.00x) | 11.8 (1.00x) |
| int8>fp32 | 9.7 (1.24x) | 10.5 (1.33x) | 8.2 (1.34x) | 9.5 (1.05x) |
| fp64>int32 | 11.9 (0.93x) | 12.3 (0.97x) | 12.0 (1.11x) | 11.8 (1.01x) |
| SR:fp64>fp32 | 5.7 (8.56x) | 2.5 (3.67x) | 1.7 (3.64x) | 2.1 (3.22x) |
| SR:fp64>fp16 | 3.7 | 1.5 | 0.9 | 1.1 |
| IP:fp64>fp32 | 13.9 (1.05x) | 13.6 (1.02x) | 11.9 (1.10x) | 11.8 (0.99x) |
| IP:fp32>fp64 | 11.0 (0.89x) | 12.0 (0.97x) | 10.6 (0.85x) | 10.3 (0.99x) |

Notes:

- FP16/BF16/FP8 conversions did not exist on x86 in HiCMA-X (only on Fugaku).
- FP64 ↔ FP32 are at the memory roofline in both; OmniCast's FP64 → FP16/BF16
  /FP8 are correctly rounded (one rounding) and vectorized.
- Stochastic rounding (FP64 → FP32, the HiCMA-X CPU case) is 8.5–9.4× faster
  with AVX-512 and 3.2–3.8× with AVX2, with identical bits.
- Float → integer saturates and maps NaN to 0 (the HiCMA-X loops were plain C
  casts, undefined out of range); with AVX2 that is now within 0.9–1.0× of the
  plain cast thanks to clamp + `cvtt` + saturating packs.
- In place: blocks through an L1 buffer only for the first 1024 elements, then
  geometrically growing chunks converted directly: within 0.9–1.18× of the
  HiCMA-X loops.

### LAPACKE dlag2s / slag2d (MKL 2024.2) against OmniCast, one thread

HiCMA-X used `LAPACKE_dlag2s`/`LAPACKE_slag2d` for most of its CPU FP64 ↔ FP32
tile conversions (43 call sites); OmniCast replaces them:

```
nb=  256  dlag2s     79.4 us  omc d2s      9.7 us (8.16x) | slag2d     27.0 us  omc s2d     11.9 us (2.27x) | omc SR d2s    131.9 us
nb=  512  dlag2s    335.1 us  omc d2s    119.3 us (2.81x) | slag2d    168.3 us  omc s2d    125.2 us (1.34x) | omc SR d2s    525.3 us
nb= 1024  dlag2s   1379.4 us  omc d2s    479.8 us (2.87x) | slag2d    732.4 us  omc s2d    505.7 us (1.45x) | omc SR d2s   2095.7 us
nb= 2048  dlag2s   7111.0 us  omc d2s   3080.7 us (2.31x) | slag2d   4666.1 us  omc s2d   4053.8 us (1.15x) | omc SR d2s   9366.5 us
```

## HiCMA-X end to end

See `docs/omnicast/README.md` in HiCMA-X: identical accuracy on every CTest case
(A100, V100, H100 including FP8 tiles) and conversion kernel time −26 % inside
the GPU Cholesky (Nsight Systems, A100, N = 32768). Factorization time, three
interleaved baseline/OmniCast runs per case:

- A100 (N = 32768, NB = 1024 and 2048): 0–1.6 % faster. Conversions are 6–8 %
  of the GPU time and overlap with GEMMs, so the gain is of the order of the
  run-to-run spread (1–3 %); four of the nine cases were faster in every run,
  none was slower beyond the spread.
- CPU (36-core Xeon Gold 6140, NB = 512): stochastic rounding 13–19 % faster
  (median, N = 8192 and 16384). The other kinds (band, adaptive, runtime
  decisions) moved by −8 % to +1 % in median, inside the ±10 % spread of these
  runs; more repetitions are needed to measure them.

## Profiling findings that shaped the code

1. **GCC does not vectorize branch-free selects under `-ftrapping-math`** (the
   default). With `-fno-trapping-math` (FP exception flags only; results
   unchanged) the generic FP8 encoders went from 1.5 to 0.33 ns per element,
   stochastic rounding from 15 to 2 ns.
2. **A runtime flag inside a loop body blocks vectorization**: saturation and
   rounding mode are template parameters of the loops.
3. **AVX2 has no vector uint64 → float convert**: the 24-bit random number
   of stochastic rounding is converted through int32 (exact), which let the
   whole AVX2 loop vectorize (0.8 → 2.5 GB/s).
4. **NVVM merged the vector store with the scalar fallback store** to the same
   address and kept the fallback's alignment (two `st.global.u32` per vector);
   Nsight showed FP64 → FP32 at 9.4 µs vs 5.1 µs for FP64 → FP16. Inline PTX
   `st.global.v2/v4` fixed it (→ 5.5 µs).
5. **Striped vector rounds** (each warp instruction one contiguous range on
   both sides) instead of each thread owning 64 contiguous bytes: widening
   conversions on small tiles went from 0.48× to 1.0× of HiCMA-X.
6. **First-case artifacts**: GPU clocks need ~0.5 s of load before timing.
