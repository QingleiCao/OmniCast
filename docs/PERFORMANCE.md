# OmniCast 2 performance tables

The comparison with the code OmniCast replaced in HiCMA-X 0.2, regenerated
on 2026-11-22 from the files in [`../results`](../results) by
`results/make_summaries.py`. [`REPORT.md`](REPORT.md) has the design, the
profiling, the in-place / out-of-place measurements for every pair and the
before/after tables of the optimizations.

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
- Machines: H100-PCIe (hexane), V100-SXM2-32GB + Xeon E5-2698 v4 (leconte),
  Radeon VII (gfx906) + EPYC 7302 (histamine2/3), Xeon Gold 6140 (methane);
  A100-PCIe-40GB (methane); CUDA 12.6, ROCm 6.4.3, GCC 11.5.

## GPU

**A100-PCIe-40GB** — geometric-mean speedup over the HiCMA-X 0.2 kernels (min–max); last column: median % of a device copy of the same bytes at nb=8192

| category | nb=1024 | nb=2048 | nb=4096 | nb=8192 | % of copy |
|---|---|---|---|---|---|
| plain (FP64/FP32/FP16/BF16) | 1.07x (0.97–1.24) | 1.30x (0.93–2.13) | 1.09x (0.95–1.21) | 1.10x (0.95–1.25) | 97% |
| stochastic rounding | 1.10x (1.05–1.20) | 1.19x (1.15–1.26) | 1.27x (1.19–1.40) | 1.29x (1.18–1.45) | 76% |
| FP8 transposed | 4.60x (4.25–4.86) | 5.32x (3.36–6.75) | 5.26x (3.71–6.70) | 5.52x (3.85–6.99) | 90% |
| FP8 transposed + dynamic scale | 2.67x (2.53–2.82) | 3.35x (2.45–4.57) | 3.16x (2.52–3.96) | 3.40x (2.64–4.37) | 51% |
| in place | 1.44x (1.36–1.54) | 2.25x (1.62–2.91) | 1.99x (1.63–2.44) | 2.05x (1.67–2.45) | 94% |

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

## CPU

One core, 256K elements (cache-resident): GB/s counting bytes read + written; in brackets the speedup over the HiCMA-X 0.2 loop where HiCMA-X had one (it had no x86 FP16/BF16/FP8 conversion).

| pair | Skylake-SP AVX-512 | Skylake-SP AVX2 path | Zen 2 AVX2 | Broadwell AVX2 |
|---|---|---|---|---|
| fp64>fp32 | 25.8 (1.00x) | 26.3 (1.00x) | 69.9 (1.06x) | 31.3 (1.05x) |
| fp32>fp64 | 24.1 (0.93x) | 26.1 (1.04x) | 60.0 (0.94x) | 25.9 (1.01x) |
| fp32>fp16 | 23.3 | 30.3 | 73.5 | 31.3 |
| fp16>fp32 | 25.5 | 28.6 | 69.0 | 25.9 |
| fp64>fp16 | 23.2 | 15.2 | 15.3 | 14.7 |
| fp32>bf16 | 24.2 | 15.9 | 27.3 | 18.3 |
| fp64>bf16 | 21.2 | 11.0 | 11.6 | 11.2 |
| fp32>e4m3 | 13.7 | 6.5 | 7.8 | 6.4 |
| e4m3>fp32 | 26.1 | 20.0 | 12.8 | 14.5 |
| fp64>e4m3 | 11.6 | 6.3 | 7.1 | 6.5 |
| fp32>int8 | 25.3 (0.88x) | 27.3 (0.88x) | 50.5 (1.12x) | 28.2 (0.95x) |
| int8>fp32 | 26.7 (3.24x) | 30.3 (3.56x) | 43.0 (4.09x) | 24.0 (2.35x) |
| fp64>int32 | 23.6 (0.92x) | 26.7 (0.98x) | 57.4 (0.84x) | 27.1 (0.91x) |
| SR:fp64>fp32 | 5.2 (9.91x) | 2.5 (3.78x) | 2.8 (4.81x) | 2.1 (3.22x) |
| SR:fp64>fp16 | 4.9 | 1.7 | 1.9 | 1.4 |
| IP:fp64>fp32 | 32.8 (1.01x) | 35.3 (1.02x) | 72.3 (1.01x) | 32.0 (1.05x) |
| IP:fp32>fp64 | 29.0 (1.48x) | 33.2 (1.54x) | 60.0 (3.07x) | 26.2 (1.10x) |

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
the GPU Cholesky (Nsight Systems, A100, N = 32768). Factorization time, interleaved
baseline/OmniCast runs:

- A100 (N = 32768, NB = 1024 and 2048): 0–1.6 % faster. Conversions are 6–8 %
  of the GPU time and overlap with GEMMs, so the gain is of the order of the
  run-to-run spread (1–3 %); four of the nine cases were faster in every run,
  none was slower beyond the spread.
- CPU (36-core Xeon Gold 6140, NB = 512, 10 interleaved runs per case):
  stochastic rounding 18–19 % faster at N = 8192 and 16384 (paired median; 95 %
  interval −26 % to −12 %; OmniCast faster in 19 of 20 pairs). The other kinds
  (band, adaptive, runtime decisions) show no measurable difference: their
  intervals contain zero, with up to 40 % between the fastest and the slowest run of one binary.

## Profiling

See [`REPORT.md`](REPORT.md), section 3: Nsight Compute on the A100, H100 and
V100, rocprofv3 on the Radeon VII, `perf` on Skylake-SP, and the measured
variants behind every kernel parameter.
