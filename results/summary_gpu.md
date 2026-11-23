
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
