
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
