#!/usr/bin/env python3
"""Build summary_gpu.md and summary_cpu.md from the benchmark outputs here."""
import re, math, os
def gpu():
    files = {'A100-PCIe-40GB': 'gpu_a100.txt', 'H100-PCIe': 'gpu_h100.txt', 'V100-SXM2-32GB': 'gpu_v100.txt', 'Radeon VII (gfx906)': 'gpu_radeonvii.txt'}
    def cat(name):
        if name.startswith('SR'): return 'stochastic rounding'
        if name.startswith('TS'): return 'FP8 transposed + dynamic scale'
        if name.startswith('T '): return 'FP8 transposed'
        if name.startswith('IP'): return 'in place'
        return 'plain (FP64/FP32/FP16/BF16)'
    cats = ['plain (FP64/FP32/FP16/BF16)', 'stochastic rounding', 'FP8 transposed', 'FP8 transposed + dynamic scale', 'in place']
    out = []
    for g, f in files.items():
        if not os.path.exists(f): continue
        data, eff = {}, {}
        for line in open(f):
            m = re.match(r'^(\S+(?: \S+)?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)%\s+(\S+)\s+(\S+)\s+(\S+)', line)
            if not m: continue
            name, nb, t, gbs, pct, tl, sp, d = m.groups()
            eff.setdefault((cat(name), int(nb)), []).append(int(pct))
            if sp == '-': continue
            data.setdefault((cat(name), int(nb)), []).append(float(sp[:-1]))
        nbs = sorted(set(k[1] for k in eff))
        out.append(f'\n**{g}** — geometric-mean speedup over the HiCMA-X 0.2 kernels (min–max); last column: median % of a device copy of the same bytes at nb=8192\n')
        out.append('| category | ' + ' | '.join(f'nb={n}' for n in nbs) + ' | % of copy |')
        out.append('|---|' + '---|' * (len(nbs) + 1))
        for c in cats:
            cells = []
            for n in nbs:
                v = data.get((c, n))
                if not v: cells.append('–'); continue
                gm = math.exp(sum(math.log(x) for x in v) / len(v))
                cells.append(f'{gm:.2f}x ({min(v):.2f}–{max(v):.2f})')
            e = sorted(eff.get((c, max(nbs)), []))
            ecell = f'{e[len(e)//2]}%' if e else '–'
            if any(x != '–' for x in cells) or e: out.append(f'| {c} | ' + ' | '.join(cells) + f' | {ecell} |')
    open('summary_gpu.md', 'w').write('\n'.join(out) + '\n')
def cpu():
    files = [('Skylake-SP AVX-512', 'cpu_skylake_avx512.txt'), ('Skylake-SP AVX2 path', 'cpu_skylake_avx2.txt'),
             ('Zen 2 AVX2', 'cpu_epyc7302_avx2.txt'), ('Broadwell AVX2', 'cpu_broadwell_avx2.txt')]
    keys = ['fp64>fp32', 'fp32>fp64', 'fp32>fp16', 'fp16>fp32', 'fp64>fp16', 'fp32>bf16', 'fp64>bf16', 'fp32>e4m3', 'e4m3>fp32',
            'fp64>e4m3', 'fp32>int8', 'int8>fp32', 'fp64>int32', 'SR:fp64>fp32', 'SR:fp64>fp16', 'IP:fp64>fp32', 'IP:fp32>fp64']
    out = []
    for size in ['262144', '33554432']:
        out.append(f'\nOne core, {int(size)//1024}K elements ({"cache-resident" if size == "262144" else "DRAM"}): GB/s counting bytes read + written; in brackets the speedup over the HiCMA-X 0.2 loop where HiCMA-X had one (it had no x86 FP16/BF16/FP8 conversion).\n')
        out.append('| pair | ' + ' | '.join(f[0] for f in files) + ' |')
        out.append('|---|' + '---|' * len(files))
        rows = {}
        for name, f in files:
            for line in open(f):
                p = line.split()
                if len(p) >= 4 and p[1] == size and p[0] in keys:
                    cell = f'{float(p[2]):.1f}'
                    if len(p) >= 7 and p[-1].endswith('x'): cell += f' ({p[-1]})'
                    rows.setdefault(p[0], {})[name] = cell
        for k in keys:
            if k in rows: out.append(f'| {k} | ' + ' | '.join(rows[k].get(n, '–') for n, _ in files) + ' |')
    open('summary_cpu.md', 'w').write('\n'.join(out) + '\n')
gpu(); cpu()
