#!/usr/bin/env python3
"""Markdown tables for docs/REPORT.md from the result files in this directory.

    python3 make_report_tables.py > report_tables.md

Inputs (missing files are skipped):
  modes_gpu_<gpu>.txt, modes_gpu_sr_<gpu>.txt   bench_modes_gpu (every pair, out / in place / copy)
  modes_cpu_<cpu>_<isa>.txt, modes_cpu_sr_...    bench_modes_cpu
  profiling/before_after_<gpu>_nb<nb>.txt        prof_gpu timings, OmniCast before / after this round
  profiling/before_after_cpu_<cpu>.txt           prof_cpu timings, before / after
"""
import collections
import glob
import os
import re
import statistics as st

GPUS = [("h100", "H100-PCIe"), ("a100", "A100-PCIe-40GB"), ("v100", "V100-SXM2-32GB"), ("radeonvii", "Radeon VII (gfx906)")]
CPUS = [("skylake", "Xeon Gold 6140 (Skylake-SP)"), ("broadwell", "Xeon E5-2698 v4 (Broadwell)"), ("zen2", "EPYC 7302 (Zen 2)")]
TYPES = ["fp64", "fp32", "fp16", "bf16", "e4m3", "e5m2", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"]


def pct(s):
    return int(s.rstrip("%"))


def dist(v):
    v = sorted(v)
    return f"{st.median(v):.0f}% ({v[len(v) // 10]}%, {v[0]}%)"


def gpu_modes():
    print("\n### GPU: every pair, out of place and in place, % of a device copy of the same bytes\n")
    print("Median over the pairs (10th percentile, minimum). In place on a strided matrix: ld = nb + 64.\n")
    print("| GPU | nb | pairs | out of place | in place | in place, ld > m | in place / out of place time (median, worst) |")
    print("|---|---|---|---|---|---|---|")
    for tag, name in GPUS:
        f = f"modes_gpu_{tag}.txt"
        if not os.path.exists(f):
            continue
        rows = [l.split() for l in open(f) if l[0] not in "#s"]
        for nb in sorted({r[2] for r in rows}, key=int):
            r = [x for x in rows if x[2] == nb]
            o, i, s = [pct(x[7]) for x in r], [pct(x[8]) for x in r], [pct(x[9]) for x in r]
            ratio = sorted(float(x[5]) / float(x[4]) for x in r)
            print(f"| {name} | {nb} | {len(r)} | {dist(o)} | {dist(i)} | {dist(s)} | {st.median(ratio):.2f}, {ratio[-1]:.2f} |")


def gpu_matrix(tag, nb):
    f = f"modes_gpu_{tag}.txt"
    if not os.path.exists(f):
        return
    rows = {(x[0], x[1]): x for x in (l.split() for l in open(f) if l[0] not in "#s") if x[2] == str(nb)}
    name = dict(GPUS)[tag]
    print(f"\n### {name}, nb = {nb}: % of copy, out of place / in place (rows: source, columns: target)\n")
    print("| | " + " | ".join(TYPES) + " |")
    print("|---|" + "---|" * len(TYPES))
    for s in TYPES:
        cells = []
        for d in TYPES:
            x = rows.get((s, d))
            cells.append(f"{pct(x[7])}/{pct(x[8])}" if x else "–")
        print(f"| **{s}** | " + " | ".join(cells) + " |")


def gpu_sr():
    print("\n### GPU stochastic rounding, % of copy (out of place / in place)\n")
    data = collections.OrderedDict()
    cols = []
    for tag, name in GPUS:
        f = f"modes_gpu_sr_{tag}.txt"
        if not os.path.exists(f):
            continue
        for l in open(f):
            x = l.split()
            if not x or x[0][0] in "#s":
                continue
            key = (x[0] + ">" + x[1], x[2])
            data.setdefault(key, {})[name] = f"{pct(x[7])}/{pct(x[8])}"
            if (name, x[2]) not in cols:
                cols.append((name, x[2]))
    if not data:
        return
    print("| pair | " + " | ".join(f"{n} nb={nb}" for n, nb in cols) + " |")
    print("|---|" + "---|" * len(cols))
    pairs = []
    for (p, nb) in data:
        if p not in pairs:
            pairs.append(p)
    for p in pairs:
        print(f"| {p} | " + " | ".join(data.get((p, nb), {}).get(n, "–") for n, nb in cols) + " |")


def before_after_gpu():
    files = sorted(glob.glob("profiling/before_after_*_nb*.txt"))
    if not files:
        return
    print("\n### GPU kernels before and after this round (prof_gpu, microseconds, best of 7)\n")
    tab = collections.OrderedDict()
    cols = []
    for f in files:
        m = re.search(r"before_after_(\w+)_nb(\d+)", f)
        if not m or m.group(1) == "cpu":
            continue
        col = (dict(GPUS).get(m.group(1), m.group(1)), m.group(2))
        cols.append(col)
        t = collections.defaultdict(dict)
        for l in open(f):
            x = l.split()
            if len(x) >= 3:
                try:
                    t[x[0]][x[1]] = float(x[2])
                except ValueError:
                    pass
        for case, d in t.items():
            after = d.get("next", d.get("cur"))
            if "before" in d and after:
                tab.setdefault(case, {})[col] = (d["before"], after)
    print("| case | " + " | ".join(f"{n} nb={nb}" for n, nb in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for case, d in tab.items():
        cells = []
        for c in cols:
            if c in d:
                b, a = d[c]
                cells.append(f"{b:.1f} → {a:.1f} ({b / a:.2f}x)")
            else:
                cells.append("–")
        print(f"| {case} | " + " | ".join(cells) + " |")


def cpu_modes():
    print("\n### Host: every pair, one thread, % of memcpy of the same bytes\n")
    print("Median over the pairs (10th percentile, minimum).\n")
    print("| CPU | path | elements | pairs | out of place | in place |")
    print("|---|---|---|---|---|---|")
    for tag, name in CPUS:
        for f in sorted(glob.glob(f"modes_cpu_{tag}_*.txt")):
            isa = f[len(f"modes_cpu_{tag}_"):-4]
            rows = [l.split() for l in open(f) if l[0] not in "#s"]
            for n in sorted({r[2] for r in rows}, key=int):
                r = [x for x in rows if x[2] == n]
                print(f"| {name} | {isa} | {n} | {len(r)} | {dist([pct(x[6]) for x in r])} | {dist([pct(x[7]) for x in r])} |")


def before_after_cpu():
    files = sorted(glob.glob("profiling/before_after_cpu_*.txt"))
    if not files:
        return
    print("\n### Host kernels before and after this round (prof_cpu, ns per element, 256K elements, one thread)\n")
    tab = collections.OrderedDict()
    cols = []
    for f in files:
        cpu = f[len("profiling/before_after_cpu_"):-4]
        for l in open(f):
            m = re.match(r"(\S+) (before|after) (\S+)(.*?) ([0-9.]+) ns/element", l.strip())
            if not m:
                continue
            isa, ba, case, opts, ns = m.groups()
            col = (dict(CPUS).get(cpu, cpu), isa)
            if col not in cols:
                cols.append(col)
            key = (case + " " + opts.strip()).strip()
            tab.setdefault(key, {}).setdefault(col, {})[ba] = float(ns)
    print("| case | " + " | ".join(f"{n} {isa}" for n, isa in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for case, d in tab.items():
        cells = []
        for c in cols:
            v = d.get(c, {})
            if "before" in v and "after" in v:
                cells.append(f"{v['before']:.2f} → {v['after']:.2f} ({v['before'] / v['after']:.2f}x)")
            else:
                cells.append("–")
        print(f"| {case} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    gpu_modes()
    gpu_matrix("h100", 8192)
    gpu_sr()
    before_after_gpu()
    cpu_modes()
    before_after_cpu()


def fill_report(path):
    """Replace {{TABLE:name}} markers in docs/REPORT.md with the tables."""
    import contextlib
    import io

    def cap(f, *a):
        b = io.StringIO()
        with contextlib.redirect_stdout(b):
            f(*a)
        return b.getvalue().strip()

    def legacy(fname):
        txt = open(fname).read().strip() if os.path.exists(fname) else ""
        return txt

    tables = {
        "gpu_modes": cap(gpu_modes),
        "gpu_matrix": cap(gpu_matrix, "h100", 8192),
        "gpu_sr": cap(gpu_sr),
        "before_after_gpu": cap(before_after_gpu),
        "cpu": cap(cpu_modes),
        "before_after_cpu": cap(before_after_cpu),
        "legacy_gpu": legacy("summary_gpu.md"),
        "legacy_cpu": legacy("summary_cpu.md")[legacy("summary_cpu.md").find("One core, 32768K"):],
    }
    s = open(path).read()
    for k, v in tables.items():
        # drop the generated "###" heading: the report has its own
        v = "\n".join(l for l in v.splitlines() if not l.startswith("### "))
        s = s.replace("{{TABLE:" + k + "}}", v.strip())
    open(path, "w").write(s)
