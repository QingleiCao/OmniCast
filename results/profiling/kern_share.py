#!/usr/bin/env python3
"""GPU kernel time by class from `nsys stats --report cuda_gpu_kern_sum --format csv`."""
import csv, glob, re, sys, os
def cls(name):
    m = re.search(r"omc::gpu::(k_\w+)<([^>]*)>", name)
    if m:
        mode = re.sub(r"\D", "", m.group(2).split(",")[-1])
        tag = {"1": " scaled", "2": " sr", "3": " sr-scaled"}.get(mode, "") if m.group(1) != "k_amax" else ""
        return "omc " + m.group(1) + tag
    n = name.lower()
    if any(k in n for k in ("gemm", "xmma", "cutlass", "sm90", "sm80", "ampere", "hopper", "cublaslt")): return "GEMM"
    if any(k in n for k in ("trsm", "potrf", "syrk", "herk", "getrf", "lapack")): return "TRSM/SYRK/POTRF"
    return "other"
for f in sorted(glob.glob(os.path.join(sys.argv[1], "kern_*_cuda_gpu_kern_sum.csv"))):
    rows = list(csv.DictReader(open(f)))
    tot = sum(float(r["Total Time (ns)"]) for r in rows)
    agg = {}
    for r in rows:
        k = cls(r["Name"]); t, n = agg.get(k, (0.0, 0)); agg[k] = (t + float(r["Total Time (ns)"]), n + int(r["Instances"]))
    name = os.path.basename(f)[5:-len("_cuda_gpu_kern_sum.csv")]
    omc = sum(t for k, (t, n) in agg.items() if k.startswith("omc"))
    print(f"\n{name}: GPU kernel time {tot/1e6:.1f} ms, OmniCast {omc/1e6:.2f} ms ({100*omc/tot:.1f} %)")
    for k, (t, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        print(f"  {k:24s} {t/1e6:9.2f} ms {100*t/tot:5.1f} %  {n:6d} launches")
