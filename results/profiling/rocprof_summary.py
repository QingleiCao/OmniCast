#!/usr/bin/env python3
"""Summarize rocprof_sweep.sh output: per case and kernel, the median over the
measured dispatches of FETCH_SIZE + WRITE_SIZE (against the minimum bytes),
MemUnitBusy, VALUBusy, the duration and the VGPR count."""
import csv
import glob
import os
import re
import statistics as st
import sys

SIZES = {"fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2, "e4m3": 1, "e5m2": 1}


TNAME = {"double": "fp64", "float": "fp32", "omc::f16": "fp16", "omc::bf16": "bf16", "omc::e4m3": "e4m3",
         "omc::e5m2": "e5m2"}


def kernel_short(name):
    """(short name, source type, destination type) of a kernel"""
    m = re.search(r"omc::gpu::(\w+)<([^>]*)>", name)
    if not m:
        return name[:20], None, None
    modes = {"0": "", "1": " scaled", "2": " sr", "3": " sr-scaled"}
    parts = [p.strip() for p in m.group(2).split(",")]
    st_ = TNAME.get(parts[0])
    dt = TNAME.get(parts[1]) if len(parts) >= 2 else None
    return m.group(1) + (modes.get(parts[2], "") if len(parts) == 3 else ""), st_, dt


def main(root, nb=8192):
    print(f"\n## {root} (nb = {nb})")
    print(f"{'case':16s} {'kernel':16s} {'us':>8s} {'bytes/min':>9s} {'MemUnitBusy%':>12s} {'VALUBusy%':>9s} {'VGPRs':>5s}")
    for d in sorted(glob.glob(os.path.join(root, "*")), key=os.path.getmtime):
        if not os.path.isdir(d):
            continue
        case = os.path.basename(d)
        m = re.search(r"(\w+)2(\w+)$", case)
        st_, dt = (m.group(1).split("_")[-1], m.group(2)) if m else ("fp64", "fp32")
        n = nb * nb
        dst = (nb + 16) ** 2 if case.startswith("T") else n
        ideal = n * SIZES.get(st_, 8) + dst * SIZES.get(dt, 4)
        vals = {}
        for c in ("FETCH_SIZE", "WRITE_SIZE", "MemUnitBusy", "VALUBusy"):
            f = glob.glob(os.path.join(d, c, "**", "*counter_collection.csv"), recursive=True)
            if not f:
                continue
            rows = list(csv.DictReader(open(f[0])))
            by = {}
            for r in rows:
                k, ks, kd = kernel_short(r["Kernel_Name"])
                # the measured kernels only (not the setup conversion FP64 -> source)
                if not k.startswith("k_") or ks != st_ or (kd is not None and kd != dt):
                    continue
                by.setdefault(k, []).append(r)
            for k, rs in by.items():
                rs = rs[-3:]                       # the measured calls
                v = vals.setdefault(k, {})
                v[c] = st.median(float(r["Counter_Value"]) for r in rs)
                v["us"] = st.median((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1000.0 for r in rs)
                v["vgpr"] = rs[-1]["VGPR_Count"]
        for k, v in vals.items():
            b = (v.get("FETCH_SIZE", 0) + v.get("WRITE_SIZE", 0)) * 1024.0
            print(f"{case:16s} {k:16s} {v.get('us', 0):8.1f} {b / ideal:9.2f} {v.get('MemUnitBusy', float('nan')):12.1f} "
                  f"{v.get('VALUBusy', float('nan')):9.1f} {v.get('vgpr', '-'):>5s}")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        main(r)
