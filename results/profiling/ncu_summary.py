#!/usr/bin/env python3
"""Summarize ncu_sweep.sh CSV files: one line per kernel launch.

Columns: duration, DRAM bytes against the minimum (read + write once), DRAM and
SM throughput (% of peak), issue-slot use, achieved occupancy, registers,
instructions per element, L2 hit rate, top stall reasons (cycles per issued
instruction).
"""
import csv
import re
import sys

SIZES = {"fp64": 8, "fp32": 4, "fp16": 2, "bf16": 2, "e4m3": 1, "e5m2": 1, "int8": 1, "int16": 2,
         "int32": 4, "int64": 8, "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8}
STALLS = ["long_scoreboard", "lg_throttle", "math_pipe_throttle", "wait", "barrier", "mio_throttle",
          "short_scoreboard", "drain", "not_selected", "selected", "no_instruction", "membar", "tex_throttle",
          "dispatch_stall", "imc_miss", "branch_resolving", "sleeping"]


def num(row, key):
    v = row.get(key, "")
    try:
        return float(v.replace(",", ""))
    except ValueError:
        return float("nan")


def short_kernel(name):
    m = re.match(r"void (\w+)<([^>]*)>", name)
    if not m:
        return name.split("(")[0]
    k, targs = m.group(1), m.group(2)
    modes = {"0": "", "1": " scaled", "2": " sr", "3": " sr-scaled"}
    parts = [p.strip() for p in targs.split(",")]
    if len(parts) == 3:
        return f"{k}{modes.get(parts[2], '')}"
    return k


def main(paths, nb_default=8192):
    for path in paths:
        nb = int(re.search(r"nb(\d+)", path).group(1)) if re.search(r"nb(\d+)", path) else nb_default
        rows = list(csv.DictReader(open(path)))
        print(f"\n## {path}  (nb = {nb})")
        print(f"{'case':16s} {'kernel':16s} {'us':>8s} {'DRAM/min':>8s} {'DRAM%':>6s} {'SM%':>5s} {'issue%':>6s} "
              f"{'occ%':>5s} {'regs':>4s} {'inst/el':>7s} {'L2hit':>5s}  top stall reasons")
        for r in rows:
            case = r["case"]
            m = re.search(r"(\w+)>(\w+)$", case)
            st, dt = (m.group(1).split("_")[-1], m.group(2)) if m else ("fp64", "fp32")
            n_el = nb * nb
            dst_el = n_el
            if case.startswith("T") and "e4m3" in dt:
                dst_el = (nb + 16) * (nb + 16)
            ideal = n_el * SIZES.get(st, 8) + dst_el * SIZES.get(dt, 4)
            dur = num(r, "gpu__time_duration.sum") / 1000.0   # ns -> us
            unit = r.get("gpu__time_duration.sum_unit", "")
            dram = num(r, "dram__bytes_read.sum") + num(r, "dram__bytes_write.sum")
            kern = short_kernel(r["Kernel Name"])
            ratio = dram / ideal if ideal else float("nan")
            stalls = sorted(((num(r, f"smsp__average_warp_latency_issue_stalled_{s}.ratio"), s) for s in STALLS),
                            reverse=True)[:3]
            inst = num(r, "smsp__inst_executed.sum") * 32 / n_el
            print(f"{case:16s} {kern:16s} {dur:8.1f} {ratio:8.2f} {num(r, 'dram__throughput.avg.pct_of_peak_sustained_elapsed'):6.1f} "
                  f"{num(r, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'):5.1f} "
                  f"{num(r, 'smsp__issue_active.avg.pct_of_peak_sustained_active'):6.1f} "
                  f"{num(r, 'sm__warps_active.avg.pct_of_peak_sustained_active'):5.1f} "
                  f"{num(r, 'launch__registers_per_thread'):4.0f} {inst:7.1f} {num(r, 'lts__t_sector_hit_rate.pct'):5.1f}  "
                  + ", ".join(f"{s}" for v, s in stalls))


if __name__ == "__main__":
    main(sys.argv[1:])
