#!/usr/bin/env python3
"""Variant timings (measure.sh output: case build us) -> table with the
speedup of each build over 'before' and the copy roofline."""
import collections
import sys

for path in sys.argv[1:]:
    t = collections.OrderedDict()
    builds = []
    for line in open(path):
        p = line.split()
        if len(p) < 3:
            continue
        case, b, v = p[0], p[1], p[2]
        if b not in builds:
            builds.append(b)
        try:
            t.setdefault(case, {})[b] = float(v)
        except ValueError:
            t.setdefault(case, {})[b] = float("nan")
    print(f"\n## {path}")
    print(f"{'case':16s} " + " ".join(f"{b:>9s}" for b in builds) + "   " + " ".join(f"{b+'/bef':>7s}" for b in builds[1:]))
    for case, d in t.items():
        base = d.get("before", float("nan"))
        print(f"{case:16s} " + " ".join(f"{d.get(b, float('nan')):9.1f}" for b in builds) + "   "
              + " ".join(f"{base / d.get(b, float('nan')):7.2f}" for b in builds[1:]))
