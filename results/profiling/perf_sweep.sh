#!/bin/bash
# perf stat sweep of OmniCast host kernels: cycles and instructions per element.
# usage: perf_sweep.sh <prof_cpu binary> <isa list> <n> > out.txt
B=$1; ISAS=$2; N=${3:-262144}
CASES=(
"fp64 fp32" "fp32 fp64" "fp32 fp16" "fp16 fp32" "fp64 fp16" "fp16 fp64" "fp32 bf16" "fp64 bf16"
"fp32 e4m3 --sat" "fp64 e4m3 --sat" "e4m3 fp32" "fp32 int8" "int8 fp32" "fp64 int32" "int64 fp64"
"fp64 fp32 --sr" "fp64 fp16 --sr" "fp32 fp16 --sr" "fp32 e4m3 --sr --sat"
"fp64 fp32 --inplace" "fp32 fp64 --inplace" "fp64 fp16 --inplace"
)
printf "%-8s %-26s %9s %9s %6s %9s %8s\n" isa case ns/el cyc/el IPC inst/el GB/s
for isa in $ISAS; do
  for c in "${CASES[@]}"; do
    set -- $c; s=$1; d=$2; shift 2
    out=$(perf stat -x, -e cycles,instructions "$B" --isa $isa --src $s --dst $d --n $N "$@" 2>&1)
    line=$(echo "$out" | grep best)
    [ -z "$line" ] && continue
    calls=$(echo "$line" | sed -E 's/.* ([0-9]+) calls.*/\1/')
    ns=$(echo "$line" | sed -E 's/.*GB\/s, ([0-9.]+) ns.*/\1/')
    gbs=$(echo "$line" | sed -E 's/.*\(([0-9.]+) GB\/s.*/\1/')
    cyc=$(echo "$out" | grep ',cycles' | cut -d, -f1)
    ins=$(echo "$out" | grep ',instructions' | cut -d, -f1)
    # restores in place (memcpy of the source) are included in the counters: noted in the report
    awk -v isa=$isa -v c="$s>$d $*" -v ns=$ns -v cyc=$cyc -v ins=$ins -v calls=$calls -v n=$N -v g=$gbs \
      'BEGIN { el = calls * n; printf "%-8s %-26s %9.3f %9.2f %6.2f %9.2f %8.2f\n", isa, c, ns, cyc/el, ins/cyc, ins/el, g }'
  done
done
