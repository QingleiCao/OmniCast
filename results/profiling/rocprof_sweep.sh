#!/bin/bash
# rocprofv3 sweep (Radeon VII): one counter per pass (gfx906 cannot collect
# these derived counters together); stdin from /dev/null (rocprofv3 would
# read the case list). usage: rocprof_sweep.sh <prof_gpu> <outdir> [nb]
B=$1; OUT=$2; NB=${3:-8192}; P=$((NB + 16))
mkdir -p $OUT
while read -r name args; do
  [ -z "$name" ] && continue
  d=$OUT/$(echo $name | tr '>' '2')
  for c in FETCH_SIZE WRITE_SIZE MemUnitBusy VALUBusy; do
    timeout 300 rocprofv3 --pmc $c --output-format csv -d $d/$c -o run -- \
      $B $args --m $NB --n $NB --reps 3 < /dev/null >> $d.log 2>&1
  done
  echo "$name: $(grep -oE '[0-9.]+ us' $d.log | head -1)"
done <<CASES
fp64>fp32 --src fp64 --dst fp32
fp32>fp64 --src fp32 --dst fp64
fp32>fp16 --src fp32 --dst fp16
fp16>fp32 --src fp16 --dst fp32
fp64>fp16 --src fp64 --dst fp16
fp32>e4m3 --src fp32 --dst e4m3 --sat
e5m2>e4m3 --src e5m2 --dst e4m3 --sat
SR_fp64>fp32 --src fp64 --dst fp32 --sr
SR_fp64>fp16 --src fp64 --dst fp16 --sr
SR_fp32>fp16 --src fp32 --dst fp16 --sr
SR_fp32>e4m3 --src fp32 --dst e4m3 --sr --sat
T_fp64>e4m3 --src fp64 --dst e4m3 --sat --trans --pad $P $P
T_fp32>e4m3 --src fp32 --dst e4m3 --sat --trans --pad $P $P
TS_fp32>e4m3 --src fp32 --dst e4m3 --sat --trans --pad $P $P --dyn
IP_fp64>fp32 --src fp64 --dst fp32 --inplace
IP_fp32>fp64 --src fp32 --dst fp64 --inplace
IPS_fp64>fp32 --src fp64 --dst fp32 --inplace --ld $((NB + 64))
CASES
