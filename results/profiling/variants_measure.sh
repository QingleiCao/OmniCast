#!/bin/bash
# time each case with every build (interleaved): prints case, build, us
# usage: measure.sh "<builds>" nb [extra env]
for v in $(env | grep -oE '^(SLURM|PMIX|PMI|OMPI_MCA_ess|PRTE)[A-Za-z0-9_]*'); do unset $v; done
BUILDS=$1; NB=${2:-8192}; P=$((NB + 16))
# OMC_BEFORE: a build of the code before the round (commit 3eee555, e.g. a git
# worktree), OMC_VARIANTS: the builds of variants_build.sh, OMC_CUR: the build
# of the current tree
V=${OMC_VARIANTS:-$HOME/omc-variants}
declare -A BIN=( [before]=${OMC_BEFORE:-$HOME/omnicast-before/build_cuda}/bench/omnicast_prof_gpu
                 [a]=$V/a/bench/omnicast_prof_gpu [b]=$V/b/bench/omnicast_prof_gpu
                 [c]=$V/c/bench/omnicast_prof_gpu [d]=$V/d/bench/omnicast_prof_gpu
                 [cur]=${OMC_CUR:-$(cd "$(dirname "$0")/../.." && pwd)/build_cuda}/bench/omnicast_prof_gpu )
while read -r name args; do
  [ -z "$name" ] && continue
  for b in $BUILDS; do
    t=$(${BIN[$b]} $args --m $NB --n $NB --reps 7 2>&1 | grep -oE '[0-9.]+ us' | head -1)
    echo "$name $b ${t:-ERR}"
  done
done <<CASES
copy_d2s --copy --src fp64 --dst fp32
copy_s2d --copy --src fp32 --dst fp64
fp64>fp32 --src fp64 --dst fp32
fp32>fp16 --src fp32 --dst fp16
fp32>fp64 --src fp32 --dst fp64
fp16>fp32 --src fp16 --dst fp32
fp16>fp64 --src fp16 --dst fp64
bf16>fp32 --src bf16 --dst fp32
e4m3>fp32 --src e4m3 --dst fp32
int8>fp32 --src int8 --dst fp32
SR_fp64>fp32 --src fp64 --dst fp32 --sr
SR_fp64>fp16 --src fp64 --dst fp16 --sr
SR_fp32>fp16 --src fp32 --dst fp16 --sr
SR_fp32>bf16 --src fp32 --dst bf16 --sr
SR_fp32>e4m3 --src fp32 --dst e4m3 --sr --sat
SR_fp16>e4m3 --src fp16 --dst e4m3 --sr --sat
SR_fp64>e4m3 --src fp64 --dst e4m3 --sr --sat
T_fp64>e4m3 --src fp64 --dst e4m3 --sat --trans --pad $P $P
T_fp32>e4m3 --src fp32 --dst e4m3 --sat --trans --pad $P $P
T_fp16>e4m3 --src fp16 --dst e4m3 --sat --trans --pad $P $P
T_fp64>fp32 --src fp64 --dst fp32 --trans
TS_fp32>e4m3 --src fp32 --dst e4m3 --sat --trans --pad $P $P --dyn
TS_fp64>e4m3 --src fp64 --dst e4m3 --sat --trans --pad $P $P --dyn
IP_fp64>fp32 --src fp64 --dst fp32 --inplace
IP_fp32>fp64 --src fp32 --dst fp64 --inplace
IP_fp32>fp16 --src fp32 --dst fp16 --inplace
IP_fp16>fp32 --src fp16 --dst fp32 --inplace
IP_fp64>fp16 --src fp64 --dst fp16 --inplace
IPS_fp64>fp32 --src fp64 --dst fp32 --inplace --ld $((NB + 64))
CASES
