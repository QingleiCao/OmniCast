#!/bin/bash
# Nsight Compute sweep of OmniCast kernels: one CSV row per kernel launch.
# usage: ncu_sweep.sh <prof_gpu binary> <out prefix> [nb]
B=$1; OUT=$2; NB=${3:-8192}
P=$((NB + 16))
MET=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed.sum,smsp__issue_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__grid_size,launch__block_size,\
lts__t_sector_hit_rate.pct,l1tex__t_bytes.sum,lts__t_bytes.sum,\
smsp__average_warp_latency_issue_stalled_long_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_lg_throttle.ratio,\
smsp__average_warp_latency_issue_stalled_math_pipe_throttle.ratio,smsp__average_warp_latency_issue_stalled_wait.ratio,\
smsp__average_warp_latency_issue_stalled_barrier.ratio,smsp__average_warp_latency_issue_stalled_mio_throttle.ratio,\
smsp__average_warp_latency_issue_stalled_short_scoreboard.ratio,smsp__average_warp_latency_issue_stalled_drain.ratio,\
smsp__average_warp_latency_issue_stalled_not_selected.ratio,smsp__average_warp_latency_issue_stalled_selected.ratio,\
smsp__average_warp_latency_issue_stalled_no_instruction.ratio,smsp__average_warp_latency_issue_stalled_membar.ratio,\
smsp__average_warp_latency_issue_stalled_tex_throttle.ratio,smsp__average_warp_latency_issue_stalled_dispatch_stall.ratio,\
smsp__average_warp_latency_issue_stalled_imc_miss.ratio,smsp__average_warp_latency_issue_stalled_branch_resolving.ratio,\
smsp__average_warp_latency_issue_stalled_sleeping.ratio,smsp__average_warp_latency_per_inst_issued.ratio,\
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_alu.avg.pct_of_peak_sustained_active,\
sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active,sm__inst_executed_pipe_lsu.avg.pct_of_peak_sustained_active
run() {   # name, args...
  local name=$1; shift
  ncu --clock-control none --profile-from-start off --metrics $MET --csv --page raw "$B" --reps 1 "$@" 2>/dev/null \
    | grep -v '^==' | sed "1,2{/^\"ID\"/!d}" | awk -v n="$name" 'NR==1 && /^"ID"/ {print "\"case\"," $0; next} /^"[0-9]/ {print "\"" n "\"," $0}'
}
{
first=1
while read -r name args; do
  [ -z "$name" ] && continue
  out=$(run "$name" $args)
  if [ $first = 1 ]; then echo "$out" | head -1; first=0; fi
  echo "$out" | grep -v '^"case"'
done <<CASES
fp64>fp32 --src fp64 --dst fp32 --m $NB --n $NB
fp32>fp64 --src fp32 --dst fp64 --m $NB --n $NB
fp32>fp16 --src fp32 --dst fp16 --m $NB --n $NB
fp16>fp32 --src fp16 --dst fp32 --m $NB --n $NB
fp64>fp16 --src fp64 --dst fp16 --m $NB --n $NB
fp16>fp64 --src fp16 --dst fp64 --m $NB --n $NB
fp32>bf16 --src fp32 --dst bf16 --m $NB --n $NB
fp64>bf16 --src fp64 --dst bf16 --m $NB --n $NB
fp32>e4m3 --src fp32 --dst e4m3 --m $NB --n $NB --sat
fp64>e4m3 --src fp64 --dst e4m3 --m $NB --n $NB --sat
e4m3>fp32 --src e4m3 --dst fp32 --m $NB --n $NB
fp32>int8 --src fp32 --dst int8 --m $NB --n $NB
int8>fp32 --src int8 --dst fp32 --m $NB --n $NB
fp64>int32 --src fp64 --dst int32 --m $NB --n $NB
fp64>int64 --src fp64 --dst int64 --m $NB --n $NB
int64>fp64 --src int64 --dst fp64 --m $NB --n $NB
SR_fp64>fp32 --src fp64 --dst fp32 --m $NB --n $NB --sr
SR_fp64>fp16 --src fp64 --dst fp16 --m $NB --n $NB --sr
SR_fp32>fp16 --src fp32 --dst fp16 --m $NB --n $NB --sr
SR_fp32>e4m3 --src fp32 --dst e4m3 --m $NB --n $NB --sr --sat
T_fp32>e4m3 --src fp32 --dst e4m3 --m $NB --n $NB --sat --trans --pad $P $P
T_fp64>e4m3 --src fp64 --dst e4m3 --m $NB --n $NB --sat --trans --pad $P $P
T_fp16>e4m3 --src fp16 --dst e4m3 --m $NB --n $NB --sat --trans --pad $P $P
T_fp64>fp32 --src fp64 --dst fp32 --m $NB --n $NB --trans
T_fp64>fp64 --src fp64 --dst fp64 --m $NB --n $NB --trans --sat
TS_fp32>e4m3 --src fp32 --dst e4m3 --m $NB --n $NB --sat --trans --pad $P $P --dyn
TS_fp64>e4m3 --src fp64 --dst e4m3 --m $NB --n $NB --sat --trans --pad $P $P --dyn
IP_fp64>fp32 --src fp64 --dst fp32 --m $NB --n $NB --inplace
IP_fp32>fp64 --src fp32 --dst fp64 --m $NB --n $NB --inplace
IP_fp64>fp16 --src fp64 --dst fp16 --m $NB --n $NB --inplace
IP_fp32>fp16 --src fp32 --dst fp16 --m $NB --n $NB --inplace
IP_fp16>fp32 --src fp16 --dst fp32 --m $NB --n $NB --inplace
IPS_fp64>fp32 --src fp64 --dst fp32 --m $NB --n $NB --ld $((NB + 64)) --inplace
IPS_fp32>fp64 --src fp32 --dst fp64 --m $NB --n $NB --ld $((NB + 64)) --inplace
IPSR_fp64>fp32 --src fp64 --dst fp32 --m $NB --n $NB --inplace --sr
CASES
} > ${OUT}.csv
echo "wrote ${OUT}.csv ($(wc -l < ${OUT}.csv) lines)"
