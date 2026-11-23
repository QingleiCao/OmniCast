#!/bin/bash
# build OmniCast variants (compile-time rounds) for sm_70 and sm_90: name SR NARROW WIDEN_MAX T_LOADS
SRC=${OMC_SRC:-$(cd "$(dirname "$0")/../.." && pwd)}       # the OmniCast source tree
OUT=${OMC_VARIANTS:-$HOME/omc-variants}                        # where the variant builds go
build() {
  local name=$1 f="-DOMC_ROUNDS_SR=$2 -DOMC_ROUNDS_NARROW=$3 -DOMC_ROUNDS_WIDEN_MAX=$4 -DOMC_T_LOADS=$5"
  cmake -S $SRC -B $OUT/$name -G Ninja -DOMNICAST_GPU_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES="70;90" \
    -DOMNICAST_BUILD_BENCHMARKS=ON -DOMNICAST_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_FLAGS="$f" > /dev/null
  ninja -C $OUT/$name omnicast_prof_gpu 2>&1 | grep -E "error|FAILED"
  echo "$name built ($f)"
}
build a 4 4 4 4
build b 2 4 16 8
build c 1 8 16 16
build d 2 8 8 16
