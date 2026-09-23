#!/usr/bin/env bash
# Run inside dgpp-spark-cross:cuda13 from this checkout. Output stays on caller's disk.
set -euo pipefail
out=${1:?pass an absolute output directory on the experiment disk}
mkdir -p "$out"
for mode in 0 1; do
  mkdir -p "$out/tmp-$mode"
  TMPDIR="$out/tmp-$mode" nvcc -ccbin=aarch64-linux-gnu-g++-13 \
    --target-directory sbsa-linux --expt-relaxed-constexpr --extended-lambda \
    -O3 -DNDEBUG -std=c++20 -arch=sm_121a -Xcompiler=-ffp-contract=off -I src \
    -DDGPP_MIMO_FP8_KV_FAST_LOAD="$mode" -ptx src/kernels/mimo_attn.cu \
    -o "$out/attention-$mode.ptx"
  ptxas -v -arch=sm_121a "$out/attention-$mode.ptx" \
    -o "$out/attention-$mode.cubin" 2> "$out/attention-$mode.log"
done
TMPDIR="$out/tmp-1" nvcc -ccbin=aarch64-linux-gnu-g++-13 \
  --target-directory sbsa-linux --expt-relaxed-constexpr -O3 -std=c++20 \
  -arch=sm_121a -I src -DDGPP_MIMO_FP8_KV_FAST_LOAD=1 \
  -ptx tests/cuda/mimo_fp8_load_test.cu -o "$out/probe.ptx"
ptxas -v -arch=sm_121a "$out/probe.ptx" -o "$out/probe.cubin" 2> "$out/probe.log"
