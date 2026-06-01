#!/usr/bin/env bash
# Google TPU v8i (2026, serving-optimized) — projected specs.
# 3D-torus ICI within a pod; ring-allreduce assumed along one torus dimension.
# Compare to examples/10_b200_blackwell.sh and 12_frontier_kimi_k2.sh for the
# B200-vs-TPU v8i story at the trillion-param frontier.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model kimi-k2 --gpu tpu-v8i --tp 8 --ep 8 \
    --scheduler chunked-prefill \
    --arrival-rate 2.0 --duration 60.0
