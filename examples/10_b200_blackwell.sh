#!/usr/bin/env bash
# B200 Blackwell — 2-3× faster than H100 on FP8 workloads.
# Compare TTFT/TPOT against examples/03_tensor_parallelism.sh (same model on H100).
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model llama-70b-fp8 --gpu b200 --tp 4 \
    --scheduler chunked-prefill \
    --arrival-rate 5.0 --duration 60.0
