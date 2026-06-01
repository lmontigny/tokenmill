#!/usr/bin/env bash
# Tensor parallelism — llama-70b sharded across 4×H100 with NVLink all-reduce.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model llama-70b --gpu h100 --tp 4 \
    --scheduler chunked-prefill \
    --arrival-rate 5.0 --duration 60.0
