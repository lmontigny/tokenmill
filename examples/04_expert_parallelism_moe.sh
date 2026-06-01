#!/usr/bin/env bash
# Expert parallelism on a MoE model.
# Mixtral 8×7B with experts sharded across 4 GPUs via NVLink all-to-all.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model mixtral-8x7b --gpu h100 --ep 4 \
    --scheduler chunked-prefill \
    --arrival-rate 8.0 --duration 60.0
