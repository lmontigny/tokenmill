#!/usr/bin/env bash
# Sweep arrival rate in parallel (rayon) and emit CSV for plotting.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model llama-8b-fp8 --gpu h100 \
    --scheduler chunked-prefill \
    --sweep-arrival-rates 5,10,20,30,40,50 \
    --duration 60.0 \
    --output csv
