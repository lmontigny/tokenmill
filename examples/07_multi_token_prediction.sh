#!/usr/bin/env bash
# Multi-token prediction — K=3 lightweight heads on the main model.
# Higher acceptance rate than spec decoding (0.9 vs 0.7) because heads are jointly trained.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model llama-8b-fp8 --gpu h100 \
    --scheduler chunked-prefill \
    --mtp-heads 3 --mtp-acceptance-rate 0.9 \
    --arrival-rate 30.0 --duration 60.0
