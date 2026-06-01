#!/usr/bin/env bash
# Speculative decoding — K=3 draft tokens from a small draft model,
# verified by one main-model pass. Expected speedup ~24% on TPOT for γ=0.75.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model llama-8b-fp8 --gpu h100 \
    --scheduler chunked-prefill \
    --spec-tokens 3 --spec-acceptance-rate 0.75 \
    --arrival-rate 30.0 --duration 60.0
