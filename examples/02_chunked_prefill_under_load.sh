#!/usr/bin/env bash
# Chunked prefill (Sarathi) — saturating arrival rate.
# Compare TTFT/p95 against 01_continuous_batch.sh to see chunked-prefill keep TTFT bounded.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model llama-8b-fp8 --gpu h100 \
    --scheduler chunked-prefill --chunk-size 512 \
    --arrival-rate 50.0 --duration 60.0
