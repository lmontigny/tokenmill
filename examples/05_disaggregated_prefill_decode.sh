#!/usr/bin/env bash
# Disaggregated prefill/decode — separate GPU pools with network KV transfer.
# Decode latency stays low even at 50 req/s because prefill bursts don't preempt decode.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model llama-8b-fp8 --gpu h100 --tp 2 \
    --disaggregate --internode-bw-gbps 200 \
    --arrival-rate 50.0 --duration 60.0
