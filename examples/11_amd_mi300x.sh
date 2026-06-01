#!/usr/bin/env bash
# AMD Instinct MI300X — 192 GB HBM3 at 5.3 TB/s.
# Bigger memory than H100 means more models fit on a single GPU (no TP needed),
# and ~1.4× higher decode throughput on memory-bound workloads.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model llama-70b-fp8 --gpu mi300x \
    --scheduler chunked-prefill \
    --arrival-rate 3.0 --duration 60.0
