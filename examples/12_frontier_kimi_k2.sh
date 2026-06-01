#!/usr/bin/env bash
# Kimi K2 — 1.026 T params total, 32 B active per token.
# Sparse MoE + MLA KV lets a trillion-param model serve at sub-millisecond TPOT on 8 GPUs.
# The 32 B *active* footprint sets the decode cost, not the 1 T total.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model kimi-k2 --gpu b200 --tp 8 --ep 8 \
    --scheduler chunked-prefill \
    --arrival-rate 2.0 --duration 60.0
