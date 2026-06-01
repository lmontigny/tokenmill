#!/usr/bin/env bash
# Continuous batching (Orca) — light load on llama-8b / H100.
# Demonstrates baseline performance under non-saturated arrival rate.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model llama-8b --gpu h100 \
    --scheduler continuous-batch \
    --arrival-rate 10.0 --duration 60.0
