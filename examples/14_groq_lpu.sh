#!/usr/bin/env bash
# Groq LPU v1 — no off-chip HBM, all weights live in 230 MB on-chip SRAM at 80 TB/s.
#
# This forces very high TP: llama-8b-fp8 needs 64 chips just to fit weights
# (8 GB / 230 MB ≈ 35, rounded up for KV headroom). llama-70b-fp8 needs ~358 chips.
#
# The deterministic compiler-scheduled architecture yields very high MFU,
# but per-hop link latency in the chip mesh is *not* modelled here — real Groq
# TPOT at hundreds of TP is higher than the simulator's spec-sheet ceiling.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model llama-8b-fp8 --gpu groq-lpu-v1 --tp 64 \
    --scheduler chunked-prefill \
    --arrival-rate 30.0 --duration 60.0
