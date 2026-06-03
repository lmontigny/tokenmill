#!/usr/bin/env bash
# NVIDIA HGX Rubin NVL8 (2026/2027) — 8× Rubin SXM in one server.
# 288 GB HBM4 per chip at 22 TB/s, NVLink 6 (3.6 TB/s/chip), 17.5 PFLOPS FP8 dense.
#
# Reference workload from NVIDIA's HGX Rubin announcement: Kimi K2 Thinking with
# ISL=OSL=4K. Compare to examples/12_frontier_kimi_k2.sh (same Kimi K2 on HGX B200).
#
# Note: NVIDIA's published "10× more token-factory throughput" claim uses sparse
# NVFP4 on Rubin vs dense NVFP4 on B200. We model FP8 dense for both, so the gap
# you see here (~1.6× faster TPOT) reflects the raw HBM-BW and FLOPS scaling
# *without* the additional sparsity / FP4 boost.
set -euo pipefail
cd "$(dirname "$0")/.."
cargo run --release -- \
    --model kimi-k2 --gpu rubin --tp 8 --ep 8 \
    --scheduler chunked-prefill --chunk-size 1024 \
    --prompt-mean 4096 --output-mean 4096 \
    --arrival-rate 5.0 --duration 60.0
