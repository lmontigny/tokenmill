#!/usr/bin/env bash
# Common inference-sim invocations.
# Run from the repo root: bash scripts/run_examples.sh

set -euo pipefail
BIN="cargo run --release --"

echo "================================================================"
echo "1. llama-8b / H100 / continuous-batch / roofline"
echo "================================================================"
$BIN --model llama-8b --gpu h100 --scheduler continuous-batch \
     --arrival-rate 10.0 --duration 60.0

echo ""
echo "================================================================"
echo "2. llama-8b / H100 / chunked-prefill / roofline"
echo "================================================================"
$BIN --model llama-8b --gpu h100 --scheduler chunked-prefill \
     --chunk-size 512 --arrival-rate 10.0 --duration 60.0

echo ""
echo "================================================================"
echo "3. llama-8b / H100 / chunked-prefill / kernel table"
echo "================================================================"
$BIN --model llama-8b --gpu h100 --scheduler chunked-prefill \
     --arrival-rate 10.0 --duration 60.0 \
     --kernel-table data/kernel_table.csv

echo ""
echo "================================================================"
echo "4. llama-70b / H100 / TP=8 / chunked-prefill"
echo "================================================================"
$BIN --model llama-70b --gpu h100 --scheduler chunked-prefill \
     --tp 8 --arrival-rate 5.0 --duration 60.0 \
     --kernel-table data/kernel_table.csv

echo ""
echo "================================================================"
echo "5. llama-70b / H100 / TP=4 PP=2 / chunked-prefill"
echo "================================================================"
$BIN --model llama-70b --gpu h100 --scheduler chunked-prefill \
     --tp 4 --pp 2 --arrival-rate 5.0 --duration 60.0

echo ""
echo "================================================================"
echo "6. Throughput sweep: arrival-rate 2 → 10 → 20 req/s"
echo "================================================================"
for rate in 2 5 10 20; do
    echo "--- arrival-rate=$rate req/s ---"
    $BIN --model llama-8b --gpu h100 --scheduler chunked-prefill \
         --arrival-rate "$rate" --duration 60.0 \
         --kernel-table data/kernel_table.csv 2>/dev/null \
    | grep -E "Throughput|TTFT|p50"
done
