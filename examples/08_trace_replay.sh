#!/usr/bin/env bash
# Replay the Mooncake conversation trace (long-context: avg prompt ~7k tokens).
# Requires running scripts/fetch_traces.sh mooncake first.
set -euo pipefail
cd "$(dirname "$0")/.."

TRACE="data/traces/mooncake_conversation.csv"
if [[ ! -f "$TRACE" ]]; then
    echo "Trace not found at $TRACE — fetching now..."
    bash scripts/fetch_traces.sh mooncake
fi

cargo run --release -- \
    --model llama-70b-fp8 --gpu h100 --tp 4 \
    --scheduler chunked-prefill \
    --workload "trace:$TRACE" \
    --duration 300.0
