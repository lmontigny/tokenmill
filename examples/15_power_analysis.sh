#!/usr/bin/env bash
# Power and energy analysis — sweep arrival rate to see how mean power and
# energy-per-token track utilization.
#
# At low load, idle power dominates → high mJ/token.
# At high load, active power dominates and per-token energy drops to its floor.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "Rate | Throughput | Mean power | Energy/token"
echo "-----|------------|------------|-------------"
for rate in 1 5 10 20 50; do
    out=$(cargo run --release -q -- \
        --model llama-8b-fp8 --gpu h100 \
        --scheduler chunked-prefill \
        --arrival-rate "$rate" --duration 30 \
        --output csv 2>/dev/null | tail -1)
    tput=$(echo "$out" | cut -d, -f9)
    mw=$(echo "$out" | cut -d, -f23)
    epot=$(echo "$out" | cut -d, -f24)
    printf "%4s | %10s | %9s kW | %8s mJ\n" "$rate" "$tput" "$mw" "$epot"
done
