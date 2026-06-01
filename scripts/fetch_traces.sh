#!/usr/bin/env bash
# Download and normalise public LLM inference traces.
#
# Usage:
#   bash scripts/fetch_traces.sh             # all traces
#   bash scripts/fetch_traces.sh azure       # Azure LLM Inference traces only
#   bash scripts/fetch_traces.sh burstgpt    # BurstGPT (GPT-3.5/4 production logs)
#   bash scripts/fetch_traces.sh mooncake    # Mooncake / Kimi (Moonshot AI)
#
# Output: data/traces/*.csv consumable by --workload trace:<path>

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/data/traces"
mkdir -p "$OUT"

# Pre-flight: need curl and python3.
command -v curl    >/dev/null || { echo "curl required" >&2; exit 1; }
command -v python3 >/dev/null || { echo "python3 required" >&2; exit 1; }

dl() {
    local url="$1" dst="$2"
    if [[ -s "$dst" ]]; then
        echo "  • already present: $dst ($(du -h "$dst" | cut -f1))"
    else
        echo "  • downloading $(basename "$dst") …"
        curl -fL --retry 3 --connect-timeout 30 -o "$dst" "$url"
    fi
}

# ── Azure LLM Inference Traces ────────────────────────────────────────────────
# Source: github.com/Azure/AzurePublicDataset (Splitwise ISCA'24, DynamoLLM HPCA'25)
# Format: TIMESTAMP,ContextTokens,GeneratedTokens — natively supported by --workload trace.
fetch_azure() {
    echo "==> Azure LLM Inference traces"
    dl "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv" \
       "$OUT/azure_code.csv"
    dl "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv" \
       "$OUT/azure_conv.csv"
    echo "  ready: --workload trace:data/traces/azure_code.csv"
    echo "  ready: --workload trace:data/traces/azure_conv.csv"
}

# ── BurstGPT ──────────────────────────────────────────────────────────────────
# Source: github.com/HPMLL/BurstGPT (Tsinghua) — real ChatGPT / GPT-4 production
# logs, captures bursty arrival patterns absent from synthetic Poisson.
# Raw format: Timestamp,Model,Request tokens,Response tokens,Total tokens,Log Type
# Full trace is 1.4 M rows (~50 MB) — pass --L to also fetch BurstGPT_2.csv.
fetch_burstgpt() {
    echo "==> BurstGPT (ChatGPT / GPT-4 production trace, Tsinghua)"
    local raw="$OUT/burstgpt_raw.csv"
    dl "https://github.com/HPMLL/BurstGPT/raw/main/data/BurstGPT_1.csv" "$raw"
    python3 - "$raw" "$OUT/burstgpt.csv" <<'PY'
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f, open(dst, "w", newline="") as g:
    r = csv.DictReader(f)
    w = csv.writer(g)
    w.writerow(["timestamp_ms", "prompt_tokens", "output_tokens"])
    n = 0
    for row in r:
        try:
            ts_ms = float(row["Timestamp"]) * 1000.0
            pt = int(float(row["Request tokens"]))
            ot = int(float(row["Response tokens"]))
            if pt > 0 and ot > 0:
                w.writerow([f"{ts_ms:.1f}", pt, ot]); n += 1
        except (KeyError, ValueError):
            continue
print(f"  normalised {n} rows → {dst}")
PY
    echo "  ready: --workload trace:data/traces/burstgpt.csv"
}

# ── Mooncake / Kimi production traces ────────────────────────────────────────
# Source: github.com/kvcache-ai/Mooncake/FAST25-release/traces (Moonshot AI,
# Mooncake FAST'25). Three workloads:
#   conversation_trace.jsonl — long-context conversations
#   synthetic_trace.jsonl    — synthetic mix, smaller
#   toolagent_trace.jsonl    — agentic tool-calling workload
# Raw format: JSONL with {timestamp, input_length, output_length, hash_ids}
fetch_mooncake() {
    echo "==> Mooncake / Kimi production traces (Moonshot AI, FAST'25)"
    local base="https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces"
    for name in conversation synthetic toolagent; do
        local raw="$OUT/mooncake_${name}_raw.jsonl"
        dl "$base/${name}_trace.jsonl" "$raw"
        python3 - "$raw" "$OUT/mooncake_${name}.csv" <<'PY'
import csv, json, sys
src, dst = sys.argv[1], sys.argv[2]
n = 0
with open(src) as f, open(dst, "w", newline="") as g:
    w = csv.writer(g)
    w.writerow(["timestamp_ms", "prompt_tokens", "output_tokens"])
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            ts  = float(d.get("timestamp", 0))
            pt  = int(d.get("input_length", 0))
            ot  = int(d.get("output_length", 0))
            if pt > 0 and ot > 0:
                w.writerow([f"{ts:.1f}", pt, ot]); n += 1
        except (json.JSONDecodeError, ValueError):
            continue
print(f"  normalised {n} rows → {dst}")
PY
        echo "  ready: --workload trace:data/traces/mooncake_${name}.csv"
    done
}

case "${1:-all}" in
    azure)    fetch_azure ;;
    burstgpt) fetch_burstgpt ;;
    mooncake) fetch_mooncake ;;
    all)      fetch_azure; echo; fetch_burstgpt; echo; fetch_mooncake ;;
    *)        echo "Usage: $0 [azure|burstgpt|mooncake|all]" >&2; exit 1 ;;
esac

echo
echo "Done. Traces under $OUT:"
ls -lh "$OUT" | awk 'NR>1 {printf "  %-30s  %s\n", $9, $5}'
