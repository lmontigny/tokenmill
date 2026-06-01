# Trace data

Two CSV formats are accepted by `--workload trace:<path>`, auto-detected from the header:

| Format | Header | Timestamp |
|--------|--------|-----------|
| **Native** | `timestamp_ms,prompt_tokens,output_tokens` | relative ms from start |
| **Azure** | `TIMESTAMP,ContextTokens,GeneratedTokens` | ISO datetime `YYYY-MM-DD HH:MM:SS.fff` |

## Public traces

Run `bash scripts/fetch_traces.sh` to download and normalise the traces below into `data/traces/`
(gitignored). Pass a name (`azure`, `burstgpt`, `mooncake`) to fetch a single source.

| Trace | Requests | Workload | Source / Paper |
|-------|---------:|---------|----------------|
| `azure_code.csv` | ~8 800 | Coding assistant (short outputs, heavy prompts) | [AzurePublicDataset](https://github.com/Azure/AzurePublicDataset) — Splitwise ISCA'24 |
| `azure_conv.csv` | ~16 000 | Conversational (longer outputs) | AzurePublicDataset — DynamoLLM HPCA'25 |
| `burstgpt.csv` | 1.4 M | Real ChatGPT / GPT-4 production logs; bursty arrivals | [HPMLL/BurstGPT](https://github.com/HPMLL/BurstGPT) (Tsinghua) |
| `mooncake_conversation.csv` | 12 031 | Long-context conversations (mean prompt ~7 k tokens) | [kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) — Mooncake FAST'25 |
| `mooncake_synthetic.csv` | 3 993 | Synthetic mixed-length | Mooncake FAST'25 |
| `mooncake_toolagent.csv` | 23 608 | Agentic tool-calling workload | Mooncake FAST'25 |

Example:
```bash
bash scripts/fetch_traces.sh mooncake
./target/release/tokenmill \
  --model llama-70b-fp8 --gpu h100 --tp 4 \
  --workload trace:data/traces/mooncake_conversation.csv \
  --duration 300
```

The conversion script collapses each upstream format to the **Native** CSV the simulator reads
natively. Azure files are kept in their original Azure format (auto-detected by the trace loader).
