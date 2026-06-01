# CLI reference

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu` | `h100` | GPU preset: `b200` \| `h100` \| `a100` \| `a10g` |
| `--model` | `llama-70b` | Model preset: `llama-70b` \| `llama-8b` \| `llama-70b-fp8` \| `llama-8b-fp8` \| `mixtral-8x7b` \| `llama4-maverick` \| `deepseek-v3` |
| `--scheduler` | `continuous-batch` | `continuous-batch` \| `chunked-prefill` |
| `--chunk-size` | `512` | Prefill chunk tokens (chunked-prefill only) |
| `--arrival-rate` | `5.0` | Requests per second (Poisson) |
| `--duration` | `60.0` | Simulation duration in seconds |
| `--prompt-mean` | `512.0` | Mean prompt length (log-normal, tokens) |
| `--output-mean` | `128.0` | Mean output length (log-normal, tokens) |
| `--tp` | `1` | Tensor parallelism degree |
| `--pp` | `1` | Pipeline parallelism degree |
| `--ep` | `1` | Expert parallelism degree (MoE models) |
| `--disaggregate` | off | Separate prefill and decode GPUs; KV transferred over network |
| `--internode-bw-gbps` | `200` | Network bandwidth for KV transfer (GB/s) |
| `--kernel-table` | — | CSV file with profiled kernel latencies |
| `--workload` | `synthetic` | `synthetic` or `trace:<path.csv>` (native or Azure format, auto-detected) |
| `--output` | `text` | `text` \| `json` \| `csv` |
| `--sweep-arrival-rates` | — | Comma-separated rates to sweep in parallel (e.g. `1,5,10,20`) |
| `--kv-block-size` | `16` | Tokens per KV cache block |
| `--kv-blocks` | `0` | KV cache blocks (0 = auto from GPU HBM) |
| `--max-batch-tokens` | `8192` | Token budget across all in-flight requests |
| `--seed` | `42` | Random seed |
| `--spec-tokens` | `0` | Speculative decoding: draft tokens per step K (0 = off) |
| `--spec-acceptance-rate` | `0.7` | Speculative decoding: per-token acceptance rate γ |
| `--draft-model` | auto | Draft model preset (defaults to 1/10-size clone of main model) |
| `--mtp-heads` | `0` | Multi-token prediction: MTP heads K (0 = off; mutually exclusive with `--spec-tokens`) |
| `--mtp-acceptance-rate` | `0.9` | Multi-token prediction: per-token acceptance rate γ |
| `--validate-kernels` | — | Compare roofline predictions against a reference CSV and report MAPE |

## Model presets

| Preset | Type | Total params | Active/token | Notes |
|--------|------|-------------|--------------|-------|
| `llama-70b` | dense | 70 B (bf16) | 70 B | GQA, 80 layers |
| `llama-8b` | dense | 8 B (bf16) | 8 B | GQA, 32 layers |
| `llama-70b-fp8` | dense | 70 B (fp8) | 70 B | FP8 quant; 2× faster decode/prefill vs bf16 |
| `llama-8b-fp8` | dense | 8 B (fp8) | 8 B | FP8 quant; matches TRT-LLM/NIM benchmark dtype |
| `mixtral-8x7b` | MoE | 46.7 B (bf16) | ~12.9 B | 8 experts top-2, all layers MoE |
| `llama4-maverick` | MoE | 400 B (fp8) | 17 B | 128 experts top-1+1 shared, 36/48 MoE layers |
| `deepseek-v3` | MoE | 671 B (fp8) | 37 B | 256 experts top-8+1 shared, MLA KV compression |

DeepSeek V3 uses **Multi-head Latent Attention (MLA)** which compresses the KV cache to a 512-dimensional
latent vector per layer — roughly 64× smaller than standard MHA — enabling long-context serving at scale.

## GPU presets

| Preset | Family | BF16 TFLOPS | FP8 TFLOPS | HBM | HBM BW | NVLink |
|---|---|---:|---:|---:|---:|---:|
| `b200` | Blackwell | 2250 | 4500 | 192 GB | 8 TB/s | 1.8 TB/s |
| `h100` | Hopper | 989 | 1978 | 80 GB | 3.35 TB/s | 900 GB/s |
| `a100` | Ampere | 312 | — | 80 GB | 2 TB/s | 600 GB/s |
| `a10g` | Ampere | 125 | — | 24 GB | 600 GB/s | — |
