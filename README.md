# inference-sim

Discrete Event Simulator (DES) for LLM inference workloads, written in Rust.

Models prefill/decode phases, KV cache, continuous batching, chunked prefill,
tensor parallelism, pipeline parallelism, and disaggregated prefill/decode.
Targets **~10% error** vs real GPU hardware.

## Build

```bash
cargo build --release
```

## Quick start

```bash
# llama-8b on H100, chunked-prefill, 10 req/s for 60s
cargo run --release -- \
  --model llama-8b --gpu h100 \
  --scheduler chunked-prefill \
  --arrival-rate 10.0 --duration 60.0

# same with profiled kernel latencies (more accurate)
cargo run --release -- \
  --model llama-8b --gpu h100 \
  --scheduler chunked-prefill \
  --arrival-rate 10.0 --duration 60.0 \
  --kernel-table data/kernel_table.csv
```

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu` | `h100` | GPU preset: `h100` \| `a100` \| `a10g` |
| `--model` | `llama-70b` | Model preset: `llama-70b` \| `llama-8b` \| `mixtral-8x7b` \| `llama4-maverick` \| `deepseek-v3` |
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
| `--workload` | `synthetic` | `synthetic` or `trace:<path.csv>` |
| `--output` | `text` | `text` \| `json` \| `csv` |
| `--sweep-arrival-rates` | — | Comma-separated rates to sweep in parallel (e.g. `1,5,10,20`) |
| `--kv-block-size` | `16` | Tokens per KV cache block |
| `--kv-blocks` | `0` | KV cache blocks (0 = auto from GPU HBM) |
| `--max-batch-tokens` | `8192` | Token budget across all in-flight requests |
| `--seed` | `42` | Random seed |

## Model presets

| Preset | Type | Total params | Active/token | Notes |
|--------|------|-------------|--------------|-------|
| `llama-70b` | dense | 70 B (bf16) | 70 B | GQA, 80 layers |
| `llama-8b` | dense | 8 B (bf16) | 8 B | GQA, 32 layers |
| `mixtral-8x7b` | MoE | 46.7 B (bf16) | ~12.9 B | 8 experts top-2, all layers MoE |
| `llama4-maverick` | MoE | 400 B (fp8) | 17 B | 128 experts top-1+1 shared, 36/48 MoE layers |
| `deepseek-v3` | MoE | 671 B (fp8) | 37 B | 256 experts top-8+1 shared, MLA KV compression |

DeepSeek V3 uses **Multi-head Latent Attention (MLA)** which compresses the KV cache to a 512-dimensional
latent vector per layer — roughly 64× smaller than standard MHA — enabling long-context serving at scale.

## Latency model

Without `--kernel-table`: **roofline** (compute-bound prefill, memory-BW-bound decode).
With `--kernel-table`: table lookup with linear interpolation on seq_len, roofline fallback on miss.

To improve accuracy, add rows to `data/kernel_table.csv` from your own profiling.
CSV format: `gpu,model,op,batch_size,seq_len,latency_ms`

## MoE accuracy model

**Prefill** (compute-bound, active FLOPs only):
```
active_flops = base_flops × active_param_fraction
active_param_fraction = active_weight_bytes / weight_bytes   (when set)
                      ≈ 1/3 (attn) + 2/3 × (dense_layers + moe_layers × top_K/n_experts)
```

**Decode** (memory-BW bound, active weights):
```
latency = (weight_bytes_active + kv_bytes × batch) / (HBM_BW × mfu_decode)
weight_bytes_active = active weight bytes for one forward pass (presets use published values)
```

**DeepSeek V3 KV cache** (MLA compression):
```
kv_bytes = n_layers × kv_lora_rank × seq_len × dtype_bytes   (vs 2 × n_layers × n_kv_heads × head_dim × seq_len)
```

**EP all-to-all** (2 per MoE layer: token dispatch + result gather):
```
latency = (ep-1)/ep × tokens_per_gpu × d_model × dtype_bytes / nvlink_bw
```

## Architecture

```
engine/        DES core: BinaryHeap event queue, SimTime clock, dispatch loop
hardware/      GPU specs, cluster topology (TP/PP), kernel time table
model/         LLM config, KV cache block manager
scheduler/     continuous-batch (Orca) and chunked-prefill (Sarathi)
workload/      Poisson synthetic arrivals, trace replay (Phase 6)
metrics/       HDR histograms for TTFT/TPOT, throughput, KV utilization
```

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ | DES engine, roofline GPU model, synthetic workload, TTFT/TPOT metrics |
| 2 | ✅ | KV cache block manager, batch decode iterations, chunked-prefill scheduler |
| 3 | ✅ | Kernel time table (CSV), linear interpolation, roofline fallback |
| 4 | ✅ | Multi-GPU: tensor parallelism + pipeline parallelism |
| 5 | ✅ | Disaggregated prefill/decode with KV transfer latency |
| 6 | ✅ | Trace replay, JSON/CSV output, parallel sweep with rayon |
| 7 | ✅ | MoE: sparse activation, expert parallelism, MLA KV compression (Llama 4 Maverick, DeepSeek V3) |
