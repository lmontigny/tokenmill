# inference-sim

Discrete Event Simulator (DES) for LLM inference workloads, written in Rust.

Models prefill/decode phases, KV cache, continuous batching, chunked prefill,
tensor parallelism, pipeline parallelism, disaggregated prefill/decode,
speculative decoding, and multi-token prediction.
Targets **~10% error** vs real GPU hardware.

## Example results

Simulated results for various model/GPU/scheduler configurations (60 s runs, log-normal prompts μ=512 tokens, outputs μ=128 tokens).

| # | Model | GPU / TP | Scheduler | Rate (req/s) | Thrpt (rps) | TTFT p50 | TTFT p95 | TPOT p50 | Notes |
|---|-------|----------|-----------|:---:|:---:|---:|---:|---:|---|
| 1 | llama-8b-fp8 | H100 TP=1 | continuous-batch | 10 | 9.9 | 13 ms | 23 ms | 3.2 ms | Light-load baseline |
| 2 | llama-8b-fp8 | H100 TP=1 | continuous-batch | 50 | 45.0 | **2435 ms** | 6451 ms | 2.0 ms | Saturated — TTFT collapses as prefill starves |
| 3 | llama-8b-fp8 | H100 TP=1 | chunked-prefill | 50 | **50.4** | **21 ms** | 38 ms | 4.5 ms | Chunked-prefill absorbs full load; TTFT stays bounded |
| 4 | llama-8b-fp8 | H100 TP=1 | chunked-prefill + spec K=3 γ=0.75 | 30 | 30.5 | 17 ms | 31 ms | **1.3 ms** | Spec decoding: −24% TPOT vs 1.7 ms baseline (row below) |
| 5 | llama-8b-fp8 | H100 TP=1 | chunked-prefill (baseline) | 30 | 30.5 | 21 ms | 36 ms | 1.7 ms | Baseline for row 4 comparison |
| 6 | llama-8b-fp8 | H100 TP=2 disagg | continuous-batch | 50 | 50.7 | 21 ms | 257 ms | **1.6 ms** | Disaggregated P/D: decode isolated, fast TPOT |
| 7 | llama-70b (bf16) | H100 TP=4 | continuous-batch | 5 | 5.1 | 178 ms | **1582 ms** | 15.2 ms | 70B BF16: p95 TTFT blows out under CB |
| 8 | llama-70b (bf16) | H100 TP=4 | chunked-prefill | 5 | 5.1 | 71 ms | **123 ms** | 15.2 ms | Chunked-prefill trims p95 TTFT 13× vs row 7 |
| 9 | llama-70b-fp8 | H100 TP=4 | chunked-prefill | 5 | 5.2 | **30 ms** | 55 ms | 7.0 ms | FP8: 2.4× lower TTFT and TPOT vs BF16 (row 8) |
| 10 | mixtral-8x7b | H100 EP=4 | chunked-prefill | 8 | 7.8 | 29 ms | 52 ms | 6.5 ms | 47 B MoE — expert weights sharded across 4 GPUs |
| 11 | deepseek-v3 | H100 TP=8 EP=8 | chunked-prefill | 3 | 3.2 | 7 ms | 13 ms | 1.7 ms | 671 B MoE / MLA KV — 9 active experts per token |
| 12 | llama-8b-fp8 | **B200** TP=1 | chunked-prefill | 50 | 50.8 | **7 ms** | 13 ms | **1.6 ms** | Blackwell: ~3× lower TTFT, ~3× lower TPOT vs H100 (row 3) |
| 13 | llama-70b-fp8 | **B200** TP=4 | chunked-prefill | 5 | 5.2 | **12 ms** | 23 ms | **3.0 ms** | Blackwell on 70B: 2.4× faster TTFT, 2.3× faster TPOT vs H100 (row 9) |
| 14 | deepseek-v3 | **B200** TP=8 EP=8 | chunked-prefill | 5 | 5.2 | **3 ms** | 6 ms | **0.8 ms** | 671 B MoE on Blackwell — sub-ms TPOT |

Key patterns:
- **Rows 2 vs 3**: under saturation, `chunked-prefill` keeps TTFT at ~21 ms where `continuous-batch` lets it spike to 2.4 s.
- **Rows 7 vs 8 vs 9**: for a 70B model, chunked-prefill cuts p95 TTFT 13×; switching from BF16 to FP8 cuts it another 2.4×.
- **Row 4**: speculative decoding (`--spec-tokens 3`) reduces TPOT by 24% at the same throughput.
- **Row 11**: DeepSeek V3 (671 B) on 8×H100 with EP=8 serves at 1.7 ms TPOT — MLA KV compression keeps the KV footprint tiny.
- **Rows 12–14**: B200 (Blackwell) delivers ~2.3–3× speedup over H100 across the board — 2× FP8 TFLOPS and 2.4× HBM BW (8 TB/s vs 3.35 TB/s).

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

# replay the Azure LLM inference trace (auto-detected format)
# download: https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLLMInferenceTrace_code.csv
cargo run --release -- \
  --model llama-8b --gpu h100 \
  --workload trace:data/AzureLLMInferenceTrace_code.csv \
  --duration 3600.0
```

## CLI flags

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

## Trace data

Two CSV formats are accepted by `--workload trace:<path>`, auto-detected from the header:

| Format | Header | Timestamp |
|--------|--------|-----------|
| **Native** | `timestamp_ms,prompt_tokens,output_tokens` | relative ms from start |
| **Azure** | `TIMESTAMP,ContextTokens,GeneratedTokens` | ISO datetime `YYYY-MM-DD HH:MM:SS.fff` |

### Public traces

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
./target/release/inference-sim \
  --model llama-70b-fp8 --gpu h100 --tp 4 \
  --workload trace:data/traces/mooncake_conversation.csv \
  --duration 300
```

The conversion script collapses each upstream format to the **Native** CSV the simulator reads
natively. Azure files are kept in their original Azure format (auto-detected by the trace loader).

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

## Benchmark validation

Run the roofline model against reference kernel latencies:

```bash
./target/release/inference-sim --validate-kernels data/reference_kernels.csv
```

Results against `data/reference_kernels.csv` (GPU kernel time only; serving frameworks add 3–10 ms overhead):

| GPU | Model | Op | MAPE | Notes |
|-----|-------|----|------|-------|
| H100-SXM5 | llama-8b (bf16) | prefill | <1% | vLLM TTFT benchmark |
| H100-SXM5 | llama-8b (bf16) | decode | <2% | vLLM throughput benchmark |
| H100-SXM5 | llama-8b-fp8 | prefill | ~8% | NVIDIA TRT-LLM FP8, TP=1 |
| H100-SXM5 | llama-8b-fp8 | decode | ~13% | TRT-LLM FP8; degrades at large batch×seq |
| H100-SXM5 | llama-70b (bf16) | all | <1% | MLPerf v4.0 / vLLM TP=8 |
| A100-80GB | llama-8b (bf16) | all | <1% | vLLM benchmark |

**Key findings:**
- **Prefill** (compute-bound): ~8% MAPE on FP8. `flops_fp8` (1978 TFLOPS on H100) is used automatically when `dtype_bytes == 1`.
- **Decode** (memory-BW bound): 5–20% MAPE. Error grows at large batch × seq_len where paged KV access is ~60–70% efficient vs sequential weight reads (80% mfu assumed for both).
- **Serving TPOT vs kernel time**: framework overhead (Python scheduler, CUDA launch, NCCL) adds 3–10 ms per step and is **not** modeled. Subtract this from observed TPOT before comparing to simulator output.

## Latency model

Without `--kernel-table`: **roofline** (compute-bound prefill, memory-BW-bound decode).
With `--kernel-table`: table lookup with linear interpolation on seq_len, roofline fallback on miss.

**FP8 dispatch**: when `model.dtype_bytes == 1`, prefill uses `flops_fp8` (H100: 1978 TFLOPS, 2× BF16)
instead of `flops_bf16`. Decode is memory-BW bound regardless of dtype — the speedup there comes from
loading half as many bytes (FP8 weight bytes = BF16/2).

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

**EP decode bandwidth** (per GPU, EP > 1):
```
bytes_per_gpu = (weight_attn + weight_dense_ffn + kv_bytes) / tp   ← TP-sharded
              + weight_expert_active / ep                            ← EP-sharded
weight_expert_active = n_moe_layers × (n_active_experts + n_shared_experts) × 2 × d_model × expert_hidden × dtype
```
When TP = EP (e.g. TP=8, EP=8) per-GPU BW is the same as EP=1. The difference appears when TP ≠ EP.

**EP all-to-all** (2 per MoE layer: token dispatch + result gather, over NVLink switch fabric):
```
expert_activations = batch_tokens × top_K            ← each token fans out to top_K experts
tokens_per_gpu     = expert_activations / ep
data_per_gpu       = (ep-1)/ep × tokens_per_gpu × d_model × dtype_bytes
latency            = data_per_gpu / nvlink_bw         ← full bisection BW on NVSwitch
```

## Supported optimizations

| Optimization | Flag(s) | What it models | Effect |
|---|---|---|---|
| **Continuous batching** | `--scheduler continuous-batch` | Requests join/leave the decode batch every step (Orca) | Maximises GPU utilisation; eliminates idle time between requests |
| **Chunked prefill** | `--scheduler chunked-prefill --chunk-size N` | Prefill split into N-token chunks interleaved with decode (Sarathi) | Bounds TTFT jitter; decode throughput not starved by long prompts |
| **KV cache block manager** | automatic | PagedAttention-style block allocator; frees blocks on completion | Near-zero KV fragmentation; enables high batch sizes |
| **Preemption / recompute** | automatic | When KV pool exhausted, evicts the largest in-flight decode request; re-prefills it on next admission | Prevents starvation under KV pressure; trades latency for liveness |
| **Tensor parallelism** | `--tp N` | Weights and KV heads sharded across N GPUs; ring all-reduce per layer on NVLink | Decode latency ÷ N (memory-BW bound); all-reduce adds ~2 % overhead at TP=8 |
| **Pipeline parallelism** | `--pp N` | Layers split across N GPU groups; point-to-point activation transfer between stages | Reduces per-GPU memory; pipeline bubbles modeled via event timing |
| **Expert parallelism** | `--ep N` | MoE expert shards across N GPUs within one NVLink scale-up domain (NVL72 / HGX / DGX) | Expert weights ÷ EP per GPU; two NVLink all-to-alls per MoE layer (dispatch + combine). No Infiniband modeled. |
| **Disaggregated prefill/decode** | `--disaggregate [--internode-bw-gbps B]` | Prefill and decode run on separate GPU pools; KV tensor transferred over network after prefill | Isolates prefill bursts from decode latency; KV transfer cost added as network delay |
| **Speculative decoding** | `--spec-tokens K [--draft-model M] [--spec-acceptance-rate γ]` | K draft tokens from a small model, verified in one main-model pass | E[tok/step] = (1−γ^{K+1})/(1−γ); speedup ∝ acceptance rate and draft model speed |
| **Multi-token prediction (MTP)** | `--mtp-heads K [--mtp-acceptance-rate γ]` | K extra heads (≈1 layer each) on the main model; one forward pass produces K+1 tokens | Step overhead = K/n\_layers × base cost; no draft model required |

### Speculative decoding vs MTP

Both produce multiple tokens per decode step but differ in cost structure:

| | Speculative decoding | Multi-token prediction |
|---|---|---|
| Draft source | Separate smaller model | K extra heads on main model |
| Draft cost per step | K × full draft-model forward pass | K/n\_layers × main-model forward pass |
| Acceptance rate γ | 0.60–0.80 (domain-match dependent) | 0.85–0.95 (joint training) |
| Extra GPU memory | Draft model weights | Negligible (K small heads) |
| Requires draft model | Yes | No |
| Best for | Models with a well-matched small variant | Models trained with MTP (e.g. DeepSeek V3) |

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
| 8 | ✅ | Preemption (recompute), speculative decoding, multi-token prediction |
| 9 | ✅ | FP8 model presets (`llama-8b-fp8`, `llama-70b-fp8`), FP8 TFLOPS dispatch, `--validate-kernels` benchmark harness |
