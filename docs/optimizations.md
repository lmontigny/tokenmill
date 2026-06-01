# Supported optimizations

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

## Speculative decoding vs MTP

Both produce multiple tokens per decode step but differ in cost structure:

| | Speculative decoding | Multi-token prediction |
|---|---|---|
| Draft source | Separate smaller model | K extra heads on main model |
| Draft cost per step | K × full draft-model forward pass | K/n\_layers × main-model forward pass |
| Acceptance rate γ | 0.60–0.80 (domain-match dependent) | 0.85–0.95 (joint training) |
| Extra GPU memory | Draft model weights | Negligible (K small heads) |
| Requires draft model | Yes | No |
| Best for | Models with a well-matched small variant | Models trained with MTP (e.g. DeepSeek V3) |
