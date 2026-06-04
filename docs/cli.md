# CLI reference

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu` | `h100` | Accelerator preset: `rubin` \| `b200` \| `h100` \| `a100` \| `a10g` \| `mi355x` \| `mi325x` \| `mi300x` \| `tpu-v8i` \| `tpu-v8t` \| `tpu-v7-ironwood` \| `groq-lpu-v1` \| `cerebras-cs3` |
| `--model` | `llama-70b` | Model preset: dense, MoE, FP8, W4A16, W4A8KV4, and sparse NVFP4 variants |
| `--scheduler` | `continuous-batch` | `continuous-batch` \| `chunked-prefill` |
| `--chunk-size` | `512` | Prefill chunk tokens (chunked-prefill only) |
| `--arrival-rate` | `5.0` | Requests per second (Poisson) |
| `--duration` | `60.0` | Simulation duration in seconds |
| `--prompt-mean` | `512.0` | Mean prompt length (log-normal, tokens) |
| `--output-mean` | `128.0` | Mean output length (log-normal, tokens) |
| `--tp` | `1` | Tensor parallelism degree |
| `--pp` | `1` | Pipeline parallelism degree |
| `--ep` | `1` | Expert parallelism degree (MoE models) |
| `--gpus-per-node` | `8` | Accelerators in one scale-up node before TP/PP/EP crosses scale-out networking |
| `--scale-out-bw-gbps` | `0` | Scale-out bandwidth for cross-node TP/PP/EP collectives in GB/s; `0` keeps legacy uniform scale-up behavior |
| `--scale-out-latency-us` | `5` | One-way scale-out latency for cross-node TP/PP/EP collectives |
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
| `llama-70b-w4a16` | dense | 70 B (W4A16) | 70 B | 4-bit weights, 16-bit activations |
| `llama-8b-w4a16` | dense | 8 B (W4A16) | 8 B | 4-bit weights, 16-bit activations |
| `llama-70b-w4a8kv4` | dense | 70 B (W4A8KV4) | 70 B | 4-bit weights, 8-bit activations, 4-bit KV |
| `llama-8b-w4a8kv4` | dense | 8 B (W4A8KV4) | 8 B | 4-bit weights, 8-bit activations, 4-bit KV |
| `llama-70b-nvfp4-sparse` | dense | 70 B (sparse NVFP4) | 70 B | W4A8KV4 + 2:4 sparsity on supporting hardware |
| `mixtral-8x7b` | MoE | 46.7 B (bf16) | ~12.9 B | 8 experts top-2, all layers MoE |
| `llama4-maverick` | MoE | 400 B (fp8) | 17 B | 128 experts top-1+1 shared, 36/48 MoE layers |
| `deepseek-v3` | MoE | 671 B (fp8) | 37 B | 256 experts top-8+1 shared, MLA KV compression |
| `kimi-k2` | MoE | **1.026 T** (fp8) | 32 B | 384 experts top-8+1 shared, MLA KV (Moonshot AI, Jul 2025) |
| `kimi-k2-nvfp4-sparse` | MoE | **1.026 T** (sparse NVFP4) | 32 B | Kimi K2 with W4A8KV4 + 2:4 sparsity |
| `llama4-behemoth` | MoE | **2 T** (fp8) | 288 B | 16 experts top-1+1 shared (Meta, announced 2025; specs approximate) |

DeepSeek V3 and Kimi K2 both use **Multi-head Latent Attention (MLA)** which compresses the KV cache to a
512-dimensional latent vector per layer — roughly 64× smaller than standard MHA — enabling long-context
serving at scale. Llama 4 Behemoth uses standard GQA (n_kv_heads=8).

## GPU presets

| Preset | Family | BF16 TFLOPS | FP8 TFLOPS | FP4 TFLOPS | Mem | Mem BW | Scale-up BW | TDP |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `rubin` | NVIDIA Rubin (2026) | 8750 | 17500 | 50000 sparse | 288 GB | 22 TB/s | 3.6 TB/s (NVLink 6) | 1500 W † |
| `b200` | NVIDIA Blackwell | 2250 | 4500 | 9000 dense | 192 GB | 8 TB/s | 1.8 TB/s (NVLink 5) | 1000 W |
| `h100` | NVIDIA Hopper | 989 | 1978 | — | 80 GB | 3.35 TB/s | 900 GB/s (NVLink 4) | 700 W |
| `a100` | NVIDIA Ampere | 312 | — | — | 80 GB | 2 TB/s | 600 GB/s (NVLink 3) | 400 W |
| `a10g` | NVIDIA Ampere | 125 | — | — | 24 GB | 600 GB/s | — | 300 W |
| `mi355x` | AMD CDNA 4 | 2500 | 5000 | — | 288 GB | 8 TB/s | 1.075 TB/s (IF Gen 4) | 1400 W |
| `mi325x` | AMD CDNA 3 refresh | 1307 | 2614 | — | 256 GB | 6 TB/s | 896 GB/s (Infinity Fabric) | 1000 W |
| `mi300x` | AMD CDNA 3 | 1307 | 2614 | — | 192 GB | 5.3 TB/s | 896 GB/s (Infinity Fabric) | 750 W |
| `tpu-v8i` | Google TPU 8i (2026, serving) | 2525 ‡ | 5050 ‡ | 10100 | 288 GB | 8.6 TB/s | 2.4 TB/s (ICI, **Boardfly**) — **384 MB on-chip SRAM** | 600 W † |
| `tpu-v8t` | Google TPU 8t (2026, training) | 3150 ‡ | 6300 ‡ | 12600 | 216 GB | 6.5 TB/s | 2.4 TB/s (ICI, 3D-torus) — 128 MB SRAM | 750 W † |
| `tpu-v7-ironwood` | Google TPU v7 (Apr 2025) | 2304 | 4614 | — | 192 GB | 7.37 TB/s | 1.2 TB/s (ICI, 3D-torus) — ~256 MB SRAM | 500 W † |
| `groq-lpu-v1` | Groq LPU v1 (GroqChip 1) | 188 | 375 | — | **230 MB** § | 80 TB/s § | ~400 GB/s (C2C) — SRAM-only | 215 W |
| `cerebras-cs3` | Cerebras CS-3 / WSE-3 | 125000 | — | — | **44 GB** ¶ | 21 PB/s ¶ | 150 GB/s (system I/O) | 23000 W † |

† Google doesn't publish per-chip TPU TDP; values are estimates from the "2× perf/W vs Ironwood" claim.

§ Groq has **no off-chip HBM**. The "HBM" columns carry the on-chip SRAM specs (230 MB / 80 TB/s). Any non-trivial model needs very high `--tp` to fit (e.g. llama-70b-fp8 needs TP ≈ 358 chips). Deterministic compiler-scheduled execution yields very high MFU but per-hop link latency in the chip mesh is not modelled — the simulator may under-estimate real-world TPOT at large TP.

¶ Cerebras CS-3 uses WSE-3 wafer-scale SRAM as its main memory: 44 GB at 21 PB/s. Multi-system TP uses the external system boundary (1.2 Tb/s = 150 GB/s), not the 214 Pb/s internal wafer fabric. [Cerebras](https://www.cerebras.ai/blog/cerebras-cs3), [Neocortex CS-3 specs](https://portal.neocortex.psc.edu/docs/CS-3/system-specifications.html).

‡ TPU 8i / 8t BF16 and FP8 figures are **derived** from the published FP4 PFLOPs (10.1 and 12.6 respectively) using the standard 2× per-precision ratio. Google publishes only FP4. [Source.](https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive)

The `scale_up_bandwidth` field is treated as a generic **scale-up fabric** bandwidth — Infinity Fabric (AMD) and ICI (TPU) reuse the same all-reduce / all-to-all formulas.

**On-chip SRAM**: each preset carries an `on_chip_sram` value (L2 cache on NVIDIA/AMD, Vmem scratchpad on TPU). When the per-chip KV working set fits in this budget, decode KV traffic is served from SRAM at ~10× HBM cost. This is the main reason TPU 8i (384 MB Vmem) outperforms B200 (≈100 MB L2) on standard MHA/GQA models at small-to-moderate batch even though their HBM bandwidths are similar.

**TPU topology caveat**: TPU 8t uses a 3D-torus ICI; TPU 8i introduces **Boardfly** — a Dragonfly-inspired hierarchical fabric (4-chip building blocks → 8-board copper-connected groups of 32 chips → 36 groups linked via Optical Circuit Switches, up to 1024 chips/pod, 7-hop diameter). The simulator's ring-allreduce formula is accurate for TP ≤ 32 (within a Boardfly group); for larger TP that crosses the OCS layer it under-estimates collective cost by ~10–20%. The on-chip CAE (Collectives Acceleration Engine) on 8i further accelerates all-reduce / all-to-all but is not separately modelled. Within-pod only — no DCN.
