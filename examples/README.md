# Examples

Runnable demos, each isolated to a single feature. Run any one directly:

```bash
bash examples/02_chunked_prefill_under_load.sh
```

| # | Script | Feature |
|---|--------|---------|
| 1 | `01_continuous_batch.sh` | Continuous batching (Orca) — light-load baseline |
| 2 | `02_chunked_prefill_under_load.sh` | Chunked prefill (Sarathi) — saturating load, bounded TTFT |
| 3 | `03_tensor_parallelism.sh` | TP=4 on 70 B model with NVLink all-reduce |
| 4 | `04_expert_parallelism_moe.sh` | Expert parallelism on Mixtral 8×7B (EP=4) |
| 5 | `05_disaggregated_prefill_decode.sh` | Disaggregated P/D with network KV transfer |
| 6 | `06_speculative_decoding.sh` | Spec decoding K=3, γ=0.75 (small draft model) |
| 7 | `07_multi_token_prediction.sh` | MTP heads K=3, γ=0.9 (DeepSeek V3-style) |
| 8 | `08_trace_replay.sh` | Mooncake long-context trace replay |
| 9 | `09_arrival_rate_sweep.sh` | Parallel arrival-rate sweep → CSV |
| 10 | `10_b200_blackwell.sh` | B200 Blackwell — 2-3× faster than H100 on FP8 |
| 11 | `11_amd_mi300x.sh` | AMD MI300X — 70B model fits on a single GPU (192 GB HBM) |
| 12 | `12_frontier_kimi_k2.sh` | Kimi K2 (1 T MoE) — trillion-param frontier on 8×B200 |
| 13 | `13_google_tpu_v8i.sh` | Google TPU v8i (2026 projected) — Kimi K2 on 3D-torus pod |
| 14 | `14_groq_lpu.sh` | Groq LPU v1 — SRAM-only, 64 chips for llama-8b-fp8 |
