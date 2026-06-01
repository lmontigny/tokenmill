# Example results

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
| 15 | llama-70b-fp8 | **MI300X** TP=**1** | chunked-prefill | 5 | 5.0 | 114 ms | 224 ms | 23.3 ms | 70 B fits on **one** AMD GPU (192 GB HBM3) — no TP needed |
| 16 | llama-8b-fp8 | **MI300X** TP=1 | chunked-prefill | 50 | 50.7 | **14 ms** | 29 ms | **3.0 ms** | MI300X vs H100 row 3: 33% lower TTFT, 33% lower TPOT |
| 17 | llama-70b-fp8 | **MI355X** TP=4 | chunked-prefill | 5 | 5.2 | 13 ms | 23 ms | 3.1 ms | MI355X (CDNA 4) essentially ties B200 on 70B FP8 (row 13) |
| 18 | deepseek-v3 | **MI355X** TP=8 EP=8 | chunked-prefill | 5 | 5.2 | 4 ms | 7 ms | **0.8 ms** | MI355X matches B200 (row 14) on 671B MoE — sub-ms TPOT |

Key patterns:
- **Rows 2 vs 3**: under saturation, `chunked-prefill` keeps TTFT at ~21 ms where `continuous-batch` lets it spike to 2.4 s.
- **Rows 7 vs 8 vs 9**: for a 70B model, chunked-prefill cuts p95 TTFT 13×; switching from BF16 to FP8 cuts it another 2.4×.
- **Row 4**: speculative decoding (`--spec-tokens 3`) reduces TPOT by 24% at the same throughput.
- **Row 11**: DeepSeek V3 (671 B) on 8×H100 with EP=8 serves at 1.7 ms TPOT — MLA KV compression keeps the KV footprint tiny.
- **Rows 12–14**: B200 (Blackwell) delivers ~2.3–3× speedup over H100 across the board — 2× FP8 TFLOPS and 2.4× HBM BW (8 TB/s vs 3.35 TB/s).
- **Rows 15–16**: MI300X's 192 GB HBM3 lets 70B-fp8 run on a single GPU (vs H100 needing TP=4). For memory-bound decode, MI300X beats H100 by ~33% at iso-config (row 16 vs row 3) — 5.3 TB/s vs 3.35 TB/s HBM, even after the 10% MFU haircut for ROCm maturity.
- **Rows 17–18**: MI355X (CDNA 4) trades blows with B200 within a millisecond on both 70B-fp8 and 671B-MoE workloads.
