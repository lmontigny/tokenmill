# Example results

Simulated results for various model/GPU/scheduler configurations (60 s runs, log-normal prompts μ=512 tokens, outputs μ=128 tokens). The **Energy/tok** column is from the 3-state power model in [`docs/power.md`](power.md).

| # | Model | GPU / TP | Scheduler | Rate (req/s) | Thrpt (rps) | TTFT p50 | TTFT p95 | TPOT p50 | Energy/tok | Notes |
|---|-------|----------|-----------|:---:|:---:|---:|---:|---:|---:|---|
| 1 | llama-8b-fp8 | H100 TP=1 | continuous-batch | 10 | 9.9 | 13 ms | 23 ms | 3.2 ms | 364 mJ | Light-load baseline (idle power dominates) |
| 2 | llama-8b-fp8 | H100 TP=1 | continuous-batch | 50 | 27.3 | **13 s** | 27 s | 3.6 ms | 140 mJ | Saturated — TTFT collapses as prefill starves |
| 3 | llama-8b-fp8 | H100 TP=1 | chunked-prefill | 50 | **50.4** | **20 ms** | 41 ms | 4.7 ms | **78 mJ** | Chunked-prefill absorbs full load; best energy/token in the table |
| 4 | llama-8b-fp8 | H100 TP=1 | chunked-prefill + spec K=3 γ=0.75 | 30 | 30.4 | 17 ms | 31 ms | 3.8 ms | 124 mJ | Spec decoding — same throughput, same energy/tok as baseline (row 5) at this load |
| 5 | llama-8b-fp8 | H100 TP=1 | chunked-prefill (baseline) | 30 | 30.5 | 20 ms | 36 ms | 1.7 ms | 123 mJ | Baseline for row 4 comparison |
| 6 | llama-8b-fp8 | H100 TP=2 disagg | continuous-batch | 50 | 50.6 | 70 ms | 478 ms | **1.7 ms** | 192 mJ | Disaggregated P/D: decode isolated; 2× chips ⇒ ~2× energy/tok |
| 7 | llama-70b (bf16) | H100 TP=4 | continuous-batch | 5 | 5.1 | 410 ms | **2286 ms** | 16.4 ms | 1.80 J | 70B BF16: p95 TTFT blows out under CB |
| 8 | llama-70b (bf16) | H100 TP=4 | chunked-prefill | 5 | 5.1 | 77 ms | 129 ms | 16.4 ms | 1.80 J | Chunked-prefill trims p95 TTFT 18× vs row 7; same energy |
| 9 | llama-70b-fp8 | H100 TP=4 | chunked-prefill | 5 | 5.2 | **34 ms** | 63 ms | 8.1 ms | 1.77 J | FP8: 2.3× faster than BF16 (row 8); near-identical energy/tok |
| 10 | mixtral-8x7b | H100 EP=4 | chunked-prefill | 8 | 7.8 | 28 ms | 51 ms | 6.6 ms | 465 mJ | 47 B MoE — expert weights sharded across 4 GPUs |
| 11 | deepseek-v3 | H100 TP=8 EP=8 | chunked-prefill | 3 | 3.2 | 14 ms | 25 ms | 3.6 ms | 4.88 J | 671 B MoE / MLA KV — 9 active experts, but 8 chips × 60 s of mostly-idle ≫ active energy |
| 12 | llama-8b-fp8 | **B200** TP=1 | chunked-prefill | 50 | 50.8 | **7 ms** | 13 ms | **1.6 ms** | 104 mJ | Blackwell: ~3× faster than H100 (row 3), ~1.3× more energy/tok (higher TDP) |
| 13 | llama-70b-fp8 | **B200** TP=4 | chunked-prefill | 5 | 5.2 | **16 ms** | 28 ms | **4.0 ms** | 2.47 J | Blackwell on 70B: 2× faster, ~1.4× higher energy than H100 (row 9) |
| 14 | deepseek-v3 | **B200** TP=8 EP=8 | chunked-prefill | 5 | 5.2 | **10 ms** | 17 ms | **2.6 ms** | 4.48 J | 671 B MoE on Blackwell — near-1ms TPOT |
| 15 | llama-70b-fp8 | **MI300X** TP=**1** | chunked-prefill | 5 | 5.0 | 115 ms | 224 ms | 23.3 ms | **791 mJ** | 70 B fits on **one** MI300X (192 GB HBM3); 4× fewer chips ⇒ best energy/tok on a 70B model |
| 16 | llama-8b-fp8 | **MI300X** TP=1 | chunked-prefill | 50 | 50.7 | 14 ms | 29 ms | 3.0 ms | 82 mJ | MI300X vs H100 row 3: 30% lower TTFT, 5% better energy/tok |
| 17 | llama-70b-fp8 | **MI355X** TP=4 | chunked-prefill | 5 | 5.2 | 17 ms | 29 ms | 4.1 ms | 3.46 J | MI355X (CDNA 4) ties B200 on latency, 40% higher energy (1400 W TDP) |
| 18 | deepseek-v3 | **MI355X** TP=8 EP=8 | chunked-prefill | 5 | 5.2 | 10 ms | 18 ms | 2.7 ms | 6.27 J | MI355X matches B200 on TPOT, draws 1.4× more energy/tok |
| 19 | **kimi-k2** (1 T) | B200 TP=8 EP=8 | chunked-prefill | 2 | 2.1 | 9 ms | 16 ms | **2.5 ms** | 10.4 J | 1.026 T MoE / 32 B active — 8 chips for 2 rps ⇒ very low utilisation, high J/tok |
| 20 | **kimi-k2** (1 T) | MI300X TP=8 EP=8 | chunked-prefill | 1 | 1.0 | 12 ms | 21 ms | 3.2 ms | 16.0 J | 1 T model fits in 8×192 GB MI300X — but at 1 rps utilisation is even lower |
| 21 | **kimi-k2** (1 T) | MI355X TP=8 EP=8 | chunked-prefill | 2 | 2.1 | 9 ms | 17 ms | 2.5 ms | 14.5 J | MI355X ties B200 on TPOT but draws 1.4× more energy/tok at this load |
| 22 | **llama4-behemoth** (2 T) | B200 TP=16 EP=16 | chunked-prefill | 1 | 1.0 | 28 ms | 46 ms | 8.0 ms | 43.2 J | 2 T / 288 B active on 16 chips — active params + chip count dominate energy |
| 23 | llama-70b-fp8 | **TPU v7 Ironwood** TP=4 | chunked-prefill | 5 | 5.2 | 15 ms | 27 ms | 3.8 ms | 1.23 J | TPU Ironwood matches MI355X on TPOT, **31% less energy/tok** (500 W vs 1400 W TDP) |
| 24 | kimi-k2 (1 T) | **TPU 8i** TP=8 EP=8 | chunked-prefill | 2 | 2.1 | **3 ms** | 6 ms | **0.9 ms** | **5.99 J** | TPU 8i (288 GB HBM, 8.6 TB/s, Boardfly + CAE) on 1 T MoE — **0.6× the energy of B200** (row 19) |
| 25 | deepseek-v3 | **TPU 8i** TP=8 EP=8 | chunked-prefill | 5 | 5.2 | **4 ms** | 7 ms | **1.0 ms** | **2.60 J** | TPU 8i on 671 B MoE — best energy/tok of any accelerator for this workload |
| 26 | llama-70b-fp8 | **TPU 8i** TP=8 | chunked-prefill | 3 | 3.2 | **6 ms** | 11 ms | **1.7 ms** | 4.06 J | Standard GQA at small batch: TPU 8i beats B200 by ~15% TPOT via 384 MB Vmem |
| 27 | llama-8b-fp8 | **Groq LPU v1** TP=**64** | chunked-prefill | 30 | 30.5 | 4 ms | 8 ms | 0.9 ms ★ | 1.24 J | 8 GB model on 64 chips: very low TPOT, but 64 chips × idle baseline ⇒ high energy/tok |
| 28 | llama-70b-fp8 | **Groq LPU v1** TP=**358** | chunked-prefill | 5 | 5.1 | 58 ms | 88 ms | 12.7 ms ★ | 40.0 J | 70 GB on 358 chips — per-hop link latency now captured; energy dominated by idle GPUs |

★ Groq numbers are **upper-bound optimistic** for TPOT (real-world serving is reported at ~2–4 ms on llama-70b), but `scale_up_latency` now captures the per-hop chip-mesh α component that previously made the simulator unrealistically fast. Calibrate via `data/kernel_table.csv` for production accuracy.

## Key patterns

### Latency / throughput
- **Rows 2 vs 3**: under saturation, `chunked-prefill` keeps TTFT at ~20 ms where `continuous-batch` lets it spike to 13 s.
- **Rows 7 vs 8 vs 9**: for a 70B model, chunked-prefill cuts p95 TTFT 18×; switching from BF16 to FP8 cuts it another 2× and TPOT in half.
- **Rows 12–14**: B200 (Blackwell) delivers ~2–3× lower latency than H100 across the board — 2× FP8 TFLOPS and 2.4× HBM BW.
- **Rows 15–16**: MI300X's 192 GB HBM3 lets 70B-fp8 run on a single GPU (vs H100 needing TP=4). At iso-config (row 16 vs row 3), MI300X beats H100 by ~30% on latency.
- **Rows 17–18**: MI355X (CDNA 4) trades blows with B200 within a millisecond on both 70B-fp8 and 671B-MoE.
- **Rows 19–21**: Sparse MoE makes the trillion-param frontier tractable. Kimi K2 (1 T total / 32 B active) serves at 0.9 ms TPOT on TPU 8i — active params, not total, set the decode wall.
- **Rows 23–26**: TPU v7 Ironwood matches H100-class FP8 on equivalent workloads. TPU 8i (288 GB / 8.6 TB/s HBM + 384 MB Vmem + Boardfly) lands within 0.1 ms of B200 on MoE workloads where MLA already makes KV tiny, but pulls ahead on standard GQA at small batch.

### Energy / sustainability
- **Best energy/token in the table**: **78 mJ** for llama-8b-fp8 on H100 chunked-prefill at full saturation (row 3) — high utilisation amortises the idle baseline.
- **Idle power dominates at low load**: row 1 (10 rps) uses **4.7×** the energy per token of row 3 (50 rps) on the same chip — chunked-prefill at saturation extracts the maximum work from each Joule.
- **MI300X wins energy/token on 70B**: row 15 shows **791 mJ/tok** for llama-70b-fp8 on a *single* MI300X (no TP) — 2.2× better than B200 TP=4 (row 13) because 4 chips × idle baseline ≫ 1 chip's full TDP.
- **TPU 8i wins energy/token on MoE**: row 25 reports **2.60 J/tok** for DeepSeek V3, **0.58× the energy** of B200 (row 14). Same TPOT, much lower power: 600 W TDP × the work done per chip-second on this MoE workload.
- **Disaggregation is an energy tax**: row 6 vs row 3 — disaggregated P/D doubles chip count and roughly doubles energy/tok in exchange for stable decode latency.
- **MI355X latency-energy trade-off**: rows 17–18 show MI355X matching B200 on TPOT but drawing ~1.4× more energy (1400 W vs 1000 W TDP).
- **Large frontier clusters are energy-expensive at low rate**: rows 19–22 — Kimi K2 and Behemoth on 8–16 chips at 1–2 rps spend most of the run idle, giving 10–43 J/tok. Pushing those clusters to higher rps would amortise the idle baseline.
