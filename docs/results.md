# Example results

Simulated results for various model/GPU/scheduler configurations (60 s runs, log-normal prompts μ=512 tokens, outputs μ=128 tokens). The **Energy/tok** column comes from the per-chip TDP model in [`docs/power.md`](power.md); the **$ / Mtok** column is computed from on-demand list prices in [`docs/cost.md`](cost.md).

This Markdown table is a fixed snapshot. For current curated comparisons across
system, GPU, model, precision, cost, latency, and energy, use the GitHub Pages
dashboard generated from `reports/manifest.json` and `reports/curated/*.json`.

| # | Model | GPU / TP | Scheduler | Rate (req/s) | Thrpt (rps) | TTFT p50 | TTFT p95 | TPOT p50 | Energy/tok | $ / Mtok | Notes |
|---|-------|----------|-----------|:---:|:---:|---:|---:|---:|---:|---:|---|
| 1 | llama-8b-fp8 | H100 TP=1 | continuous-batch | 10 | 9.9 | 13 ms | 23 ms | 3.2 ms | 364 mJ | $0.77 | Light-load baseline (idle power dominates → high $/Mtok) |
| 2 | llama-8b-fp8 | H100 TP=1 | continuous-batch | 50 | 27.3 | **13 s** | 27 s | 3.6 ms | 140 mJ | $0.28 | Saturated — TTFT collapses but throughput is high |
| 3 | llama-8b-fp8 | H100 TP=1 | chunked-prefill | 50 | **50.4** | **20 ms** | 41 ms | 4.7 ms | **78 mJ** | **$0.15** | Best in table — saturated chunked-prefill amortises every cost |
| 4 | llama-8b-fp8 | H100 TP=1 | chunked-prefill + spec K=3 γ=0.75 | 30 | 30.4 | 17 ms | 31 ms | 3.8 ms | 124 mJ | $0.25 | Spec decoding — same throughput & cost as baseline at this load |
| 5 | llama-8b-fp8 | H100 TP=1 | chunked-prefill (baseline) | 30 | 30.5 | 20 ms | 36 ms | 1.7 ms | 123 mJ | $0.25 | Baseline for row 4 comparison |
| 6 | llama-8b-fp8 | H100 TP=2 disagg | continuous-batch | 50 | 50.6 | 70 ms | 478 ms | **1.7 ms** | 192 mJ | $0.60 | Disagg P/D: 2× pools × 2 chips = 4 chips ⇒ ~2× $/Mtok |
| 7 | llama-70b (bf16) | H100 TP=4 | continuous-batch | 5 | 5.1 | 410 ms | **2286 ms** | 16.4 ms | 1.80 J | $5.76 | 70B BF16: p95 TTFT blows out under CB |
| 8 | llama-70b (bf16) | H100 TP=4 | chunked-prefill | 5 | 5.1 | 77 ms | 129 ms | 16.4 ms | 1.80 J | $5.76 | Chunked-prefill trims p95 TTFT 18× vs row 7; same cost |
| 9 | llama-70b-fp8 | H100 TP=4 | chunked-prefill | 5 | 5.2 | **34 ms** | 63 ms | 8.1 ms | 1.77 J | $5.72 | FP8: 2.3× faster than BF16 (row 8); near-identical $/Mtok |
| 10 | mixtral-8x7b | H100 EP=4 | chunked-prefill | 8 | 7.8 | 28 ms | 51 ms | 6.6 ms | 1.19 J | $3.85 | 47 B MoE — experts on 4 GPUs ⇒ 4× idle baseline at 8 rps |
| 11 | deepseek-v3 | H100 TP=8 EP=8 | chunked-prefill | 3 | 3.2 | 14 ms | 25 ms | 3.6 ms | 4.88 J | $17.78 | 671 B MoE: 8 chips × 60 s × low rate ⇒ idle baseline dominates |
| 12 | llama-8b-fp8 | **B200** TP=1 | chunked-prefill | 50 | 50.8 | **7 ms** | 13 ms | **1.6 ms** | 104 mJ | $0.28 | Blackwell: 3× faster than H100 (row 3), 1.8× higher $/Mtok (B200 lists at $6.50/hr) |
| 13 | llama-70b-fp8 | **B200** TP=4 | chunked-prefill | 5 | 5.2 | **16 ms** | 28 ms | **4.0 ms** | 2.47 J | $10.56 | Blackwell 70B: 2× faster, ~1.8× more expensive than H100 (row 9) |
| 14 | deepseek-v3 | **B200** TP=8 EP=8 | chunked-prefill | 5 | 5.2 | **10 ms** | 17 ms | **2.6 ms** | 4.48 J | $21.12 | 671 B MoE on Blackwell — fast, but list price burns through dollars |
| 15 | llama-70b-fp8 | **MI300X** TP=**1** | chunked-prefill | 5 | 5.0 | 115 ms | 224 ms | 23.3 ms | **791 mJ** | **$1.46** | 70 B on **one** MI300X — **7× cheaper $/Mtok** than B200 TP=4 (row 13) |
| 16 | llama-8b-fp8 | **MI300X** TP=1 | chunked-prefill | 50 | 50.7 | 14 ms | 29 ms | 3.0 ms | 82 mJ | **$0.15** | MI300X vs H100 (row 3): same $/Mtok, 30% lower TTFT — sweet spot |
| 17 | llama-70b-fp8 | **MI355X** TP=4 | chunked-prefill | 5 | 5.2 | 17 ms | 29 ms | 4.1 ms | 3.46 J | $9.75 | MI355X ties B200 on TPOT but 8% cheaper $/Mtok (lower list price) |
| 18 | deepseek-v3 | **MI355X** TP=8 EP=8 | chunked-prefill | 5 | 5.2 | 10 ms | 18 ms | 2.7 ms | 6.27 J | $19.50 | MI355X DSV3 ≈ B200 (row 14) on cost; higher energy from 1400 W TDP |
| 19 | **kimi-k2** (1 T) | B200 TP=8 EP=8 | chunked-prefill | 2 | 2.1 | 9 ms | 16 ms | **2.5 ms** | 10.4 J | $50.50 | 1 T MoE: 8 × B200 at 2 rps ⇒ heavy idle baseline ⇒ steep $/Mtok |
| 20 | **kimi-k2** (1 T) | MI300X TP=8 EP=8 | chunked-prefill | 1 | 1.0 | 12 ms | 21 ms | 3.2 ms | 16.0 J | $56.88 | Same model, 8 chips × 1 rps — even worse utilisation |
| 21 | **kimi-k2** (1 T) | MI355X TP=8 EP=8 | chunked-prefill | 2 | 2.1 | 9 ms | 17 ms | 2.5 ms | 14.5 J | $46.62 | MI355X ties B200 on TPOT, **8% cheaper** $/Mtok |
| 22 | **llama4-behemoth** (2 T) | B200 TP=16 EP=16 | chunked-prefill | 1 | 1.0 | 28 ms | 46 ms | 8.0 ms | 43.2 J | **$214** | 16 chips × $6.50/hr × low rate ⇒ frontier serving is expensive |
| 23 | llama-70b-fp8 | **TPU v7 Ironwood** TP=4 | chunked-prefill | 5 | 5.2 | 15 ms | 27 ms | 3.8 ms | 1.23 J | **$6.50** | TPU Ironwood matches MI355X on TPOT, **33% cheaper** than B200 (row 13) |
| 24 | kimi-k2 (1 T) | **TPU 8i** TP=8 EP=8 | chunked-prefill | 2 | 2.1 | **3 ms** | 6 ms | **0.9 ms** | **5.99 J** | **$34.71** | TPU 8i on 1 T MoE — **0.69× the cost** of B200 (row 19) at lower TPOT |
| 25 | deepseek-v3 | **TPU 8i** TP=8 EP=8 | chunked-prefill | 5 | 5.2 | **4 ms** | 7 ms | **1.0 ms** | **2.60 J** | **$14.62** | TPU 8i on DSV3 — **best cost/perf on MoE**: 0.69× B200 (row 14) cost |
| 26 | llama-70b-fp8 | **TPU 8i** TP=8 | chunked-prefill | 3 | 3.2 | **6 ms** | 11 ms | **1.7 ms** | 4.06 J | $22.86 | Standard GQA at small batch — TPU 8i beats B200 by ~15% TPOT |
| 27 | llama-8b-fp8 | **Groq LPU v1** TP=**64** | chunked-prefill | 30 | 30.5 | 4 ms | 8 ms | 0.9 ms ★ | 1.24 J | **$1.36** | 64 chips × $0.30/hr = $19/hr cluster — cheap per chip, expensive per token |
| 28 | llama-70b-fp8 | **Groq LPU v1** TP=**358** | chunked-prefill | 5 | 5.1 | 58 ms | 88 ms | 12.7 ms ★ | 40.0 J | $44.22 | 358 chips × $0.30/hr = $107/hr cluster — frontier scale on tiny chips |
| 29 | kimi-k2 (Thinking, 4K/4K) | **HGX B200** (8× B200) | chunked-prefill | 5 | 4.1 | 29 ms | 55 ms | 3.1 ms | 195 mJ | $0.90 | NVIDIA HGX B200 reference workload for Kimi K2 Thinking (ISL=OSL=4K) |
| 30 | kimi-k2 (Thinking, 4K/4K) | **HGX Rubin NVL8** (8× Rubin) | chunked-prefill | 5 | 4.6 | **16 ms** | 26 ms | **1.9 ms** | 247 mJ | $1.29 | Rubin vs B200: 1.8× lower TTFT, 1.6× lower TPOT, 1.2× more throughput at FP8 dense ‡ |

★ Groq numbers are **upper-bound optimistic** for TPOT (real-world serving is reported at ~2–4 ms on llama-70b), but `scale_up_latency` now captures the per-hop chip-mesh α component. Calibrate via `data/kernel_table.csv` for production accuracy.

‡ NVIDIA's own slides claim **10× more token-factory throughput** for HGX Rubin NVL8 vs HGX B200 on this exact workload. The headline 10× uses **Rubin with sparse NVFP4 vs B200 with dense NVFP4**. Row 30 remains the older FP8-dense comparison; use `kimi-k2-nvfp4-sparse` to model the sparse NVFP4 path.

## Cheapest serving rates in this table

For reference, vendor list prices for llama-70B serving in 2026 run roughly: Together $0.88/Mtok, Fireworks $0.90/Mtok, Groq $0.79/Mtok, OpenAI GPT-4o $10/Mtok. Our cheapest 70B configs are an order of magnitude above the heavily-batched serving rates because the table runs at moderate batch (5 rps), not the 10× higher concurrency that production services extract.

| Rank | Config | $/Mtok | Why |
|---|---|---:|---|
| 1 | llama-8b-fp8, H100 chunked saturated (row 3) | $0.15 | Smallest model + highest utilisation |
| 1= | llama-8b-fp8, MI300X chunked saturated (row 16) | $0.15 | Cheaper chip / higher TDP cancels out |
| 3 | llama-8b-fp8, Groq TP=64 (row 27) | $1.36 | 64 cheap chips × low utilisation = expensive per token |
| 4 | llama-70b-fp8, MI300X single chip (row 15) | $1.46 | 70B on one GPU — no TP overhead, no idle chips |
| 5 | llama-70b-fp8, TPU v7 Ironwood TP=4 (row 23) | $6.50 | Cheapest 70B-TP4 — $4/hr/chip beats H100 / B200 |

## Key patterns

### Latency / throughput
- **Rows 2 vs 3**: under saturation, `chunked-prefill` keeps TTFT at ~20 ms where `continuous-batch` lets it spike to 13 s.
- **Rows 7 vs 8 vs 9**: for a 70B model, chunked-prefill cuts p95 TTFT 18×; switching from BF16 to FP8 cuts it another 2× and TPOT in half.
- **Rows 12–14**: B200 (Blackwell) delivers ~2–3× lower latency than H100 across the board — 2× FP8 TFLOPS and 2.4× HBM BW.
- **Rows 15–16**: MI300X's 192 GB HBM3 lets 70B-fp8 run on a single GPU (vs H100 needing TP=4). At iso-config (row 16 vs row 3), MI300X beats H100 by ~30% on latency at the **same** $/Mtok.
- **Rows 17–18**: MI355X (CDNA 4) trades blows with B200 within a millisecond on both 70B-fp8 and 671B-MoE.
- **Rows 19–21**: Sparse MoE makes the trillion-param frontier tractable. Kimi K2 (1 T total / 32 B active) serves at 0.9 ms TPOT on TPU 8i — active params, not total, set the decode wall.
- **Rows 23–26**: TPU v7 Ironwood matches H100-class FP8 on equivalent workloads. TPU 8i (288 GB / 8.6 TB/s HBM + 384 MB Vmem + Boardfly) lands within 0.1 ms of B200 on MoE workloads where MLA already makes KV tiny, but pulls ahead on standard GQA at small batch.

### Energy / sustainability
- **Best energy/token**: **78 mJ** for llama-8b-fp8 on H100 chunked-prefill saturated (row 3) — high utilisation amortises the idle baseline.
- **Idle dominates at low load**: row 1 (10 rps) uses **4.7×** the energy per token of row 3 (50 rps) on the same chip.
- **MI300X wins energy/token on 70B**: row 15 — 791 mJ/tok on a *single* MI300X (no TP), 2.2× better than B200 TP=4 (row 13) because 4 idle chips ≫ 1 working chip.
- **TPU 8i wins energy/token on MoE**: row 25 — 2.60 J/tok for DeepSeek V3, **0.58× B200's** energy at the same TPOT.

### Cost
- **Cheapest serving in the table**: H100 *and* MI300X tie at **$0.15/Mtok** for llama-8b-fp8 at saturation (rows 3 and 16). The cheaper chip ($3.50/hr) with higher TDP balances the more expensive chip ($3.50/hr same) with lower utilisation — both extract the most tokens per dollar at full load.
- **MI300X wins 70B cost**: row 15 — **$1.46/Mtok** on a single MI300X, **7× cheaper** than B200 TP=4 (row 13). One chip's bill < four chips' bill.
- **TPU v7 Ironwood is the value pick at TP=4**: row 23 — **$6.50/Mtok**, 33% cheaper than B200 (row 13) at similar TPOT, because GCP TPU lists at $4/chip/hr vs $6.50 for B200.
- **TPU 8i wins frontier-MoE cost**: rows 24–25 — **$14.62/Mtok on DSV3, $34.71 on Kimi K2** — ~30% cheaper than B200 at lower TPOT.
- **Groq is cheap per chip, expensive per token at low utilisation**: row 27 — 64 chips × $0.30/hr = a $19/hr cluster serving 8B at $1.36/Mtok. To match Groq's published $0.79/Mtok rate, you'd need ~2× the throughput per chip — which their actual production deployments achieve via aggressive batching not modelled here.
- **Frontier clusters at low rate are expensive**: row 22 — Behemoth at 1 rps on 16 chips = **$214/Mtok**. Pushing the rate up amortises the idle cluster cost.
