# Roadmap

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
| 10 | ✅ | B200 Blackwell GPU preset, public trace fetcher (Azure / BurstGPT / Mooncake), library exports + tests, CI |
| 11 | ✅ | AMD Instinct presets — MI300X (CDNA 3, 192 GB HBM3), MI325X (CDNA 3 refresh, 256 GB HBM3e), MI355X (CDNA 4, B200 competitor). Infinity Fabric reuses the scale-up fabric formulas. |
| 12 | ✅ | Frontier-class model presets — Kimi K2 (1.026 T MoE, 384 experts, MLA KV) and Llama 4 Behemoth (2 T MoE, 16 huge experts). Demonstrates that the simulator handles trillion-param clusters. |
| 13 | ✅ | Google TPU presets — v7 Ironwood (2025), v8t (training, 3D torus), v8i (serving, Boardfly). Official 8t/8i specs from Google Cloud blog; FP8/BF16 derived from FP4. |
| 14 | ✅ | On-chip SRAM modelling — `on_chip_sram` field on `GpuSpec`. When per-chip KV fits in SRAM, decode KV traffic is served at 1/10 the HBM cost. Captures the TPU 8i 384 MB Vmem advantage on low-latency serving. |
| 15 | ✅ | Split presets by vendor — TPU into `src/hardware/tpu.rs`, Groq LPU v1 into `src/hardware/groq.rs`. Groq has no off-chip HBM; the "HBM" fields carry the 230 MB / 80 TB/s on-chip SRAM, so any non-trivial model needs very high `--tp`. Deterministic-flow per-hop latency is not modelled (results are upper-bound optimistic at large TP). |
| 16 | ✅ | Vendor-neutral memory naming (`memory_bandwidth` / `memory_capacity`) and per-hop link latency (`scale_up_latency`) on every preset — fixes Groq's artificial speed at high TP and works for any SRAM-only architecture. |
| 17 | ✅ | Power and energy reporting — `src/hardware/power.rs` with a 3-state per-chip model (prefill 0.90 / decode 0.65 / idle 0.35 × TDP). Each preset carries `tdp_watts`; the sim tracks per-GPU prefill/decode busy time and emits `total_energy_kj`, `mean_power_kw`, `energy_per_token_mj`, `energy_per_request_j` in text / JSON / CSV output. See [`docs/power.md`](power.md). |
| 18 | ✅ | Cost reporting — `src/hardware/cost.rs` computes cluster $/hour, $/Mtok, $/request from `cost_per_hour_usd` (on-demand list price per chip-hour) on every preset. Same units cloud vendors and inference APIs publish; see [`docs/cost.md`](cost.md). |
| 19 | ✅ | `docs/topology.md` — consolidates the single-rack / single-pod scope assumption that was previously scattered across comments. Documents what scale-up fabric is modelled per vendor (NVLink, Infinity Fabric, ICI, Boardfly, Groq C2C) and the three cases where the assumption matters (Boardfly OCS-spanning TP, Groq very-high TP, multi-rack clusters). README opener rewritten to lead with the value prop. |
| 20 | ✅ | NVIDIA Rubin preset (2026/2027 generation) — 288 GB HBM4 at 22 TB/s, 17.5 PFLOPS FP8 dense, NVLink 6 at 3.6 TB/s per chip. Per-chip specs from nvidia.com/en-us/data-center/hgx/. `--gpu rubin --tp 8` models one HGX Rubin NVL8 server. NVIDIA's headline "10× HGX B200" uses sparse NVFP4 (not yet modelled); our FP8-dense comparison shows the ~1.6–2.5× raw HBM-BW + FLOPS scaling. |
