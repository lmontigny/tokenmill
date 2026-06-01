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
