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
