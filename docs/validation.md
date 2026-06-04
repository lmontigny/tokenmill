# Benchmark validation

Run the roofline model against reference kernel latencies:

```bash
./target/release/tokenmill --validate-kernels data/reference_kernels.csv
```

For Cerebras-specific online validation, see
[`cerebras-validation.md`](cerebras-validation.md). The CS-3 preset currently
matches public hardware specs but intentionally behaves as a roofline ceiling,
not as a calibrated Cerebras Inference API predictor.

Results against `data/reference_kernels.csv` (GPU kernel time only; serving frameworks add 3–10 ms overhead):

| GPU | Model | Op | MAPE | Notes |
|-----|-------|----|------|-------|
| H100-SXM5 | llama-8b (bf16) | prefill | <1% | vLLM TTFT benchmark |
| H100-SXM5 | llama-8b (bf16) | decode | <2% | vLLM throughput benchmark |
| H100-SXM5 | llama-8b-fp8 | prefill | ~8% | NVIDIA TRT-LLM FP8, TP=1 |
| H100-SXM5 | llama-8b-fp8 | decode | ~13% | TRT-LLM FP8; degrades at large batch×seq |
| H100-SXM5 | llama-70b (bf16) | all | <1% | MLPerf v4.0 / vLLM TP=8 |
| A100-80GB | llama-8b (bf16) | all | <1% | vLLM benchmark |
| H200-SXM5 | — | — | _spec-backed, not kernel-validated_ | Same Hopper tensor core peak as H100; decode uses larger/faster HBM3e |
| B200-SXM / Rubin-SXM | — | — | _spec-backed, partially validated_ | B200 FP4/NVFP4 comparison lives in [`quantization-validation.md`](quantization-validation.md); Rubin is public-spec only |
| MI300X / MI325X / MI355X | — | — | _not yet validated_ | Add rows to `reference_kernels.csv` if you have ROCm/vLLM profiling data |
| TPU / Groq / Cerebras | — | — | _not kernel-validated_ | Presets are public-spec roofline models; production calibration needs vendor-specific traces or kernel tables |
| H200-SXM5 / MI300X | DeepSeek-V3 | decode | _not validated_ | Public SGLang/vLLM benchmarks report tens to hundreds of ms TPOT; current roofline predicts ~3 ms and is too optimistic for frontier MoE serving |

## DeepSeek-V3 online validation

Public DeepSeek-V3 serving results show that the current generic roofline model
is **not calibrated** for large MoE serving latency:

| Source | Hardware / setup | Public result | Local comparable scratch run |
|---|---|---|---|
| [Verda H200 SGLang FP8](https://verda.com/blog/deepseek-v3-llm-nvidia-h200-gpu-inference-benchmarking) | 8x H200, TP=8, FP8, 1024 input / 1024 output, RPS 1 | median TTFT 563 ms, median TPOT 144 ms | H200 TP=8, 1024/1024, RPS 1: p50 TTFT 20.6 ms, p50 TPOT 3.0 ms |
| [Moreh MI300X vLLM](https://moreh.io/assets/moreh-vllm-performance-evaluation-deepseek-v3-r1-671b-on-amd-instinct-mi300x-gpus-250829.pdf) | 8x MI300X, TP=8, 1024/1024, concurrency 1 | mean TTFT 68.9 ms, mean TPOT 14.7 ms | MI300X TP=8, 1024/1024, RPS 1: p50 TTFT 21.4 ms, p50 TPOT 3.3 ms |
| [GPUStack DeepSeek-V3.2 H200](https://docs.gpustack.ai/2.0/performance-lab/deepseek-v3.2/h200/) | 8x H200, TP=8, optimized SGLang variants | median TPOT ranges from ~85 ms to hundreds of ms depending runtime flags | current roofline remains single-digit ms |

Treat DeepSeek-V3, Kimi K2, and other frontier-MoE dashboard rows as
**optimistic roofline estimates** for relative hardware direction only. Absolute
TTFT/TPOT needs a kernel table or model-specific calibration for MoE routing,
expert all-to-all, framework scheduler overhead, CUDA/ROCm graph behavior,
DeepGEMM/FlashMLA kernels, and batching policy.

## Key findings

- **Prefill** (compute-bound): ~8% MAPE on FP8. `flops_fp8` and `flops_fp4` are selected from `weight_bits`; FP4 paths should be calibrated with kernel tables before production use.
- **Decode** (memory-BW bound): 5–20% MAPE. Error grows at large batch × seq_len where paged KV access is ~60–70% efficient vs sequential weight reads (80% mfu assumed for both).
- **Serving TPOT vs kernel time**: framework overhead (Python scheduler, CUDA launch, NCCL) adds latency per step and is **not** modeled. For small dense models this can be a few milliseconds; for frontier MoE models public serving benchmarks show much larger gaps without model-specific calibration.
- **AMD MI series**: the MFU constants (`mfu_prefill=0.65`, `mfu_decode=0.72`) are set ~10% below NVIDIA equivalents as a placeholder for the ROCm/vLLM vs CUDA kernel-maturity gap. Real-world MI300X has been reported anywhere from 0.45 (early ROCm) to 0.75 (recent vLLM ROCm) for decode — calibrate with `--validate-kernels` against your own measurements before drawing conclusions on AMD numbers.

## Calibrating to your hardware

Profile a real GPU with `nsys` or your serving framework's instrumentation and add rows to
`data/kernel_table.csv` to override the roofline for those (gpu, model, op, batch, seq) points:

```
gpu,model,op,batch_size,seq_len,latency_ms
H100-SXM5,llama-8b-fp8,decode,32,1024,4.2
```

The simulator falls back to the roofline only when there is no matching row.
