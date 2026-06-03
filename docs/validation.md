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
| MI300X / MI325X / MI355X | — | — | _not yet validated_ | Add rows to `reference_kernels.csv` if you have ROCm/vLLM profiling data |

## Key findings

- **Prefill** (compute-bound): ~8% MAPE on FP8. `flops_fp8` and `flops_fp4` are selected from `weight_bits`; FP4 paths should be calibrated with kernel tables before production use.
- **Decode** (memory-BW bound): 5–20% MAPE. Error grows at large batch × seq_len where paged KV access is ~60–70% efficient vs sequential weight reads (80% mfu assumed for both).
- **Serving TPOT vs kernel time**: framework overhead (Python scheduler, CUDA launch, NCCL) adds 3–10 ms per step and is **not** modeled. Subtract this from observed TPOT before comparing to simulator output.
- **AMD MI series**: the MFU constants (`mfu_prefill=0.65`, `mfu_decode=0.72`) are set ~10% below NVIDIA equivalents as a placeholder for the ROCm/vLLM vs CUDA kernel-maturity gap. Real-world MI300X has been reported anywhere from 0.45 (early ROCm) to 0.75 (recent vLLM ROCm) for decode — calibrate with `--validate-kernels` against your own measurements before drawing conclusions on AMD numbers.

## Calibrating to your hardware

Profile a real GPU with `nsys` or your serving framework's instrumentation and add rows to
`data/kernel_table.csv` to override the roofline for those (gpu, model, op, batch, seq) points:

```
gpu,model,op,batch_size,seq_len,latency_ms
H100-SXM5,llama-8b-fp8,decode,32,1024,4.2
```

The simulator falls back to the roofline only when there is no matching row.
