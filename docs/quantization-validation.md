# Mixed-precision validation

This page compares the mixed-precision simulator path against public NVIDIA
Blackwell / TensorRT-LLM results. The closest public table is TensorRT-LLM's
performance overview, which reports **output tokens per second per GPU** from
`trtllm-bench`.

## Sources checked

| Source | Relevant public number |
|---|---:|
| [TensorRT-LLM performance overview](https://nvidia.github.io/TensorRT-LLM/latest/developer-guide/perf-overview.html) | Llama 3.3 70B FP4 on B200: 6,920 tok/s/GPU at 1K/1K; 1,362 at 8K/1K; 274 at 32K/1K |
| [NVIDIA InferenceMAX v1 blog](https://developer.nvidia.com/blog/?p=106975) | B200 reaches about 10,000 tok/s/GPU at 50 TPS/user on Llama 3.3 70B 1K/1K |
| [NVIDIA NVFP4 blog](https://developer.nvidia.com/blog/?p=102000) | B200 low-precision peak: 9 PFLOPS dense / 18 PFLOPS sparse FP4 |
| [NVIDIA NVFP4 acceleration blog](https://developer.nvidia.com/blog/3-ways-nvfp4-accelerates-ai-training-and-inference/) | Blackwell Ultra dense NVFP4 is 3x FP8; Rubin reaches 50 PFLOPS NVFP4 inference |

## B200 FP4 Llama 70B comparison

Simulator command shape:

```bash
cargo run --release -- \
  --model llama-70b-w4a8kv4 --gpu b200 \
  --scheduler chunked-prefill \
  --max-batch-tokens <budget> \
  --prompt-mean <ISL> --output-mean <OSL> \
  --arrival-rate <rate> --duration <seconds> \
  --output json
```

| Workload | NVIDIA TRT-LLM | Simulator | Error | Notes |
|---|---:|---:|---:|---|
| 1K/1K | 6,920 tok/s/GPU | 6,958 tok/s/GPU | +0.5% | Good match under high batching |
| 8K/1K | 1,362 tok/s/GPU | 940 tok/s/GPU | -31.0% | Conservative; synthetic arrivals leave more prefill/queue overhead than `trtllm-bench` steady state |
| 32K/1K | 274 tok/s/GPU | 259 tok/s/GPU | -5.5% | Good match when arrival rate avoids queue collapse |

The 1K/1K match validates the core FP4 decode path: `weight_bits = 4`,
`kv_bits = 4`, B200 `flops_fp4`, and memory traffic are in the right regime.
The 32K/1K case is also close. The 8K/1K case is conservative because the
discrete-event synthetic workload includes queueing and chunked-prefill
interleaving, while the NVIDIA table reports steady-state `trtllm-bench`
per-GPU output throughput.

## H100 FP8 baseline check

For the same TensorRT-LLM table, Llama 3.3 70B FP8 on H100 TP2 reports
2,209 tok/s/GPU at 1K/1K. The simulator with `llama-70b-fp8 --gpu h100 --tp 2`
reported 1,465 tok/s/GPU under a non-overloaded chunked-prefill run, about 34%
low. That is consistent with the existing validation caveat: the roofline path
is conservative for serving-stack-optimized kernels unless calibrated with
`--kernel-table`.

## Takeaway

Mixed precision is wired correctly enough for architectural sweeps:

- FP4 B200 1K/1K is within 1% of TensorRT-LLM.
- Long-context FP4 is within 6% at 32K/1K when load is controlled.
- Mid-context 8K/1K needs calibration if exact throughput matters.

For production comparisons, add `data/kernel_table.csv` rows from
`trtllm-bench` for the exact model, sequence length, and batching policy.
