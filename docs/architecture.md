# Architecture

Discrete-event simulator. Each model decode step, prefill kernel, KV transfer
and request arrival is a typed event with a sim-time timestamp; the engine pops
them off a min-heap in order and dispatches each to the right handler.

## Module layout

```
engine/        DES core: BinaryHeap event queue, SimTime clock, dispatch loop
hardware/      accelerator specs, cluster topology (TP/PP/EP), kernel time table
model/         LLM config (dense + MoE), KV cache block manager
scheduler/     continuous-batch (Orca) and chunked-prefill (Sarathi)
workload/      Poisson synthetic arrivals, trace replay (Azure + native CSV)
metrics/       HDR histograms for TTFT/TPOT, throughput, KV utilization
```

## Engine

- `engine::event::EventPayload` — typed event payloads (RequestArrival, PrefillStart, PrefillDone, DecodeStep, RequestComplete, KvTransferDone).
- `engine::queue::SimQueue` — `BinaryHeap<Reverse<Event>>` ordered by sim-time.
- `engine::sim::Simulator` — owns GPUs, KV cache, scheduler, metrics; loops `pop event → handle → push follow-ups` until `now ≥ until`.

## Hardware

- `GpuSpec` — peak FLOPS (BF16/FP8/FP4), memory bandwidth + capacity, on-chip SRAM, scale-up fabric bandwidth/latency, TDP, chip-hour cost, and MFU calibration. Presets cover NVIDIA, AMD, Google TPU, Groq, and Cerebras accelerators.
- `ClusterConfig` — TP/PP/EP degrees, scale-up bandwidth, optional scale-out bandwidth/latency, node size, and disaggregation flag. All collective formulas (hierarchical ring-allreduce, EP all-to-all, PP transfer, KV transfer) live here.
- `KernelTable` — optional CSV of profiled kernel latencies, looked up by (gpu, model, op, batch, seq_len) with linear interpolation on `seq_len`. Roofline fallback on miss.

## Model

- `LlmConfig` — dense or MoE topology (layers, d_model, n_kv_heads, head_dim, FFN/expert hidden, MLA `kv_lora_rank`) plus mixed precision (`weight_bits`, `activation_bits`, `kv_bits`, `weight_sparsity`). Knows its own `weight_bytes`, `weight_bytes_active`, `expert_weight_bytes_active`, `kv_bytes(seq_len)`.
- `KvCacheManager` — PagedAttention block allocator. Grows requests in fixed-size blocks; preempts and recomputes when the pool is exhausted.

## Schedulers

- `ContinuousBatchScheduler` (Orca) — admits requests one at a time during prefill; once a request reaches decode it joins the batch and stays for every step until completion.
- `ChunkedPrefillScheduler` (Sarathi) — chops prefill into `--chunk-size` chunks and interleaves them with decode batches in a single per-step token budget, so long prompts never starve decode.

## Workload

- `SyntheticWorkload` — Poisson arrivals, log-normal prompt + output lengths.
- `TraceReplay` — reads CSV traces. Auto-detects Azure `TIMESTAMP,ContextTokens,GeneratedTokens` vs native `timestamp_ms,prompt_tokens,output_tokens` from the header. Records are sorted and normalised so the first arrival is at t = 0.

## Metrics

- `MetricsCollector` — HDR histograms (`hdrhistogram` crate) for TTFT, prefill latency, KV transfer time, TPOT. Tracks completions, token throughput, mean KV utilisation.
- `RunSummary` — serializable struct emitted as text / JSON / CSV (`--output`).
