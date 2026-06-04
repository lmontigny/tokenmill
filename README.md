# tokenmill

A discrete-event simulator for LLM inference clusters, written in Rust.
Useful for capacity planning, hardware shortlisting, and what-if analysis
before you provision the GPUs.

**Built-in support for:**
- **Schedulers** — continuous batching (Orca), chunked prefill (Sarathi), preemption / recompute
- **Parallelism** — tensor (TP), pipeline (PP), expert (EP), disaggregated prefill/decode
- **Speedups** — speculative decoding, multi-token prediction, paged KV cache, MLA KV compression
- **Models** — dense (Llama 8B / 70B + FP8) and MoE (Mixtral, Llama 4 Maverick / Behemoth, DeepSeek V3, Kimi K2 1 T)
- **Hardware** — NVIDIA rubin (2026) / b200 / h200 / h100 / a100 / a10g, AMD mi300x / mi325x / mi355x, Google TPU v7-ironwood / 8t / 8i, Groq lpu-v1, Cerebras CS-3 / WSE-3
- **Systems** — DGX H100 / H200 / B200 presets with 8-GPU scale-up nodes and NDR-class scale-out defaults
- **Latency prediction** — TTFT / TPOT histograms (p50 / p95 / p99), throughput, KV utilization, preemption counts
- **Energy prediction** — per-chip TDP model, total kJ, mean kW, energy per output token
- **Cost prediction** — GPU-hour pricing, total $, $ per million tokens, $ per request

Targets ~10% error vs real GPU kernel time on validated configs (see [docs/validation.md](docs/validation.md)).
Collective formulas model one scale-up domain by default and can add a scale-out
Ethernet / InfiniBand tier for multi-node TP / PP / EP; see [docs/topology.md](docs/topology.md).

## Build

```bash
cargo build --release
```

## Quick start

```bash
# llama-8b on H100, chunked-prefill, 10 req/s for 60s
cargo run --release -- \
  --model llama-8b --gpu h100 \
  --scheduler chunked-prefill \
  --arrival-rate 10.0 --duration 60.0

# same with profiled kernel latencies (more accurate)
cargo run --release -- \
  --model llama-8b --gpu h100 \
  --scheduler chunked-prefill \
  --arrival-rate 10.0 --duration 60.0 \
  --kernel-table data/kernel_table.csv

# replay the Azure LLM inference trace (auto-detected format)
bash scripts/fetch_traces.sh azure
cargo run --release -- \
  --model llama-8b --gpu h100 \
  --workload trace:data/traces/azure_code.csv \
  --duration 3600.0
```

## Docs

- **[CLI reference](docs/cli.md)** — all flags, model presets, GPU presets
- **[Latency model](docs/latency-model.md)** — roofline + MoE + EP math
- **[Supported optimizations](docs/optimizations.md)** — batching, parallelism, spec/MTP
- **[Quantization](docs/quantization.md)** — FP8, FP4, W4A16, W4A8KV4, and sparse NVFP4 modeling
- **[Traces](docs/traces.md)** — public trace sources and the fetch script
- **[Example results](docs/results.md)** — 14 representative simulation runs
- **[Benchmark validation](docs/validation.md)** — MAPE vs real NVIDIA GPUs
- **[Power and energy](docs/power.md)** — per-chip TDP model, energy per token
- **[Cost](docs/cost.md)** — $/Mtok from GPU-hour pricing, comparison to vendor rates
- **[Topology and scope](docs/topology.md)** — what scale-up fabric is modelled per vendor, and why DCN is out of scope
- **[Architecture](docs/architecture.md)** — module layout and DES engine
- **[Roadmap](docs/roadmap.md)** — phase history

## Examples and tests

```bash
bash examples/02_chunked_prefill_under_load.sh   # one-feature-per-script demos
cargo test --release                              # integration tests under tests/
```

See [`examples/README.md`](examples/README.md) for the full list (16 scripts).

## License

Apache-2.0 — see [LICENSE](LICENSE).
