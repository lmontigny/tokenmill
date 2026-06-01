# inference-sim

Discrete-event simulator for LLM inference workloads, written in Rust.

Models prefill/decode phases, KV cache, continuous batching, chunked prefill,
tensor / pipeline / expert parallelism, disaggregated prefill/decode,
speculative decoding, and multi-token prediction.
Targets **~10% error** vs real GPU hardware (see [docs/validation.md](docs/validation.md)).

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
- **[Traces](docs/traces.md)** — public trace sources and the fetch script
- **[Example results](docs/results.md)** — 14 representative simulation runs
- **[Benchmark validation](docs/validation.md)** — MAPE vs real NVIDIA GPUs
- **[Architecture](docs/architecture.md)** — module layout and DES engine
- **[Roadmap](docs/roadmap.md)** — phase history

## Examples and tests

```bash
bash examples/02_chunked_prefill_under_load.sh   # one-feature-per-script demos
cargo test --release                              # integration tests under tests/
```

See [`examples/README.md`](examples/README.md) for the full list (10 scripts).

## License

Apache-2.0 — see [LICENSE](LICENSE).
