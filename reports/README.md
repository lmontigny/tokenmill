# Reports

Use this directory for generated study reports.

## Layout

- `dashboard.html` — the single GitHub Pages report UI.
- `curated/` — tracked, intentionally published JSON datasets with stable commands.
- `scratch/` — local exploratory reports; ignored by Git except `.gitkeep`.
- `manifest.json` — metadata and dataset paths consumed by `dashboard.html`.

Do not commit every experiment. HTML and JSON reports can get large and noisy.
Commit only reports that are useful as project examples or recurring validation
artifacts.

## Curated Report Workflow

Current curated reports:

- [`curated/accelerator-llama8b-coverage.json`](curated/accelerator-llama8b-coverage.json)
  compares Llama 8B FP8 and W4A8KV4 across every non-Groq accelerator preset.
- [`curated/dgx-system-precision-coverage.json`](curated/dgx-system-precision-coverage.json)
  compares DGX H100, H200, and B200 systems for Llama 70B FP8, W4A8KV4, and sparse NVFP4.
- [`curated/b200-model-coverage.json`](curated/b200-model-coverage.json)
  runs every supported model preset on B200.
- [`curated/frontier-moe-accelerator-coverage.json`](curated/frontier-moe-accelerator-coverage.json)
  compares DeepSeek V3 and Kimi K2 variants across high-end accelerator presets.
- [`curated/frontier-moe-dgx-coverage.json`](curated/frontier-moe-dgx-coverage.json)
  compares DeepSeek V3 and Kimi K2 variants on DGX H100, H200, and B200 systems.
- [`curated/groq-fit-examples.json`](curated/groq-fit-examples.json)
  shows Groq LPU v1 examples at high TP, where on-chip SRAM capacity drives chip count.

Generate a curated dataset:

```bash
cargo run -- \
  --study-models llama-70b-fp8,llama-70b-w4a8kv4 \
  --study-gpus h100,h200,b200,mi300x \
  --study-tps 1,2,4,8 \
  --study-arrival-rates 1,5,10 \
  --scheduler chunked-prefill \
  --prompt-mean 1024 \
  --output-mean 512 \
  --duration 60 \
  --json-out reports/curated/llama70b-hardware.json
```

For local exploration, write to `scratch/`:

```bash
cargo run -- \
  --study-models llama-8b-fp8,llama-8b-w4a8kv4 \
  --study-gpus h100,h200,b200 \
  --study-arrival-rates 1,5 \
  --duration 30 \
  --json-out reports/scratch/test.json \
  --html reports/scratch/test.html
```

When adding a curated report, include the exact command in the commit message or
PR description so the report can be regenerated. Also add a `manifest.json`
entry with title, description, models, hardware/systems, precision, tags, row
count, JSON path, and the regeneration command.

## Regenerating Current Reports

```bash
cargo run --release -- \
  --study-models llama-8b-fp8,llama-8b-w4a8kv4 \
  --study-gpus rubin,b200,h200,h100,a100,a10g,mi355x,mi325x,mi300x,tpu-v8i,tpu-v8t,tpu-v7-ironwood,cerebras-cs3 \
  --study-tps 1 \
  --study-arrival-rates 5 \
  --scheduler chunked-prefill \
  --prompt-mean 512 \
  --output-mean 128 \
  --duration 10 \
  --json-out reports/curated/accelerator-llama8b-coverage.json
```

```bash
cargo run --release -- \
  --study-models llama-70b-fp8,llama-70b-w4a8kv4,llama-70b-nvfp4-sparse \
  --study-systems dgx-h100,dgx-h200,dgx-b200 \
  --study-tps 8,16 \
  --study-arrival-rates 1 \
  --scheduler chunked-prefill \
  --prompt-mean 1024 \
  --output-mean 256 \
  --duration 10 \
  --json-out reports/curated/dgx-system-precision-coverage.json
```

```bash
cargo run --release -- \
  --study-models llama-8b,llama-8b-fp8,llama-8b-w4a16,llama-8b-w4a8kv4,llama-70b,llama-70b-fp8,llama-70b-w4a16,llama-70b-w4a8kv4,llama-70b-nvfp4-sparse,mixtral-8x7b,llama4-maverick,deepseek-v3,kimi-k2,kimi-k2-nvfp4-sparse,llama4-behemoth \
  --study-gpus b200 \
  --study-tps 8 \
  --study-arrival-rates 1 \
  --ep 8 \
  --scheduler chunked-prefill \
  --prompt-mean 1024 \
  --output-mean 256 \
  --duration 10 \
  --json-out reports/curated/b200-model-coverage.json
```

```bash
cargo run --release -- \
  --study-models deepseek-v3,kimi-k2,kimi-k2-nvfp4-sparse \
  --study-gpus rubin,b200,h200,h100,mi355x,mi325x,mi300x,tpu-v8i,tpu-v8t,tpu-v7-ironwood,cerebras-cs3 \
  --study-tps 8 \
  --study-arrival-rates 1,3 \
  --ep 8 \
  --scheduler chunked-prefill \
  --prompt-mean 2048 \
  --output-mean 512 \
  --duration 10 \
  --json-out reports/curated/frontier-moe-accelerator-coverage.json
```

```bash
cargo run --release -- \
  --study-models deepseek-v3,kimi-k2,kimi-k2-nvfp4-sparse \
  --study-systems dgx-h100,dgx-h200,dgx-b200 \
  --study-tps 8,16 \
  --study-arrival-rates 1,3 \
  --ep 8 \
  --scheduler chunked-prefill \
  --prompt-mean 2048 \
  --output-mean 512 \
  --duration 10 \
  --json-out reports/curated/frontier-moe-dgx-coverage.json
```

```bash
cargo run --release -- \
  --study-models llama-8b-fp8 \
  --study-gpus groq-lpu-v1 \
  --study-tps 64 \
  --study-arrival-rates 5 \
  --scheduler chunked-prefill \
  --prompt-mean 512 \
  --output-mean 128 \
  --duration 10 \
  --json-out reports/scratch/groq-llama8b.json

cargo run --release -- \
  --study-models llama-70b-fp8 \
  --study-gpus groq-lpu-v1 \
  --study-tps 358 \
  --study-arrival-rates 1 \
  --scheduler chunked-prefill \
  --prompt-mean 1024 \
  --output-mean 256 \
  --duration 10 \
  --json-out reports/scratch/groq-llama70b.json

jq -s add reports/scratch/groq-llama8b.json reports/scratch/groq-llama70b.json \
  > reports/curated/groq-fit-examples.json
```
