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

- [`curated/llama8b-precision-hardware.json`](curated/llama8b-precision-hardware.json)
  compares Llama 8B FP8 vs W4A8KV4 across H100, H200, and B200 at 1 and 5 rps.
- [`curated/dgx-llama70b-comparison.json`](curated/dgx-llama70b-comparison.json)
  compares DGX H200 vs DGX B200 for Llama 70B FP8 and W4A8KV4 at TP=8 and TP=16.

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
cargo run -- \
  --study-models llama-8b-fp8,llama-8b-w4a8kv4 \
  --study-gpus h100,h200,b200 \
  --study-tps 1 \
  --study-arrival-rates 1,5 \
  --scheduler chunked-prefill \
  --prompt-mean 512 \
  --output-mean 128 \
  --duration 10 \
  --json-out reports/curated/llama8b-precision-hardware.json
```

```bash
cargo run -- \
  --study-models llama-70b-fp8,llama-70b-w4a8kv4 \
  --study-systems dgx-h200,dgx-b200 \
  --study-tps 8,16 \
  --study-arrival-rates 1 \
  --scheduler chunked-prefill \
  --prompt-mean 1024 \
  --output-mean 256 \
  --duration 10 \
  --json-out reports/curated/dgx-llama70b-comparison.json
```
