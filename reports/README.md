# Reports

Use this directory for generated study reports.

## Layout

- `curated/` — tracked, intentionally published reports with stable commands.
- `scratch/` — local exploratory reports; ignored by Git except `.gitkeep`.

Do not commit every experiment. HTML and JSON reports can get large and noisy.
Commit only reports that are useful as project examples or recurring validation
artifacts.

## Curated Report Workflow

Generate a report into `curated/`:

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
  --json-out reports/curated/llama70b-hardware.json \
  --html reports/curated/llama70b-hardware.html
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
PR description so the report can be regenerated.
