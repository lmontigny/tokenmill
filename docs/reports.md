# Study reports

Large comparison tables should be generated as HTML reports instead of being
kept by hand in Markdown. Report mode runs a matrix over model, hardware/system,
TP degree, and arrival rate, then writes:

- a normalized JSON result file for automation
- a self-contained HTML file for review and sharing

The GitHub Pages site uses `reports/dashboard.html` as the single public report
UI. Curated datasets should be committed as JSON under `reports/curated/` and
listed in `reports/manifest.json`; standalone HTML reports are best kept under
`reports/scratch/` for local review. The root Pages entry point redirects to
that dashboard.

Example:

```bash
tokenmill \
  --study-models llama-70b-fp8,llama-70b-w4a8kv4,llama-70b-nvfp4-sparse \
  --study-gpus h100,h200,b200,mi300x \
  --study-tps 1,2,4,8 \
  --study-arrival-rates 1,5,10 \
  --scheduler chunked-prefill \
  --prompt-mean 1024 \
  --output-mean 512 \
  --duration 60 \
  --json-out reports/llama70b-hardware.json \
  --html reports/llama70b-hardware.html
```

DGX/system comparison:

```bash
tokenmill \
  --study-models llama-70b-fp8,llama-70b-w4a8kv4 \
  --study-systems dgx-h200,dgx-b200 \
  --study-tps 8,16 \
  --study-arrival-rates 1,5 \
  --duration 60 \
  --json-out reports/scratch/dgx-comparison.json \
  --html reports/scratch/dgx-comparison.html
```

The HTML report includes summary cards, a sortable/filterable table, and simple
embedded SVG charts for cost-vs-latency, throughput, and energy per token.

The report uses the same simulation path as normal CLI runs. Any option not
covered by the matrix flags, such as scheduler, prompt/output length, seed,
chunk size, speculative decoding, scale-out fabric, or kernel table, is inherited
from the command line and applied to every matrix run.
