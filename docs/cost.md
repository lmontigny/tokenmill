# Cost reporting

Every simulation run reports cluster GPU-hour cost in the same units cloud providers and inference-as-a-service vendors (NVIDIA NIM, OpenAI, Anthropic, Together, Fireworks, …) actually publish:

- **$ / 1 M tokens** — the industry-standard headline cost-efficiency number
- **$ / request** — for cost-per-conversation framing
- **$ / hour for the cluster** — for capacity planning
- **Total $ for the simulated run** — for the run as a whole

## Why GPU-hour pricing, not energy × electricity

Cloud GPU prices bundle four things into one $/hour number: capex amortisation, facility / rack overhead, cooling, and energy. Energy alone accounts for only ~10–20% of cloud GPU TCO, so multiplying our energy figures by an electricity rate would answer only the *sustainability* question, not the *what does it actually cost* question. The simulator therefore uses on-demand list prices per accelerator per chip-hour:

```text
chip_hours          = n_chips × sim_duration / 3600
total_cost_usd      = chip_hours × cost_per_hour_usd
cost_per_Mtok_usd   = total_cost_usd × 1_000_000 / tokens_generated
cost_per_request    = total_cost_usd / requests
```

`n_chips` covers both pools in disaggregated prefill/decode mode.

## Prices used

These are 2026 on-demand list prices, drawn from AWS / GCP / Lambda / Vultr / Hot Aisle. They drift quickly — reserved instances and spot pricing run 30–60% cheaper, and absolute numbers shift quarterly. Use them for **relative** comparisons; calibrate the field if you have negotiated rates.

| Preset | $/chip/hr | Source |
|---|---:|---|
| `b200` | 6.50 | AWS / Lambda preview pricing, limited availability |
| `h200` | 4.50 | 2026 estimate; premium over H100, below B200 |
| `h100` | 3.50 | AWS p5, GCP A3, Lambda |
| `a100` | 2.50 | AWS p4, GCP A2 |
| `a10g` | 1.20 | AWS g5.2xl |
| `mi300x` | 3.50 | Vultr, Hot Aisle, MicroCloud |
| `mi325x` | 4.50 | newer, limited availability |
| `mi355x` | 6.00 | 2026 estimate |
| `tpu-v8i` | 4.50 | estimated 2026 GCP on-demand |
| `tpu-v8t` | 5.50 | estimated, marketed for training |
| `tpu-v7-ironwood` | 4.00 | extrapolated from GCP TPU v5p |
| `groq-lpu-v1` | 0.30 | back-calculated from Groq's per-token pricing |
| `cerebras-cs3` | 0.00 | disabled; no stable public per-system on-demand list price |

## Output

The text report adds one line at the bottom whenever the preset has a non-zero price:

```
Cost                $0.06 total   $0.247/Mtok   $0.000033/req   ($3.50/hr cluster)
```

CSV and JSON output carry four new fields: `total_cost_usd`, `cost_per_million_tokens_usd`, `cost_per_request_usd`, `cluster_cost_per_hour_usd`.

## Comparing to vendor pricing

For reference, a few published 2026 inference-as-a-service rates per 1 M output tokens:

| Provider | Model | $/Mtok output |
|---|---|---:|
| OpenAI | GPT-4o | $10 |
| Anthropic | Claude Sonnet 4.5 | $15 |
| Together | Llama-3.3-70B Turbo | $0.88 |
| Fireworks | Llama-70B | $0.90 |
| Groq | Llama-70B | $0.79 |

A simulated H100 TP=4 serving llama-70b-fp8 lands around $10/Mtok in our model (row 9 of [results.md](results.md)) — close to OpenAI's API pricing, well above the highly-batched serving rates from Together / Fireworks / Groq, which run at 5–10× higher utilisation than our examples.

## Caveats

- **Prices change quarterly** — and vary by region, reservation tier, and committed-use discounts. Update the field on the preset for your actual contract.
- **Compute only** — host CPUs, NICs, storage, egress bandwidth, and inference-framework licensing are out of scope.
- **No batching premium** — real production inference services batch aggressively and amortise per-chip cost across many concurrent users; our defaults run at moderate batch.
- **Idle time is paid time** — at low arrival rates, $/Mtok is dominated by under-utilised GPUs. The simulator reflects this honestly: it's a real serving cost. Calibrate by re-running at the rate your traffic actually delivers.

To opt out, set `cost_per_hour_usd = 0` on the preset; the cost line disappears from the text report and the CSV columns are zero.
