# Power and energy

Every simulation run reports total energy used by the GPU cluster, mean power, and per-token / per-request energy. The model is intentionally simple — a three-state per-chip power profile — but enough to capture the first-order story (idle dominates at low load; prefill is more expensive than decode; perf-per-watt differences between accelerators).

## The model

Each chip is treated as drawing one of three power levels at any moment:

| State | Fraction of TDP | When |
|---|---:|---|
| **Prefill** | `0.90 × TDP` | GPU is doing prefill compute (tensor cores saturated) |
| **Decode** | `0.65 × TDP` | GPU is doing decode (memory-bandwidth bound) |
| **Idle** | `0.35 × TDP` | GPU is waiting (HBM PLLs and link clocks still active) |

Coefficients are calibrated against published NVIDIA / AMD profiling data (NVML telemetry on H100 shows decode at 60–75% TDP and prefill at 80–95%). They live in `src/hardware/power.rs` and can be tuned per workload.

The cluster total is:

```
total_energy = sum_over_chips(
    prefill_secs × TDP × 0.90 +
    decode_secs  × TDP × 0.65 +
    idle_secs    × TDP × 0.35
)
mean_power   = total_energy / sim_duration
```

The simulator tracks `prefill_busy_secs` and `decode_busy_secs` per chip by hooking the two places where `busy_until` advances. `idle_secs` is the remainder of `sim_duration × n_chips` (in disaggregated mode, the prefill and decode pools are both counted).

## TDP values

| Accelerator | TDP | Source |
|---|---:|---|
| B200 SXM | 1000 W | NVIDIA |
| H200 SXM | 700 W | NVIDIA |
| H100 SXM5 | 700 W | NVIDIA |
| A100 80GB SXM | 400 W | NVIDIA |
| A10G | 300 W | NVIDIA |
| MI300X | 750 W | AMD |
| MI325X | 1000 W | AMD |
| MI355X | 1400 W | AMD |
| TPU 8i | 600 W | estimate (Google does not publish per-chip TDP; "2× perf/W vs Ironwood" claim) |
| TPU 8t | 750 W | estimate |
| TPU v7 Ironwood | 500 W | estimate |
| Groq LPU v1 | 215 W | Groq published typical (peak ~300 W) |

## Output

The text report adds one line at the bottom whenever the GPU preset has a non-zero TDP:

```
Energy              total=14.20 kJ   mean=0.47 kW   per-token=350.88 mJ   per-req=46.3 J
```

CSV and JSON output carry the same fields:

- `total_energy_kj`
- `mean_power_kw`
- `energy_per_token_mj`
- `energy_per_request_j`

## Caveats

- **Static voltage/frequency** — real GPUs throttle / boost based on workload; we don't model DVFS.
- **No memory-controller breakdown** — high-bandwidth memory keeps drawing power even at idle; the 0.35 × TDP baseline absorbs that into one number.
- **No interconnect-link power** — NVLink / IF / ICI / C2C links draw real power that isn't broken out here.
- **Cluster-level only** — host CPUs, NICs, cooling, and rack overhead are out of scope. To estimate facility-wide power, multiply by a PUE factor (typical hyperscaler PUE ≈ 1.1–1.3).

To recover the old "no power" behaviour, set `tdp_watts = 0` on the preset; the energy fields then print as zero and the text report omits the Energy line.
