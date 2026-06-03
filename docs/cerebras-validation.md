# Cerebras validation

This page validates the `cerebras-cs3` preset against public Cerebras Inference
numbers. The short version: the CS-3 hardware fields are correct, but the current
roofline model is a hardware ceiling, not an API-serving predictor, for Cerebras.

## Sources checked

| Source | Relevant public number |
|---|---:|
| [Cerebras Inference launch](https://www.cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed) | Llama 3.1 8B: 1,800 output tok/s; Llama 3.1 70B: 450 output tok/s; native 16-bit weights |
| [Cerebras 70B update](https://www.cerebras.ai/blog/cerebras-inference-3x-faster) | Llama 3.1 70B: 2,100 output tok/s average; speculative decoding enabled; +/-20% variation expected |
| [Cerebras 405B result](https://www.cerebras.ai/blog/llama-405b-inference) | Llama 3.1 405B: 969 output tok/s at 1K input, 539 output tok/s at 100K input; 240 ms TTFT |
| [Neocortex CS-3 specs](https://portal.neocortex.psc.edu/docs/CS-3/system-specifications.html) | WSE-3: 44 GB SRAM, 21 PB/s memory bandwidth, 214 Pb/s on-wafer interconnect, 1.2 Tb/s system I/O |

## Simulator comparisons

Commands were run on 2026-06-04 against the `cerebras-cs3` preset.

| Case | Public result | Simulator shape | Simulator result | Verdict |
|---|---:|---|---:|---|
| Llama 3.1 8B, 16-bit | 1,800 tok/s, 0.56 ms/token | `llama-8b`, 1 CS-3, chunked prefill | `tpot_p50_ms = 0.001` (histogram floor), >=1,000,000 tok/s | Not valid as API predictor; missing fixed per-token service overhead |
| Llama 3.1 70B, 16-bit launch | 450 tok/s, 2.22 ms/token | `llama-70b`, `--pp 4`, chunked prefill | `tpot_p50_ms = 0.008`, 125,000 tok/s | Not valid as API predictor; pure layer-split roofline ceiling |
| Llama 3.1 70B, 16-bit current | 2,100 tok/s, 0.48 ms/token | `llama-70b`, `--pp 4`, chunked prefill | `tpot_p50_ms = 0.008`, 125,000 tok/s | Not valid as API predictor; public result includes optimized stack and speculative decoding |
| Llama 3.1 405B, 16-bit | 969 tok/s at 1K input; 539 tok/s at 100K input | no repo preset | not run | Needs a `llama-405b` preset before comparison |

## Interpretation

The spec fields in `src/hardware/cerebras.rs` match public CS-3/WSE-3 hardware
data: 44 GB SRAM, 21 PB/s SRAM bandwidth, 214 Pb/s on-wafer fabric, and 1.2 Tb/s
system I/O. Those numbers are useful for upper-bound architectural studies.

They are not enough for production API latency. Cerebras Inference has a large
fixed service-time component that the current roofline model does not represent:
runtime scheduling, host/API overhead, pipeline fill/drain, wafer I/O behavior,
token sampling, and serving-stack limits. For 70B, Cerebras also reports
speculative decoding in the 2,100 tok/s result, while the simulator only models
generic speculative decoding when the user explicitly passes `--spec-tokens`.

## Calibration target

To make `cerebras-cs3` predictive, add one of these before using it for API
capacity planning:

- A Cerebras-specific per-token latency floor calibrated near 0.5 ms/token for
  Llama 3.1 8B/70B current API behavior.
- Cerebras kernel/service-time table entries for the exact model, context, and
  serving mode being compared.
- A `llama-405b` preset plus empirical CS-3 pipeline calibration for the 405B
  public numbers.

Until then, treat `cerebras-cs3` results as a hardware roofline ceiling.
