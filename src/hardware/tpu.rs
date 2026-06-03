//! Google TPU presets.
//!
//! TPUs reuse the same [`GpuSpec`] shape as NVIDIA / AMD chips — the roofline
//! math is vendor-agnostic. Vendor-specific quirks (Boardfly topology, on-chip
//! Vmem scratchpad, ICI vs NVLink) are captured by field values plus comments;
//! the collective formulas in `cluster.rs` are reused unchanged.
//!
//! Split out from `gpu.rs` so more accelerator families (Trainium, Gaudi, …)
//! can land in their own files without bloating one match arm.

use super::gpu::GpuSpec;

/// Look up a TPU preset by short name (e.g. `"tpu-v8i"`).
///
/// Returns `None` for unknown names so the caller can chain other vendor lookups.
pub fn preset(name: &str) -> Option<GpuSpec> {
    match name {
        // ── TPU 8i — 2026 serving-optimized ───────────────────────────────────
        // Official specs from Google Cloud blog:
        //   https://cloud.google.com/blog/products/compute/tpu-8t-and-tpu-8i-technical-deep-dive
        //
        // FP8/BF16 derived from the published FP4 PFLOPs using the standard 2× per-precision
        // ratio (FP4 → FP8 halves, FP8 → BF16 halves). Google publishes only FP4.
        //
        // Network topology: **Boardfly** — Dragonfly-inspired hierarchical fabric:
        //   - Building Block: 4-chip ring (internal ICI)
        //   - Group: 8 boards copper-connected (32 chips)
        //   - Pod: 36 groups via Optical Circuit Switches, up to 1024 chips
        //   - Diameter: 7 hops (56% lower than v7 Ironwood's 16-hop torus)
        // On-chip CAE (Collectives Acceleration Engine) accelerates all-reduce / all-to-all.
        // The simulator's ring-allreduce formula is accurate for TP ≤ 32 (within a group);
        // for larger TP spanning the OCS layer it under-estimates by ~10-20%.
        "tpu-v8i" => Some(GpuSpec {
            name: "TPU-8i".into(),
            flops_bf16: 2525e12,  // FP4 / 4 = 2.525 PFLOPS BF16 (derived)
            flops_fp8: 5050e12,   // FP4 / 2 = 5.05 PFLOPS FP8 (derived)
            flops_fp4: 10_100e12, // published FP4
            supports_2to4_sparsity: false,
            memory_bandwidth: 8.601e12,       // 8601 GB/s — official
            memory_capacity: 288_000_000_000, // 288 GB — official
            on_chip_sram: 384_000_000,        // 384 MB Vmem — official, 3× TPU 8t
            scale_up_bandwidth: 2400e9,       // ICI ~2.4 TB/s aggregate (2× v7 Ironwood per blog)
            scale_up_latency: 200e-9,         // Boardfly 7-hop diameter ⇒ very low per-hop α
            tdp_watts: 600.0, // TPU 8i estimate (Google "2× perf/W" claim implies modest TDP)
            cost_per_hour_usd: 4.50, // estimated 2026 GCP on-demand
            mfu_prefill: 0.72,
            mfu_decode: 0.80, // CAE + huge Vmem → strong decode efficiency
        }),
        // ── TPU 8t — 2026 training-focused ────────────────────────────────────
        // 3D torus, 9600-chip superpod. Same FP4 derivation as 8i.
        "tpu-v8t" => Some(GpuSpec {
            name: "TPU-8t".into(),
            flops_bf16: 3150e12,  // FP4 / 4 = 3.15 PFLOPS BF16 (derived)
            flops_fp8: 6300e12,   // FP4 / 2 = 6.3 PFLOPS FP8 (derived)
            flops_fp4: 12_600e12, // published FP4
            supports_2to4_sparsity: false,
            memory_bandwidth: 6.528e12,       // 6528 GB/s — official
            memory_capacity: 216_000_000_000, // 216 GB — official
            on_chip_sram: 128_000_000,        // 128 MB Vmem — official
            scale_up_bandwidth: 2400e9,       // ICI 2× v7 Ironwood (blog: "2x scale-up bandwidth")
            scale_up_latency: 500e-9,         // 3D-torus hop ~500 ns
            tdp_watts: 750.0,                 // TPU 8t estimate (training chip, more compute)
            cost_per_hour_usd: 5.50,          // estimate; TPU 8t marketed for training workloads
            mfu_prefill: 0.70,
            mfu_decode: 0.75,
        }),
        // ── TPU v7 Ironwood — April 2025, inference-focused ───────────────────
        // 3D-torus ICI.
        "tpu-v7-ironwood" => Some(GpuSpec {
            name: "TPU-v7-Ironwood".into(),
            flops_bf16: 2304e12,
            flops_fp8: 4614e12,
            flops_fp4: 0.0,
            supports_2to4_sparsity: false,
            memory_bandwidth: 7.37e12,
            memory_capacity: 192_000_000_000,
            on_chip_sram: 256_000_000, // ~256 MB Vmem (estimate; between v5p and 8t)
            scale_up_bandwidth: 1200e9, // ICI ~1.2 TB/s aggregate
            scale_up_latency: 500e-9,  // 3D-torus hop ~500 ns
            tdp_watts: 500.0,          // v7 Ironwood estimate
            cost_per_hour_usd: 4.00,   // extrapolated from GCP TPU v5p pricing
            mfu_prefill: 0.70,
            mfu_decode: 0.75,
        }),
        _ => None,
    }
}
