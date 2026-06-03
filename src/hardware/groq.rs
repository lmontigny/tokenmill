//! Groq LPU presets.
//!
//! Groq's deterministic streaming dataflow architecture differs from GPUs/TPUs
//! in two important ways:
//!
//! 1. **No HBM.** All weights and activations live in on-chip SRAM. The
//!    `memory_bandwidth` field carries the SRAM bandwidth (~80 TB/s) and
//!    `memory_capacity` is the SRAM size (~230 MB on GroqChip 1) — orders of
//!    magnitude smaller than any GPU. Any non-trivial model needs very high
//!    `--tp` (e.g. llama-70b-fp8 at TP ≈ 358) just to fit the weights.
//!
//! 2. **Compiler-scheduled deterministic execution.** Groq's tensor streaming
//!    processor runs on a fixed cycle schedule, so there is no caching or
//!    runtime overhead — published MFU is very high. Setting `on_chip_sram`
//!    to 0 keeps the simulator from double-counting (the SRAM IS the main
//!    memory, not a faster tier on top of it).
//!
//! ### Modelling caveats
//! The ring-allreduce and EP all-to-all formulas in `cluster.rs` are
//! approximations of Groq's actual chip-to-chip dataflow. At very high TP
//! (hundreds of chips), real-world Groq latency is also limited by per-hop
//! link latency in the mesh, which this simulator does not model.

use super::gpu::GpuSpec;

/// Look up a Groq accelerator preset by short name.
pub fn preset(name: &str) -> Option<GpuSpec> {
    match name {
        // ── GroqChip 1 (LPU v1) ────────────────────────────────────────────────
        // 14 nm GlobalFoundries, 725 mm² die, 215-300 W TDP.
        // 750 TOPS INT8 / 188 TFLOPS FP16 / 230 MB on-chip SRAM at 80 TB/s.
        // No off-chip memory — chips are clustered (GroqRack = 8 chips,
        // GroqNode = 8 chips, pods of 256+ chips) with high-speed C2C links.
        "groq-lpu-v1" => Some(GpuSpec {
            name: "GroqChip-1".into(),
            flops_bf16: 188e12, // 188 TFLOPS FP16 (BF16-equivalent for the roofline)
            flops_fp8: 375e12,  // FP8 ≈ 2× FP16; INT8 is 750 TOPS (Groq's primary serving dtype)
            flops_fp4: 0.0,
            supports_2to4_sparsity: false,
            memory_bandwidth: 80e12, // 80 TB/s on-chip SRAM (THE memory; no off-chip DRAM)
            memory_capacity: 230_000_000, // 230 MB — tiny vs GPU; high `--tp` is mandatory
            on_chip_sram: 0,         // No two-tier memory: HBM field already IS the SRAM
            scale_up_bandwidth: 400e9, // C2C: 16 ports × ~25 GB/s aggregate per chip (approx.)
            // Per-hop C2C latency ~100 ns — this is what lets the simulator capture
            // why Groq's real TPOT at TP=358 is dominated by the chip-mesh diameter,
            // not by SRAM bandwidth (which is essentially infinite for this purpose).
            scale_up_latency: 100e-9,
            tdp_watts: 215.0, // GroqChip 1 — published typical (peak ~300W)
            // Groq doesn't sell per-chip; pricing is per-rack. $0.30/hr/chip is
            // back-calculated from their published per-token rates and pod size.
            cost_per_hour_usd: 0.30,
            mfu_prefill: 0.85, // Deterministic compiler-scheduled execution → very high util
            mfu_decode: 0.90,
        }),
        _ => None,
    }
}
