//! Cerebras CS-3 / WSE-3 presets.
//!
//! Cerebras differs from GPU-style accelerators in two ways that matter here:
//!
//! 1. **Wafer-scale SRAM.** A CS-3 system exposes 44 GB of on-wafer SRAM at
//!    21 PB/s. The simulator stores that in the main memory fields, like Groq,
//!    and sets `on_chip_sram = 0` to avoid treating the SRAM as a second cache.
//!
//! 2. **System-level scale-out.** The 214 Pb/s WSE fabric is internal to one
//!    wafer. Tensor parallelism across multiple CS-3 systems uses the external
//!    system fabric, so `scale_up_bandwidth` uses the public 1.2 Tb/s system I/O
//!    value as a conservative cross-system boundary.

use super::gpu::GpuSpec;

/// Look up a Cerebras accelerator preset by short name.
pub fn preset(name: &str) -> Option<GpuSpec> {
    match name {
        // ── Cerebras CS-3 with WSE-3 ─────────────────────────────────────────
        // Public WSE-3 / CS-3 specs:
        //   - 900,000 AI cores, 4T transistors
        //   - 125 PFLOPS AI compute
        //   - 44 GB on-chip SRAM at 21 PB/s
        //   - 214 Pb/s internal wafer interconnect
        //   - 1.2 Tb/s system I/O
        //
        // TDP is a system-level estimate; Cerebras states CS-3 doubles CS-2
        // performance at the same power/cost, and public CS-2 installations are
        // typically described around the low-20 kW range.
        "cerebras-cs3" | "cs3" | "wse3" => Some(GpuSpec {
            name: "Cerebras-CS3-WSE3".into(),
            flops_bf16: 125_000e12,
            flops_fp8: 0.0, // WSE-3 peak is published as FP16/BF16-class AI compute.
            flops_fp4: 0.0,
            supports_2to4_sparsity: false,
            memory_bandwidth: 21_000e12, // 21 PB/s on-wafer SRAM bandwidth
            memory_capacity: 44_000_000_000, // 44 GB on-wafer SRAM
            on_chip_sram: 0,             // Main memory already is SRAM.
            scale_up_bandwidth: 150e9,   // 1.2 Tb/s system I/O = 150 GB/s
            scale_up_latency: 2e-6,      // conservative system-level hop estimate
            tdp_watts: 23_000.0,
            cost_per_hour_usd: 0.0, // no stable public per-system on-demand price
            mfu_prefill: 0.75,
            mfu_decode: 0.90,
        }),
        _ => None,
    }
}
