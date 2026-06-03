//! Power and energy model for an accelerator cluster.
//!
//! We approximate per-chip power as a 3-state model:
//!
//! * **Prefill** (compute-bound) — tensor cores active, ~`PREFILL_COEF` × TDP.
//! * **Decode** (memory-bound) — HBM saturated, cores partially active,
//!   ~`DECODE_COEF` × TDP.
//! * **Idle** — clocks down but HBM PLLs and links active,
//!   ~`IDLE_COEF` × TDP.
//!
//! Coefficients are calibrated against published NVIDIA / AMD profiling data
//! (e.g. NVML telemetry on H100): real serving runs draw 60–75% TDP during
//! decode and 80–95% TDP during prefill, ~35% TDP at idle.
//!
//! Energy per chip = prefill_secs × prefill_W + decode_secs × decode_W + idle_secs × idle_W.
//! For a cluster we multiply by the number of chips (TP × PP × disagg pools).

/// Fraction of TDP drawn when the GPU is in active prefill (compute-bound).
pub const PREFILL_COEF: f64 = 0.90;

/// Fraction of TDP drawn when the GPU is in active decode (memory-bound).
pub const DECODE_COEF: f64 = 0.65;

/// Baseline TDP fraction drawn while idle (HBM PLLs, link clocks, etc.).
pub const IDLE_COEF: f64 = 0.35;

/// Aggregate energy report for a simulation run.
#[derive(Debug, Clone, Default)]
pub struct PowerReport {
    pub total_energy_j: f64,
    pub mean_power_w: f64,
    pub prefill_energy_j: f64,
    pub decode_energy_j: f64,
    pub idle_energy_j: f64,
    /// Convenience: total energy divided by tokens generated (millijoules per output token).
    pub energy_per_token_mj: f64,
    /// Convenience: total energy divided by completed requests (joules per request).
    pub energy_per_request_j: f64,
}

/// Compute a `PowerReport` for a cluster from per-chip timing accumulators.
///
/// All `*_secs` values are summed across **every** chip in the cluster
/// (so passing TP × PP × N_pools chips yields the cluster-wide total).
///
/// * `sim_duration_s` — wall time of the run, used to compute mean power.
/// * `n_chips`        — total chips in the cluster (for idle accounting).
/// * `tdp_watts`      — per-chip TDP from the GpuSpec.
/// * `tokens`         — total output tokens generated.
/// * `requests`       — completed requests.
pub fn compute_report(
    prefill_secs_cluster: f64,
    decode_secs_cluster: f64,
    sim_duration_s: f64,
    n_chips: u32,
    tdp_watts: f64,
    tokens: u64,
    requests: u64,
) -> PowerReport {
    if tdp_watts <= 0.0 || sim_duration_s <= 0.0 || n_chips == 0 {
        return PowerReport::default();
    }
    let total_chip_seconds = sim_duration_s * n_chips as f64;
    let active_secs = prefill_secs_cluster + decode_secs_cluster;
    let idle_secs = (total_chip_seconds - active_secs).max(0.0);

    let prefill_energy_j = prefill_secs_cluster * tdp_watts * PREFILL_COEF;
    let decode_energy_j = decode_secs_cluster * tdp_watts * DECODE_COEF;
    let idle_energy_j = idle_secs * tdp_watts * IDLE_COEF;
    let total_energy_j = prefill_energy_j + decode_energy_j + idle_energy_j;

    PowerReport {
        total_energy_j,
        mean_power_w: total_energy_j / sim_duration_s,
        prefill_energy_j,
        decode_energy_j,
        idle_energy_j,
        energy_per_token_mj: if tokens > 0 {
            total_energy_j * 1000.0 / tokens as f64
        } else {
            0.0
        },
        energy_per_request_j: if requests > 0 {
            total_energy_j / requests as f64
        } else {
            0.0
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_idle_run_uses_only_idle_power() {
        // 60 s on 1 chip @ 700 W, no work → energy = 60 × 700 × 0.35 = 14_700 J.
        let r = compute_report(0.0, 0.0, 60.0, 1, 700.0, 0, 0);
        assert!((r.total_energy_j - 14_700.0).abs() < 1.0);
        assert!((r.mean_power_w - 700.0 * IDLE_COEF).abs() < 1.0);
    }

    #[test]
    fn fully_loaded_prefill_uses_prefill_coef() {
        // 60 s on 1 chip @ 1000 W, fully prefill → 60 × 1000 × 0.90 = 54_000 J.
        let r = compute_report(60.0, 0.0, 60.0, 1, 1000.0, 0, 0);
        assert!((r.total_energy_j - 54_000.0).abs() < 1.0);
    }

    #[test]
    fn per_token_energy_is_nonzero() {
        let r = compute_report(0.0, 30.0, 60.0, 8, 700.0, 1_000, 10);
        assert!(r.energy_per_token_mj > 0.0);
        assert!(r.energy_per_request_j > 0.0);
    }

    #[test]
    fn zero_tdp_disables_reporting() {
        let r = compute_report(10.0, 10.0, 60.0, 1, 0.0, 100, 5);
        assert_eq!(r.total_energy_j, 0.0);
    }
}
