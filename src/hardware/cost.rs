//! Cost reporting from cluster GPU-hour pricing.
//!
//! Matches how cloud users actually pay: a flat $/hour per accelerator that
//! bundles capex amortisation + facility + power + ops. Energy alone accounts
//! for only ~10-20% of cloud GPU TCO, so the headline cost-efficiency number
//! used by NVIDIA / OpenAI / Together / Fireworks pricing pages is **$ per
//! 1 M tokens**, computed here as:
//!
//! ```text
//! chip_hours          = n_chips × sim_duration_s / 3600
//! total_cost_usd      = chip_hours × cost_per_hour_usd
//! cost_per_mtok_usd   = total_cost_usd / tokens_generated × 1_000_000
//! cost_per_request_usd = total_cost_usd / requests
//! ```

#[derive(Debug, Clone, Default)]
pub struct CostReport {
    pub total_cost_usd: f64,
    pub cost_per_million_tokens_usd: f64,
    pub cost_per_request_usd: f64,
    pub cluster_cost_per_hour_usd: f64,
}

/// Compute cost metrics. Returns an all-zero report when pricing is unset.
pub fn compute_report(
    sim_duration_s: f64,
    n_chips: u32,
    cost_per_hour_usd: f64,
    tokens: u64,
    requests: u64,
) -> CostReport {
    if cost_per_hour_usd <= 0.0 || sim_duration_s <= 0.0 || n_chips == 0 {
        return CostReport::default();
    }
    let cluster_cost_per_hour = cost_per_hour_usd * n_chips as f64;
    let chip_hours = n_chips as f64 * sim_duration_s / 3600.0;
    let total_cost = chip_hours * cost_per_hour_usd;

    CostReport {
        total_cost_usd: total_cost,
        cost_per_million_tokens_usd: if tokens > 0 {
            total_cost * 1_000_000.0 / tokens as f64
        } else {
            0.0
        },
        cost_per_request_usd: if requests > 0 {
            total_cost / requests as f64
        } else {
            0.0
        },
        cluster_cost_per_hour_usd: cluster_cost_per_hour,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_min_h100_costs_pro_rated_dollars() {
        // 1 chip @ $3.50/hr × 600s = $0.5833.
        let r = compute_report(600.0, 1, 3.50, 100, 5);
        assert!((r.total_cost_usd - 0.5833).abs() < 0.001);
    }

    #[test]
    fn cluster_hourly_scales_with_chip_count() {
        let r = compute_report(60.0, 8, 4.00, 1, 1);
        assert!((r.cluster_cost_per_hour_usd - 32.0).abs() < 1e-6);
    }

    #[test]
    fn zero_price_disables_reporting() {
        let r = compute_report(60.0, 8, 0.0, 1_000, 10);
        assert_eq!(r.total_cost_usd, 0.0);
        assert_eq!(r.cost_per_million_tokens_usd, 0.0);
    }

    #[test]
    fn per_million_tokens_units_are_correct() {
        // 1 chip × $3.60/hr × 1 hour = $3.60 for 1M tokens → $3.60/Mtok.
        let r = compute_report(3600.0, 1, 3.60, 1_000_000, 100);
        assert!((r.cost_per_million_tokens_usd - 3.60).abs() < 0.01);
    }
}
