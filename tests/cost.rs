//! End-to-end cost-reporting checks through `MetricsCollector::to_summary`.

use tokenmill::metrics::collector::MetricsCollector;

#[test]
fn h100_single_chip_one_hour_costs_three_fifty() {
    let mut m = MetricsCollector::new();
    m.sim_duration = 3600.0;
    m.tokens_generated = 1_000_000;
    m.completions = 1000;
    // tdp_watts=0 (skip energy), cost=$3.50/hr.
    let s = m.to_summary(
        "m", "g", "sched", 1, 1, 1, false, 0.0, "roofline", 0.0, 3.50,
    );
    assert!((s.total_cost_usd - 3.50).abs() < 0.001);
    assert!((s.cost_per_million_tokens_usd - 3.50).abs() < 0.01);
    assert!((s.cluster_cost_per_hour_usd - 3.50).abs() < 0.001);
}

#[test]
fn cost_scales_with_chip_count() {
    let mut m = MetricsCollector::new();
    m.sim_duration = 60.0;
    m.tokens_generated = 10_000;
    m.completions = 100;
    // 4 chips × $4/hr × 60s = $0.2667
    let s = m.to_summary(
        "m", "g", "sched", 4, 1, 1, false, 0.0, "roofline", 0.0, 4.00,
    );
    assert!((s.total_cost_usd - 0.2667).abs() < 0.001);
    assert!((s.cluster_cost_per_hour_usd - 16.0).abs() < 0.001);
}

#[test]
fn disaggregated_doubles_chip_count_for_cost() {
    let mut m = MetricsCollector::new();
    m.sim_duration = 3600.0;
    m.tokens_generated = 1_000_000;
    m.completions = 1000;
    // 4 chips × 2 pools × $3.50/hr × 1hr = $28
    let s = m.to_summary("m", "g", "sched", 4, 1, 1, true, 0.0, "roofline", 0.0, 3.50);
    assert!((s.total_cost_usd - 28.0).abs() < 0.01);
}

#[test]
fn ep_greater_than_tp_counts_max_chips() {
    let mut m = MetricsCollector::new();
    m.sim_duration = 3600.0;
    m.tokens_generated = 1_000_000;
    m.completions = 1000;
    // tp=1, ep=4 → max=4 chips, $3.50/hr each, 1 hour → $14
    let s = m.to_summary(
        "m", "g", "sched", 1, 1, 4, false, 0.0, "roofline", 0.0, 3.50,
    );
    assert!((s.total_cost_usd - 14.0).abs() < 0.01);
    assert!((s.cluster_cost_per_hour_usd - 14.0).abs() < 0.01);
}

#[test]
fn ep_equal_tp_doesnt_double_count() {
    let mut m = MetricsCollector::new();
    m.sim_duration = 3600.0;
    m.tokens_generated = 1_000_000;
    m.completions = 1000;
    // tp=8, ep=8 → max=8 chips (same set), not 64
    let s = m.to_summary(
        "m", "g", "sched", 8, 1, 8, false, 0.0, "roofline", 0.0, 3.50,
    );
    assert!((s.total_cost_usd - 28.0).abs() < 0.01);
}

#[test]
fn zero_price_disables_cost_reporting() {
    let mut m = MetricsCollector::new();
    m.sim_duration = 60.0;
    m.tokens_generated = 10_000;
    m.completions = 100;
    let s = m.to_summary("m", "g", "sched", 1, 1, 1, false, 0.0, "roofline", 0.0, 0.0);
    assert_eq!(s.total_cost_usd, 0.0);
    assert_eq!(s.cost_per_million_tokens_usd, 0.0);
}
