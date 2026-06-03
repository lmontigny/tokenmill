//! End-to-end power-reporting checks through `MetricsCollector::to_summary`.

use tokenmill::hardware::power;
use tokenmill::metrics::collector::MetricsCollector;

#[test]
fn summary_carries_energy_when_tdp_is_set() {
    let mut m = MetricsCollector::new();
    m.sim_duration = 60.0;
    m.tokens_generated = 12_000;
    m.completions = 100;
    m.add_prefill_busy(15.0);
    m.add_decode_busy(30.0);

    // 1 GPU at 700 W TDP. Idle = 60 - 15 - 30 = 15 s.
    // Energy = 15·700·0.90 + 30·700·0.65 + 15·700·0.35 = 9450 + 13650 + 3675 = 26 775 J.
    let s = m.to_summary(
        "m", "g", "sched", 1, 1, 1, false, 5.0, "roofline", 700.0, 0.0,
    );
    assert!(
        (s.total_energy_kj - 26.775).abs() < 0.01,
        "expected 26.775 kJ, got {:.3}",
        s.total_energy_kj
    );
    assert!((s.mean_power_kw - 26.775 / 60.0).abs() < 0.01);
    assert!(s.energy_per_token_mj > 0.0);
    assert!(s.energy_per_request_j > 0.0);
}

#[test]
fn summary_zero_energy_when_tdp_is_zero() {
    let mut m = MetricsCollector::new();
    m.sim_duration = 60.0;
    m.add_prefill_busy(10.0);
    m.add_decode_busy(10.0);
    let s = m.to_summary("m", "g", "sched", 1, 1, 1, false, 5.0, "roofline", 0.0, 0.0);
    assert_eq!(s.total_energy_kj, 0.0);
    assert_eq!(s.mean_power_kw, 0.0);
}

#[test]
fn disaggregated_doubles_chip_count_for_idle_accounting() {
    let mut m = MetricsCollector::new();
    m.sim_duration = 60.0;
    // Two pools of 4 GPUs each, both fully idle.
    // Idle energy = 60 × 8 × 1000 × 0.35 = 168 000 J = 168 kJ.
    let s = m.to_summary(
        "m", "g", "sched", 4, 1, 1, true, 0.0, "roofline", 1000.0, 0.0,
    );
    assert!(
        (s.total_energy_kj - 168.0).abs() < 0.1,
        "expected 168 kJ idle, got {:.2}",
        s.total_energy_kj
    );
}

#[test]
fn coefficients_are_monotonic() {
    // Idle < decode < prefill, as a basic sanity check.
    const _: () = assert!(power::IDLE_COEF < power::DECODE_COEF);
    const _: () = assert!(power::DECODE_COEF < power::PREFILL_COEF);
}
