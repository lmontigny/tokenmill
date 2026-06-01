//! Speculative decoding and multi-token prediction math.

use inference_sim::engine::sim::{MtpConfig, SpecConfig};
use inference_sim::model::llm_config::LlmConfig;

fn close(actual: f64, expected: f64, tol: f64) -> bool {
    (actual - expected).abs() <= tol
}

#[test]
fn spec_tokens_per_step_formula() {
    // E[tok/step] = (1 - γ^(K+1)) / (1 - γ)
    // K=3, γ=0.7 → (1 - 0.7^4) / (1 - 0.7) = (1 - 0.2401) / 0.3 = 2.533
    let draft = LlmConfig::preset("llama-8b").unwrap();
    let spec = SpecConfig {
        draft_tokens: 3,
        acceptance_rate: 0.7,
        draft_model: draft,
    };
    assert!(
        close(spec.tokens_per_step(), 2.533, 0.01),
        "expected 2.533, got {:.4}",
        spec.tokens_per_step()
    );
}

#[test]
fn spec_tokens_at_zero_acceptance_is_one() {
    let draft = LlmConfig::preset("llama-8b").unwrap();
    let spec = SpecConfig {
        draft_tokens: 4,
        acceptance_rate: 0.0,
        draft_model: draft,
    };
    // No draft tokens accepted → still one main-model token committed per step.
    assert!(
        close(spec.tokens_per_step(), 1.0, 1e-6),
        "expected 1.0, got {}",
        spec.tokens_per_step()
    );
}

#[test]
fn mtp_tokens_per_step_formula() {
    // Same formula as spec but with main-model heads. K=3, γ=0.9 → 3.439
    let mtp = MtpConfig {
        num_heads: 3,
        acceptance_rate: 0.9,
    };
    assert!(
        close(mtp.tokens_per_step(), 3.439, 0.01),
        "expected 3.439, got {:.4}",
        mtp.tokens_per_step()
    );
}

#[test]
fn mtp_overhead_fraction() {
    // overhead = K / n_layers. With K=3 heads on a 32-layer model → 3/32 = 0.09375
    let mtp = MtpConfig {
        num_heads: 3,
        acceptance_rate: 0.9,
    };
    assert!(close(mtp.overhead_fraction(32), 3.0 / 32.0, 1e-9));
}

#[test]
fn spec_high_acceptance_approaches_k_plus_one() {
    // At γ near 1, expected tokens per step approaches K + 1.
    let draft = LlmConfig::preset("llama-8b").unwrap();
    let spec = SpecConfig {
        draft_tokens: 4,
        acceptance_rate: 0.99,
        draft_model: draft,
    };
    let t = spec.tokens_per_step();
    assert!(t > 4.8 && t < 5.0, "expected ~5.0, got {:.3}", t);
}
