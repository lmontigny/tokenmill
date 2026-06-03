//! Regression fixtures for the public B200 FP4 validation in
//! `docs/quantization-validation.md`.
//!
//! These tests intentionally use pinned public benchmark numbers instead of
//! fetching vendor docs during CI.

use std::process::Command;

use serde_json::Value;

struct Fixture {
    name: &'static str,
    prompt_mean: &'static str,
    output_mean: &'static str,
    max_batch_tokens: &'static str,
    arrival_rate: &'static str,
    public_tok_s_per_gpu: f64,
    rel_tol: f64,
}

fn close(actual: f64, expected: f64, rel_tol: f64) -> bool {
    (actual - expected).abs() / expected.abs() <= rel_tol
}

fn run_fixture(f: &Fixture) -> f64 {
    let output = Command::new(env!("CARGO_BIN_EXE_tokenmill"))
        .args([
            "--model",
            "llama-70b-w4a8kv4",
            "--gpu",
            "b200",
            "--scheduler",
            "chunked-prefill",
            "--max-batch-tokens",
            f.max_batch_tokens,
            "--prompt-mean",
            f.prompt_mean,
            "--output-mean",
            f.output_mean,
            "--duration",
            "60",
            "--arrival-rate",
            f.arrival_rate,
            "--output",
            "json",
        ])
        .output()
        .expect("failed to run tokenmill");

    assert!(
        output.status.success(),
        "tokenmill failed for {}:\nstdout:\n{}\nstderr:\n{}",
        f.name,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let summary: Value =
        serde_json::from_slice(&output.stdout).expect("tokenmill did not emit valid JSON");
    summary["token_throughput"]
        .as_f64()
        .expect("summary missing token_throughput")
}

#[test]
fn b200_fp4_llama70b_matches_public_trtllm_throughput_regime() {
    // Public TensorRT-LLM table for Llama 3.3 70B FP4 on B200 reports output
    // tokens per second per GPU. The simulator runs a deterministic synthetic
    // workload whose output throughput is comparable for single-GPU fixtures.
    let fixtures = [
        Fixture {
            name: "1K/1K",
            prompt_mean: "1024",
            output_mean: "1024",
            max_batch_tokens: "8192",
            arrival_rate: "8",
            public_tok_s_per_gpu: 6_920.0,
            rel_tol: 0.10,
        },
        Fixture {
            name: "8K/1K",
            prompt_mean: "8192",
            output_mean: "1024",
            max_batch_tokens: "32768",
            arrival_rate: "1",
            public_tok_s_per_gpu: 1_362.0,
            rel_tol: 0.35,
        },
        Fixture {
            name: "32K/1K",
            prompt_mean: "32768",
            output_mean: "1024",
            max_batch_tokens: "32768",
            arrival_rate: "0.25",
            public_tok_s_per_gpu: 274.0,
            rel_tol: 0.15,
        },
    ];

    for fixture in fixtures {
        let actual = run_fixture(&fixture);
        assert!(
            close(actual, fixture.public_tok_s_per_gpu, fixture.rel_tol),
            "{}: expected {:.0} tok/s/GPU +/- {:.0}%, got {:.1}",
            fixture.name,
            fixture.public_tok_s_per_gpu,
            fixture.rel_tol * 100.0,
            actual
        );
    }
}
