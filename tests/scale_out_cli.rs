//! CLI coverage for named scale-out fabric presets.

use std::process::Command;

use serde_json::Value;

fn run_tokenmill(args: &[&str]) -> Value {
    let output = Command::new(env!("CARGO_BIN_EXE_tokenmill"))
        .args(args)
        .output()
        .expect("failed to run tokenmill");

    assert!(
        output.status.success(),
        "tokenmill failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    serde_json::from_slice(&output.stdout).expect("tokenmill did not emit valid JSON")
}

#[test]
fn quantum_x800_preset_matches_manual_800g_values() {
    let common = [
        "--gpu",
        "b200",
        "--model",
        "llama-70b-fp8",
        "--tp",
        "144",
        "--gpus-per-node",
        "72",
        "--duration",
        "2",
        "--arrival-rate",
        "1",
        "--prompt-mean",
        "128",
        "--output-mean",
        "32",
        "--output",
        "json",
    ];
    let preset = run_tokenmill(
        &common
            .iter()
            .copied()
            .chain(["--scale-out-fabric", "quantum-x800"])
            .collect::<Vec<_>>(),
    );
    let manual = run_tokenmill(
        &common
            .iter()
            .copied()
            .chain(["--scale-out-bw-gbps", "100", "--scale-out-latency-us", "2"])
            .collect::<Vec<_>>(),
    );

    assert_eq!(preset["token_throughput"], manual["token_throughput"]);
    assert_eq!(preset["tpot_p50_ms"], manual["tpot_p50_ms"]);
    assert_eq!(preset["ttft_p50_ms"], manual["ttft_p50_ms"]);
}

#[test]
fn dgx_h200_system_preset_resolves_to_h200_gpu() {
    let summary = run_tokenmill(&[
        "--system",
        "dgx-h200",
        "--model",
        "llama-70b-fp8",
        "--duration",
        "3",
        "--arrival-rate",
        "1",
        "--prompt-mean",
        "128",
        "--output-mean",
        "32",
        "--output",
        "json",
    ]);

    assert_eq!(summary["gpu"], "H200-SXM5");
    assert_eq!(summary["cluster_cost_per_hour_usd"], 4.5);
}

#[test]
fn dgx_b200_system_preset_matches_manual_b200_ndr_node_defaults() {
    let common = [
        "--model",
        "llama-70b-fp8",
        "--tp",
        "16",
        "--duration",
        "2",
        "--arrival-rate",
        "1",
        "--prompt-mean",
        "128",
        "--output-mean",
        "32",
        "--output",
        "json",
    ];
    let system = run_tokenmill(
        &common
            .iter()
            .copied()
            .chain(["--system", "dgx-b200"])
            .collect::<Vec<_>>(),
    );
    let manual = run_tokenmill(
        &common
            .iter()
            .copied()
            .chain([
                "--gpu",
                "b200",
                "--gpus-per-node",
                "8",
                "--scale-out-fabric",
                "ndr-400",
            ])
            .collect::<Vec<_>>(),
    );

    assert_eq!(system["gpu"], "B200-SXM");
    assert_eq!(system["token_throughput"], manual["token_throughput"]);
    assert_eq!(system["tpot_p50_ms"], manual["tpot_p50_ms"]);
}
