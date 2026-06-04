//! CLI coverage for study JSON and HTML report generation.

use std::fs;
use std::process::Command;

#[test]
fn study_mode_writes_json_and_html_report() {
    let dir = tempfile::tempdir().unwrap();
    let json_path = dir.path().join("reports").join("study.json");
    let html_path = dir.path().join("reports").join("study.html");

    let output = Command::new(env!("CARGO_BIN_EXE_tokenmill"))
        .args([
            "--study-models",
            "llama-8b-fp8,llama-8b-w4a8kv4",
            "--study-gpus",
            "h100,h200",
            "--study-tps",
            "1",
            "--study-arrival-rates",
            "1",
            "--duration",
            "2",
            "--prompt-mean",
            "64",
            "--output-mean",
            "16",
            "--json-out",
            json_path.to_str().unwrap(),
            "--html",
            html_path.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run tokenmill");

    assert!(
        output.status.success(),
        "tokenmill failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let json = fs::read_to_string(&json_path).unwrap();
    let rows: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(rows.as_array().unwrap().len(), 4);
    assert!(json.contains("llama-8b-fp8"));
    assert!(json.contains("H200-SXM5"));

    let html = fs::read_to_string(&html_path).unwrap();
    assert!(html.contains("tokenmill study report"));
    assert!(html.contains("Cost vs TPOT p95"));
    assert!(html.contains("llama-8b-w4a8kv4"));
}
