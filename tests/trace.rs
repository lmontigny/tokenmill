//! Trace replay: native CSV and Azure datetime auto-detection.

use std::io::Write;

use tokenmill::workload::trace_replay::TraceReplay;

fn write_temp(name: &str, contents: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(contents.as_bytes()).unwrap();
    path
}

#[test]
fn native_csv_loads() {
    let csv = "timestamp_ms,prompt_tokens,output_tokens\n\
               0.0,512,128\n\
               1000.0,1024,64\n\
               2500.5,256,32\n";
    let path = write_temp("tokenmill_test_native.csv", csv);
    let trace = TraceReplay::from_csv(&path).unwrap();
    assert_eq!(trace.len(), 3);
}

#[test]
fn azure_csv_loads_and_normalises() {
    let csv = "TIMESTAMP,ContextTokens,GeneratedTokens\n\
               2023-11-16 18:17:03.000,4808,10\n\
               2023-11-16 18:17:04.500,256,128\n";
    let path = write_temp("tokenmill_test_azure.csv", csv);
    let trace = TraceReplay::from_csv(&path).unwrap();
    assert_eq!(trace.len(), 2);
}

#[test]
fn trace_skips_comment_lines() {
    let csv = "# this is a comment\n\
               timestamp_ms,prompt_tokens,output_tokens\n\
               # so is this\n\
               0.0,512,128\n";
    let path = write_temp("tokenmill_test_comments.csv", csv);
    let trace = TraceReplay::from_csv(&path).unwrap();
    assert_eq!(trace.len(), 1);
}

#[test]
fn empty_trace_loads() {
    let csv = "timestamp_ms,prompt_tokens,output_tokens\n";
    let path = write_temp("tokenmill_test_empty.csv", csv);
    let trace = TraceReplay::from_csv(&path).unwrap();
    assert_eq!(trace.len(), 0);
}
