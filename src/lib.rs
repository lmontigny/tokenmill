//! tokenmill — discrete-event simulator for LLM inference workloads.
//!
//! The binary CLI lives in `main.rs`; this lib.rs exists so integration tests
//! under `tests/` (and downstream tooling) can pull in the same modules.

pub mod engine;
pub mod hardware;
pub mod metrics;
pub mod model;
pub mod scheduler;
pub mod workload;
