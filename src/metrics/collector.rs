use hdrhistogram::Histogram;

use super::report::RunSummary;
use crate::hardware::power;

pub struct MetricsCollector {
    ttft: Histogram<u64>,        // microseconds — arrival → first output token
    prefill_lat: Histogram<u64>, // microseconds — arrival → prefill done
    kv_transfer: Histogram<u64>, // microseconds — KV transfer time (disaggregated only)
    tpot: Histogram<u64>,        // microseconds
    pub completions: u64,
    pub tokens_generated: u64,
    pub preemptions: u64,
    pub sim_duration: f64,
    kv_util_sum: f64,
    kv_util_samples: u64,
    pub kv_util_final: f64,
    pub disaggregated: bool,
    /// Sum across the prefill pool of seconds the GPU(s) were doing prefill compute.
    /// Multiplied by `n_prefill_chips` already if more than one prefill GPU runs in parallel.
    pub prefill_busy_secs: f64,
    /// Same for decode. In aggregated mode this counts the single GPU's decode time.
    pub decode_busy_secs: f64,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            ttft: Histogram::new(3).unwrap(),
            prefill_lat: Histogram::new(3).unwrap(),
            kv_transfer: Histogram::new(3).unwrap(),
            tpot: Histogram::new(3).unwrap(),
            completions: 0,
            tokens_generated: 0,
            preemptions: 0,
            sim_duration: 0.0,
            kv_util_sum: 0.0,
            kv_util_samples: 0,
            kv_util_final: 0.0,
            disaggregated: false,
            prefill_busy_secs: 0.0,
            decode_busy_secs: 0.0,
        }
    }

    pub fn add_prefill_busy(&mut self, secs: f64) {
        if secs > 0.0 {
            self.prefill_busy_secs += secs;
        }
    }

    pub fn add_decode_busy(&mut self, secs: f64) {
        if secs > 0.0 {
            self.decode_busy_secs += secs;
        }
    }

    pub fn record_ttft(&mut self, secs: f64) {
        let _ = self.ttft.record((secs * 1_000_000.0).max(1.0) as u64);
    }

    pub fn record_prefill_latency(&mut self, secs: f64) {
        let _ = self
            .prefill_lat
            .record((secs * 1_000_000.0).max(1.0) as u64);
    }

    pub fn record_kv_transfer(&mut self, secs: f64) {
        if secs > 0.0 {
            let _ = self
                .kv_transfer
                .record((secs * 1_000_000.0).max(1.0) as u64);
        }
    }

    pub fn record_tpot(&mut self, secs: f64) {
        let _ = self.tpot.record((secs * 1_000_000.0).max(1.0) as u64);
    }

    pub fn record_completion(&mut self, output_tokens: u32) {
        self.completions += 1;
        self.tokens_generated += output_tokens as u64;
    }

    pub fn record_preemption(&mut self) {
        self.preemptions += 1;
    }

    pub fn record_kv_util(&mut self, util: f64) {
        self.kv_util_sum += util;
        self.kv_util_samples += 1;
    }

    pub fn kv_util_mean(&self) -> f64 {
        if self.kv_util_samples == 0 {
            0.0
        } else {
            self.kv_util_sum / self.kv_util_samples as f64
        }
    }

    fn pct(h: &Histogram<u64>, q: f64) -> f64 {
        h.value_at_quantile(q) as f64 / 1000.0 // µs → ms
    }

    #[allow(clippy::too_many_arguments)]
    pub fn to_summary(
        &self,
        model: &str,
        gpu: &str,
        scheduler: &str,
        tp: u32,
        pp: u32,
        disaggregate: bool,
        arrival_rate: f64,
        latency_mode: &str,
        tdp_watts: f64,
    ) -> RunSummary {
        let throughput = self.completions as f64 / self.sim_duration.max(1e-9);
        let tok_throughput = self.tokens_generated as f64 / self.sim_duration.max(1e-9);

        // Total chips = TP × PP per pool, doubled if prefill / decode pools are disaggregated.
        let pools = if disaggregate { 2u32 } else { 1 };
        let n_chips = tp.saturating_mul(pp).saturating_mul(pools);
        let pr = power::compute_report(
            self.prefill_busy_secs,
            self.decode_busy_secs,
            self.sim_duration,
            n_chips,
            tdp_watts,
            self.tokens_generated,
            self.completions,
        );
        RunSummary {
            model: model.into(),
            gpu: gpu.into(),
            scheduler: scheduler.into(),
            tp,
            pp,
            disaggregate,
            arrival_rate,
            duration_s: self.sim_duration,
            latency_mode: latency_mode.into(),
            completions: self.completions,
            preemptions: self.preemptions,
            throughput_rps: throughput,
            token_throughput: tok_throughput,
            kv_util_mean_pct: self.kv_util_mean() * 100.0,
            ttft_p50_ms: Self::pct(&self.ttft, 0.50),
            ttft_p95_ms: Self::pct(&self.ttft, 0.95),
            ttft_p99_ms: Self::pct(&self.ttft, 0.99),
            prefill_p50_ms: Self::pct(&self.prefill_lat, 0.50),
            prefill_p95_ms: Self::pct(&self.prefill_lat, 0.95),
            prefill_p99_ms: Self::pct(&self.prefill_lat, 0.99),
            kv_transfer_p50_ms: Self::pct(&self.kv_transfer, 0.50),
            kv_transfer_p95_ms: Self::pct(&self.kv_transfer, 0.95),
            kv_transfer_p99_ms: Self::pct(&self.kv_transfer, 0.99),
            tpot_p50_ms: Self::pct(&self.tpot, 0.50),
            tpot_p95_ms: Self::pct(&self.tpot, 0.95),
            tpot_p99_ms: Self::pct(&self.tpot, 0.99),
            total_energy_kj: pr.total_energy_j / 1000.0,
            mean_power_kw: pr.mean_power_w / 1000.0,
            energy_per_token_mj: pr.energy_per_token_mj,
            energy_per_request_j: pr.energy_per_request_j,
        }
    }

    pub fn print_report(&self) {
        let throughput = self.completions as f64 / self.sim_duration.max(1e-9);
        let tok_throughput = self.tokens_generated as f64 / self.sim_duration.max(1e-9);

        println!("=== tokenmill results ===");
        println!("Requests completed : {}", self.completions);
        println!("Throughput         : {:.2} req/s", throughput);
        println!("Token throughput   : {:.0} tok/s", tok_throughput);
        println!(
            "KV utilization     : mean={:.1}%  final={:.1}%",
            self.kv_util_mean() * 100.0,
            self.kv_util_final * 100.0
        );
        println!();

        if !self.ttft.is_empty() {
            println!(
                "TTFT (ms)           p50={:.1}  p95={:.1}  p99={:.1}",
                Self::pct(&self.ttft, 0.50),
                Self::pct(&self.ttft, 0.95),
                Self::pct(&self.ttft, 0.99)
            );
        }

        if !self.prefill_lat.is_empty() {
            println!(
                "Prefill time (ms)   p50={:.1}  p95={:.1}  p99={:.1}",
                Self::pct(&self.prefill_lat, 0.50),
                Self::pct(&self.prefill_lat, 0.95),
                Self::pct(&self.prefill_lat, 0.99)
            );
        }

        if self.disaggregated && !self.kv_transfer.is_empty() {
            println!(
                "KV transfer (ms)    p50={:.1}  p95={:.1}  p99={:.1}",
                Self::pct(&self.kv_transfer, 0.50),
                Self::pct(&self.kv_transfer, 0.95),
                Self::pct(&self.kv_transfer, 0.99)
            );
        }

        if !self.tpot.is_empty() {
            println!(
                "TPOT (ms)           p50={:.1}  p95={:.1}  p99={:.1}",
                Self::pct(&self.tpot, 0.50),
                Self::pct(&self.tpot, 0.95),
                Self::pct(&self.tpot, 0.99)
            );
        }
    }
}
