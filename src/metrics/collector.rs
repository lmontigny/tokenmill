use hdrhistogram::Histogram;

pub struct MetricsCollector {
    ttft: Histogram<u64>,           // microseconds — arrival → first output token
    prefill_lat: Histogram<u64>,    // microseconds — arrival → prefill done
    kv_transfer: Histogram<u64>,    // microseconds — KV transfer time (disaggregated only)
    tpot: Histogram<u64>,           // microseconds
    pub completions: u64,
    pub tokens_generated: u64,
    pub sim_duration: f64,
    kv_util_sum: f64,
    kv_util_samples: u64,
    pub kv_util_final: f64,
    pub disaggregated: bool,
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
            sim_duration: 0.0,
            kv_util_sum: 0.0,
            kv_util_samples: 0,
            kv_util_final: 0.0,
            disaggregated: false,
        }
    }

    pub fn record_ttft(&mut self, secs: f64) {
        let _ = self.ttft.record((secs * 1_000_000.0).max(1.0) as u64);
    }

    pub fn record_prefill_latency(&mut self, secs: f64) {
        let _ = self.prefill_lat.record((secs * 1_000_000.0).max(1.0) as u64);
    }

    pub fn record_kv_transfer(&mut self, secs: f64) {
        if secs > 0.0 {
            let _ = self.kv_transfer.record((secs * 1_000_000.0).max(1.0) as u64);
        }
    }

    pub fn record_tpot(&mut self, secs: f64) {
        let _ = self.tpot.record((secs * 1_000_000.0).max(1.0) as u64);
    }

    pub fn record_completion(&mut self, output_tokens: u32) {
        self.completions += 1;
        self.tokens_generated += output_tokens as u64;
    }

    pub fn record_kv_util(&mut self, util: f64) {
        self.kv_util_sum += util;
        self.kv_util_samples += 1;
    }

    pub fn kv_util_mean(&self) -> f64 {
        if self.kv_util_samples == 0 { 0.0 } else { self.kv_util_sum / self.kv_util_samples as f64 }
    }

    fn pct(h: &Histogram<u64>, q: f64) -> f64 {
        h.value_at_quantile(q) as f64 / 1000.0 // µs → ms
    }

    pub fn print_report(&self) {
        let throughput = self.completions as f64 / self.sim_duration.max(1e-9);
        let tok_throughput = self.tokens_generated as f64 / self.sim_duration.max(1e-9);

        println!("=== inference-sim results ===");
        println!("Requests completed : {}", self.completions);
        println!("Throughput         : {:.2} req/s", throughput);
        println!("Token throughput   : {:.0} tok/s", tok_throughput);
        println!("KV utilization     : mean={:.1}%  final={:.1}%",
            self.kv_util_mean() * 100.0, self.kv_util_final * 100.0);
        println!();

        if self.ttft.len() > 0 {
            println!("TTFT (ms)           p50={:.1}  p95={:.1}  p99={:.1}",
                Self::pct(&self.ttft, 0.50),
                Self::pct(&self.ttft, 0.95),
                Self::pct(&self.ttft, 0.99));
        }

        if self.prefill_lat.len() > 0 {
            println!("Prefill time (ms)   p50={:.1}  p95={:.1}  p99={:.1}",
                Self::pct(&self.prefill_lat, 0.50),
                Self::pct(&self.prefill_lat, 0.95),
                Self::pct(&self.prefill_lat, 0.99));
        }

        if self.disaggregated && self.kv_transfer.len() > 0 {
            println!("KV transfer (ms)    p50={:.1}  p95={:.1}  p99={:.1}",
                Self::pct(&self.kv_transfer, 0.50),
                Self::pct(&self.kv_transfer, 0.95),
                Self::pct(&self.kv_transfer, 0.99));
        }

        if self.tpot.len() > 0 {
            println!("TPOT (ms)           p50={:.1}  p95={:.1}  p99={:.1}",
                Self::pct(&self.tpot, 0.50),
                Self::pct(&self.tpot, 0.95),
                Self::pct(&self.tpot, 0.99));
        }
    }
}
