use hdrhistogram::Histogram;

pub struct MetricsCollector {
    ttft: Histogram<u64>,   // microseconds
    tpot: Histogram<u64>,   // microseconds
    pub completions: u64,
    pub tokens_generated: u64,
    pub sim_duration: f64,
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            ttft: Histogram::new(3).unwrap(),
            tpot: Histogram::new(3).unwrap(),
            completions: 0,
            tokens_generated: 0,
            sim_duration: 0.0,
        }
    }

    pub fn record_ttft(&mut self, secs: f64) {
        let us = (secs * 1_000_000.0) as u64;
        let _ = self.ttft.record(us.max(1));
    }

    pub fn record_tpot(&mut self, secs: f64) {
        let us = (secs * 1_000_000.0) as u64;
        let _ = self.tpot.record(us.max(1));
    }

    pub fn record_completion(&mut self, output_tokens: u32) {
        self.completions += 1;
        self.tokens_generated += output_tokens as u64;
    }

    pub fn print_report(&self) {
        let throughput = self.completions as f64 / self.sim_duration.max(1e-9);
        let tok_throughput = self.tokens_generated as f64 / self.sim_duration.max(1e-9);

        println!("=== inference-sim results ===");
        println!("Requests completed : {}", self.completions);
        println!("Throughput         : {:.1} req/s", throughput);
        println!("Token throughput   : {:.0} tok/s", tok_throughput);
        println!();

        if self.ttft.len() > 0 {
            println!("TTFT (ms)");
            println!("  p50 : {:.1}", self.ttft.value_at_quantile(0.50) as f64 / 1000.0);
            println!("  p95 : {:.1}", self.ttft.value_at_quantile(0.95) as f64 / 1000.0);
            println!("  p99 : {:.1}", self.ttft.value_at_quantile(0.99) as f64 / 1000.0);
        }

        if self.tpot.len() > 0 {
            println!("TPOT (ms)");
            println!("  p50 : {:.1}", self.tpot.value_at_quantile(0.50) as f64 / 1000.0);
            println!("  p95 : {:.1}", self.tpot.value_at_quantile(0.95) as f64 / 1000.0);
            println!("  p99 : {:.1}", self.tpot.value_at_quantile(0.99) as f64 / 1000.0);
        }
    }
}
