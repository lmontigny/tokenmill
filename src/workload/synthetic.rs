use rand::SeedableRng;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Exp, LogNormal};

use crate::engine::event::SimTime;

use super::request::InferenceRequest;

pub struct SyntheticWorkload {
    arrival_rate: f64,
    prompt_dist: LogNormal<f64>,
    output_dist: LogNormal<f64>,
    rng: SmallRng,
    clock: SimTime,
    next_id: u64,
    end_time: SimTime,
}

impl SyntheticWorkload {
    /// arrival_rate: requests/second. prompt_mean/output_mean: tokens (log-normal mean).
    pub fn new(arrival_rate: f64, prompt_mean: f64, output_mean: f64, end_time: SimTime, seed: u64) -> Self {
        // LogNormal params: mu/sigma in log-space. We derive from desired mean.
        // mean = exp(mu + sigma²/2). For sigma=0.5: mu = ln(mean) - 0.125
        let sigma = 0.5_f64;
        let prompt_mu = prompt_mean.ln() - sigma * sigma / 2.0;
        let output_mu = output_mean.ln() - sigma * sigma / 2.0;
        Self {
            arrival_rate,
            prompt_dist: LogNormal::new(prompt_mu, sigma).unwrap(),
            output_dist: LogNormal::new(output_mu, sigma).unwrap(),
            rng: SmallRng::seed_from_u64(seed),
            clock: 0.0,
            next_id: 0,
            end_time,
        }
    }

    pub fn next_arrival(&mut self) -> Option<(SimTime, InferenceRequest)> {
        let gap = Exp::new(self.arrival_rate).unwrap().sample(&mut self.rng);
        self.clock += gap;
        if self.clock > self.end_time {
            return None;
        }
        let prompt_tokens = (self.prompt_dist.sample(&mut self.rng) as u32).max(1);
        let output_tokens = (self.output_dist.sample(&mut self.rng) as u32).max(1);
        let req = InferenceRequest {
            req_id: self.next_id,
            prompt_tokens,
            max_output_tokens: output_tokens,
            arrival_time: self.clock,
        };
        self.next_id += 1;
        Some((self.clock, req))
    }
}
