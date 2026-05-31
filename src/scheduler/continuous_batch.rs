use crate::workload::request::{InferenceRequest, RequestState};

pub struct BatchDecision {
    pub admit: Vec<u64>,    // req_ids to move from waiting → prefilling
    pub decode: Vec<u64>,   // req_ids already in decode, continue
}

pub struct ContinuousBatchScheduler {
    pub max_batch_tokens: u32,
}

impl ContinuousBatchScheduler {
    pub fn new(max_batch_tokens: u32) -> Self {
        Self { max_batch_tokens }
    }

    pub fn schedule(
        &self,
        waiting: &[InferenceRequest],
        running: &[RequestState],
        _now: f64,
    ) -> BatchDecision {
        // Count tokens currently in flight
        let in_flight: u32 = running
            .iter()
            .map(|s| s.req.prompt_tokens + s.req.max_output_tokens)
            .sum();

        let mut admit = Vec::new();
        let mut budget = self.max_batch_tokens.saturating_sub(in_flight);

        for req in waiting {
            let needed = req.prompt_tokens + req.max_output_tokens;
            if needed <= budget {
                admit.push(req.req_id);
                budget -= needed;
            }
        }

        let decode = running.iter().map(|s| s.req.req_id).collect();

        BatchDecision { admit, decode }
    }
}
