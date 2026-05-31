use crate::workload::request::{InferenceRequest, RequestState};

pub struct BatchDecision {
    pub admit: Vec<u64>,    // req_ids to move from waiting → prefilling
    pub preempt: Vec<u64>,  // req_ids to evict from running (KV pressure)
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
        kv_free_blocks: u32,
        kv_block_size: u32,
        _now: f64,
    ) -> BatchDecision {
        let in_flight: u32 = running
            .iter()
            .map(|s| s.req.prompt_tokens + s.req.max_output_tokens)
            .sum();

        let mut admit = Vec::new();
        let mut token_budget = self.max_batch_tokens.saturating_sub(in_flight);
        let mut kv_budget = kv_free_blocks;

        for req in waiting {
            let tokens = req.prompt_tokens + req.max_output_tokens;
            let kv_blocks_needed = tokens.div_ceil(kv_block_size);
            if tokens <= token_budget && kv_blocks_needed <= kv_budget {
                admit.push(req.req_id);
                token_budget -= tokens;
                kv_budget -= kv_blocks_needed;
            }
        }

        BatchDecision { admit, preempt: Vec::new() }
    }
}
