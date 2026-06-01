use crate::workload::request::{InferenceRequest, RequestPhase, RequestState};

pub struct ChunkedBatchDecision {
    /// Requests newly admitted this tick, paired with how many tokens to prefill first.
    pub admit: Vec<(u64, u32)>,
    /// Requests to preempt (KV pressure).
    pub preempt: Vec<u64>,
}

/// Sarathi-style chunked prefill: splits long prefills into `chunk_size`-token slices
/// so decode requests are not head-of-line blocked by a single long prefill.
pub struct ChunkedPrefillScheduler {
    pub chunk_size: u32,
    pub max_batch_tokens: u32,
}

impl ChunkedPrefillScheduler {
    pub fn new(chunk_size: u32, max_batch_tokens: u32) -> Self {
        Self { chunk_size, max_batch_tokens }
    }

    pub fn schedule(
        &self,
        waiting: &[InferenceRequest],
        running: &[RequestState],
        kv_free_blocks: u32,
        kv_block_size: u32,
        _now: f64,
    ) -> ChunkedBatchDecision {
        // Preempt largest decode request when KV is fully exhausted.
        if kv_free_blocks == 0 && !waiting.is_empty() {
            if let Some(victim) = running.iter()
                .filter(|s| matches!(s.phase, RequestPhase::Decoding))
                .max_by_key(|s| s.req.prompt_tokens + s.req.max_output_tokens)
            {
                return ChunkedBatchDecision { admit: vec![], preempt: vec![victim.req.req_id] };
            }
        }

        // Token budget split: reserve half for decode, half for prefill chunks.
        let decode_tokens: u32 = running.len() as u32; // 1 token per running req per step
        let prefill_budget = self.max_batch_tokens.saturating_sub(decode_tokens);

        let mut admit = Vec::new();
        let mut token_budget = prefill_budget;
        let mut kv_budget = kv_free_blocks;

        for req in waiting {
            if token_budget == 0 {
                break;
            }
            let kv_needed = req.prompt_tokens.div_ceil(kv_block_size);
            if kv_needed > kv_budget {
                continue;
            }
            // Admit with at most chunk_size tokens in this iteration
            let chunk = req.prompt_tokens.min(self.chunk_size).min(token_budget);
            admit.push((req.req_id, chunk));
            token_budget = token_budget.saturating_sub(chunk);
            kv_budget -= kv_needed;
        }

        ChunkedBatchDecision { admit, preempt: Vec::new() }
    }
}
