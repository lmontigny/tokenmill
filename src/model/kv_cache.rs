use rustc_hash::FxHashMap;

#[derive(Debug)]
pub enum KvError {
    OutOfMemory { needed: u32, free: u32 },
    UnknownRequest(u64),
}

impl std::fmt::Display for KvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KvError::OutOfMemory { needed, free } => {
                write!(f, "KV OOM: need {} blocks, only {} free", needed, free)
            }
            KvError::UnknownRequest(id) => write!(f, "unknown request {}", id),
        }
    }
}

pub struct KvCacheManager {
    pub block_size: u32, // tokens per block
    pub total_blocks: u32,
    pub free_blocks: u32,
    allocated: FxHashMap<u64, u32>, // req_id → blocks currently held
}

impl KvCacheManager {
    pub fn new(total_blocks: u32, block_size: u32) -> Self {
        Self {
            block_size,
            total_blocks,
            free_blocks: total_blocks,
            allocated: FxHashMap::default(),
        }
    }

    /// Allocate blocks for `n_tokens` for a new request. Fails if not enough free.
    pub fn allocate(&mut self, req_id: u64, n_tokens: u32) -> Result<(), KvError> {
        let n_blocks = n_tokens.div_ceil(self.block_size);
        if n_blocks > self.free_blocks {
            return Err(KvError::OutOfMemory {
                needed: n_blocks,
                free: self.free_blocks,
            });
        }
        *self.allocated.entry(req_id).or_insert(0) += n_blocks;
        self.free_blocks -= n_blocks;
        Ok(())
    }

    /// Grow allocation for a running request by `n_new_tokens`.
    pub fn grow(&mut self, req_id: u64, n_new_tokens: u32) -> Result<(), KvError> {
        let current_tokens = self.allocated.get(&req_id).copied().unwrap_or(0) * self.block_size;
        let total_tokens = current_tokens + n_new_tokens;
        let new_blocks_needed = total_tokens.div_ceil(self.block_size);
        let current_blocks = self.allocated.get(&req_id).copied().unwrap_or(0);
        let extra = new_blocks_needed.saturating_sub(current_blocks);
        if extra > self.free_blocks {
            return Err(KvError::OutOfMemory {
                needed: extra,
                free: self.free_blocks,
            });
        }
        *self.allocated.entry(req_id).or_insert(0) += extra;
        self.free_blocks -= extra;
        Ok(())
    }

    /// Release all blocks held by a completed or preempted request.
    pub fn free(&mut self, req_id: u64) {
        if let Some(blocks) = self.allocated.remove(&req_id) {
            self.free_blocks += blocks;
        }
    }

    /// Blocks needed to hold `n_tokens`.
    pub fn blocks_for(&self, n_tokens: u32) -> u32 {
        n_tokens.div_ceil(self.block_size)
    }

    pub fn utilization(&self) -> f64 {
        1.0 - self.free_blocks as f64 / self.total_blocks as f64
    }

    pub fn can_fit(&self, n_tokens: u32) -> bool {
        self.blocks_for(n_tokens) <= self.free_blocks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_free() {
        let mut mgr = KvCacheManager::new(100, 16);
        assert!(mgr.allocate(1, 64).is_ok()); // 4 blocks
        assert_eq!(mgr.free_blocks, 96);
        mgr.free(1);
        assert_eq!(mgr.free_blocks, 100);
    }

    #[test]
    fn oom_returns_error() {
        let mut mgr = KvCacheManager::new(4, 16);
        assert!(mgr.allocate(1, 100).is_err()); // needs 7 blocks, only 4
    }

    #[test]
    fn grow_allocates_extra_blocks() {
        let mut mgr = KvCacheManager::new(100, 16);
        mgr.allocate(1, 16).unwrap(); // 1 block
        assert_eq!(mgr.free_blocks, 99);
        mgr.grow(1, 16).unwrap(); // still fits in same block (16+16=32 → 2 blocks)
        assert_eq!(mgr.free_blocks, 98);
    }
}
