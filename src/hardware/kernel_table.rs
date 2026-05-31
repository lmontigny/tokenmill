use std::path::Path;

use rustc_hash::FxHashMap;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelOp {
    Prefill,
    Decode,
}

impl KernelOp {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "prefill" => Some(Self::Prefill),
            "decode" => Some(Self::Decode),
            _ => None,
        }
    }
}

// Lookup key — exact match first, then we interpolate on seq_len.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct KernelKey {
    gpu: String,
    model: String,
    op: KernelOp,
    batch: u32,
    seq_len: u32,
}

// Partial key for the interpolation index (without seq_len).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PartialKey {
    gpu: String,
    model: String,
    op: KernelOp,
    batch: u32,
}

#[derive(Debug, Deserialize)]
struct CsvRecord {
    gpu: String,
    model: String,
    op: String,
    batch_size: u32,
    seq_len: u32,
    latency_ms: f64,
}

pub struct KernelTable {
    exact: FxHashMap<KernelKey, f64>,
    // For interpolation: partial key → sorted vec of (seq_len, latency_s)
    by_seq: FxHashMap<PartialKey, Vec<(u32, f64)>>,
}

impl KernelTable {
    pub fn from_csv(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .comment(Some(b'#'))
            .trim(csv::Trim::All)
            .from_path(path)?;

        let mut table = Self::empty();
        for result in rdr.deserialize() {
            let rec: CsvRecord = result?;
            if let Some(op) = KernelOp::from_str(&rec.op) {
                table.insert(&rec.gpu, &rec.model, op, rec.batch_size, rec.seq_len, rec.latency_ms / 1000.0);
            }
        }
        // Sort each interpolation vec by seq_len
        for v in table.by_seq.values_mut() {
            v.sort_by_key(|&(s, _)| s);
        }
        Ok(table)
    }

    fn empty() -> Self {
        Self { exact: FxHashMap::default(), by_seq: FxHashMap::default() }
    }

    fn insert(&mut self, gpu: &str, model: &str, op: KernelOp, batch: u32, seq_len: u32, latency_s: f64) {
        self.exact.insert(
            KernelKey { gpu: gpu.into(), model: model.into(), op, batch, seq_len },
            latency_s,
        );
        self.by_seq
            .entry(PartialKey { gpu: gpu.into(), model: model.into(), op, batch })
            .or_default()
            .push((seq_len, latency_s));
    }

    /// Returns profiled latency in seconds, or None if no data for this gpu+model+op+batch combo.
    /// Uses exact match first, then linear interpolation on seq_len.
    pub fn lookup(&self, gpu: &str, model: &str, op: KernelOp, batch: u32, seq_len: u32) -> Option<f64> {
        // Exact hit
        let key = KernelKey { gpu: gpu.into(), model: model.into(), op, batch, seq_len };
        if let Some(&v) = self.exact.get(&key) {
            return Some(v);
        }

        // Interpolate: find the two bracketing seq_len entries
        let pkey = PartialKey { gpu: gpu.into(), model: model.into(), op, batch };
        let entries = self.by_seq.get(&pkey)?;
        if entries.is_empty() {
            return None;
        }

        // Below the lowest entry → extrapolate from first two (or clamp)
        if seq_len <= entries[0].0 {
            return Some(entries[0].1);
        }
        // Above the highest → extrapolate from last two (or clamp)
        if seq_len >= entries[entries.len() - 1].0 {
            return Some(entries[entries.len() - 1].1);
        }

        // Binary search for the bracketing pair
        let pos = entries.partition_point(|&(s, _)| s <= seq_len);
        let (s0, l0) = entries[pos - 1];
        let (s1, l1) = entries[pos];
        // Linear interpolation
        let t = (seq_len - s0) as f64 / (s1 - s0) as f64;
        Some(l0 + t * (l1 - l0))
    }

    /// Find nearest batch size with data, then interpolate on seq_len.
    /// Used when the exact batch size isn't in the table.
    pub fn lookup_nearest_batch(&self, gpu: &str, model: &str, op: KernelOp, batch: u32, seq_len: u32) -> Option<f64> {
        // Try exact batch first
        if let Some(v) = self.lookup(gpu, model, op, batch, seq_len) {
            return Some(v);
        }

        // Find all batch sizes available for this gpu/model/op
        let available_batches: Vec<u32> = self.by_seq.keys()
            .filter(|k| k.gpu == gpu && k.model == model && k.op == op)
            .map(|k| k.batch)
            .collect();

        if available_batches.is_empty() {
            return None;
        }

        // Pick the nearest batch size
        let nearest = *available_batches.iter().min_by_key(|&&b| b.abs_diff(batch))?;
        self.lookup(gpu, model, op, nearest, seq_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn make_table() -> KernelTable {
        let csv = "gpu,model,op,batch_size,seq_len,latency_ms\n\
                   H100,llama-8b,prefill,1,128,1.0\n\
                   H100,llama-8b,prefill,1,512,4.0\n\
                   H100,llama-8b,decode,1,128,16.0\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        KernelTable::from_csv(f.path()).unwrap()
    }

    #[test]
    fn exact_lookup() {
        let t = make_table();
        assert!((t.lookup("H100", "llama-8b", KernelOp::Prefill, 1, 128).unwrap() - 0.001).abs() < 1e-9);
    }

    #[test]
    fn interpolated_lookup() {
        let t = make_table();
        // seq=320 is halfway between 128 and 512 → latency should be between 1ms and 4ms
        let v = t.lookup("H100", "llama-8b", KernelOp::Prefill, 1, 320).unwrap();
        assert!(v > 0.001 && v < 0.004);
    }

    #[test]
    fn missing_returns_none() {
        let t = make_table();
        assert!(t.lookup("A100", "llama-8b", KernelOp::Prefill, 1, 128).is_none());
    }
}
