use std::path::Path;

use serde::Deserialize;

use crate::engine::event::SimTime;

use super::request::InferenceRequest;
use super::traits::WorkloadSource;

#[derive(Debug, Deserialize)]
struct TraceRecord {
    timestamp_ms: f64,
    prompt_tokens: u32,
    output_tokens: u32,
}

pub struct TraceReplay {
    records: Vec<(SimTime, InferenceRequest)>,
    cursor: usize,
}

impl TraceReplay {
    /// Load a CSV trace file. Columns: timestamp_ms, prompt_tokens, output_tokens.
    /// Timestamps are replayed as-is (relative to the first record = time 0).
    pub fn from_csv(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .comment(Some(b'#'))
            .trim(csv::Trim::All)
            .from_path(path)?;

        let mut rows: Vec<TraceRecord> = rdr.deserialize().collect::<Result<_, _>>()?;
        rows.sort_by(|a, b| a.timestamp_ms.partial_cmp(&b.timestamp_ms).unwrap());

        let base = rows.first().map(|r| r.timestamp_ms).unwrap_or(0.0);

        let records = rows
            .into_iter()
            .enumerate()
            .map(|(i, r)| {
                let t = (r.timestamp_ms - base) / 1000.0; // ms → seconds
                let req = InferenceRequest {
                    req_id: i as u64,
                    prompt_tokens: r.prompt_tokens.max(1),
                    max_output_tokens: r.output_tokens.max(1),
                    arrival_time: t,
                };
                (t, req)
            })
            .collect();

        Ok(Self { records, cursor: 0 })
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
}

impl WorkloadSource for TraceReplay {
    fn next_arrival(&mut self) -> Option<(SimTime, InferenceRequest)> {
        if self.cursor >= self.records.len() {
            return None;
        }
        let item = self.records[self.cursor].clone();
        self.cursor += 1;
        Some(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn loads_and_replays_in_order() {
        let csv = "timestamp_ms,prompt_tokens,output_tokens\n\
                   200.0,512,128\n\
                   100.0,256,64\n\
                   300.0,1024,256\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        let mut replay = TraceReplay::from_csv(f.path()).unwrap();
        assert_eq!(replay.len(), 3);
        // Should be sorted: first arrival at t=0 (base=100ms)
        let (t0, r0) = replay.next_arrival().unwrap();
        assert!((t0 - 0.0).abs() < 1e-9);
        assert_eq!(r0.prompt_tokens, 256);
        let (t1, _) = replay.next_arrival().unwrap();
        assert!((t1 - 0.1).abs() < 1e-9); // 100ms gap
    }
}
