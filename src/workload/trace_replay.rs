use std::path::Path;

use crate::engine::event::SimTime;

use super::request::InferenceRequest;
use super::traits::WorkloadSource;

pub struct TraceReplay {
    records: Vec<(SimTime, InferenceRequest)>,
    cursor: usize,
}

/// Convert Gregorian date to Julian Day Number (Richards' algorithm).
/// Used purely for computing day offsets; any monotonic mapping works here
/// because we subtract the base timestamp before using the value.
fn gregorian_to_jdn(year: i32, month: i32, day: i32) -> i32 {
    let a = (14 - month) / 12;
    let y = year + 4800 - a;
    let m = month + 12 * a - 3;
    day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045
}

/// Parse Azure trace timestamp "2023-11-16 18:17:03.9799600" → seconds (arbitrary epoch).
/// The absolute value doesn't matter; the caller subtracts the base of the first record.
fn parse_iso_datetime(s: &str) -> Result<f64, Box<dyn std::error::Error>> {
    let (date, time) = s.split_once(' ')
        .ok_or_else(|| format!("expected 'YYYY-MM-DD HH:MM:SS' but got '{}'", s))?;

    let mut dp = date.split('-');
    let year: i32  = dp.next().ok_or("missing year")?.trim().parse()?;
    let month: i32 = dp.next().ok_or("missing month")?.trim().parse()?;
    let day: i32   = dp.next().ok_or("missing day")?.trim().parse()?;

    let mut tp = time.splitn(3, ':');
    let h: f64  = tp.next().ok_or("missing hour")?.trim().parse()?;
    let mi: f64 = tp.next().ok_or("missing minute")?.trim().parse()?;
    let sec: f64 = tp.next().ok_or("missing second")?.trim().parse()?;

    Ok(gregorian_to_jdn(year, month, day) as f64 * 86400.0 + h * 3600.0 + mi * 60.0 + sec)
}

impl TraceReplay {
    /// Load a CSV trace file.
    ///
    /// Two formats are supported, detected automatically from the header row:
    ///
    /// **Native** (our format):
    /// ```text
    /// timestamp_ms,prompt_tokens,output_tokens
    /// 0.0,512,128
    /// ```
    ///
    /// **Azure** (AzurePublicDataset LLM inference traces):
    /// ```text
    /// TIMESTAMP,ContextTokens,GeneratedTokens
    /// 2023-11-16 18:17:03.979,4808,10
    /// ```
    ///
    /// Timestamps are normalised: the first record becomes t=0, all others
    /// are relative offsets in seconds from that point.
    pub fn from_csv(path: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let mut rdr = csv::ReaderBuilder::new()
            .comment(Some(b'#'))
            .trim(csv::Trim::All)
            .from_path(path)?;

        // Detect format from the first header column name.
        let headers = rdr.headers()?.clone();
        let is_azure = headers.get(0).map(|h| h == "TIMESTAMP").unwrap_or(false);

        // Parse all rows → (raw_time, prompt_tokens, output_tokens).
        // raw_time is milliseconds for native format, seconds for Azure format.
        let mut rows: Vec<(f64, u32, u32)> = Vec::new();
        for result in rdr.records() {
            let rec = result?;
            let raw_time = if is_azure {
                parse_iso_datetime(rec.get(0).ok_or("missing TIMESTAMP column")?)?
            } else {
                rec.get(0).ok_or("missing timestamp_ms column")?.parse::<f64>()?
            };
            let prompt: u32 = rec.get(1).ok_or("missing prompt/context column")?.parse()?;
            let output: u32 = rec.get(2).ok_or("missing output column")?.parse()?;
            rows.push((raw_time, prompt, output));
        }

        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let base = rows.first().map(|r| r.0).unwrap_or(0.0);

        let records = rows
            .into_iter()
            .enumerate()
            .map(|(i, (raw_time, prompt, output))| {
                // Convert to seconds relative to first record.
                let t = if is_azure {
                    raw_time - base              // Azure: already in seconds
                } else {
                    (raw_time - base) / 1000.0   // native: milliseconds → seconds
                };
                let req = InferenceRequest {
                    req_id: i as u64,
                    prompt_tokens: prompt.max(1),
                    max_output_tokens: output.max(1),
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
    fn native_format_loads_sorted() {
        let csv = "timestamp_ms,prompt_tokens,output_tokens\n\
                   200.0,512,128\n\
                   100.0,256,64\n\
                   300.0,1024,256\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        let mut replay = TraceReplay::from_csv(f.path()).unwrap();
        assert_eq!(replay.len(), 3);
        let (t0, r0) = replay.next_arrival().unwrap();
        assert!((t0 - 0.0).abs() < 1e-9, "first arrival should be t=0");
        assert_eq!(r0.prompt_tokens, 256);
        let (t1, _) = replay.next_arrival().unwrap();
        assert!((t1 - 0.1).abs() < 1e-9, "100 ms gap should be 0.1 s");
    }

    #[test]
    fn azure_format_parses_datetime() {
        let csv = "TIMESTAMP,ContextTokens,GeneratedTokens\n\
                   2023-11-16 18:17:03.979960,512,10\n\
                   2023-11-16 18:17:04.031960,256,8\n\
                   2023-11-16 18:17:05.000000,1024,20\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        let mut replay = TraceReplay::from_csv(f.path()).unwrap();
        assert_eq!(replay.len(), 3);
        let (t0, r0) = replay.next_arrival().unwrap();
        assert!((t0 - 0.0).abs() < 1e-6, "first arrival should be t=0");
        assert_eq!(r0.prompt_tokens, 512);
        // Second record: 18:17:04.031960 - 18:17:03.979960 = 0.052000 s
        let (t1, _) = replay.next_arrival().unwrap();
        assert!((t1 - 0.052).abs() < 1e-4, "gap should be ~52 ms, got {}", t1);
        // Third record: 18:17:05.000000 - 18:17:03.979960 = 1.020040 s
        let (t2, _) = replay.next_arrival().unwrap();
        assert!((t2 - 1.02004).abs() < 1e-4, "gap should be ~1.020 s, got {}", t2);
    }

    #[test]
    fn azure_multiday_timestamps_are_monotonic() {
        // Ensure the day boundary (Nov 30 → Dec 1) doesn't break ordering.
        let csv = "TIMESTAMP,ContextTokens,GeneratedTokens\n\
                   2023-11-30 23:59:59.000000,100,5\n\
                   2023-12-01 00:00:01.000000,200,10\n";
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(csv.as_bytes()).unwrap();
        let mut replay = TraceReplay::from_csv(f.path()).unwrap();
        let (t0, _) = replay.next_arrival().unwrap();
        let (t1, _) = replay.next_arrival().unwrap();
        assert!(t1 > t0, "Dec 1 should be after Nov 30");
        assert!((t1 - t0 - 2.0).abs() < 1e-4, "gap should be 2 s, got {}", t1 - t0);
    }
}
