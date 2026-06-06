//! Deterministic stats over a trace JSONL: streams line-by-line (never loads
//! the whole file), tolerant of torn/malformed lines and unknown event types.
//! This is the human `/stats` summary AND the measurement substrate the
//! self-improvement ladder builds on. Reads ONLY `agent` and `ui` records —
//! `core_trace` is a raw mirror, never parsed. All aggregations are
//! order-independent: file line order is NOT authoritative under concurrent
//! producers (the `seq` field is).

use std::collections::BTreeMap;
use std::io::BufRead;

use serde::{Deserialize, Serialize};

/// Per-tool aggregate.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolStat {
    pub count: usize,
    pub errors: usize,
    /// errors / count (0.0 for no calls) — the rung-3 measurement substrate.
    pub error_rate: f64,
    pub p50_ms: u64,
    pub p95_ms: u64,
    #[serde(skip)]
    pub(crate) durations: Vec<u64>,
}

/// The deterministic summary of one trace file.
/// `Clone + Deserialize` because it now travels inside [`crate::cells::Cell`]
/// (session persistence round-trips the transcript as JSON).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceStats {
    pub records: usize,
    pub skipped_lines: usize,
    pub user_inputs: usize,
    pub turns: usize,
    pub llm_calls: usize,
    pub total_input_tokens: u64,
    /// Prompt-cache reads summed across LLM calls (0 when caching is off) —
    /// the cache hit-rate numerator for the `/stats` card.
    pub total_cache_read_tokens: u64,
    pub total_output_tokens: u64,
    pub llm_latency_p50_ms: u64,
    pub llm_latency_p95_ms: u64,
    pub ttft_p50_ms: u64,
    pub ttft_p95_ms: u64,
    pub tools: BTreeMap<String, ToolStat>,
    pub retries: usize,
    pub doom_loops: usize,
    pub guardrail_denied: usize,
    pub guardrail_warned: usize,
    pub approvals: usize,
    pub approval_denials: usize,
    pub approval_mean_latency_ms: u64,
    pub interrupts: usize,
    pub compactions: usize,
    pub prunes: usize,
    pub run_completed: usize,
    pub run_failed: usize,
    /// Wall-clock span of the trace: first record `ts` → last record `ts`,
    /// in milliseconds (0 when fewer than two parseable timestamps).
    pub duration_ms: u64,
    /// Input tokens per LLM call, in trace order — the context-growth curve
    /// (the rung-3 measurement substrate; /analyze gets it in the JSON).
    pub turn_input_tokens: Vec<u64>,
}

/// Nearest-rank percentile over a SORTED slice (0 for empty).
fn pct(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let rank = ((p * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

/// Stream a trace and aggregate. Tolerant by design: torn/malformed lines and
/// unknown event types/versions are counted, never fatal.
pub fn compute(reader: impl std::io::Read) -> TraceStats {
    let mut s = TraceStats::default();
    let mut llm_latencies: Vec<u64> = Vec::new();
    let mut ttfts: Vec<u64> = Vec::new();
    let mut approval_latencies: Vec<u64> = Vec::new();
    // Wall-clock span: min/max record timestamp (epoch ms). Order-independent
    // (file order is not authoritative), tolerant of unparseable ts.
    let mut ts_min: Option<i64> = None;
    let mut ts_max: Option<i64> = None;
    for line in std::io::BufReader::new(reader).lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(rec) = serde_json::from_str::<serde_json::Value>(&line) else {
            s.skipped_lines += 1;
            continue;
        };
        let (Some(src), Some(ev)) = (rec["src"].as_str(), rec.get("event")) else {
            s.skipped_lines += 1;
            continue;
        };
        s.records += 1;
        if let Some(ts) = rec["ts"]
            .as_str()
            .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
        {
            let ms = ts.timestamp_millis();
            ts_min = Some(ts_min.map_or(ms, |m| m.min(ms)));
            ts_max = Some(ts_max.map_or(ms, |m| m.max(ms)));
        }
        let ty = ev["type"].as_str().unwrap_or("");
        match (src, ty) {
            ("ui", "user_input") => s.user_inputs += 1,
            ("ui", "approval") => {
                s.approvals += 1;
                if ev["decision"].as_str().unwrap_or("").contains("deny") {
                    s.approval_denials += 1;
                }
                approval_latencies.push(ev["latency_ms"].as_u64().unwrap_or(0));
            }
            // cp1 only — cp2 mirrors the same user action.
            ("ui", "interrupt_requested")
                if ev["checkpoint"].as_str().unwrap_or("").starts_with("cp1") =>
            {
                s.interrupts += 1;
            }
            ("agent", "turn_started") => s.turns += 1,
            ("agent", "llm_response") => {
                s.llm_calls += 1;
                let inp = ev["usage"]["input_tokens"].as_u64().unwrap_or(0);
                s.turn_input_tokens.push(inp);
                s.total_input_tokens += inp;
                s.total_output_tokens += ev["usage"]["output_tokens"].as_u64().unwrap_or(0);
                s.total_cache_read_tokens +=
                    ev["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0);
                llm_latencies.push(ev["latency_ms"].as_u64().unwrap_or(0));
                let ttft = ev["time_to_first_token_ms"].as_u64().unwrap_or(0);
                if ttft > 0 {
                    ttfts.push(ttft);
                }
            }
            ("agent", "tool_call_completed") => {
                let name = ev["tool_name"].as_str().unwrap_or("?").to_string();
                let stat = s.tools.entry(name).or_default();
                stat.count += 1;
                if ev["is_error"].as_bool().unwrap_or(false) {
                    stat.errors += 1;
                }
                stat.durations.push(ev["duration_ms"].as_u64().unwrap_or(0));
            }
            ("agent", "retry_attempt") => s.retries += 1,
            ("agent", "doom_loop_detected") | ("agent", "fuzzy_doom_loop_detected") => {
                s.doom_loops += 1;
            }
            ("agent", "guardrail_denied") => s.guardrail_denied += 1,
            ("agent", "guardrail_warned") => s.guardrail_warned += 1,
            ("agent", "auto_compaction_triggered") | ("agent", "context_summarized") => {
                s.compactions += 1;
            }
            ("agent", "session_pruned") => s.prunes += 1,
            ("agent", "run_completed") => s.run_completed += 1,
            ("agent", "run_failed") => s.run_failed += 1,
            _ => {} // unknown type/src: counted in records, otherwise ignored
        }
    }
    llm_latencies.sort_unstable();
    ttfts.sort_unstable();
    s.llm_latency_p50_ms = pct(&llm_latencies, 0.5);
    s.llm_latency_p95_ms = pct(&llm_latencies, 0.95);
    s.ttft_p50_ms = pct(&ttfts, 0.5);
    s.ttft_p95_ms = pct(&ttfts, 0.95);
    if !approval_latencies.is_empty() {
        s.approval_mean_latency_ms =
            approval_latencies.iter().sum::<u64>() / approval_latencies.len() as u64;
    }
    for stat in s.tools.values_mut() {
        stat.durations.sort_unstable();
        stat.p50_ms = pct(&stat.durations, 0.5);
        stat.p95_ms = pct(&stat.durations, 0.95);
        if stat.count > 0 {
            stat.error_rate = stat.errors as f64 / stat.count as f64;
        }
    }
    if let (Some(min), Some(max)) = (ts_min, ts_max) {
        s.duration_ms = u64::try_from(max - min).unwrap_or(0);
    }
    s
}

impl TraceStats {
    /// A fixed-width text table (rendered into the transcript in a code fence).
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "records {:>6}   skipped {:>3}   user msgs {:>4}\n",
            self.records, self.skipped_lines, self.user_inputs
        ));
        out.push_str(&format!(
            "turns   {:>6}   llm calls {:>4}   completed {} / failed {}   wall-clock {:.1}s\n",
            self.turns,
            self.llm_calls,
            self.run_completed,
            self.run_failed,
            self.duration_ms as f64 / 1000.0
        ));
        out.push_str(&format!(
            "tokens  in {} / out {}\n",
            self.total_input_tokens, self.total_output_tokens
        ));
        if let (Some(first), Some(last)) = (
            self.turn_input_tokens.first(),
            self.turn_input_tokens.last(),
        ) {
            let max = self.turn_input_tokens.iter().max().copied().unwrap_or(0);
            out.push_str(&format!(
                "context growth: {first} → {last} per call (max {max}) over {} calls\n",
                self.turn_input_tokens.len()
            ));
        }
        out.push_str(&format!(
            "llm latency p50/p95  {}ms / {}ms   ttft p50/p95  {}ms / {}ms\n",
            self.llm_latency_p50_ms, self.llm_latency_p95_ms, self.ttft_p50_ms, self.ttft_p95_ms
        ));
        out.push_str(&format!(
            "friction: retries {}  doom-loops {}  guardrail deny/warn {}/{}  interrupts {}  compactions {}  prunes {}\n",
            self.retries,
            self.doom_loops,
            self.guardrail_denied,
            self.guardrail_warned,
            self.interrupts,
            self.compactions,
            self.prunes
        ));
        out.push_str(&format!(
            "approvals {} (denied {})  mean human latency {}ms\n",
            self.approvals, self.approval_denials, self.approval_mean_latency_ms
        ));
        if !self.tools.is_empty() {
            out.push_str("tools:\n");
            for (name, t) in &self.tools {
                out.push_str(&format!(
                    "  {:<14} ×{:<4} errors {:<3} p50/p95 {}ms/{}ms\n",
                    name, t.count, t.errors, t.p50_ms, t.p95_ms
                ));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic-but-valid trace (the golden fixture).
    fn fixture() -> String {
        let lines = [
            r#"{"v":1,"seq":0,"ts":"2026-06-06T10:00:00.000Z","src":"ui","event":{"type":"session_started","version":"0.1.0","session_id":"s1","model":"m","permission_mode":"normal","mcp_servers":[],"context_recall":true,"verify_command":null}}"#,
            r#"{"v":1,"seq":1,"ts":"2026-06-06T10:00:01.000Z","src":"ui","event":{"type":"user_input","text":"do the thing"}}"#,
            r#"{"v":1,"seq":2,"ts":"2026-06-06T10:00:01.100Z","src":"agent","event":{"type":"turn_started","agent":"entry","turn":1,"max_turns":300}}"#,
            r#"{"v":1,"seq":3,"ts":"2026-06-06T10:00:03.000Z","src":"agent","event":{"type":"llm_response","agent":"entry","turn":1,"usage":{"input_tokens":1000,"output_tokens":50},"stop_reason":"tool_use","tool_call_count":1,"latency_ms":1900,"time_to_first_token_ms":400}}"#,
            r#"{"v":1,"seq":4,"ts":"2026-06-06T10:00:03.100Z","src":"ui","event":{"type":"approval","tools":["bash"],"decision":"allow","latency_ms":2500,"mode":"normal"}}"#,
            r#"{"v":1,"seq":5,"ts":"2026-06-06T10:00:06.000Z","src":"agent","event":{"type":"tool_call_completed","agent":"entry","tool_name":"bash","tool_call_id":"t1","is_error":false,"duration_ms":300,"output":"ok"}}"#,
            r#"{"v":1,"seq":6,"ts":"2026-06-06T10:00:06.500Z","src":"agent","event":{"type":"retry_attempt","agent":"(provider)","attempt":1,"max_retries":3,"delay_ms":1000,"error_class":"rate_limited"}}"#,
            r#"{"v":1,"seq":7,"ts":"2026-06-06T10:00:08.000Z","src":"agent","event":{"type":"turn_started","agent":"entry","turn":2,"max_turns":300}}"#,
            r#"{"v":1,"seq":8,"ts":"2026-06-06T10:00:09.000Z","src":"agent","event":{"type":"llm_response","agent":"entry","turn":2,"usage":{"input_tokens":1200,"output_tokens":80},"stop_reason":"end_turn","tool_call_count":0,"latency_ms":900,"time_to_first_token_ms":200}}"#,
            r#"{"v":1,"seq":9,"ts":"2026-06-06T10:00:09.100Z","src":"agent","event":{"type":"tool_call_completed","agent":"entry","tool_name":"bash","tool_call_id":"t2","is_error":true,"duration_ms":700,"output":"boom"}}"#,
            r#"{"v":1,"seq":10,"ts":"2026-06-06T10:00:09.200Z","src":"ui","event":{"type":"interrupt_requested","checkpoint":"cp1_effect_dequeued","running":true}}"#,
            r#"{"v":1,"seq":12,"ts":"2026-06-06T10:00:09.250Z","src":"ui","event":{"type":"interrupt_requested","checkpoint":"cp2_handle_interrupted","running":true}}"#,
            r#"{"v":1,"seq":11,"ts":"2026-06-06T10:00:09.300Z","src":"agent","event":{"type":"run_completed","agent":"entry","total_usage":{"input_tokens":2200,"output_tokens":130},"tool_calls_made":2}}"#,
            "{ this line is torn garba",
        ];
        lines.join("\n")
    }

    #[test]
    fn compute_sums_cache_read_tokens() {
        let trace = [
            r#"{"v":1,"seq":0,"ts":"2026-06-06T10:00:00.000Z","src":"agent","event":{"type":"llm_response","agent":"o","turn":1,"usage":{"input_tokens":100,"output_tokens":10,"cache_read_input_tokens":40},"tool_call_count":0,"latency_ms":10,"time_to_first_token_ms":5}}"#,
            r#"{"v":1,"seq":1,"ts":"2026-06-06T10:00:01.000Z","src":"agent","event":{"type":"llm_response","agent":"o","turn":2,"usage":{"input_tokens":200,"output_tokens":10,"cache_read_input_tokens":150},"tool_call_count":0,"latency_ms":10,"time_to_first_token_ms":5}}"#,
        ]
        .join("\n");
        let s = compute(trace.as_bytes());
        assert_eq!(s.total_cache_read_tokens, 190);
        // Absent field (the golden fixture has none) sums to 0 — no panic.
        assert_eq!(compute(fixture().as_bytes()).total_cache_read_tokens, 0);
    }

    #[test]
    fn golden_fixture_computes_exact_stats() {
        let s = compute(fixture().as_bytes());
        assert_eq!(s.records, 13);
        assert_eq!(s.skipped_lines, 1);
        assert_eq!(s.user_inputs, 1);
        assert_eq!(s.turns, 2);
        assert_eq!(s.llm_calls, 2);
        assert_eq!(s.total_input_tokens, 2200);
        assert_eq!(s.total_output_tokens, 130);
        // sorted latencies [900, 1900]: p50 = 900 (nearest-rank), p95 = 1900
        assert_eq!(s.llm_latency_p50_ms, 900);
        assert_eq!(s.llm_latency_p95_ms, 1900);
        assert_eq!(s.ttft_p50_ms, 200);
        let bash = s.tools.get("bash").expect("bash stats");
        assert_eq!(bash.count, 2);
        assert_eq!(bash.errors, 1);
        assert_eq!(bash.p50_ms, 300);
        assert_eq!(bash.p95_ms, 700);
        assert_eq!(s.retries, 1);
        assert_eq!(s.approvals, 1);
        assert_eq!(s.approval_denials, 0);
        assert_eq!(s.approval_mean_latency_ms, 2500);
        assert_eq!(s.interrupts, 1); // cp1 only — cp2 must not double-count
        assert_eq!(s.run_completed, 1);
        assert_eq!(s.run_failed, 0);
        assert_eq!(s.doom_loops, 0);
        // wall-clock: first ts 10:00:00.000Z → last ts 10:00:09.300Z = 9300ms
        assert_eq!(s.duration_ms, 9300);
        // per-turn input tokens (the context-growth curve — rung-3 substrate)
        assert_eq!(s.turn_input_tokens, vec![1000, 1200]);
        assert!(s.render().contains("context growth"), "{}", s.render());
        // bash: 1 error out of 2 calls
        let bash = s.tools.get("bash").expect("bash stats");
        assert!((bash.error_rate - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn empty_input_yields_zeroed_stats() {
        let s = compute(&b""[..]);
        assert_eq!(s.records, 0);
        assert_eq!(s.turns, 0);
        assert_eq!(s.llm_latency_p50_ms, 0);
        assert!(s.tools.is_empty());
    }

    #[test]
    fn unknown_types_and_versions_are_counted_not_fatal() {
        let input = [
            r#"{"v":99,"seq":0,"ts":"t","src":"ui","event":{"type":"from_the_future"}}"#,
            r#"{"v":1,"seq":1,"ts":"t","src":"agent","event":{"type":"turn_started","agent":"a","turn":1,"max_turns":5}}"#,
        ]
        .join("\n");
        let s = compute(input.as_bytes());
        assert_eq!(s.records, 2); // parsed envelope = a record, even if unknown
        assert_eq!(s.turns, 1);
    }

    #[test]
    fn percentile_is_nearest_rank() {
        assert_eq!(pct(&[], 0.5), 0);
        assert_eq!(pct(&[10], 0.95), 10);
        assert_eq!(pct(&[10, 20, 30, 40], 0.5), 20);
        assert_eq!(pct(&[10, 20, 30, 40], 0.95), 40);
    }

    #[test]
    fn render_is_a_readable_table() {
        let s = compute(fixture().as_bytes());
        let out = s.render();
        assert!(out.contains("turns"), "got: {out}");
        assert!(out.contains("bash"), "got: {out}");
        assert!(out.contains("2200"), "tokens visible: {out}");
        assert!(out.contains("retries"), "got: {out}");
    }
}
