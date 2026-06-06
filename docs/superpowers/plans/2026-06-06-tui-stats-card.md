# TUI `/stats` Styled Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `/stats` renders a styled transcript card (colored sections, human units, context sparkline, red tool errors, cache %) instead of a raw preformatted block.

**Architecture:** The edge sends the `TraceStats` struct (not a pre-rendered string) through `Msg::StatsReady`; a new `Cell::Stats` variant renders the card in `cells.rs` (testable pure code); `/export` falls back to the kept plain `render()`.

**Tech Stack:** Rust, ratatui 0.29.

Spec: `docs/superpowers/specs/2026-06-06-tui-stats-card-design.md`.

Verified anchors: `main.rs:1138` `Ok::<String, String>(trace_stats::compute(file).render())`; `app.rs:809` `Msg::StatsReady(Ok(table))` → `Cell::Agent("```\n…```")`; `Cell` derives `Debug, Clone, Serialize, Deserialize`; `session.rs:101-124` export match over cells; existing reducer test at app.rs:2609 asserts the old string path.

---

### Task 1: cache aggregation + derives in `trace_stats.rs`

- [x] **Step 1 (red):** test beside the existing compute tests:

```rust
    #[test]
    fn compute_sums_cache_read_tokens() {
        let trace = r#"{"v":1,"seq":0,"ts":"2026-06-06T10:00:00.000Z","src":"agent","event":{"type":"llm_response","agent":"o","turn":1,"model":"m","text":"","tool_call_count":0,"latency_ms":10,"time_to_first_token_ms":5,"usage":{"input_tokens":100,"output_tokens":10,"cache_read_input_tokens":40,"cache_creation_input_tokens":0,"reasoning_tokens":0}}}
{"v":1,"seq":1,"ts":"2026-06-06T10:00:01.000Z","src":"agent","event":{"type":"llm_response","agent":"o","turn":2,"model":"m","text":"","tool_call_count":0,"latency_ms":10,"time_to_first_token_ms":5,"usage":{"input_tokens":200,"output_tokens":10,"cache_read_input_tokens":150,"cache_creation_input_tokens":0,"reasoning_tokens":0}}}"#;
        let stats = compute(std::io::Cursor::new(trace));
        assert_eq!(stats.total_cache_read_tokens, 190);
    }
```

(Match the EXACT envelope shape the existing compute tests use — copy one of
their fixture lines and add the `cache_read_input_tokens` values.)

- [x] **Step 2:** field `pub total_cache_read_tokens: u64` on `TraceStats`; sum it where `total_input_tokens` is summed from `llm_response.usage`; add `Clone, Serialize, Deserialize` (and `Default` if absent) to `TraceStats` AND `ToolStat` derives.
- [x] **Step 3:** `cargo test -p heartbit-tui trace_stats` PASS. Commit `feat(tui): stats aggregate cache-read tokens + serde derives`.

### Task 2: helpers + `Cell::Stats` card renderer in `cells.rs`

- [x] **Step 1 (red):** tests:

```rust
    #[test]
    fn human_units_format() {
        assert_eq!(fmt_tokens(982), "982");
        assert_eq!(fmt_tokens(4_400), "4.4k");
        assert_eq!(fmt_tokens(1_200_000), "1.2M");
        assert_eq!(fmt_ms(2), "2ms");
        assert_eq!(fmt_ms(355), "355ms");
        assert_eq!(fmt_ms(71_100), "1m11s");
        assert_eq!(fmt_ms(19_822), "19.8s");
    }

    #[test]
    fn sparkline_scales_and_downsamples() {
        assert_eq!(sparkline(&[0, 7], 8), "▁█");
        let s = sparkline(&(0..100u64).collect::<Vec<_>>(), 8);
        assert_eq!(s.chars().count(), 8, "{s}");
        assert!(s.ends_with('█'));
        assert_eq!(sparkline(&[], 8), "");
    }

    #[test]
    fn stats_cell_renders_the_card() {
        let mut stats = crate::trace_stats::TraceStats::default();
        stats.llm_calls = 21;
        stats.duration_ms = 107_000;
        stats.total_input_tokens = 86_700;
        stats.total_output_tokens = 6_900;
        stats.total_cache_read_tokens = 35_000;
        stats.turn_input_tokens = vec![4_400, 5_000, 7_100, 13_250, 7_126];
        stats.run_completed = 8;
        stats.tools.insert("webfetch".into(), crate::trace_stats::ToolStat {
            count: 11, errors: 2, p50_ms: 355, p95_ms: 2_957,
        });
        let text: String = Cell::Stats { label: "abc-1".into(), stats: Box::new(stats) }
            .to_lines()
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>() + "\n")
            .collect();
        assert!(text.contains("stats — session abc-1"), "{text}");
        assert!(text.contains("21 llm calls"), "{text}");
        assert!(text.contains("86.7k"), "{text}");
        assert!(text.contains("cache 40%"), "{text}");
        assert!(text.contains("webfetch"), "{text}");
        assert!(text.contains("2 ⚠"), "{text}");
        assert!(text.contains('▁') || text.contains('█'), "sparkline:\n{text}");
        assert!(text.contains("8 ok"), "{text}");
        assert!(text.contains("none"), "friction none:\n{text}");
    }
```

(`ToolStat` field names: verify against trace_stats.rs and adapt. `TraceStats`
needs `Default` — Task 1 added it.)

- [x] **Step 2:** implement in `cells.rs`:

```rust
/// `982` · `4.4k` · `1.2M` — compact token counts.
fn fmt_tokens(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", n as f64 / 1_000.0)
    } else {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    }
}

/// `2ms` · `19.8s` · `1m47s` — human durations.
fn fmt_ms(ms: u64) -> String {
    if ms < 1_000 {
        format!("{ms}ms")
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f64 / 1_000.0)
    } else {
        format!("{}m{}s", ms / 60_000, (ms % 60_000) / 1_000)
    }
}

/// Downsample to ≤ `width` buckets (mean), scale to the max, render ▁▂▃▄▅▆▇█.
fn sparkline(values: &[u64], width: usize) -> String {
    const GLYPHS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if values.is_empty() || width == 0 {
        return String::new();
    }
    let chunk = values.len().div_ceil(width);
    let buckets: Vec<u64> = values
        .chunks(chunk)
        .map(|c| c.iter().sum::<u64>() / c.len() as u64)
        .collect();
    let max = buckets.iter().copied().max().unwrap_or(0).max(1);
    buckets
        .iter()
        .map(|&v| GLYPHS[((v * 7) / max) as usize])
        .collect()
}
```

Variant (Box to keep `Cell` small): `Stats { label: String, stats: Box<crate::trace_stats::TraceStats> }`.

`to_lines()` arm — build the card per the spec visual: header line (`▎ stats — session {label} · {fmt_ms(duration_ms)} · {llm_calls} llm calls`, marker+title magenta bold, meta dim); blank; `tokens` / `context` (+ `sparkline(&turn_input_tokens, 24)` dim, row skipped when `turn_input_tokens.len() < 2`) / `latency` / `runs` (failed red when > 0) / `friction` ("none" green, else the non-zero counters joined with ` · ` yellow) / `approvals` ("instant" when mean < 100ms); blank; tools header row dim + one aligned row per tool (`{:<14} {:<4} {:<5} {:<7} {}`), err column `—` dim or `{n} ⚠` red. Section labels (`tokens`, `context`, …) dim, `{:<10}`-padded; values default-styled.

- [x] **Step 3:** PASS + commit `feat(tui): Cell::Stats styled card — human units, sparkline, error highlighting`.

### Task 3: wiring — msg, edge, reducer, export

- [x] **Step 1 (red):** update the reducer test at app.rs:2609 to the new shape:

```rust
        app.update(Msg::StatsReady(Ok((
            "t1".into(),
            Box::new(crate::trace_stats::TraceStats::default()),
        ))));
        assert!(matches!(app.history.last(), Some(Cell::Stats { .. })));
```

…and an export test in session.rs tests: a `Cell::Stats` exports a fenced block containing `tools:`.

- [x] **Step 2:** `msg.rs`: `StatsReady(Result<(String, Box<crate::trace_stats::TraceStats>), String>)`. `main.rs:1138`: `Ok::<_, String>((label.clone(), Box::new(trace_stats::compute(file))))` — `label` = the resolved trace id already computed in that handler (verify its variable name in context; derive from the path stem if absent). `app.rs:809`: push `Cell::Stats { label, stats }`. `session.rs` export arm: `Cell::Stats { label, stats } => out.push_str(&format!("**stats — {label}**\n\n```\n{}```\n\n", stats.render()))`.
- [x] **Step 3:** `cargo test -p heartbit-tui` ALL PASS. Commit `feat(tui): /stats renders the styled card — struct through the Msg, plain render for export`.

### Task 4: gate + live pty

- [x] Full workspace gate green.
- [x] `cargo build -p heartbit-tui`; pty against the bridged research trace (`/stats 6a2473a4-2570678` with `HEARTBIT_TUI_CONFIG` pointing at `/tmp/claude-1000/tuitracebridge-nxc6ed2i/isolated-tui.toml`): settled frame contains `▎ stats`, `webfetch`, `⚠`, sparkline glyphs. Mark checkboxes; report.

## Self-review

1. **Spec coverage:** cache % (T1+T2) · human units (T2) · sparkline (T2) · red errors / green-yellow friction (T2) · struct through Msg (T3) · export fallback via kept `render()` (T3) · old-session compat (serde derive, T1) · live validation (T4). ✓
2. **Placeholders:** the two "verify against the code" notes (ToolStat field names, label variable in main.rs) name exactly what to check and where — acceptable for inline execution.
3. **Type consistency:** `Cell::Stats { label: String, stats: Box<TraceStats> }` in T2/T3; `fmt_tokens`/`fmt_ms`/`sparkline` signatures consistent between tests and impl.
