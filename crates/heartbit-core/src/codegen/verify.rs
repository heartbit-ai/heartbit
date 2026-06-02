//! Deterministic verification: the `verify` tool + the pure render/parse of its
//! machine-greppable `VERIFY_RESULT:` sentinel.
//!
//! Format (the LAST line of the whole output is always the overall verdict, so it
//! survives in the goal judge's transcript tail):
//!
//! ```text
//! $ cargo test --workspace
//! exit_code=0
//! --- output (tail, 3000 of 8123 bytes) ---
//! <tail of combined stdout+stderr>
//! VERIFY_RESULT: PASS exit_code=0 command=cargo test --workspace
//! ```

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;

use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

/// Default per-command timeout for [`VerifyCommandTool`] (builds/tests are slow).
const DEFAULT_VERIFY_TIMEOUT: Duration = Duration::from_secs(300);
/// Default output tail kept per command (enough for a failure summary).
const DEFAULT_VERIFY_TAIL: usize = 4_000;

/// The parsed verdict of one verification command. `passed == (exit_code == 0)`
/// (borrowed from claw-code's `TestCommandProvenance::passed`, but wired).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifyOutcome {
    /// Whether the command exited 0.
    pub passed: bool,
    /// The process exit code (`-1` if the process was killed without a code).
    pub exit_code: i32,
    /// The command that was run (as configured).
    pub command: String,
}

/// The machine-greppable sentinel prefix. A deterministic completion gate (or the
/// goal judge) keys on the LAST occurrence of this in the transcript.
pub const VERIFY_SENTINEL: &str = "VERIFY_RESULT:";

/// Render one verify command's result block, ending with the sentinel line.
///
/// `output` is the combined stdout+stderr; only its last `max_tail` bytes are
/// shown (char-boundary safe), with a header noting how much was elided.
pub fn render_verify_result(
    command: &str,
    exit_code: i32,
    output: &str,
    max_tail: usize,
) -> String {
    let verdict = if exit_code == 0 { "PASS" } else { "FAIL" };
    let total = output.len();
    let tail = tail_chars(output, max_tail);
    let body_header = if tail.len() < total {
        format!("--- output (tail, {} of {total} bytes) ---", tail.len())
    } else {
        "--- output ---".to_string()
    };
    format!(
        "$ {command}\nexit_code={exit_code}\n{body_header}\n{tail}\n\
         {VERIFY_SENTINEL} {verdict} exit_code={exit_code} command={command}"
    )
}

/// The last `max` bytes of `s`, advanced to the nearest char boundary so the
/// slice is always valid UTF-8.
fn tail_chars(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut start = s.len() - max;
    while start < s.len() && !s.is_char_boundary(start) {
        start += 1;
    }
    &s[start..]
}

/// Find the LAST `VERIFY_RESULT:` sentinel anywhere in `transcript` and parse it.
/// Returns `None` if no sentinel is present. This is the deterministic-gate seam.
pub fn parse_latest_verify(transcript: &str) -> Option<VerifyOutcome> {
    // Return the LAST *canonical* sentinel — one carrying the structured
    // `exit_code=` field. The model may echo "VERIFY_RESULT: PASS" in prose
    // (possibly a stale verdict); a bare echo has no `exit_code=`, so it can't
    // spoof a gate. Scan occurrences from the end until a canonical one is found.
    let mut search_end = transcript.len();
    while let Some(idx) = transcript[..search_end].rfind(VERIFY_SENTINEL) {
        let after = &transcript[idx + VERIFY_SENTINEL.len()..];
        let line = after.lines().next().unwrap_or("").trim();
        if line.contains("exit_code=") {
            return Some(parse_sentinel_line(line));
        }
        search_end = idx; // not canonical; keep looking earlier
    }
    None
}

/// Parse the text following the `VERIFY_SENTINEL` on a canonical line, e.g.
/// `"PASS exit_code=0 command=cargo test --workspace"`.
fn parse_sentinel_line(line: &str) -> VerifyOutcome {
    let (verdict, rest) = match line.split_once(char::is_whitespace) {
        Some((v, r)) => (v, r),
        None => (line, ""),
    };
    let passed = verdict.eq_ignore_ascii_case("PASS");

    let exit_code = rest
        .split_whitespace()
        .find_map(|tok| tok.strip_prefix("exit_code="))
        .and_then(|n| n.parse::<i32>().ok())
        .unwrap_or(if passed { 0 } else { -1 });

    let command = rest
        .find("command=")
        .map(|i| rest[i + "command=".len()..].trim().to_string())
        .unwrap_or_default();

    VerifyOutcome {
        passed,
        exit_code,
        command,
    }
}

/// The `verify` tool: runs the harness-**configured** verification command(s) in
/// the workspace and reports a structured, gate-able PASS/FAIL. Its two reasons
/// to exist over raw `bash`: (1) the command is configured by the harness (the
/// agent never guesses what "passing" means), and (2) the result is a
/// machine-greppable [`VERIFY_SENTINEL`] the goal judge / a deterministic gate
/// can key on. Commands run sequentially and **fail-fast** (stop at the first
/// non-zero exit), so the final sentinel is the overall verdict.
///
/// Trust model: the command is **operator-configured**, not agent-supplied, so it
/// runs with ambient authority — like a CI command — and is deliberately NOT
/// sandboxed the way the agent-facing `bash` tool is (it does not apply
/// `BashTool::with_sandbox`'s Landlock / `EnvPolicy` / `CorePathPolicy`). It
/// intentionally does *not* reuse `BashTool` for that reason: the agent's `bash`
/// sandbox and the operator's verify authority are kept separate, and the small
/// spawn/timeout plumbing here is the price of that separation.
pub struct VerifyCommandTool {
    commands: Vec<String>,
    workspace: PathBuf,
    timeout: Duration,
    max_tail: usize,
}

impl VerifyCommandTool {
    /// Create a verify tool that runs `commands` (in order, fail-fast) with the
    /// workspace as the working directory.
    pub fn new(workspace: impl Into<PathBuf>, commands: Vec<String>) -> Self {
        Self {
            commands,
            workspace: workspace.into(),
            timeout: DEFAULT_VERIFY_TIMEOUT,
            max_tail: DEFAULT_VERIFY_TAIL,
        }
    }

    /// Override the per-command timeout (default 300s).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Override the per-command output tail cap (default 4000 bytes).
    pub fn max_tail(mut self, max_tail: usize) -> Self {
        self.max_tail = max_tail;
        self
    }
}

impl Tool for VerifyCommandTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "verify".into(),
            description: "Run the project's configured verification command(s) (build/test/lint) \
                          in the workspace and report PASS/FAIL with the exit code. Takes no \
                          arguments — the command is fixed by the harness. Call this after making \
                          changes; the task is only complete when this reports `VERIFY_RESULT: PASS`."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {},
                "additionalProperties": false
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        _input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            if self.commands.is_empty() {
                return Ok(ToolOutput::error(
                    "no verify command configured for this code harness",
                ));
            }

            let mut blocks: Vec<String> = Vec::with_capacity(self.commands.len());
            for command in &self.commands {
                let child = tokio::process::Command::new("bash")
                    .arg("-c")
                    .arg(command)
                    .current_dir(&self.workspace)
                    .stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn();

                let child = match child {
                    Ok(c) => c,
                    // Could not even spawn the shell — a genuine framework-level
                    // error (misconfiguration / missing shell), not a FAIL verdict.
                    Err(e) => {
                        return Ok(ToolOutput::error(format!(
                            "verify command `{command}` could not be spawned: {e}"
                        )));
                    }
                };

                match tokio::time::timeout(self.timeout, child.wait_with_output()).await {
                    Ok(Ok(output)) => {
                        let exit_code = output.status.code().unwrap_or(-1);
                        let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        if !stderr.is_empty() {
                            if !combined.is_empty() {
                                combined.push('\n');
                            }
                            combined.push_str(&stderr);
                        }
                        blocks.push(render_verify_result(
                            command,
                            exit_code,
                            &combined,
                            self.max_tail,
                        ));
                        if exit_code != 0 {
                            break; // fail-fast: the overall verdict is this FAIL
                        }
                    }
                    Ok(Err(e)) => {
                        return Ok(ToolOutput::error(format!(
                            "verify command `{command}` failed to run: {e}"
                        )));
                    }
                    Err(_) => {
                        // Timed out: surface it as a FAIL verdict (the agent should
                        // know its change made verification hang), not a tool error.
                        // Known limitation (mirrors BashTool): the timed-out child is
                        // not killed, so a slow `cargo build`/`test` can leave a heavy
                        // orphan until it exits on its own.
                        let note = format!("(timed out after {}s)", self.timeout.as_secs());
                        blocks.push(render_verify_result(command, -1, &note, self.max_tail));
                        break;
                    }
                }
            }

            Ok(ToolOutput::success(blocks.join("\n\n")))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_pass_includes_sentinel_and_header() {
        let s = render_verify_result("true", 0, "all good", 3000);
        assert!(s.contains("$ true"), "header missing: {s}");
        assert!(s.contains("exit_code=0"), "exit code missing: {s}");
        assert!(s.contains("all good"), "output missing: {s}");
        assert!(
            s.contains("VERIFY_RESULT: PASS exit_code=0 command=true"),
            "sentinel missing/wrong: {s}"
        );
    }

    #[test]
    fn render_fail_sentinel() {
        let s = render_verify_result("cargo test", 101, "boom", 3000);
        assert!(
            s.contains("VERIFY_RESULT: FAIL exit_code=101 command=cargo test"),
            "fail sentinel wrong: {s}"
        );
    }

    #[test]
    fn render_sentinel_is_last_line() {
        let s = render_verify_result("x", 0, "noise\nmore", 3000);
        let last = s.lines().last().unwrap_or("");
        assert!(
            last.starts_with("VERIFY_RESULT:"),
            "last line not sentinel: {last:?}"
        );
    }

    #[test]
    fn render_caps_long_output_tail() {
        let big = "x".repeat(10_000);
        let s = render_verify_result("c", 0, &big, 100);
        // The elision is noted, and the rendered body must not contain the full 10k.
        assert!(s.contains("of 10000 bytes"), "elision note missing: {s}");
        assert!(s.len() < 1_000, "tail not capped, len={}", s.len());
        assert!(
            s.contains("VERIFY_RESULT: PASS"),
            "sentinel survived cap: {s}"
        );
    }

    #[test]
    fn parse_roundtrips_render_pass() {
        let s = render_verify_result("cargo test --workspace", 0, "ok", 3000);
        let got = parse_latest_verify(&s).expect("should parse");
        assert_eq!(
            got,
            VerifyOutcome {
                passed: true,
                exit_code: 0,
                command: "cargo test --workspace".to_string()
            }
        );
    }

    #[test]
    fn parse_roundtrips_render_fail() {
        let s = render_verify_result("npm test", 1, "fail", 3000);
        let got = parse_latest_verify(&s).expect("should parse");
        assert!(!got.passed);
        assert_eq!(got.exit_code, 1);
        assert_eq!(got.command, "npm test");
    }

    #[test]
    fn parse_returns_latest_of_many() {
        // A FAIL then a later PASS — the latest verdict wins.
        let transcript = format!(
            "{}\n{}",
            render_verify_result("cargo test", 101, "first attempt failed", 3000),
            render_verify_result("cargo test", 0, "now passes", 3000),
        );
        let got = parse_latest_verify(&transcript).expect("should parse");
        assert!(got.passed, "latest should be PASS, got {got:?}");
        assert_eq!(got.exit_code, 0);
    }

    #[test]
    fn parse_none_when_absent() {
        assert_eq!(parse_latest_verify("no verification happened here"), None);
    }

    #[test]
    fn parse_ignores_prose_echo_and_trusts_canonical_sentinel() {
        // The verify TOOL emits a canonical sentinel (carrying `exit_code=`). The
        // model may later ECHO "VERIFY_RESULT: ..." in prose, possibly a stale
        // FAIL. The parser must trust the canonical tool sentinel, never a bare
        // prose mention — otherwise a deterministic gate is spoofable.
        let transcript = "[tool] VERIFY_RESULT: PASS exit_code=0 command=pytest\n\
                          assistant: Earlier it showed VERIFY_RESULT: FAIL but I fixed it; passes now.";
        let got = parse_latest_verify(transcript).expect("canonical sentinel present");
        assert!(
            got.passed,
            "must trust the canonical tool sentinel, not the prose 'FAIL': {got:?}"
        );
        assert_eq!(got.exit_code, 0);
        assert_eq!(got.command, "pytest");
    }

    #[test]
    fn parse_none_when_only_a_prose_echo_exists() {
        // A bare prose mention with no structured fields is not a verdict.
        assert_eq!(
            parse_latest_verify("the docs say to look for VERIFY_RESULT: PASS lines"),
            None
        );
    }

    #[test]
    fn parse_handles_sentinel_embedded_in_a_tool_result_line() {
        // conversation_text may inline a tool result; the sentinel can be mid-text,
        // but it terminates at the newline.
        let transcript =
            "[Tool result: ...\nVERIFY_RESULT: PASS exit_code=0 command=cargo check\nmore text]";
        let got = parse_latest_verify(transcript).expect("should parse");
        assert!(got.passed);
        assert_eq!(got.command, "cargo check");
    }

    // ---- VerifyCommandTool ----

    fn run_verify(tool: &VerifyCommandTool) -> ToolOutput {
        // Drive the tool's async execute on a fresh runtime (the tool spawns
        // subprocesses; no LLM/network involved).
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(tool.execute(&crate::ExecutionContext::default(), json!({})))
            .expect("tool execute should not error at the framework level")
    }

    #[test]
    fn definition_is_verify_with_no_args() {
        let tool = VerifyCommandTool::new(std::env::temp_dir(), vec!["true".into()]);
        let def = tool.definition();
        assert_eq!(def.name, "verify");
        // No agent-supplied arguments — the command is harness-configured.
        assert_eq!(def.input_schema["properties"], json!({}));
    }

    #[test]
    fn verify_passes_on_true() {
        let dir = tempfile::tempdir().unwrap();
        let tool = VerifyCommandTool::new(dir.path(), vec!["true".into()]);
        let out = run_verify(&tool);
        assert!(
            !out.is_error,
            "a passing verify is not a tool error: {}",
            out.content
        );
        let parsed = parse_latest_verify(&out.content).expect("sentinel present");
        assert!(parsed.passed);
        assert_eq!(parsed.exit_code, 0);
    }

    #[test]
    fn verify_reports_fail_without_tool_error() {
        let dir = tempfile::tempdir().unwrap();
        let tool = VerifyCommandTool::new(dir.path(), vec!["exit 3".into()]);
        let out = run_verify(&tool);
        // A failing verification is a SUCCESSFUL tool call that reports FAIL —
        // the agent must read it and repair, not treat it as a tool malfunction.
        assert!(
            !out.is_error,
            "FAIL must not be a tool error: {}",
            out.content
        );
        let parsed = parse_latest_verify(&out.content).expect("sentinel present");
        assert!(!parsed.passed);
        assert_eq!(parsed.exit_code, 3);
    }

    #[test]
    fn verify_runs_in_workspace_cwd() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker_file.txt"), "x").unwrap();
        let tool = VerifyCommandTool::new(dir.path(), vec!["ls".into()]);
        let out = run_verify(&tool);
        assert!(
            out.content.contains("marker_file.txt"),
            "verify should run in the workspace cwd: {}",
            out.content
        );
    }

    #[test]
    fn verify_fail_fast_skips_later_commands() {
        let dir = tempfile::tempdir().unwrap();
        let tool = VerifyCommandTool::new(
            dir.path(),
            vec!["false".into(), "echo SHOULD_NOT_RUN".into()],
        );
        let out = run_verify(&tool);
        assert!(
            !out.content.contains("SHOULD_NOT_RUN"),
            "fail-fast must stop before the second command: {}",
            out.content
        );
        assert!(!parse_latest_verify(&out.content).unwrap().passed);
    }

    #[test]
    fn verify_all_pass_overall_pass() {
        let dir = tempfile::tempdir().unwrap();
        let tool = VerifyCommandTool::new(dir.path(), vec!["true".into(), "true".into()]);
        let out = run_verify(&tool);
        assert!(parse_latest_verify(&out.content).unwrap().passed);
    }

    #[test]
    fn verify_no_command_configured_is_tool_error() {
        let tool = VerifyCommandTool::new(std::env::temp_dir(), vec![]);
        let out = run_verify(&tool);
        assert!(out.is_error, "an unconfigured verify is a misconfiguration");
        assert!(out.content.to_lowercase().contains("no verify"));
    }
}
