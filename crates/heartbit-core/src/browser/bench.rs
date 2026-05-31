//! Live browser-agent benchmark — the "go to a real site and perform actions"
//! smoke test, as a repeatable evaluation process.
//!
//! The unit tests prove navigation works; this proves the *agent loop* works
//! against live, uncontrolled web pages: a real LLM (Kimi K2 via OpenRouter)
//! drives real Chrome through the full [`BrowserAgentBuilder`] stack to complete
//! multi-step interactive tasks (log in, wait for async content, navigate +
//! extract), and each outcome is graded by an INDEPENDENT deterministic oracle
//! — we re-snapshot the real page after the run and check a verbatim ground-truth
//! signal, rather than trusting the agent's own "done" claim (the
//! Online-Mind2Web / "Illusion of Progress" lesson: agents over-report success).
//!
//! Design mirrors the rest of the module: the gradable core ([`Oracle::grade`],
//! [`scorecard`]) is pure and unit-tested; the live driver ([`run_bench`]) and
//! the `#[ignore]` live suite are the thin shells over real Chrome + a real model.

use std::sync::Arc;
use std::time::Instant;

use crate::execution_context::ExecutionContext;
use crate::llm::LlmProvider;
use crate::tool::Tool;

use super::builder::BrowserAgentBuilder;

/// A deterministic success oracle, graded against the REAL post-run page and the
/// agent's answer — independent of whatever the agent claims it did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Oracle {
    /// The final-page accessibility snapshot must contain this verbatim substring
    /// (e.g. a success banner that only renders once the task is complete).
    FinalPageContains(String),
    /// The agent's text answer must contain this substring (case-insensitive) —
    /// for extraction tasks where the proof is the value reported, not page state.
    AgentAnswerContains(String),
    /// The final page URL must contain this fragment (e.g. `/secure` after login).
    UrlContains(String),
}

impl Oracle {
    /// Grade an outcome. `snapshot` is an independent post-run `take_snapshot` of
    /// the live page; `answer` is the agent's final text. Pure + unit-testable.
    pub fn grade(&self, snapshot: &str, answer: &str) -> bool {
        match self {
            Oracle::FinalPageContains(s) => snapshot.contains(s.as_str()),
            Oracle::AgentAnswerContains(s) => answer.to_lowercase().contains(&s.to_lowercase()),
            Oracle::UrlContains(s) => {
                snapshot_url(snapshot).is_some_and(|u| u.contains(s.as_str()))
            }
        }
    }
}

/// Extract the `RootWebArea` `url="..."` value from a snapshot, if present.
fn snapshot_url(snapshot: &str) -> Option<&str> {
    let line = snapshot.lines().find(|l| l.contains("RootWebArea"))?;
    let key = "url=\"";
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// One benchmark task: a natural-language goal for the agent + a deterministic
/// oracle for grading.
#[derive(Debug, Clone)]
pub struct BenchTask {
    /// Short identifier.
    pub name: String,
    /// Difficulty label (easy/medium/hard/hardest) for the scorecard.
    pub difficulty: String,
    /// Hosts the agent is allowed to navigate to (deny-by-default otherwise).
    pub allow_hosts: Vec<String>,
    /// The task handed to the browser agent.
    pub instruction: String,
    /// Deterministic success check.
    pub oracle: Oracle,
    /// Turn cap for this task's ReAct loop.
    pub max_turns: usize,
}

/// Outcome of running one [`BenchTask`].
#[derive(Debug, Clone)]
pub struct BenchResult {
    /// Task name.
    pub name: String,
    /// Difficulty label.
    pub difficulty: String,
    /// Whether the independent oracle judged the task complete.
    pub passed: bool,
    /// Tool calls the agent made.
    pub tool_calls: usize,
    /// Input tokens consumed.
    pub input_tokens: u32,
    /// Output tokens produced.
    pub output_tokens: u32,
    /// Estimated cost in USD, if the provider reported it.
    pub cost_usd: Option<f64>,
    /// Wall-clock duration in milliseconds.
    pub millis: u128,
    /// First ~160 chars of the agent's answer (for the trace).
    pub answer_excerpt: String,
    /// Error, if the build or run failed before grading.
    pub error: Option<String>,
}

/// Run a benchmark suite live. For each task, builds a fresh agent over the
/// SHARED MCP `tools` (one Chrome, reused), runs it, then INDEPENDENTLY grades by
/// taking a fresh snapshot of the real page. Never panics on a task failure — a
/// build/run error is captured into [`BenchResult::error`] and graded as failed,
/// so one bad task can't abort the suite.
pub async fn run_bench<P: LlmProvider>(
    provider: Arc<P>,
    tools: Vec<Arc<dyn Tool>>,
    tasks: &[BenchTask],
) -> Vec<BenchResult> {
    let ctx = ExecutionContext::default();
    let snapshot_tool = tools
        .iter()
        .find(|t| t.definition().name == "take_snapshot")
        .cloned();

    let mut results = Vec::with_capacity(tasks.len());
    for task in tasks {
        let started = Instant::now();
        let mut r = BenchResult {
            name: task.name.clone(),
            difficulty: task.difficulty.clone(),
            passed: false,
            tool_calls: 0,
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: None,
            millis: 0,
            answer_excerpt: String::new(),
            error: None,
        };

        match BrowserAgentBuilder::new(Arc::clone(&provider))
            .name(task.name.clone())
            .allow_hosts(task.allow_hosts.clone())
            .max_turns(task.max_turns)
            .build_with_tools(tools.clone())
        {
            Err(e) => r.error = Some(format!("build: {e}")),
            Ok(agent) => match agent.execute(&task.instruction).await {
                Err(e) => r.error = Some(format!("run: {e}")),
                Ok(out) => {
                    let snap = match &snapshot_tool {
                        Some(t) => t
                            .execute(&ctx, serde_json::json!({}))
                            .await
                            .map(|o| o.content)
                            .unwrap_or_default(),
                        None => String::new(),
                    };
                    r.passed = task.oracle.grade(&snap, &out.result);
                    r.tool_calls = out.tool_calls_made;
                    r.input_tokens = out.tokens_used.input_tokens;
                    r.output_tokens = out.tokens_used.output_tokens;
                    r.cost_usd = out.estimated_cost_usd;
                    r.answer_excerpt = out.result.chars().take(160).collect();
                }
            },
        }
        r.millis = started.elapsed().as_millis();
        results.push(r);
    }
    results
}

/// Render a human-readable scorecard from results.
pub fn scorecard(results: &[BenchResult]) -> String {
    use std::fmt::Write as _;
    let passed = results.iter().filter(|r| r.passed).count();
    let mut s = String::new();
    let _ = writeln!(
        s,
        "=== Browser-agent benchmark: {}/{} tasks passed ===",
        passed,
        results.len()
    );
    let _ = writeln!(
        s,
        "{:<24} {:<8} {:<5} {:>6} {:>8} {:>8} {:>9}",
        "task", "diff", "pass", "calls", "in_tok", "out_tok", "ms"
    );
    for r in results {
        let _ = writeln!(
            s,
            "{:<24} {:<8} {:<5} {:>6} {:>8} {:>8} {:>9}",
            r.name,
            r.difficulty,
            if r.passed { "PASS" } else { "FAIL" },
            r.tool_calls,
            r.input_tokens,
            r.output_tokens,
            r.millis
        );
        if let Some(e) = &r.error {
            let _ = writeln!(s, "    ! {e}");
        } else if !r.answer_excerpt.is_empty() {
            let _ = writeln!(s, "    answer: {}", r.answer_excerpt.replace('\n', " "));
        }
    }
    s
}

/// The default tiered benchmark suite: a sanity task plus three genuinely
/// multi-step interactive tasks — form login, async wait-for-stability, and
/// multi-page navigation + extraction. Every oracle value was verified verbatim
/// against the live sites (2026-05-31 recon), so a pass means the agent actually
/// achieved the goal, not that it claimed to.
pub fn bench_suite() -> Vec<BenchTask> {
    vec![
        // Sanity: navigate + read. Confirms the harness is wired end-to-end.
        BenchTask {
            name: "example_extract".into(),
            difficulty: "easy".into(),
            allow_hosts: vec!["example.com".into()],
            instruction: "Go to https://example.com and report the main heading text shown on \
                           the page."
                .into(),
            oracle: Oracle::FinalPageContains("Example Domain".into()),
            max_turns: 6,
        },
        // Form auth: fill two fields, submit, verify the state transition to the
        // secure area (the success banner only renders after a correct login).
        BenchTask {
            name: "the_internet_login".into(),
            difficulty: "medium".into(),
            allow_hosts: vec!["the-internet.herokuapp.com".into()],
            instruction: "Go to https://the-internet.herokuapp.com/login and log in with \
                           username \"tomsmith\" and password \"SuperSecretPassword!\". Confirm \
                           you reached the secure area."
                .into(),
            oracle: Oracle::FinalPageContains("You logged into a secure area!".into()),
            max_turns: 12,
        },
        // Async settle: click Start, wait out a 5s spinner, read the revealed text
        // (not in the initial DOM) — exercises the wait-for-stability discipline.
        BenchTask {
            name: "dynamic_loading_settle".into(),
            difficulty: "hard".into(),
            allow_hosts: vec!["the-internet.herokuapp.com".into()],
            instruction: "Go to https://the-internet.herokuapp.com/dynamic_loading/2 , click the \
                           Start button, wait for the content to finish loading, and report the \
                           text that appears."
                .into(),
            oracle: Oracle::FinalPageContains("Hello World!".into()),
            max_turns: 14,
        },
        // Multi-step navigation + extraction across pages: home -> Travel category
        // -> the specific book -> read its price. Graded on the reported value.
        BenchTask {
            name: "books_travel_price".into(),
            difficulty: "hardest".into(),
            allow_hosts: vec!["books.toscrape.com".into()],
            instruction: "Go to https://books.toscrape.com , open the Travel category, find the \
                           book titled \"Under the Tuscan Sun\", open its page, and report its \
                           exact price."
                .into(),
            oracle: Oracle::AgentAnswerContains("37.33".into()),
            max_turns: 16,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOGIN_OK_SNAP: &str = r#"uid=1_0 RootWebArea "The Internet" url="https://the-internet.herokuapp.com/secure"
  uid=1_1 StaticText "You logged into a secure area!"
  uid=1_2 button "Logout""#;

    const LOGIN_FAIL_SNAP: &str = r#"uid=1_0 RootWebArea "The Internet" url="https://the-internet.herokuapp.com/login"
  uid=1_1 StaticText "Your username is invalid!"
  uid=1_2 textbox "Username""#;

    #[test]
    fn final_page_contains_grades_on_banner() {
        let o = Oracle::FinalPageContains("You logged into a secure area!".into());
        assert!(o.grade(LOGIN_OK_SNAP, "I logged in."));
        assert!(!o.grade(LOGIN_FAIL_SNAP, "I logged in.")); // agent claims success, page says no
    }

    #[test]
    fn url_contains_grades_on_redirect() {
        let o = Oracle::UrlContains("/secure".into());
        assert!(o.grade(LOGIN_OK_SNAP, ""));
        assert!(!o.grade(LOGIN_FAIL_SNAP, ""));
    }

    #[test]
    fn agent_answer_contains_is_case_insensitive() {
        let o = Oracle::AgentAnswerContains("£51.77".into());
        assert!(o.grade("", "The price is £51.77 incl. tax."));
        assert!(!o.grade("", "I could not find the price."));
        let o2 = Oracle::AgentAnswerContains("Hello World".into());
        assert!(o2.grade("", "the revealed text was HELLO WORLD!"));
    }

    #[test]
    fn oracle_does_not_trust_agent_over_page() {
        // The whole point: a lying "done" answer must still FAIL when the page
        // doesn't show the success signal.
        let o = Oracle::FinalPageContains("secure area".into());
        assert!(
            !o.grade(LOGIN_FAIL_SNAP, "Done! Successfully logged in."),
            "page is the source of truth, not the agent's claim"
        );
    }

    #[test]
    fn snapshot_url_extracts_root_url() {
        assert_eq!(
            snapshot_url(LOGIN_OK_SNAP),
            Some("https://the-internet.herokuapp.com/secure")
        );
        assert_eq!(snapshot_url("uid=1_0 RootWebArea \"x\""), None); // no url=
        assert_eq!(snapshot_url(""), None);
    }

    #[test]
    fn bench_suite_is_wellformed() {
        let suite = bench_suite();
        assert_eq!(suite.len(), 4, "tiered suite has 4 tasks");
        for t in &suite {
            assert!(!t.name.is_empty());
            assert!(
                !t.allow_hosts.is_empty(),
                "{} must allowlist a host",
                t.name
            );
            assert!(
                t.instruction.contains("http"),
                "{} instruction must name a URL",
                t.name
            );
            assert!(t.max_turns >= 4, "{} max_turns too low", t.name);
            // The task's allowlisted host should appear in its instruction URL.
            assert!(
                t.allow_hosts
                    .iter()
                    .any(|h| t.instruction.contains(h.as_str())),
                "{} instruction must target its allowlisted host",
                t.name
            );
        }
        let diffs: Vec<_> = suite.iter().map(|t| t.difficulty.as_str()).collect();
        assert!(
            diffs.contains(&"easy") && diffs.contains(&"hardest"),
            "suite must span easy..hardest, got {diffs:?}"
        );
    }

    /// LIVE: Kimi K2 (OpenRouter) drives real Chrome through the whole benchmark
    /// suite on real sites, each graded by an independent oracle (a fresh
    /// post-run snapshot of the real page, not the agent's claim). This is the
    /// "go to a website and perform actions" smoke test. Run:
    ///
    /// ```text
    /// OPENROUTER_API_KEY=sk-or-... cargo test -p heartbit-core --lib \
    ///   browser::bench::tests::live_kimi_browser_benchmark -- --ignored --nocapture
    /// ```
    #[tokio::test]
    #[ignore = "live: needs OpenRouter key + spawns real Chrome; hits 3 external sites"]
    async fn live_kimi_browser_benchmark() {
        let key = std::env::var("LLM_API_KEY")
            .or_else(|_| std::env::var("OPENROUTER_API_KEY"))
            .expect("set LLM_API_KEY or OPENROUTER_API_KEY to run this live benchmark");
        let provider = Arc::new(crate::OpenRouterProvider::new(
            key,
            "moonshotai/kimi-k2-0905",
        ));

        let chrome = "/usr/bin/google-chrome";
        let extra: Vec<String> = if std::path::Path::new(chrome).exists() {
            vec!["--executable-path".to_string(), chrome.to_string()]
        } else {
            Vec::new()
        };
        let tools = crate::connect_preset_with_args("chrome-devtools", &extra)
            .await
            .expect("connect chrome-devtools preset");

        let suite = bench_suite();
        let results = run_bench(provider, tools, &suite).await;
        eprintln!("\n{}", scorecard(&results));

        // Sanity gate: the harness must work end-to-end, so the easy extract task
        // must pass. The harder tasks are MEASURED, not gated — their pass/fail is
        // the benchmark signal, printed in the scorecard above.
        let easy = results
            .iter()
            .find(|r| r.name == "example_extract")
            .expect("easy task ran");
        assert!(
            easy.passed,
            "harness sanity: the easy extract task must pass (see scorecard above)"
        );
    }

    #[test]
    fn scorecard_counts_and_formats() {
        let results = vec![
            BenchResult {
                name: "login".into(),
                difficulty: "hard".into(),
                passed: true,
                tool_calls: 5,
                input_tokens: 1200,
                output_tokens: 80,
                cost_usd: Some(0.01),
                millis: 4200,
                answer_excerpt: "logged in".into(),
                error: None,
            },
            BenchResult {
                name: "basket".into(),
                difficulty: "hardest".into(),
                passed: false,
                tool_calls: 9,
                input_tokens: 3000,
                output_tokens: 140,
                cost_usd: None,
                millis: 9000,
                answer_excerpt: String::new(),
                error: Some("run: timeout".into()),
            },
        ];
        let card = scorecard(&results);
        assert!(card.contains("1/2 tasks passed"));
        assert!(card.contains("login"));
        assert!(card.contains("PASS"));
        assert!(card.contains("FAIL"));
        assert!(card.contains("! run: timeout"));
    }
}
