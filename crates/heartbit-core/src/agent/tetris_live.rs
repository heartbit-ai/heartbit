//! LIVE demo: a goal-gated **dynamic workflow** that *builds* a Tetris web app
//! and then *verifies it actually works in a real browser* — with the browser
//! check baked into a [`GoalCondition`](super::goal::GoalCondition).
//!
//! This ties together three heartbit features end-to-end against `qwen3-235b`:
//!
//! 1. **Dynamic workflow** ([`flow::agent`](super::flow::agent)): the build step
//!    is a workflow leaf carrying file tools (`write`/`bash`) rooted at a tempdir,
//!    gated by a build goal whose independent judge reads the `wc`/`grep`
//!    validation evidence from the transcript (not the agent's claim).
//! 2. **Browser harness** ([`BrowserAgentBuilder`](crate::browser::BrowserAgentBuilder)):
//!    drives real headless Chrome (chrome-devtools MCP) against the freshly-built
//!    page served from an in-process static server.
//! 3. **Goal as browser test**: the verify goal is met *only* when the transcript
//!    shows — via `evaluate_script` reads of `window.__tetris` around a `press_key`
//!    — that an arrow key moved the active piece's **column**, proving the game
//!    responds to keyboard input. Gravity is frozen first (`paused = true`) so the
//!    row auto-dropping can't masquerade as a response (the WebJudge / deterministic
//!    -evidence pattern — see `tasks/browser-bot-sota-2026-05-31.md`).
//!
//! Run it (real Chrome + network + OpenRouter key required, hence `#[ignore]`):
//! ```text
//! OPENROUTER_API_KEY=sk-or-... cargo test -p heartbit-core --lib \
//!   agent::tetris_live::live_goal_workflow_builds_and_browser_verifies_tetris_qwen \
//!   -- --ignored --nocapture
//! ```
//!
//! Iterate the *verify* phase cheaply (skip the build, serve a fixed game):
//! ```text
//! HEARTBIT_TETRIS_HTML=/tmp/heartbit-tetris-latest.html OPENROUTER_API_KEY=... \
//!   cargo test ... -- --ignored --nocapture
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::agent::events::{AgentEvent, OnEvent};
use crate::agent::flow::{WorkflowCtx, agent, log, phase};
use crate::agent::goal::GoalCondition;
use crate::llm::BoxedProvider;
use crate::tool::Tool;
use crate::tool::builtins::{BuiltinToolsConfig, builtin_tools};
use crate::{OnWorkflowEvent, WorkflowEvent};

const MODEL: &str = "qwen/qwen3-235b-a22b-2507";

/// What the dynamic-workflow build leaf is asked to produce. The contract is
/// deliberately minimal-but-real (spawn / move / gravity / lock / line-clear)
/// plus two testability affordances the browser verifier relies on: DOM-rendered
/// cells with class `filled`, and a `window.__tetris` state object exposing a
/// **settable `paused`** flag that halts gravity (but not the keyboard).
///
/// Critically it FORBIDS `requestAnimationFrame` / any continuous render loop:
/// chrome-devtools MCP waits for the page to go idle before `evaluate_script`,
/// and a perpetually-animating page never settles, so the verifier's state reads
/// would time out. In-place, event-driven DOM updates keep the page idle.
const BUILD_PROMPT: &str = r#"Build a small but REAL, playable Tetris game as a SINGLE self-contained static
web page and save it to `index.html` in your working directory with the `write`
tool. No build step, no external files, no CDN — inline CSS and JavaScript only,
so it runs by simply loading the file over HTTP.

Required mechanics (keep it minimal but actually functioning):
- A 10-column x 20-row playfield RENDERED AS DOM ELEMENTS (a grid of <div> cells,
  NOT a <canvas>), so the board is visible in the accessibility tree. Create the
  200 cell <div>s ONCE at startup; thereafter render by TOGGLING the CSS class
  "filled" on those existing cells in place (e.g. <div class="cell filled">). An
  occupied cell has class "filled"; an empty cell does not.
- Tetromino pieces spawn at the top, near the horizontal middle (so they have
  room to move left).
- Gravity: the active piece falls one row at a time driven by a SINGLE
  setInterval at 800ms. When a piece can fall no further it locks, then a new
  piece spawns.
- Full rows clear and increment a line counter.
- Keyboard controls bound on the document: ArrowLeft moves the active piece one
  column left, ArrowRight one column right, ArrowDown soft-drops one row, ArrowUp
  rotates. All moves clamp to the walls.

CRITICAL performance/testability rule — the page MUST go idle between events:
- Do NOT use requestAnimationFrame. Do NOT use any continuous render/animation
  loop. Render ONLY in response to an event (a gravity tick or a keypress), by
  updating the existing cells' "filled" classes in place, then stop. Between the
  800ms gravity ticks the page must be completely idle (no timers mutating the
  DOM, no rAF).

Required TESTABILITY HOOK — expose the live game state on `window.__tetris`,
re-synced whenever the state changes (each gravity tick and each keypress), with
AT LEAST these fields:
    window.__tetris = {
      cols, rows, score, lines, gameOver,
      paused,                       // settable boolean; see below
      activePiece: { type, row, col }   // col = the active piece's left-most column
    }
Rules the hook MUST obey:
- Pressing ArrowLeft decreases activePiece.col by exactly 1 (unless against the
  left wall); ArrowRight increases it by exactly 1 (unless against the right wall).
- `window.__tetris.paused` is settable from outside. When it is set to true, the
  gravity setInterval callback must do NOTHING (no piece drop, no DOM change), but
  the ARROW KEYS MUST STILL MOVE the piece. So while paused the page is fully idle
  yet still keyboard-responsive — this lets a test freeze the board and prove the
  keyboard works in isolation.

After writing index.html, VALIDATE it with the `bash` tool and SHOW the output:
    wc -c index.html
    grep -nE "window.__tetris|keydown|ArrowLeft|ArrowRight|filled|paused" index.html
Also confirm you did NOT use requestAnimationFrame:
    grep -c "requestAnimationFrame" index.html   # must print 0
Then stop. You are done when index.html exists, the validation output concretely
shows the file is non-trivial and contains the window.__tetris hook, a keydown
handler, ArrowLeft/ArrowRight handling, the "filled" cell class, and the paused
flag, AND the requestAnimationFrame count is 0."#;

/// The build goal's objective, graded by an INDEPENDENT judge over the transcript.
const BUILD_OBJECTIVE: &str = "A single self-contained static `index.html` Tetris game has been written to the \
     workspace AND validated by commands whose output is visible in the transcript. \
     The objective is met ONLY when the transcript shows the `wc -c index.html` size \
     (non-trivial, not a stub), `grep` output proving index.html contains ALL of: the \
     `window.__tetris` state hook, a `keydown` handler, `ArrowLeft` and `ArrowRight` \
     handling, the DOM `filled` cell class, and the settable `paused` flag, AND a \
     `grep -c requestAnimationFrame` result of 0 (no continuous render loop). The agent \
     merely asserting it wrote the file is NOT evidence — require the command output.";

/// The browser verify task. `URL_PLACEHOLDER` is replaced with the live server URL.
/// The protocol freezes gravity first, then proves a keypress moves the COLUMN —
/// a gravity-independent observable.
const VERIFY_TASK: &str = r#"A Tetris web game is being served at URL_PLACEHOLDER . Verify IN THE REAL
BROWSER that it is a working interactive Tetris that responds to keyboard input —
not a static page. Follow this protocol exactly:

1. navigate_page to URL_PLACEHOLDER
2. take_snapshot and confirm a grid-shaped Tetris board is rendered.
3. FREEZE GRAVITY and read the starting column with ONE synchronous evaluate_script
   call (the function must return a plain value, not a Promise):
     () => { window.__tetris.paused = true; return { col: window.__tetris.activePiece.col, paused: window.__tetris.paused }; }
   Record the returned `col` as BEFORE_COL.
4. If BEFORE_COL is 0 the piece is on the left wall: press_key "ArrowRight".
   Otherwise: press_key "ArrowLeft".
5. Read the column again:
     () => window.__tetris.activePiece.col
   Record it as AFTER_COL.
6. Confirm the COLUMN moved by exactly one in the direction of the key you pressed
   (ArrowLeft => AFTER_COL == BEFORE_COL - 1 ; ArrowRight => AFTER_COL == BEFORE_COL + 1).
   Because gravity is paused, only the keypress can change the state, so a column
   move proves the keyboard drives the game.

If an evaluate_script call ever errors or times out, immediately take_snapshot to
see the page state, then retry the same call once. Report BEFORE_COL, the key you
pressed, and AFTER_COL explicitly in your answer."#;

/// The browser goal's objective. The judge keys the verdict on the gravity-
/// independent column delta — never on the row, never on the page merely rendering.
const VERIFY_OBJECTIVE: &str = "The Tetris game served in the browser is proven to be a WORKING, keyboard- \
     interactive game. The objective is met ONLY when the transcript shows, with \
     gravity paused (window.__tetris.paused set true), two evaluate_script reads of \
     window.__tetris.activePiece.col surrounding a single press_key of an arrow key, \
     AND the column value moved by exactly one in the direction of that arrow \
     (ArrowLeft => col decreased by 1, ArrowRight => col increased by 1). A change in \
     the piece's ROW does NOT count (gravity moves the row). The page merely rendering, \
     or the agent claiming it works without the before/after column evidence, does NOT \
     satisfy the objective.";

/// Locate a real Chrome/Chromium executable for the live test.
fn live_chrome_path() -> Option<String> {
    for env_key in ["CHROME_PATH", "GOOGLE_CHROME_BIN", "CHROME_BIN"] {
        if let Ok(v) = std::env::var(env_key)
            && !v.is_empty()
            && std::path::Path::new(&v).exists()
        {
            return Some(v);
        }
    }
    for cand in [
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ] {
        if std::path::Path::new(cand).exists() {
            return Some(cand.to_string());
        }
    }
    None
}

/// Aborts the in-process static server task when the test scope ends — no child
/// process to leak or `kill`, so the never-kill-running-servers rule is moot.
struct ServerGuard(tokio::task::JoinHandle<()>);

impl Drop for ServerGuard {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Serve a single HTML document from an in-process tokio listener on
/// `127.0.0.1:<ephemeral>`. Returns the bound port and a guard that stops the
/// server on drop. `/` and `/index.html` return the page; everything else 404s
/// (sidesteps MIME strictness for a single inline-everything file).
async fn serve_html(html: String) -> (u16, ServerGuard) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let port = listener.local_addr().expect("local_addr").port();

    let handle = tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(_) => break,
            };
            let body = html.clone();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                let first = String::from_utf8_lossy(&buf[..n]);
                let path = first
                    .lines()
                    .next()
                    .unwrap_or("")
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("/");
                let (status, ctype, payload): (&str, &str, Vec<u8>) =
                    if path == "/" || path == "/index.html" {
                        ("200 OK", "text/html; charset=utf-8", body.into_bytes())
                    } else {
                        (
                            "404 Not Found",
                            "text/plain; charset=utf-8",
                            b"not found".to_vec(),
                        )
                    };
                let header = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\n\
                     Cache-Control: no-store\r\nConnection: close\r\n\r\n",
                    payload.len()
                );
                let _ = sock.write_all(header.as_bytes()).await;
                let _ = sock.write_all(&payload).await;
                let _ = sock.flush().await;
            });
        }
    });

    (port, ServerGuard(handle))
}

/// Build the Tetris game via the dynamic-workflow `agent()` leaf + build goal,
/// returning the page HTML. Honors `HEARTBIT_TETRIS_HTML` to skip the build and
/// serve a fixed game (cheap verify-phase iteration). Persists the built game to
/// `<tmp>/heartbit-tetris-latest.html` for inspection.
async fn build_tetris_html(key: &str) -> (String, u64, String) {
    if let Ok(path) = std::env::var("HEARTBIT_TETRIS_HTML")
        && std::path::Path::new(&path).exists()
    {
        let html = std::fs::read_to_string(&path).expect("read HEARTBIT_TETRIS_HTML");
        eprintln!("[build] skipped — serving fixed game from {path}");
        return (html, 0, format!("(build skipped: {path})"));
    }

    let build_provider = Arc::new(BoxedProvider::new(crate::OpenRouterProvider::new(
        key.to_string(),
        MODEL,
    )));
    let build_judge = Arc::new(BoxedProvider::new(crate::OpenRouterProvider::new(
        key.to_string(),
        MODEL,
    )));

    let tmp = tempfile::tempdir().expect("tempdir");
    let work = tmp.path().to_path_buf();

    let wf_events = Arc::new(std::sync::Mutex::new(Vec::<WorkflowEvent>::new()));
    let sink = Arc::clone(&wf_events);
    let on_wf: Arc<OnWorkflowEvent> = Arc::new(move |e: WorkflowEvent| {
        sink.lock().expect("wf events lock").push(e);
    });

    let ctx = WorkflowCtx::builder(build_provider)
        .max_concurrency(2)
        .max_agents(50)
        .on_event(on_wf)
        .build()
        .expect("build workflow ctx");

    let build_text = {
        let _guard = phase(&ctx, "build");
        log(
            &ctx,
            "building a self-contained DOM Tetris with a window.__tetris hook",
        );

        let builtins: Vec<Arc<dyn Tool>> = builtin_tools(BuiltinToolsConfig {
            workspace: Some(work.clone()),
            dangerous_tools: true,
            ..Default::default()
        });

        let build_goal =
            GoalCondition::new(BUILD_OBJECTIVE, build_judge).with_max_continuations(10);

        // A goal leaf surfaces only its text; the on-disk file (below) is the real
        // success signal, so a continuation-cap / max-turns outcome must NOT panic.
        match agent(&ctx, BUILD_PROMPT)
            .label("tetris-builder")
            .tools(builtins)
            .goal(build_goal)
            .run()
            .await
        {
            Ok(Some(text)) => text,
            Ok(None) => "(build leaf was skipped)".to_string(),
            Err(e) => format!("(build leaf ended with: {e})"),
        }
    };

    let index_path = work.join("index.html");
    let html = std::fs::read_to_string(&index_path).unwrap_or_else(|e| {
        panic!("build did not produce index.html at {index_path:?}: {e}\nbuild said:\n{build_text}")
    });

    // Persist a copy for inspection / cheap re-runs (best-effort).
    let debug_path = std::env::temp_dir().join("heartbit-tetris-latest.html");
    let _ = std::fs::write(&debug_path, &html);
    eprintln!("[build] persisted game to {}", debug_path.display());

    let build_tokens: u64 = wf_events
        .lock()
        .expect("wf events lock")
        .iter()
        .filter_map(|e| match e {
            WorkflowEvent::AgentFinished { usage, .. } => {
                Some(usage.input_tokens as u64 + usage.output_tokens as u64)
            }
            _ => None,
        })
        .sum();

    (html, build_tokens, build_text)
}

/// LIVE: a goal-gated dynamic workflow builds a Tetris web app, then the browser
/// harness verifies — as part of a second goal — that it really plays in Chrome.
#[tokio::test]
#[ignore = "live: needs OpenRouter key + spawns real Chrome + network"]
async fn live_goal_workflow_builds_and_browser_verifies_tetris_qwen() {
    let key = std::env::var("OPENROUTER_API_KEY")
        .or_else(|_| std::env::var("LLM_API_KEY"))
        .expect("set OPENROUTER_API_KEY to run this live test");

    // ---- PHASE 1: build the game (dynamic workflow + build goal) ----
    let (html, build_tokens, build_text) = build_tetris_html(&key).await;
    assert!(
        html.len() > 500,
        "index.html is suspiciously small ({} bytes); build said:\n{build_text}",
        html.len()
    );
    assert!(
        html.contains("__tetris"),
        "index.html is missing the window.__tetris testability hook; build said:\n{build_text}"
    );

    // ---- serve it from an in-process static server ----
    let (port, _server) = serve_html(html.clone()).await;
    let url = format!("http://127.0.0.1:{port}/");

    // ---- PHASE 2: browser harness verifies the game IS the goal ----
    let verify_provider = Arc::new(crate::OpenRouterProvider::new(key.clone(), MODEL));
    let verify_judge = Arc::new(BoxedProvider::new(crate::OpenRouterProvider::new(
        key.clone(),
        MODEL,
    )));

    let turns = Arc::new(AtomicUsize::new(0));
    let tool_log = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let turn_sink = Arc::clone(&turns);
    let log_sink = Arc::clone(&tool_log);
    let on_event: Arc<OnEvent> = Arc::new(move |e: AgentEvent| match e {
        AgentEvent::TurnStarted { turn, .. } => {
            turn_sink.fetch_max(turn, Ordering::SeqCst);
        }
        AgentEvent::ToolCallCompleted {
            tool_name,
            is_error,
            duration_ms,
            output,
            ..
        } => {
            let mut out = output.replace('\n', " ");
            out.truncate(160);
            log_sink.lock().expect("tool log lock").push(format!(
                "{:>6}ms {} {:<18} {}",
                duration_ms,
                if is_error { "ERR" } else { "ok " },
                tool_name,
                out
            ));
        }
        _ => {}
    });

    let verify_task = VERIFY_TASK.replace("URL_PLACEHOLDER", &url);
    let verify_goal = GoalCondition::new(VERIFY_OBJECTIVE, verify_judge).with_max_continuations(5);

    let mut builder = crate::browser::BrowserAgentBuilder::new(verify_provider)
        .name("tetris-verifier")
        .allow_hosts(["127.0.0.1", "localhost"])
        .max_turns(24)
        .max_identical_tool_calls(3)
        .on_event(on_event)
        .goal(verify_goal);
    if let Some(chrome) = live_chrome_path() {
        builder = builder.chrome_executable(chrome);
    }
    let verifier = builder
        .connect()
        .await
        .expect("connect chrome-devtools preset + assemble verifier");

    let out = verifier
        .execute(&verify_task)
        .await
        .expect("browser verify run should succeed");

    // ---- report ----
    eprintln!("\n=== live_goal_workflow_builds_and_browser_verifies_tetris_qwen ===");
    eprintln!("index.html bytes : {}", html.len());
    eprintln!("served at        : {url}");
    eprintln!("build tokens     : {build_tokens}");
    eprintln!("verify turns     : {}", turns.load(Ordering::SeqCst));
    eprintln!("verify toolcalls : {}", out.tool_calls_made);
    eprintln!("verify goal_met  : {:?}", out.goal_met);
    eprintln!("verify tokens    : {:?}", out.tokens_used);
    eprintln!("--- browser tool log ---");
    for line in tool_log.lock().expect("tool log lock").iter() {
        eprintln!("  {line}");
    }
    eprintln!("verify answer    : {}", out.result.trim());

    // Assert the MECHANISMS fired (not that qwen necessarily succeeded — that is
    // what we print and observe): a browser goal ran, and the agent actually drove
    // Chrome (navigate + evaluate_script + press_key is >= 3 tool calls).
    assert!(
        out.goal_met.is_some(),
        "a browser goal was set, so goal_met must be Some"
    );
    assert!(
        out.tool_calls_made >= 3,
        "expected the verifier to drive the browser (navigate + evaluate + press_key), \
         but only {} tool calls were made",
        out.tool_calls_made
    );
}
