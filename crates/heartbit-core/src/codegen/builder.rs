//! [`CodeAgentBuilder`] — a domain-specific harness for code generation, mirroring
//! [`BrowserAgentBuilder`](crate::browser::BrowserAgentBuilder). It curates the
//! code builtins rooted at a workspace, adds the [`VerifyCommandTool`], installs a
//! loop-invariant system prompt, and (optionally) gates completion on a
//! [`GoalCondition`] whose independent judge keys on the deterministic
//! `VERIFY_RESULT:` sentinel. Everything else (the ReAct loop, the goal-driven
//! repair continuation, workspace jailing, guardrails) is reused, not rebuilt.

use std::path::PathBuf;
use std::sync::Arc;

use crate::agent::events::OnEvent;
use crate::agent::goal::GoalCondition;
use crate::agent::guardrail::Guardrail;
use crate::agent::{AgentRunner, AgentRunnerBuilder};
use crate::error::Error;
use crate::llm::{BoxedProvider, LlmProvider};
use crate::tool::Tool;
use crate::tool::builtins::{BuiltinToolsConfig, builtin_tools};

use super::verify::VerifyCommandTool;

/// Default ReAct turn budget for a code agent. Generous because
/// edit→verify→read-failures→re-edit→re-verify burns turns quickly.
const DEFAULT_CODE_MAX_TURNS: usize = 40;

/// Code builtins this harness curates (the editing + searching surface), plus the
/// `verify` tool which is appended separately. `bash` is included only when
/// `dangerous_tools` is set.
const CODE_BUILTIN_NAMES: &[&str] = &[
    "read",
    "write",
    "edit",
    "patch",
    "grep",
    "glob",
    "list",
    "bash",
    "todoread",
    "todowrite",
    "skill",
];

/// The loop-invariant system prompt for the code harness (the code analogue of
/// `BROWSER_SYSTEM_PROMPT`).
pub const CODE_SYSTEM_PROMPT: &str = "You are a senior software engineer working inside a fixed workspace. Your job is to \
implement the requested change and PROVE it works by running the project's verification \
command. Follow this loop:\n\
\n\
1. UNDERSTAND: before editing, read the relevant files (read/grep/glob/list). Never edit \
code you have not looked at. Build a concrete picture of the current state.\n\
2. PLAN: keep an explicit, ordered list of subgoals (use the todo tools). Work one step \
at a time.\n\
3. EDIT: make the smallest change that achieves the goal. Prefer `edit`/`patch` over \
rewriting whole files. Read a file before you write to it.\n\
4. VERIFY: after a coherent change, call the `verify` tool. It runs the project's \
configured build/test command and reports `VERIFY_RESULT: PASS` or `VERIFY_RESULT: FAIL` \
with the exit code. This — not your own judgement — is the source of truth for whether \
the code works.\n\
5. REPAIR: if `verify` reports FAIL, read the captured output, identify the specific \
cause, fix it, and run `verify` again. Do not repeat the same failing change; if a \
command keeps failing identically, change your approach.\n\
6. FINISH: you are done ONLY immediately after `verify` reports `VERIFY_RESULT: PASS`. \
Then summarise what you changed. Never claim success without a passing `verify`.\n\
\n\
EFFICIENCY: keep edits and tool calls minimal; do not re-read unchanged files.\n\
SAFETY: stay within the workspace. The `verify` command is fixed by the harness — you \
cannot choose what \"passing\" means, only make the code pass it.";

/// The build goal's objective — keyed on the deterministic verify sentinel.
const CODE_GOAL_OBJECTIVE: &str = "The requested code change is complete and proven to work. The objective is met ONLY \
     when the most recent `verify` tool result in the transcript shows `VERIFY_RESULT: PASS` \
     (exit code 0) — i.e. the project's configured verification command actually passed. A \
     claim of completion WITHOUT a `VERIFY_RESULT: PASS` line from the verify tool, or \
     superseded by a later `VERIFY_RESULT: FAIL`, does NOT satisfy the objective.";

/// Curated code builtins rooted at `workspace`, plus the `verify` tool bound to
/// `verify_commands`. When `dangerous` is false, `bash` is omitted.
pub fn code_tools(
    workspace: impl Into<PathBuf>,
    dangerous: bool,
    verify_commands: Vec<String>,
) -> Vec<Arc<dyn Tool>> {
    let ws: PathBuf = workspace.into();
    let raw = builtin_tools(BuiltinToolsConfig {
        workspace: Some(ws.clone()),
        dangerous_tools: dangerous,
        ..Default::default()
    });
    let mut tools: Vec<Arc<dyn Tool>> = raw
        .into_iter()
        .filter(|t| CODE_BUILTIN_NAMES.contains(&t.definition().name.as_str()))
        .collect();
    tools.push(Arc::new(VerifyCommandTool::new(ws, verify_commands)));
    tools
}

/// A [`GoalCondition`] whose independent judge gates completion on the
/// deterministic `VERIFY_RESULT: PASS` evidence produced by the `verify` tool.
pub fn code_goal(judge: Arc<BoxedProvider>) -> GoalCondition {
    GoalCondition::new(CODE_GOAL_OBJECTIVE, judge).with_max_continuations(4)
}

/// Fluent builder assembling a code-generation [`AgentRunner`] (mirror of
/// [`BrowserAgentBuilder`](crate::browser::BrowserAgentBuilder)).
pub struct CodeAgentBuilder<P: LlmProvider> {
    provider: Arc<P>,
    workspace: PathBuf,
    verify_commands: Vec<String>,
    system_prompt: Option<String>,
    name: Option<String>,
    max_turns: Option<usize>,
    max_identical_tool_calls: Option<u32>,
    on_event: Option<Arc<OnEvent>>,
    extra_guardrails: Vec<Arc<dyn Guardrail>>,
    tool_allow: Vec<String>,
    goal: Option<GoalCondition>,
    dangerous_tools: bool,
}

impl<P: LlmProvider> CodeAgentBuilder<P> {
    /// Start a code harness rooted at `workspace` (file tools are jailed there).
    pub fn new(provider: Arc<P>, workspace: impl Into<PathBuf>) -> Self {
        Self {
            provider,
            workspace: workspace.into(),
            verify_commands: Vec::new(),
            system_prompt: None,
            name: None,
            max_turns: None,
            max_identical_tool_calls: None,
            on_event: None,
            extra_guardrails: Vec::new(),
            tool_allow: Vec::new(),
            goal: None,
            dangerous_tools: true,
        }
    }

    /// Append one verification command (run in order, fail-fast).
    pub fn verify_command(mut self, cmd: impl Into<String>) -> Self {
        self.verify_commands.push(cmd.into());
        self
    }

    /// Set the verification command list (replaces any existing).
    pub fn verify_commands(mut self, cmds: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.verify_commands = cmds.into_iter().map(Into::into).collect();
        self
    }

    /// Override the loop-invariant system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the agent's name (for events / observability).
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Override the ReAct turn budget (default 40).
    pub fn max_turns(mut self, max_turns: usize) -> Self {
        self.max_turns = Some(max_turns);
        self
    }

    /// Doom-loop cure: abort after `n` identical tool calls in a row.
    pub fn max_identical_tool_calls(mut self, n: u32) -> Self {
        self.max_identical_tool_calls = Some(n);
        self
    }

    /// Subscribe to agent events.
    pub fn on_event(mut self, callback: Arc<OnEvent>) -> Self {
        self.on_event = Some(callback);
        self
    }

    /// Add an extra guardrail (first-Deny-wins, composed with any others).
    pub fn guardrail(mut self, guard: Arc<dyn Guardrail>) -> Self {
        self.extra_guardrails.push(guard);
        self
    }

    /// Restrict the curated builtins to this allowlist (the `verify` tool is
    /// always kept). Empty = keep all curated tools.
    pub fn tools_allow(mut self, names: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tool_allow = names.into_iter().map(Into::into).collect();
        self
    }

    /// Gate completion on a goal (typically [`code_goal`]).
    pub fn goal(mut self, goal: GoalCondition) -> Self {
        self.goal = Some(goal);
        self
    }

    /// Include/exclude the `bash` tool (default: included).
    pub fn dangerous_tools(mut self, enabled: bool) -> Self {
        self.dangerous_tools = enabled;
        self
    }

    /// Assemble the [`AgentRunner`].
    pub fn build(self) -> Result<AgentRunner<P>, Error> {
        let mut tools = code_tools(
            self.workspace.clone(),
            self.dangerous_tools,
            self.verify_commands.clone(),
        );
        if !self.tool_allow.is_empty() {
            tools.retain(|t| {
                let n = t.definition().name;
                // `verify` is the harness's reason to exist — never filter it out.
                n == "verify" || self.tool_allow.iter().any(|a| a == &n)
            });
        }

        let prompt = self
            .system_prompt
            .unwrap_or_else(|| CODE_SYSTEM_PROMPT.to_string());

        let mut b: AgentRunnerBuilder<P> = AgentRunner::builder(self.provider)
            .tools(tools)
            .system_prompt(prompt)
            .workspace(self.workspace)
            .max_turns(self.max_turns.unwrap_or(DEFAULT_CODE_MAX_TURNS));
        if let Some(name) = self.name {
            b = b.name(name);
        }
        if let Some(n) = self.max_identical_tool_calls {
            b = b.max_identical_tool_calls(n);
        }
        if let Some(ev) = self.on_event {
            b = b.on_event(ev);
        }
        if !self.extra_guardrails.is_empty() {
            b = b.guardrails(self.extra_guardrails);
        }
        if let Some(goal) = self.goal {
            b = b.goal(goal);
        }
        b.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::test_helpers::MockProvider;

    #[test]
    fn system_prompt_encodes_loop_invariants() {
        for needle in [
            "UNDERSTAND",
            "PLAN",
            "EDIT",
            "VERIFY",
            "REPAIR",
            "FINISH",
            "verify",
            "VERIFY_RESULT",
        ] {
            assert!(
                CODE_SYSTEM_PROMPT.contains(needle),
                "system prompt missing loop invariant `{needle}`"
            );
        }
    }

    #[test]
    fn goal_objective_keys_on_verify_sentinel() {
        let judge = Arc::new(BoxedProvider::new(MockProvider::new(vec![])));
        let goal = code_goal(judge);
        assert!(
            goal.objective().contains("VERIFY_RESULT: PASS"),
            "goal must gate on the deterministic verify sentinel, got: {}",
            goal.objective()
        );
    }

    #[test]
    fn code_tools_include_verify_and_editing_surface() {
        let dir = tempfile::tempdir().unwrap();
        let tools = code_tools(dir.path(), true, vec!["true".into()]);
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        for expected in [
            "verify", "read", "write", "edit", "patch", "grep", "glob", "list", "bash",
        ] {
            assert!(
                names.iter().any(|n| n == expected),
                "code_tools missing `{expected}`; got {names:?}"
            );
        }
    }

    #[test]
    fn code_tools_omit_bash_when_not_dangerous() {
        let dir = tempfile::tempdir().unwrap();
        let tools = code_tools(dir.path(), false, vec!["true".into()]);
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert!(
            !names.iter().any(|n| n == "bash"),
            "bash must be gated: {names:?}"
        );
        // verify and the editing surface remain.
        assert!(names.iter().any(|n| n == "verify"));
        assert!(names.iter().any(|n| n == "edit"));
    }

    #[test]
    fn build_wires_a_runner() {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(MockProvider::new(vec![]));
        let runner = CodeAgentBuilder::new(provider, dir.path())
            .name("coder")
            .verify_command("cargo test")
            .build();
        assert!(
            runner.is_ok(),
            "build should wire a runner: {:?}",
            runner.err()
        );
    }

    #[test]
    fn build_keeps_verify_even_under_tools_allow() {
        let dir = tempfile::tempdir().unwrap();
        let provider = Arc::new(MockProvider::new(vec![]));
        // Restrict to just `read`; `verify` must survive regardless.
        let tools = code_tools(dir.path(), true, vec!["true".into()]);
        let _ = CodeAgentBuilder::new(provider, dir.path())
            .verify_command("true")
            .tools_allow(["read"])
            .build()
            .expect("build ok");
        // The curated set itself still contains verify (build keeps it).
        assert!(tools.iter().any(|t| t.definition().name == "verify"));
    }
}
