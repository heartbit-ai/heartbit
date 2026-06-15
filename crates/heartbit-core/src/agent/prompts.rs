//! Built-in system prompt fragments for multi-agent collaboration.

/// Selection guidance for the orchestration primitives (frontier finding #9).
///
/// The 2026 multi-agent literature (Cemri et al., "Why Do Multi-Agent LLM
/// Systems Fail?", arXiv 2503.13657) is a caution against reflexively reaching
/// for more agents: coordination overhead, inter-agent error propagation, and
/// specification/verification gaps mean a multi-agent design frequently performs
/// *worse* than a single well-prompted agent with good tools. The library keeps
/// its combinators (Sequential/Parallel/Loop/DAG, Debate/Voting/Mixture,
/// Evaluator-Optimizer, Orchestrator delegation, and the durable `flow` engine)
/// because each has a justified niche — but choosing among them should follow:
///
/// 1. **Default to a single agent** with tools, memory, and a goal/judge. Most
///    tasks do not need multiple agents.
/// 2. **Reach for determinism before autonomy.** When the steps are known, a
///    `flow` pipeline / `SequentialAgent` / `DagAgent` is cheaper and far more
///    reliable than autonomous delegation — no LLM coordination cost, replayable.
/// 3. **Use parallel fan-out (`ParallelAgent`, orchestrator squads) only for
///    genuinely independent subtasks.** Interdependent work serialised across
///    agents accumulates handoff errors.
/// 4. **Use ensemble combinators (`VotingAgent`, `DebateAgent`,
///    `MixtureOfAgentsAgent`) to trade cost for reliability on hard, verifiable
///    questions** — and prefer a [`Verifier`](super::Verifier)-graded best-of-N
///    when you have a reward signal.
/// 5. **Prefer the durable `flow` engine for long-horizon orchestration** so a
///    failure is recoverable (checkpoint/replay) rather than restarting the run.
///
/// In short: add agents to *decompose for coverage* or *cross-check for
/// confidence*, never by default.
pub const MULTI_AGENT_SELECTION_GUIDANCE: &str = "\
Orchestration selection (multi-agent designs often underperform a single good \
agent — Cemri et al. 2503.13657):\n\
1. Default to ONE agent with tools + memory + a goal/judge.\n\
2. Prefer determinism (flow pipeline / Sequential / DAG) over autonomous \
delegation when steps are known.\n\
3. Parallel fan-out (ParallelAgent / squads) only for INDEPENDENT subtasks.\n\
4. Ensembles (Voting / Debate / Mixture, or Verifier best-of-N) trade cost for \
reliability on hard, verifiable questions.\n\
5. Use the durable flow engine for long-horizon work so failures replay.\n\
Add agents to decompose for coverage or cross-check for confidence — never by \
default.";

/// Multi-agent collaboration prompt appended to sub-agent system prompts.
///
/// Contains `{name}` and `{description}` placeholders that must be replaced
/// before injection using `.replace()`.
pub const MULTI_AGENT_COLLAB_PROMPT: &str = "\n\n\
--- MULTI-AGENT COLLABORATION PROTOCOL ---

You are agent `{name}` with role: {description}.

## Blackboard Protocol
- **Before starting**: read the blackboard to check what other agents have already produced.
- **During work**: write intermediate results to the blackboard so other agents can see your progress.
- **After completion**: write your final results to the blackboard under the key `agent:{name}`.

## Deduplication
- Before executing your task, verify it has not already been completed by another agent.
- If the blackboard already contains a satisfactory answer for your task, report that instead of redoing the work.

## Cross-Verification
- After producing your results, compare them against any related outputs from other agents on the blackboard.
- Flag contradictions or inconsistencies in your final output.

## Execution Loop: Perceive -> Plan -> Act -> Reflect
1. **Perceive**: read the blackboard and understand the current state of the shared workspace.
2. **Plan**: decide what steps are needed, considering what other agents have done.
3. **Act**: execute your plan using available tools.
4. **Reflect**: review your output for correctness and consistency with other agents' work.

## Result Sharing
- Write your final output to the blackboard with a clear, descriptive key.
- Include a brief summary so other agents can quickly understand your contribution.
";

/// Replace `{name}` and `{description}` placeholders in the collaboration prompt.
pub fn render_collab_prompt(name: &str, description: &str) -> String {
    MULTI_AGENT_COLLAB_PROMPT
        .replace("{name}", name)
        .replace("{description}", description)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_is_non_empty() {
        assert!(!MULTI_AGENT_COLLAB_PROMPT.is_empty());
    }

    #[test]
    fn prompt_under_reasonable_size() {
        // Should be under 4 KB — it's supplemental guidance, not a novel
        assert!(MULTI_AGENT_COLLAB_PROMPT.len() < 4096);
    }

    #[test]
    fn prompt_contains_expected_sections() {
        let prompt = MULTI_AGENT_COLLAB_PROMPT;
        assert!(prompt.contains("Blackboard Protocol"));
        assert!(prompt.contains("Deduplication"));
        assert!(prompt.contains("Cross-Verification"));
        assert!(prompt.contains("Perceive"));
        assert!(prompt.contains("Plan"));
        assert!(prompt.contains("Act"));
        assert!(prompt.contains("Reflect"));
        assert!(prompt.contains("Result Sharing"));
    }

    #[test]
    fn prompt_has_placeholders() {
        assert!(MULTI_AGENT_COLLAB_PROMPT.contains("{name}"));
        assert!(MULTI_AGENT_COLLAB_PROMPT.contains("{description}"));
    }

    #[test]
    fn render_replaces_placeholders() {
        let rendered = render_collab_prompt("researcher", "Finds relevant papers");
        assert!(!rendered.contains("{name}"));
        assert!(!rendered.contains("{description}"));
        assert!(rendered.contains("`researcher`"));
        assert!(rendered.contains("Finds relevant papers"));
        assert!(rendered.contains("agent:researcher"));
    }

    #[test]
    fn render_handles_special_characters_in_name() {
        let rendered = render_collab_prompt("code-analyzer", "Analyzes {code} structures");
        assert!(rendered.contains("`code-analyzer`"));
        assert!(rendered.contains("Analyzes {code} structures"));
    }
}
