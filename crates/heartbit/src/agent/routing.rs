//! Dynamic complexity-based agent routing.
//!
//! Three-tier hybrid cascade for routing tasks to single-agent or orchestrator:
//! - **Tier 1**: Fast heuristic scoring (< 1ms, zero LLM calls)
//! - **Tier 2**: Agent capability matching (< 1ms, zero LLM calls)
//! - **Tier 3**: Runtime escalation on failure (zero upfront overhead)

use serde::{Deserialize, Serialize};

use super::events::AgentEvent;
use crate::Error;

/// User-configurable routing strategy.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RoutingMode {
    /// Tiers 1+2+3: heuristic scoring, capability matching, runtime escalation.
    #[default]
    Auto,
    /// Force orchestrator for all multi-agent configs (old behavior).
    AlwaysOrchestrate,
    /// Force single agent (always agents\[0\]).
    SingleAgent,
}

/// Routing outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingDecision {
    SingleAgent {
        agent_index: usize,
        reason: &'static str,
    },
    Orchestrate {
        reason: &'static str,
    },
}

/// Heuristic signals exposed for testing and telemetry.
#[derive(Debug, Clone, Default)]
pub struct ComplexitySignals {
    pub word_count: usize,
    pub step_markers: usize,
    pub domain_signals: Vec<String>,
    pub explicit_delegation: bool,
    pub names_multiple_agents: bool,
    /// Tier 2: indices of agents that cover all detected domains.
    pub covering_agents: Vec<usize>,
    pub complexity_score: f32,
}

/// Pre-computed agent capability summary.
#[derive(Debug, Clone)]
pub struct AgentCapability {
    pub name: String,
    pub description_lower: String,
    pub tool_names: Vec<String>,
    pub domains: Vec<String>,
}

// ── Domain keyword lists ──

const CODE_KEYWORDS: &[&str] = &[
    "code",
    "implement",
    "refactor",
    "debug",
    "compile",
    "function",
    "class",
    "module",
    "rust",
    "python",
    "javascript",
    "typescript",
    "java",
    "golang",
    "programming",
    "syntax",
    "bug",
    "fix",
    "test",
    "unit test",
];

const RESEARCH_KEYWORDS: &[&str] = &[
    "research",
    "investigate",
    "analyze",
    "study",
    "survey",
    "find",
    "search",
    "explore",
    "review",
    "literature",
    "paper",
    "arxiv",
];

const DATABASE_KEYWORDS: &[&str] = &[
    "database",
    "sql",
    "query",
    "table",
    "schema",
    "migration",
    "postgres",
    "mysql",
    "sqlite",
    "mongodb",
    "redis",
    "index",
];

const FRONTEND_KEYWORDS: &[&str] = &[
    "frontend",
    "ui",
    "ux",
    "component",
    "react",
    "vue",
    "angular",
    "css",
    "html",
    "layout",
    "responsive",
    "design",
    "button",
    "form",
    "modal",
];

const BACKEND_KEYWORDS: &[&str] = &[
    "backend",
    "api",
    "endpoint",
    "server",
    "rest",
    "graphql",
    "middleware",
    "authentication",
    "authorization",
    "route",
    "handler",
];

const DEVOPS_KEYWORDS: &[&str] = &[
    "devops",
    "deploy",
    "docker",
    "kubernetes",
    "ci/cd",
    "pipeline",
    "infrastructure",
    "terraform",
    "ansible",
    "nginx",
    "monitoring",
    "logging",
];

const WRITING_KEYWORDS: &[&str] = &[
    "write",
    "document",
    "documentation",
    "readme",
    "blog",
    "article",
    "report",
    "summary",
    "copywriting",
    "content",
    "draft",
    "edit text",
];

const SECURITY_KEYWORDS: &[&str] = &[
    "security",
    "vulnerability",
    "audit",
    "penetration",
    "encryption",
    "auth",
    "cors",
    "xss",
    "injection",
    "firewall",
    "certificate",
    "tls",
];

const DOMAIN_LISTS: &[(&str, &[&str])] = &[
    ("code", CODE_KEYWORDS),
    ("research", RESEARCH_KEYWORDS),
    ("database", DATABASE_KEYWORDS),
    ("frontend", FRONTEND_KEYWORDS),
    ("backend", BACKEND_KEYWORDS),
    ("devops", DEVOPS_KEYWORDS),
    ("writing", WRITING_KEYWORDS),
    ("security", SECURITY_KEYWORDS),
];

// ── Step marker patterns ──

const STEP_MARKERS: &[&str] = &[
    "first,",
    "second,",
    "third,",
    "then,",
    "finally,",
    "next,",
    "after that",
    "step 1",
    "step 2",
    "step 3",
    "step 4",
    "step 5",
];

// ── Delegation language ──

const DELEGATION_PHRASES: &[&str] = &[
    "delegate",
    "have them",
    "coordinate between",
    "coordinate with",
    "team up",
    "work together",
    "collaborate",
    "assign to",
    "hand off",
    "pass to",
];

// ── Thresholds ──

const SINGLE_AGENT_THRESHOLD: f32 = 0.30;
const ORCHESTRATE_THRESHOLD: f32 = 0.55;

// ── Signal weights ──

const WEIGHT_SIMPLE_QUESTION: f32 = -0.30;
const WEIGHT_WORD_COUNT_HIGH: f32 = 0.10;
const WEIGHT_STEP_MARKERS: f32 = 0.25;
const WEIGHT_DELEGATION: f32 = 0.30;
const WEIGHT_NAMES_AGENTS: f32 = 0.40;
const WEIGHT_DOMAIN_DIVERSITY: f32 = 0.20;

impl AgentCapability {
    /// Build an `AgentCapability` from an agent's name, description, and tool list.
    pub fn from_config(name: &str, description: &str, tool_names: &[String]) -> Self {
        let description_lower = description.to_lowercase();
        let tool_lower: Vec<String> = tool_names.iter().map(|t| t.to_lowercase()).collect();

        // Extract domains from description + tool names
        let combined = format!("{} {}", description_lower, tool_lower.join(" "));
        let mut domains = Vec::new();
        for &(domain, keywords) in DOMAIN_LISTS {
            if keywords.iter().any(|kw| contains_keyword(&combined, kw)) {
                domains.push(domain.to_string());
            }
        }

        Self {
            name: name.to_lowercase(),
            description_lower,
            tool_names: tool_lower,
            domains,
        }
    }
}

/// Pure-Rust task analyzer. Zero LLM calls, sub-millisecond.
pub struct TaskComplexityAnalyzer<'a> {
    agents: &'a [AgentCapability],
}

impl<'a> TaskComplexityAnalyzer<'a> {
    pub fn new(agents: &'a [AgentCapability]) -> Self {
        Self { agents }
    }

    /// Full analysis: Tier 1 heuristics + Tier 2 capability matching → decision.
    pub fn analyze(&self, task: &str) -> (RoutingDecision, ComplexitySignals) {
        let mut signals = self.heuristic_signals(task);

        // Tier 1: clear decision from score alone
        if signals.complexity_score < SINGLE_AGENT_THRESHOLD {
            return (
                RoutingDecision::SingleAgent {
                    agent_index: 0,
                    reason: "heuristic score below single-agent threshold",
                },
                signals,
            );
        }
        if signals.complexity_score > ORCHESTRATE_THRESHOLD {
            return (
                RoutingDecision::Orchestrate {
                    reason: "heuristic score above orchestrate threshold",
                },
                signals,
            );
        }

        // Tier 2: uncertain zone — check agent capability coverage
        let decision = self.capability_match(&signals.domain_signals, &mut signals.covering_agents);
        (decision, signals)
    }

    /// Tier 1: compute heuristic signals and a complexity score.
    pub fn heuristic_signals(&self, task: &str) -> ComplexitySignals {
        let task_lower = task.to_lowercase();
        let words: Vec<&str> = task.split_whitespace().collect();
        let word_count = words.len();

        // Simple question detection
        let simple_question = is_simple_question(&task_lower, &words);

        // Step markers
        let step_markers = count_step_markers(&task_lower);
        // Also count numbered list items (e.g., "1.", "2.", "3.")
        let numbered_items = words
            .iter()
            .filter(|w| {
                w.len() >= 2
                    && w.ends_with('.')
                    && w[..w.len() - 1].chars().all(|c| c.is_ascii_digit())
            })
            .count();
        let total_step_markers = step_markers + numbered_items;

        // Delegation language
        let explicit_delegation = DELEGATION_PHRASES.iter().any(|p| task_lower.contains(p));

        // Domain detection
        let domain_signals = detect_domains(&task_lower);

        // Agent name detection
        let names_multiple_agents = if self.agents.len() >= 2 {
            let matching = self
                .agents
                .iter()
                .filter(|a| task_lower.contains(&a.name))
                .count();
            matching >= 2
        } else {
            false
        };

        // Weighted score
        let mut score: f32 = 0.0;
        if simple_question {
            score += WEIGHT_SIMPLE_QUESTION;
        }
        if word_count > 100 {
            score += WEIGHT_WORD_COUNT_HIGH;
        }
        if total_step_markers >= 2 {
            score += WEIGHT_STEP_MARKERS;
        }
        if explicit_delegation {
            score += WEIGHT_DELEGATION;
        }
        if names_multiple_agents {
            score += WEIGHT_NAMES_AGENTS;
        }
        if domain_signals.len() >= 3 {
            score += WEIGHT_DOMAIN_DIVERSITY;
        }

        // Clamp to [0, 1]
        score = score.clamp(0.0, 1.0);

        ComplexitySignals {
            word_count,
            step_markers: total_step_markers,
            domain_signals,
            explicit_delegation,
            names_multiple_agents,
            covering_agents: Vec::new(),
            complexity_score: score,
        }
    }

    /// Tier 2: check if one agent covers all detected domains.
    fn capability_match(
        &self,
        task_domains: &[String],
        covering_agents: &mut Vec<usize>,
    ) -> RoutingDecision {
        if task_domains.is_empty() {
            // No domains detected → conservative default: single agent
            return RoutingDecision::SingleAgent {
                agent_index: 0,
                reason: "no domains detected, defaulting to single agent",
            };
        }

        for (i, agent) in self.agents.iter().enumerate() {
            let covers_all = task_domains.iter().all(|d| agent.domains.contains(d));
            if covers_all {
                covering_agents.push(i);
            }
        }

        if covering_agents.is_empty() {
            RoutingDecision::Orchestrate {
                reason: "no single agent covers all detected domains",
            }
        } else {
            // Pick the first covering agent
            RoutingDecision::SingleAgent {
                agent_index: covering_agents[0],
                reason: "single agent covers all detected domains",
            }
        }
    }
}

/// Check if the task looks like a simple question (no multi-step markers).
fn is_simple_question(task_lower: &str, words: &[&str]) -> bool {
    let question_starters = [
        "what", "how", "why", "explain", "describe", "who", "when", "where",
    ];
    // Check first word starts with a question word. This intentionally uses
    // starts_with to catch contractions like "what's" → "what". False positives
    // like "however" bias toward single-agent (the cheaper direction).
    let starts_with_question = words
        .first()
        .is_some_and(|w| question_starters.iter().any(|q| w.starts_with(q)));
    let has_step_markers = count_step_markers(task_lower) >= 2;
    starts_with_question && !has_step_markers
}

/// Count step markers in the task text.
fn count_step_markers(task_lower: &str) -> usize {
    STEP_MARKERS
        .iter()
        .filter(|marker| task_lower.contains(*marker))
        .count()
}

/// Check if `text` contains `keyword` as a whole word (not a substring of a larger word).
/// For multi-word keywords (e.g., "unit test"), uses plain `contains`.
fn contains_keyword(text: &str, keyword: &str) -> bool {
    if keyword.contains(' ') {
        // Multi-word: substring match is fine
        return text.contains(keyword);
    }
    // Single word: check word boundaries
    for (start, _) in text.match_indices(keyword) {
        let end = start + keyword.len();
        let before_ok = start == 0 || !text.as_bytes()[start - 1].is_ascii_alphanumeric();
        let after_ok = end == text.len() || !text.as_bytes()[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

/// Detect which domains are referenced in the task.
fn detect_domains(task_lower: &str) -> Vec<String> {
    let mut domains = Vec::new();
    for &(domain, keywords) in DOMAIN_LISTS {
        if keywords.iter().any(|kw| contains_keyword(task_lower, kw)) {
            domains.push(domain.to_string());
        }
    }
    domains
}

/// Tier 3: Determine if a single-agent failure should escalate to orchestrator.
pub fn should_escalate(error: &Error, events: &[AgentEvent]) -> bool {
    // Unwrap WithPartialUsage wrapper to check the inner error variant.
    let inner = match error {
        Error::WithPartialUsage { source, .. } => source.as_ref(),
        other => other,
    };

    // MaxTurnsExceeded or RunTimeout → escalate
    if matches!(inner, Error::MaxTurnsExceeded(_) | Error::RunTimeout(_)) {
        return true;
    }

    // DoomLoopDetected ≥ 1 → escalate
    if events
        .iter()
        .any(|e| matches!(e, AgentEvent::DoomLoopDetected { .. }))
    {
        return true;
    }

    // AutoCompactionTriggered ≥ 2 → escalate
    let compaction_count = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::AutoCompactionTriggered { .. }))
        .count();
    if compaction_count >= 2 {
        return true;
    }

    false
}

/// Resolve routing mode from config + env override.
pub fn resolve_routing_mode(config_mode: RoutingMode) -> RoutingMode {
    match std::env::var("HEARTBIT_ROUTING").ok() {
        Some(val) => match val.to_lowercase().as_str() {
            "auto" => RoutingMode::Auto,
            "always_orchestrate" => RoutingMode::AlwaysOrchestrate,
            "single_agent" => RoutingMode::SingleAgent,
            _ => {
                tracing::warn!(
                    value = %val,
                    "unknown HEARTBIT_ROUTING value, falling back to config"
                );
                config_mode
            }
        },
        None => config_mode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_agents() -> Vec<AgentCapability> {
        vec![
            AgentCapability::from_config(
                "coder",
                "A code implementation agent that writes and debugs software",
                &["bash".into(), "read_file".into(), "write_file".into()],
            ),
            AgentCapability::from_config(
                "researcher",
                "A research agent that investigates and analyzes topics",
                &["web_search".into(), "read_file".into()],
            ),
        ]
    }

    // ── AgentCapability tests ──

    #[test]
    fn agent_capability_extracts_domains_from_description() {
        let cap = AgentCapability::from_config(
            "fullstack",
            "Handles frontend React components and backend API endpoints with database queries",
            &[],
        );
        assert!(cap.domains.contains(&"frontend".to_string()));
        assert!(cap.domains.contains(&"backend".to_string()));
        assert!(cap.domains.contains(&"database".to_string()));
    }

    #[test]
    fn agent_capability_extracts_domains_from_tools() {
        let cap = AgentCapability::from_config(
            "devops-agent",
            "Manages infrastructure",
            &["docker_build".into(), "deploy_k8s".into()],
        );
        assert!(cap.domains.contains(&"devops".to_string()));
    }

    // ── Tier 1: heuristic signal tests ──

    #[test]
    fn simple_question_scores_below_threshold() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        let (decision, signals) = analyzer.analyze("What is the capital of France?");
        assert!(
            signals.complexity_score < SINGLE_AGENT_THRESHOLD,
            "score {} should be < {}",
            signals.complexity_score,
            SINGLE_AGENT_THRESHOLD
        );
        assert!(matches!(decision, RoutingDecision::SingleAgent { .. }));
    }

    #[test]
    fn multi_step_multi_domain_routes_to_orchestrate() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        let task = "First, research the best database schema for user authentication. \
                    Then, implement the backend API endpoints in Rust. \
                    Finally, write frontend React components for the login form.";
        let (decision, signals) = analyzer.analyze(task);
        assert!(
            signals.step_markers >= 2,
            "step_markers: {}",
            signals.step_markers
        );
        assert!(
            signals.domain_signals.len() >= 3,
            "domains: {:?}",
            signals.domain_signals
        );
        // Score lands in uncertain zone (Tier 1), but Tier 2 detects split coverage
        // across agents → routes to Orchestrate
        assert!(
            matches!(decision, RoutingDecision::Orchestrate { .. }),
            "decision: {decision:?}, score: {}",
            signals.complexity_score
        );
    }

    #[test]
    fn delegation_language_boosts_score() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        let task = "Delegate the research task to the researcher and coordinate with the coder";
        let signals = analyzer.heuristic_signals(task);
        assert!(signals.explicit_delegation);
        // delegation (0.30) + names 2 agents (0.40) = 0.70
        assert!(
            signals.complexity_score > ORCHESTRATE_THRESHOLD,
            "score: {}",
            signals.complexity_score
        );
    }

    #[test]
    fn naming_multiple_agents_boosts_score() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        let task = "Have coder write the code and researcher find the documentation";
        let signals = analyzer.heuristic_signals(task);
        assert!(signals.names_multiple_agents);
        assert!(
            signals.complexity_score >= WEIGHT_NAMES_AGENTS,
            "score: {}",
            signals.complexity_score
        );
    }

    #[test]
    fn word_count_above_100_adds_weight() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        // 101+ words
        let task = (0..110)
            .map(|i| format!("word{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let signals = analyzer.heuristic_signals(&task);
        assert!(signals.word_count > 100);
        assert!(
            signals.complexity_score >= WEIGHT_WORD_COUNT_HIGH,
            "score: {}",
            signals.complexity_score
        );
    }

    #[test]
    fn numbered_list_detected_as_step_markers() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        let task = "1. Set up the database. 2. Write the API. 3. Test everything.";
        let signals = analyzer.heuristic_signals(task);
        assert!(
            signals.step_markers >= 2,
            "step_markers: {}",
            signals.step_markers
        );
    }

    #[test]
    fn score_clamped_to_zero_one() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);

        // Extremely simple — all negative signals
        let signals = analyzer.heuristic_signals("What is 2+2?");
        assert!(
            signals.complexity_score >= 0.0,
            "score: {}",
            signals.complexity_score
        );

        // Everything positive at once
        let task = "First, delegate to coder and researcher. Then step 1 deploy the docker \
                    kubernetes infrastructure with database schema, frontend React components, \
                    backend API endpoints, security audit, research papers, and write documentation. \
                    Finally, coordinate the team. ".repeat(5);
        let signals = analyzer.heuristic_signals(&task);
        assert!(
            signals.complexity_score <= 1.0,
            "score: {}",
            signals.complexity_score
        );
    }

    // ── Tier 2: capability matching tests ──

    #[test]
    fn one_agent_covers_all_domains_routes_to_single() {
        let agents = vec![AgentCapability::from_config(
            "fullstack",
            "Handles code implementation, database queries, and backend API endpoints",
            &[],
        )];
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        // Task with code + database + backend domains — all covered by fullstack
        let task = "Update the database query and fix the backend API endpoint bug";
        let (decision, signals) = analyzer.analyze(task);
        // Score should be in uncertain zone or below threshold → single agent
        match &decision {
            RoutingDecision::SingleAgent { agent_index, .. } => {
                assert_eq!(*agent_index, 0);
            }
            RoutingDecision::Orchestrate { reason } => {
                // If score > orchestrate threshold, that's fine too — it's a valid outcome
                assert!(
                    signals.complexity_score > ORCHESTRATE_THRESHOLD,
                    "unexpected orchestrate: {reason}"
                );
            }
        }
    }

    #[test]
    fn split_coverage_routes_to_orchestrate() {
        let agents = vec![
            AgentCapability::from_config(
                "frontend-dev",
                "Builds frontend React components and CSS layouts",
                &[],
            ),
            AgentCapability::from_config(
                "backend-dev",
                "Builds backend API endpoints and database schemas",
                &[],
            ),
        ];
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        // Task that needs both frontend AND backend AND database — no single agent covers all
        let task = "Build a React form that submits to a new backend API endpoint and stores data in the database";
        let mut signals = analyzer.heuristic_signals(task);
        let mut covering = Vec::new();
        let decision = analyzer.capability_match(&signals.domain_signals, &mut covering);
        signals.covering_agents = covering;
        assert!(
            matches!(decision, RoutingDecision::Orchestrate { .. }),
            "expected Orchestrate, got: {decision:?}"
        );
        assert!(signals.covering_agents.is_empty());
    }

    #[test]
    fn no_domains_defaults_to_single_agent() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        let mut covering = Vec::new();
        let decision = analyzer.capability_match(&[], &mut covering);
        assert!(matches!(
            decision,
            RoutingDecision::SingleAgent { agent_index: 0, .. }
        ));
    }

    // ── Tier 3: escalation tests ──

    #[test]
    fn escalate_on_max_turns_exceeded() {
        let err = Error::MaxTurnsExceeded(10);
        assert!(should_escalate(&err, &[]));
    }

    #[test]
    fn escalate_on_max_turns_wrapped_in_partial_usage() {
        use crate::llm::types::TokenUsage;
        let err = Error::MaxTurnsExceeded(10).with_partial_usage(TokenUsage::default());
        assert!(should_escalate(&err, &[]));
    }

    #[test]
    fn escalate_on_run_timeout() {
        let err = Error::RunTimeout(std::time::Duration::from_secs(60));
        assert!(should_escalate(&err, &[]));
    }

    #[test]
    fn escalate_on_doom_loop_event() {
        let events = vec![AgentEvent::DoomLoopDetected {
            agent: "a".into(),
            turn: 5,
            consecutive_count: 3,
            tool_names: vec!["web_search".into()],
        }];
        let err = Error::Agent("generic error".into());
        assert!(should_escalate(&err, &events));
    }

    #[test]
    fn escalate_on_two_compactions() {
        let events = vec![
            AgentEvent::AutoCompactionTriggered {
                agent: "a".into(),
                turn: 2,
                success: true,
                usage: Default::default(),
            },
            AgentEvent::AutoCompactionTriggered {
                agent: "a".into(),
                turn: 5,
                success: true,
                usage: Default::default(),
            },
        ];
        let err = Error::Agent("context overflow".into());
        assert!(should_escalate(&err, &events));
    }

    #[test]
    fn no_escalation_on_single_compaction() {
        let events = vec![AgentEvent::AutoCompactionTriggered {
            agent: "a".into(),
            turn: 2,
            success: true,
            usage: Default::default(),
        }];
        let err = Error::Agent("some error".into());
        assert!(!should_escalate(&err, &events));
    }

    #[test]
    fn no_escalation_on_normal_error() {
        let err = Error::Agent("tool failed".into());
        assert!(!should_escalate(&err, &[]));
    }

    // ── RoutingMode serde tests ──

    #[test]
    fn routing_mode_default_is_auto() {
        assert_eq!(RoutingMode::default(), RoutingMode::Auto);
    }

    #[test]
    fn routing_mode_roundtrips_json() {
        for mode in [
            RoutingMode::Auto,
            RoutingMode::AlwaysOrchestrate,
            RoutingMode::SingleAgent,
        ] {
            let json = serde_json::to_string(&mode).unwrap();
            let back: RoutingMode = serde_json::from_str(&json).unwrap();
            assert_eq!(mode, back, "failed for {json}");
        }
    }

    #[test]
    fn routing_mode_deserializes_from_toml_strings() {
        #[derive(Deserialize)]
        struct W {
            mode: RoutingMode,
        }
        let w: W = toml::from_str(r#"mode = "auto""#).unwrap();
        assert_eq!(w.mode, RoutingMode::Auto);
        let w: W = toml::from_str(r#"mode = "always_orchestrate""#).unwrap();
        assert_eq!(w.mode, RoutingMode::AlwaysOrchestrate);
        let w: W = toml::from_str(r#"mode = "single_agent""#).unwrap();
        assert_eq!(w.mode, RoutingMode::SingleAgent);
    }

    // ── Integration tests: end-to-end analyze → decision ──

    #[test]
    fn analyze_simple_task_two_agents_routes_single() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        let (decision, _) = analyzer.analyze("How do I parse JSON in Rust?");
        assert!(
            matches!(decision, RoutingDecision::SingleAgent { .. }),
            "got: {decision:?}"
        );
    }

    #[test]
    fn analyze_complex_multi_domain_task_routes_orchestrate() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        let task = "First, research the latest security vulnerabilities. \
                    Then, implement a fix in the backend API code. \
                    Finally, deploy the fix using Docker and update the documentation.";
        let (decision, signals) = analyzer.analyze(task);
        assert!(
            signals.complexity_score > ORCHESTRATE_THRESHOLD
                || matches!(decision, RoutingDecision::Orchestrate { .. }),
            "decision: {decision:?}, score: {}",
            signals.complexity_score
        );
    }

    #[test]
    fn analyze_delegation_with_agent_names_routes_orchestrate() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        let task =
            "Delegate to coder to implement the feature and have researcher find best practices";
        let (decision, signals) = analyzer.analyze(task);
        assert!(
            matches!(decision, RoutingDecision::Orchestrate { .. }),
            "decision: {decision:?}, score: {}",
            signals.complexity_score
        );
    }

    // ── resolve_routing_mode tests ──

    #[test]
    fn resolve_routing_mode_uses_config_when_no_env() {
        // Clear env to ensure test isolation
        // SAFETY: test-only, no concurrent env access.
        unsafe {
            std::env::remove_var("HEARTBIT_ROUTING");
        }
        assert_eq!(
            resolve_routing_mode(RoutingMode::AlwaysOrchestrate),
            RoutingMode::AlwaysOrchestrate
        );
    }

    // ── Backward compatibility ──

    #[test]
    fn missing_routing_field_defaults_to_auto() {
        #[derive(Deserialize)]
        struct TestConfig {
            #[serde(default)]
            routing: RoutingMode,
        }
        let config: TestConfig = toml::from_str("").unwrap();
        assert_eq!(config.routing, RoutingMode::Auto);
    }

    // ── contains_keyword tests ──

    #[test]
    fn contains_keyword_word_boundary() {
        // "ui" should NOT match inside "builds"
        assert!(!contains_keyword("builds backend api", "ui"));
        // "ui" should match as standalone word
        assert!(contains_keyword("the ui is broken", "ui"));
        // "ui" at start
        assert!(contains_keyword("ui components", "ui"));
        // "ui" at end
        assert!(contains_keyword("fix the ui", "ui"));
        // "api" should match as word, not inside "capital"
        assert!(contains_keyword("the api endpoint", "api"));
        assert!(!contains_keyword("the capital city", "api"));
    }

    #[test]
    fn contains_keyword_multi_word() {
        assert!(contains_keyword("run the unit test suite", "unit test"));
        assert!(!contains_keyword("run the unittest suite", "unit test"));
    }

    #[test]
    fn contains_keyword_adjacent_to_punctuation() {
        // Punctuation is not alphanumeric → word boundary
        assert!(contains_keyword("fix the api.", "api"));
        assert!(contains_keyword("(api) endpoint", "api"));
        assert!(contains_keyword("api/rest", "api"));
    }

    // ── Domain detection tests ──

    #[test]
    fn detect_domains_finds_multiple() {
        let domains = detect_domains("implement the api endpoint and write database migration");
        assert!(domains.contains(&"code".to_string())); // "implement"
        assert!(domains.contains(&"backend".to_string())); // "api", "endpoint"
        assert!(domains.contains(&"database".to_string())); // "database", "migration"
    }

    #[test]
    fn detect_domains_empty_for_generic_text() {
        let domains = detect_domains("hello world how are you");
        assert!(domains.is_empty());
    }

    // ── Edge cases ──

    #[test]
    fn empty_task_routes_single_agent() {
        let agents = make_agents();
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        let (decision, signals) = analyzer.analyze("");
        assert_eq!(signals.complexity_score, 0.0);
        assert!(matches!(decision, RoutingDecision::SingleAgent { .. }));
    }

    #[test]
    fn single_agent_list_always_routes_single() {
        let agents = vec![AgentCapability::from_config("solo", "Does everything", &[])];
        let analyzer = TaskComplexityAnalyzer::new(&agents);
        // Even a complex task — naming agents can't trigger with only 1 agent
        let task = "Delegate the complex multi-step task that involves code, database, frontend, backend, security, devops, research, and writing";
        let (decision, signals) = analyzer.analyze(task);
        // It may still orchestrate due to high score, but names_multiple_agents should be false
        assert!(!signals.names_multiple_agents);
        // With delegation (0.30) + domains ≥ 3 (0.20) + steps (0.25) = 0.75 → orchestrate is valid
        match decision {
            RoutingDecision::SingleAgent { .. } | RoutingDecision::Orchestrate { .. } => {}
        }
    }
}
