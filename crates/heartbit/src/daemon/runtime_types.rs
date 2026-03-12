use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agent::guardrails::pii::PiiAction;
use crate::agent::workflow::WorkflowType;
use crate::config::DispatchMode;

/// Provider type for runtime execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeProviderType {
    Anthropic,
    Openrouter,
}

/// MCP server configuration for runtime execution.
#[derive(Clone, Serialize, Deserialize)]
pub struct RuntimeMcpServer {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<String>,
}

impl std::fmt::Debug for RuntimeMcpServer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeMcpServer")
            .field("url", &self.url)
            .field(
                "auth_header",
                &self.auth_header.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

/// Provider configuration for runtime execution.
#[derive(Clone, Serialize, Deserialize)]
pub struct RuntimeProviderConfig {
    pub provider_type: RuntimeProviderType,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub prompt_caching: bool,
}

impl std::fmt::Debug for RuntimeProviderConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuntimeProviderConfig")
            .field("provider_type", &self.provider_type)
            .field("api_key", &"[REDACTED]")
            .field("model", &self.model)
            .field("prompt_caching", &self.prompt_caching)
            .finish()
    }
}

/// Advanced agent configuration for runtime execution.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RuntimeAdvancedConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_total_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_timeout_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_timeout_seconds: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_identical_tool_calls: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_strategy: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summarize_threshold: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_reflection: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_prune: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recursive_summarization: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub consolidate_on_exit: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tools_per_turn: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tool_output_bytes: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_cache_size: Option<usize>,
}

/// Agent configuration for runtime execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeAgentConfig {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_prompt: Option<String>,
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub advanced: RuntimeAdvancedConfig,
}

fn default_max_turns() -> usize {
    50
}

fn default_max_tokens() -> u32 {
    4096
}

fn default_orch_max_turns() -> usize {
    10
}

/// Sub-agent configuration for multi-agent orchestration via RuntimeRequest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSubAgentConfig {
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    #[serde(default = "default_max_turns")]
    pub max_turns: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub builtin_tools: Vec<String>,
    #[serde(default)]
    pub mcp_servers: Vec<RuntimeMcpServer>,
}

/// Orchestrator settings for multi-agent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeOrchestratorConfig {
    #[serde(default = "default_orch_max_turns")]
    pub max_turns: usize,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub enable_squads: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dispatch_mode: Option<DispatchMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spawn: Option<RuntimeSpawnConfig>,
}

/// Dynamic agent spawning configuration for runtime execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeSpawnConfig {
    /// Maximum number of agents that can be spawned per run.
    #[serde(default = "default_max_spawned_agents")]
    pub max_spawned_agents: u32,
    /// Tool names spawned agents may use (from builtin pool).
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
    /// Maximum turns per spawned agent.
    #[serde(default = "default_spawn_max_turns")]
    pub max_turns: usize,
    /// Maximum tokens per LLM call for spawned agents.
    #[serde(default = "default_spawn_max_tokens")]
    pub max_tokens: u32,
    /// Cumulative token budget across ALL spawned agents.
    #[serde(default = "default_spawn_max_total_tokens")]
    pub max_total_tokens: u64,
}

fn default_max_spawned_agents() -> u32 {
    3
}

fn default_spawn_max_turns() -> usize {
    15
}

fn default_spawn_max_tokens() -> u32 {
    4096
}

fn default_spawn_max_total_tokens() -> u64 {
    50_000
}

/// Guardrail configuration for runtime execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeGuardrailConfig {
    #[serde(default = "default_true")]
    pub injection: bool,
    #[serde(default = "default_true")]
    pub pii: bool,
    /// What to do when PII is detected: redact (default), deny, or warn.
    #[serde(default)]
    pub pii_action: PiiAction,
    #[serde(default = "default_injection_threshold")]
    pub injection_threshold: f32,
}

fn default_true() -> bool {
    true
}

fn default_injection_threshold() -> f32 {
    0.5
}

/// Memory configuration for runtime execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reflection_threshold: Option<u32>,
    #[serde(default)]
    pub consolidate_on_exit: bool,
}

/// Full execution request from cloud to runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeRequest {
    pub task_id: Uuid,
    pub prompt: String,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory: Option<RuntimeMemoryConfig>,
    pub agent: RuntimeAgentConfig,
    pub provider: RuntimeProviderConfig,
    #[serde(default)]
    pub mcp_servers: Vec<RuntimeMcpServer>,
    #[serde(default)]
    pub builtin_tools: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub guardrails: Option<RuntimeGuardrailConfig>,
    #[serde(default)]
    pub messages: Vec<crate::llm::types::Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<Uuid>,
    /// Sub-agents for multi-agent orchestration. When non-empty, the execute
    /// endpoint builds an Orchestrator instead of a single AgentRunner.
    #[serde(default)]
    pub sub_agents: Vec<RuntimeSubAgentConfig>,
    /// Orchestrator settings (only used when `sub_agents` is non-empty).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orchestrator: Option<RuntimeOrchestratorConfig>,
    /// Workflow configuration for deterministic execution (DAG, sequential, etc.).
    /// When present, the execute endpoint builds a workflow agent instead of
    /// a single AgentRunner or Orchestrator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub workflow: Option<RuntimeWorkflowConfig>,
    /// Multimodal content blocks for the initial user message (e.g. image + text).
    /// When non-empty, the execute endpoint calls `execute_with_content` instead of
    /// `execute(&prompt)`. The `prompt` field is ignored in that case.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub initial_content: Vec<crate::llm::types::ContentBlock>,
}

/// Workflow execution configuration for the runtime execute endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeWorkflowConfig {
    pub workflow_type: WorkflowType,
    pub nodes: Vec<RuntimeWorkflowNode>,
    #[serde(default)]
    pub edges: Vec<RuntimeWorkflowEdge>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_iterations: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_pattern: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rounds: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layers: Option<u32>,
}

/// A node in a runtime workflow definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeWorkflowNode {
    pub name: String,
    pub agent_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
}

/// An edge in a runtime workflow DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeWorkflowEdge {
    pub from: String,
    pub to: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<EdgeConditionSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<EdgeTransform>,
}

/// Transform to apply to edge data between workflow nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeTransform {
    Uppercase,
    Lowercase,
    ExtractJson,
    Trim,
}

/// Pattern type for workflow edge conditions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeConditionPattern {
    Contains,
    NotContains,
    StartsWith,
    Regex,
}

/// Condition specification for a workflow edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeConditionSpec {
    pub pattern: EdgeConditionPattern,
    pub value: String,
}

/// Execution response from runtime to cloud.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeResponse {
    pub result: String,
    pub usage: crate::llm::types::TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Agent events collected during execution (sync path only).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub events: Vec<serde_json::Value>,
}

/// SSE event types streamed from runtime to cloud.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeSseEvent {
    Delta {
        content: String,
    },
    Done(RuntimeResponse),
    Error {
        message: String,
    },
    Event {
        name: String,
        data: serde_json::Value,
    },
}

// ---------------------------------------------------------------------------
// Eval types (cloud → runtime eval execution)
// ---------------------------------------------------------------------------

/// Scorer selection and configuration for eval runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeScorerConfig {
    /// Scorer names: "trajectory", "keyword", "similarity",
    /// "cost", "latency", "tool_call_count", "safety"
    pub scorers: Vec<String>,
    #[serde(default = "default_max_cost")]
    pub max_cost_usd: f64,
    #[serde(default = "default_max_latency")]
    pub max_latency_ms: u64,
    #[serde(default = "default_max_tool_calls")]
    pub max_tool_calls: usize,
}

fn default_max_cost() -> f64 {
    0.10
}

fn default_max_latency() -> u64 {
    30_000
}

fn default_max_tool_calls() -> usize {
    20
}

/// Eval execution request (cloud → runtime).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEvalRequest {
    pub eval_id: Uuid,
    /// Agent config — reuses RuntimeRequest (prompt/stream fields ignored).
    pub agent_config: RuntimeRequest,
    pub cases: Vec<crate::eval::EvalCase>,
    pub scoring: RuntimeScorerConfig,
    #[serde(default)]
    pub stream: bool,
    /// Optional baseline for A/B comparison.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline: Option<Vec<crate::eval::EvalResult>>,
}

/// Eval response (runtime → cloud).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeEvalResponse {
    pub eval_id: Uuid,
    pub results: Vec<crate::eval::EvalResult>,
    pub summary: crate::eval::EvalSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison: Option<crate::eval::EvalComparison>,
}

/// SSE events for streaming eval.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RuntimeEvalSseEvent {
    CaseResult(crate::eval::EvalResult),
    Done(RuntimeEvalResponse),
    Error { message: String },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json;

    #[test]
    fn runtime_request_round_trip() {
        let request = RuntimeRequest {
            task_id: Uuid::new_v4(),
            prompt: "Hello, world!".to_string(),
            stream: true,
            tenant_id: Some(Uuid::new_v4()),
            user_id: Some("user-1".to_string()),
            memory: Some(RuntimeMemoryConfig {
                enabled: true,
                reflection_threshold: Some(100),
                consolidate_on_exit: true,
            }),
            agent: RuntimeAgentConfig {
                name: "test-agent".to_string(),
                system_prompt: Some("You are helpful.".to_string()),
                max_turns: 30,
                max_tokens: 8192,
                advanced: RuntimeAdvancedConfig {
                    reasoning_effort: Some("high".to_string()),
                    tool_profile: Some("full".to_string()),
                    ..Default::default()
                },
            },
            provider: RuntimeProviderConfig {
                provider_type: RuntimeProviderType::Anthropic,
                api_key: "sk-test".to_string(),
                model: "claude-sonnet-4-20250514".to_string(),
                prompt_caching: true,
            },
            mcp_servers: vec![RuntimeMcpServer {
                url: "https://mcp.example.com".to_string(),
                auth_header: Some("Bearer token".to_string()),
            }],
            builtin_tools: vec!["bash".to_string(), "read".to_string()],
            guardrails: Some(RuntimeGuardrailConfig {
                injection: true,
                pii: false,
                pii_action: Default::default(),
                injection_threshold: 0.7,
            }),
            messages: vec![],
            session_id: Some(Uuid::new_v4()),
            sub_agents: vec![],
            orchestrator: None,
            workflow: None,
            initial_content: vec![],
        };

        let json = serde_json::to_string(&request).expect("serialize");
        let deserialized: RuntimeRequest = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.task_id, request.task_id);
        assert_eq!(deserialized.prompt, "Hello, world!");
        assert!(deserialized.stream);
        assert_eq!(deserialized.agent.name, "test-agent");
        assert_eq!(deserialized.agent.max_turns, 30);
        assert_eq!(deserialized.agent.max_tokens, 8192);
        assert_eq!(
            deserialized.agent.advanced.reasoning_effort.as_deref(),
            Some("high")
        );
        assert_eq!(deserialized.provider.model, "claude-sonnet-4-20250514");
        assert!(deserialized.provider.prompt_caching);
        assert_eq!(deserialized.mcp_servers.len(), 1);
        assert_eq!(deserialized.builtin_tools, vec!["bash", "read"]);
        assert_eq!(
            deserialized
                .guardrails
                .as_ref()
                .unwrap()
                .injection_threshold,
            0.7
        );
    }

    #[test]
    fn runtime_sse_event_delta_deserialization() {
        let json = r#"{"type":"delta","content":"Hello"}"#;
        let event: RuntimeSseEvent = serde_json::from_str(json).expect("deserialize");
        match event {
            RuntimeSseEvent::Delta { content } => assert_eq!(content, "Hello"),
            other => panic!("expected Delta, got {:?}", other),
        }
    }

    #[test]
    fn runtime_sse_event_done_deserialization() {
        let json =
            r#"{"type":"done","result":"All done","usage":{"input_tokens":10,"output_tokens":20}}"#;
        let event: RuntimeSseEvent = serde_json::from_str(json).expect("deserialize");
        match event {
            RuntimeSseEvent::Done(resp) => {
                assert_eq!(resp.result, "All done");
                assert_eq!(resp.usage.input_tokens, 10);
                assert_eq!(resp.usage.output_tokens, 20);
                assert!(resp.model_name.is_none());
            }
            other => panic!("expected Done, got {:?}", other),
        }
    }

    #[test]
    fn runtime_sse_event_error_deserialization() {
        let json = r#"{"type":"error","message":"something broke"}"#;
        let event: RuntimeSseEvent = serde_json::from_str(json).expect("deserialize");
        match event {
            RuntimeSseEvent::Error { message } => assert_eq!(message, "something broke"),
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn runtime_sse_event_event_deserialization() {
        let json = r#"{"type":"event","name":"tool_start","data":{"tool":"bash"}}"#;
        let event: RuntimeSseEvent = serde_json::from_str(json).expect("deserialize");
        match event {
            RuntimeSseEvent::Event { name, data } => {
                assert_eq!(name, "tool_start");
                assert_eq!(data["tool"], "bash");
            }
            other => panic!("expected Event, got {:?}", other),
        }
    }

    #[test]
    fn runtime_advanced_config_defaults() {
        let config = RuntimeAdvancedConfig::default();
        assert!(config.reasoning_effort.is_none());
        assert!(config.tool_profile.is_none());
        assert!(config.max_total_tokens.is_none());
        assert!(config.run_timeout_seconds.is_none());
        assert!(config.tool_timeout_seconds.is_none());
        assert!(config.max_identical_tool_calls.is_none());
        assert!(config.context_strategy.is_none());
        assert!(config.summarize_threshold.is_none());
        assert!(config.enable_reflection.is_none());
        assert!(config.session_prune.is_none());
        assert!(config.recursive_summarization.is_none());
        assert!(config.consolidate_on_exit.is_none());
        assert!(config.max_tools_per_turn.is_none());
        assert!(config.max_tool_output_bytes.is_none());
        assert!(config.response_cache_size.is_none());

        // Should serialize to empty object (all None fields skipped)
        let json = serde_json::to_string(&config).expect("serialize");
        assert_eq!(json, "{}");
    }

    #[test]
    fn runtime_provider_type_serializes_lowercase() {
        let anthropic = serde_json::to_string(&RuntimeProviderType::Anthropic).expect("serialize");
        assert_eq!(anthropic, r#""anthropic""#);

        let openrouter =
            serde_json::to_string(&RuntimeProviderType::Openrouter).expect("serialize");
        assert_eq!(openrouter, r#""openrouter""#);

        // Round-trip
        let parsed: RuntimeProviderType =
            serde_json::from_str(r#""anthropic""#).expect("deserialize");
        assert!(matches!(parsed, RuntimeProviderType::Anthropic));
    }

    #[test]
    fn runtime_response_round_trip() {
        let response = RuntimeResponse {
            result: "Task completed successfully.".to_string(),
            usage: crate::llm::types::TokenUsage {
                input_tokens: 100,
                output_tokens: 250,
                cache_creation_input_tokens: 50,
                cache_read_input_tokens: 30,
                reasoning_tokens: 0,
            },
            model_name: Some("claude-sonnet-4-20250514".to_string()),
            events: vec![],
        };

        let json = serde_json::to_string(&response).expect("serialize");
        let deserialized: RuntimeResponse = serde_json::from_str(&json).expect("deserialize");

        assert_eq!(deserialized.result, "Task completed successfully.");
        assert_eq!(deserialized.usage.input_tokens, 100);
        assert_eq!(deserialized.usage.output_tokens, 250);
        assert_eq!(deserialized.usage.cache_creation_input_tokens, 50);
        assert_eq!(deserialized.usage.cache_read_input_tokens, 30);
        assert_eq!(
            deserialized.model_name.as_deref(),
            Some("claude-sonnet-4-20250514")
        );
    }

    #[test]
    fn runtime_response_backward_compat_no_events() {
        // Old JSON without `events` field should deserialize with empty vec
        let json = r#"{"result":"done","usage":{"input_tokens":10,"output_tokens":5}}"#;
        let resp: RuntimeResponse = serde_json::from_str(json).expect("deserialize");
        assert_eq!(resp.result, "done");
        assert!(resp.events.is_empty());
    }

    #[test]
    fn runtime_response_events_round_trip() {
        let response = RuntimeResponse {
            result: "ok".to_string(),
            usage: crate::llm::types::TokenUsage::default(),
            model_name: None,
            events: vec![serde_json::json!({"type": "run_started", "agent": "a", "task": "t"})],
        };
        let json = serde_json::to_string(&response).expect("serialize");
        assert!(json.contains("run_started"), "json: {json}");
        let back: RuntimeResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.events.len(), 1);
    }

    #[test]
    fn runtime_response_empty_events_omitted() {
        let response = RuntimeResponse {
            result: "ok".to_string(),
            usage: crate::llm::types::TokenUsage::default(),
            model_name: None,
            events: vec![],
        };
        let json = serde_json::to_string(&response).expect("serialize");
        // Empty events should be omitted via skip_serializing_if
        assert!(!json.contains("events"), "json: {json}");
    }

    #[test]
    fn runtime_request_minimal_fields() {
        let json = serde_json::json!({
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "Do something",
            "agent": {
                "name": "minimal-agent"
            },
            "provider": {
                "provider_type": "anthropic",
                "api_key": "sk-key",
                "model": "claude-sonnet-4-20250514"
            }
        });

        let request: RuntimeRequest =
            serde_json::from_value(json).expect("deserialize minimal request");

        assert_eq!(request.prompt, "Do something");
        assert!(!request.stream);
        assert!(request.tenant_id.is_none());
        assert!(request.user_id.is_none());
        assert!(request.memory.is_none());
        assert!(request.guardrails.is_none());
        assert!(request.session_id.is_none());
        assert!(request.messages.is_empty());
        assert!(request.mcp_servers.is_empty());
        assert!(request.builtin_tools.is_empty());

        // Agent defaults
        assert_eq!(request.agent.name, "minimal-agent");
        assert!(request.agent.system_prompt.is_none());
        assert_eq!(request.agent.max_turns, 50);
        assert_eq!(request.agent.max_tokens, 4096);

        // Provider defaults
        assert!(!request.provider.prompt_caching);

        // Multi-agent defaults
        assert!(request.sub_agents.is_empty());
        assert!(request.orchestrator.is_none());
    }

    #[test]
    fn runtime_sub_agent_config_round_trip() {
        let config = RuntimeSubAgentConfig {
            name: "researcher".to_string(),
            description: "Looks up data".to_string(),
            system_prompt: "You are a researcher.".to_string(),
            max_turns: 10,
            max_tokens: 2048,
            builtin_tools: vec!["read".to_string()],
            mcp_servers: vec![RuntimeMcpServer {
                url: "http://localhost:9000/mcp".to_string(),
                auth_header: None,
            }],
        };

        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: RuntimeSubAgentConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.name, "researcher");
        assert_eq!(parsed.max_turns, 10);
        assert_eq!(parsed.builtin_tools, vec!["read"]);
        assert_eq!(parsed.mcp_servers.len(), 1);
    }

    #[test]
    fn runtime_sub_agent_config_defaults() {
        let json = serde_json::json!({
            "name": "agent-a",
            "description": "Does stuff",
            "system_prompt": "You help."
        });
        let config: RuntimeSubAgentConfig =
            serde_json::from_value(json).expect("deserialize with defaults");
        assert_eq!(config.max_turns, 50);
        assert_eq!(config.max_tokens, 4096);
        assert!(config.builtin_tools.is_empty());
        assert!(config.mcp_servers.is_empty());
    }

    #[test]
    fn runtime_orchestrator_config_defaults() {
        let json = serde_json::json!({});
        let config: RuntimeOrchestratorConfig =
            serde_json::from_value(json).expect("deserialize empty");
        assert_eq!(config.max_turns, 10);
        assert_eq!(config.max_tokens, 4096);
        assert!(!config.enable_squads);
        assert!(config.dispatch_mode.is_none());
    }

    #[test]
    fn runtime_scorer_config_defaults() {
        let json = r#"{"scorers":["keyword","trajectory"]}"#;
        let config: RuntimeScorerConfig = serde_json::from_str(json).expect("deserialize");
        assert_eq!(config.scorers, vec!["keyword", "trajectory"]);
        assert!((config.max_cost_usd - 0.10).abs() < f64::EPSILON);
        assert_eq!(config.max_latency_ms, 30_000);
        assert_eq!(config.max_tool_calls, 20);
    }

    #[test]
    fn runtime_eval_request_round_trip() {
        let req = RuntimeEvalRequest {
            eval_id: Uuid::new_v4(),
            agent_config: RuntimeRequest {
                task_id: Uuid::new_v4(),
                prompt: String::new(),
                stream: false,
                tenant_id: None,
                user_id: None,
                memory: None,
                agent: RuntimeAgentConfig {
                    name: "eval-agent".into(),
                    system_prompt: Some("You help.".into()),
                    max_turns: 10,
                    max_tokens: 2048,
                    advanced: RuntimeAdvancedConfig::default(),
                },
                provider: RuntimeProviderConfig {
                    provider_type: RuntimeProviderType::Anthropic,
                    api_key: "sk-test".into(),
                    model: "claude-sonnet-4-20250514".into(),
                    prompt_caching: false,
                },
                mcp_servers: vec![],
                builtin_tools: vec![],
                guardrails: None,
                messages: vec![],
                session_id: None,
                sub_agents: vec![],
                orchestrator: None,
                workflow: None,
                initial_content: vec![],
            },
            cases: vec![crate::eval::EvalCase::new("greet", "Say hi").expect_output_contains("hi")],
            scoring: RuntimeScorerConfig {
                scorers: vec!["keyword".into()],
                max_cost_usd: 0.05,
                max_latency_ms: 10_000,
                max_tool_calls: 5,
            },
            stream: false,
            baseline: None,
        };

        let json = serde_json::to_string(&req).expect("serialize");
        let parsed: RuntimeEvalRequest = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.eval_id, req.eval_id);
        assert_eq!(parsed.cases.len(), 1);
        assert_eq!(parsed.cases[0].name, "greet");
        assert_eq!(parsed.scoring.scorers, vec!["keyword"]);
    }

    #[test]
    fn runtime_eval_response_round_trip() {
        let resp = RuntimeEvalResponse {
            eval_id: Uuid::new_v4(),
            results: vec![crate::eval::EvalResult {
                case_name: "test".into(),
                passed: true,
                scores: vec![],
                actual_tools: vec![],
                actual_output: "hello".into(),
                error: None,
            }],
            summary: crate::eval::EvalSummary {
                total: 1,
                passed: 1,
                failed: 0,
                errors: 0,
                avg_score: 1.0,
                scorer_averages: vec![],
            },
            comparison: None,
        };
        let json = serde_json::to_string(&resp).expect("serialize");
        let parsed: RuntimeEvalResponse = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.eval_id, resp.eval_id);
        assert_eq!(parsed.results.len(), 1);
        assert!(parsed.results[0].passed);
    }

    #[test]
    fn runtime_eval_sse_event_case_result() {
        let event = RuntimeEvalSseEvent::CaseResult(crate::eval::EvalResult {
            case_name: "c".into(),
            passed: true,
            scores: vec![],
            actual_tools: vec![],
            actual_output: String::new(),
            error: None,
        });
        let json = serde_json::to_string(&event).expect("serialize");
        assert!(json.contains(r#""type":"case_result""#));
        let parsed: RuntimeEvalSseEvent = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(parsed, RuntimeEvalSseEvent::CaseResult(_)));
    }

    #[test]
    fn runtime_eval_sse_event_error() {
        let json = r#"{"type":"error","message":"boom"}"#;
        let event: RuntimeEvalSseEvent = serde_json::from_str(json).expect("deserialize");
        match event {
            RuntimeEvalSseEvent::Error { message } => assert_eq!(message, "boom"),
            other => panic!("expected Error, got {:?}", other),
        }
    }

    #[test]
    fn runtime_request_with_sub_agents_round_trip() {
        let json = serde_json::json!({
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "Research and write a report",
            "agent": {
                "name": "orchestrator"
            },
            "provider": {
                "provider_type": "anthropic",
                "api_key": "sk-key",
                "model": "claude-sonnet-4-20250514"
            },
            "sub_agents": [
                {
                    "name": "researcher",
                    "description": "Gathers data",
                    "system_prompt": "You research things."
                },
                {
                    "name": "writer",
                    "description": "Writes reports",
                    "system_prompt": "You write reports.",
                    "max_turns": 5,
                    "builtin_tools": ["write"]
                }
            ],
            "orchestrator": {
                "max_turns": 8,
                "enable_squads": true,
                "dispatch_mode": "parallel"
            }
        });

        let request: RuntimeRequest =
            serde_json::from_value(json).expect("deserialize multi-agent request");

        assert_eq!(request.sub_agents.len(), 2);
        assert_eq!(request.sub_agents[0].name, "researcher");
        assert_eq!(request.sub_agents[0].max_turns, 50); // default
        assert_eq!(request.sub_agents[1].name, "writer");
        assert_eq!(request.sub_agents[1].max_turns, 5);
        assert_eq!(request.sub_agents[1].builtin_tools, vec!["write"]);

        let orch = request.orchestrator.as_ref().unwrap();
        assert_eq!(orch.max_turns, 8);
        assert!(orch.enable_squads);
        assert_eq!(orch.dispatch_mode, Some(DispatchMode::Parallel));
    }

    #[test]
    fn runtime_workflow_config_round_trip() {
        let config = RuntimeWorkflowConfig {
            workflow_type: WorkflowType::Dag,
            nodes: vec![
                RuntimeWorkflowNode {
                    name: "a".into(),
                    agent_name: "researcher".into(),
                    role: None,
                },
                RuntimeWorkflowNode {
                    name: "b".into(),
                    agent_name: "writer".into(),
                    role: None,
                },
            ],
            edges: vec![RuntimeWorkflowEdge {
                from: "a".into(),
                to: "b".into(),
                condition: Some(EdgeConditionSpec {
                    pattern: EdgeConditionPattern::Contains,
                    value: "success".into(),
                }),
                transform: None,
            }],
            max_iterations: None,
            stop_pattern: None,
            rounds: None,
            layers: None,
        };
        let json = serde_json::to_string(&config).expect("serialize");
        let parsed: RuntimeWorkflowConfig = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.nodes.len(), 2);
        assert_eq!(parsed.edges.len(), 1);
        assert!(matches!(parsed.workflow_type, WorkflowType::Dag));
    }

    #[test]
    fn runtime_request_backward_compat_no_workflow() {
        let json = serde_json::json!({
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "Do something",
            "agent": {
                "name": "minimal-agent"
            },
            "provider": {
                "provider_type": "anthropic",
                "api_key": "sk-key",
                "model": "claude-sonnet-4-20250514"
            }
        });
        let request: RuntimeRequest =
            serde_json::from_value(json).expect("deserialize without workflow");
        assert!(request.workflow.is_none());
    }

    #[test]
    fn runtime_request_with_workflow_round_trip() {
        let json = serde_json::json!({
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "prompt": "Run the workflow",
            "agent": { "name": "orchestrator" },
            "provider": {
                "provider_type": "anthropic",
                "api_key": "sk-key",
                "model": "claude-sonnet-4-20250514"
            },
            "sub_agents": [
                {
                    "name": "researcher",
                    "description": "Gathers data",
                    "system_prompt": "You research."
                },
                {
                    "name": "writer",
                    "description": "Writes reports",
                    "system_prompt": "You write."
                }
            ],
            "workflow": {
                "workflow_type": "dag",
                "nodes": [
                    { "name": "a", "agent_name": "researcher" },
                    { "name": "b", "agent_name": "writer" }
                ],
                "edges": [
                    { "from": "a", "to": "b" }
                ]
            }
        });
        let request: RuntimeRequest =
            serde_json::from_value(json).expect("deserialize with workflow");
        let wf = request.workflow.as_ref().unwrap();
        assert!(matches!(wf.workflow_type, WorkflowType::Dag));
        assert_eq!(wf.nodes.len(), 2);
        assert_eq!(wf.edges.len(), 1);
        assert_eq!(wf.edges[0].from, "a");
        assert_eq!(wf.edges[0].to, "b");
    }

    #[test]
    fn edge_condition_spec_round_trip() {
        let spec = EdgeConditionSpec {
            pattern: EdgeConditionPattern::Regex,
            value: "\\d+".into(),
        };
        let json = serde_json::to_string(&spec).expect("serialize");
        assert!(json.contains("regex"), "json: {json}");
        let parsed: EdgeConditionSpec = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(parsed.pattern, EdgeConditionPattern::Regex));
        assert_eq!(parsed.value, "\\d+");
    }

    #[test]
    fn edge_transform_round_trip() {
        let t = EdgeTransform::ExtractJson;
        let json = serde_json::to_string(&t).expect("serialize");
        assert_eq!(json, r#""extract_json""#);
        let parsed: EdgeTransform = serde_json::from_str(&json).expect("deserialize");
        assert!(matches!(parsed, EdgeTransform::ExtractJson));
    }

    #[test]
    fn workflow_node_with_role() {
        let json = r#"{"name":"judge","agent_name":"judge-agent","role":"judge"}"#;
        let node: RuntimeWorkflowNode = serde_json::from_str(json).expect("deserialize");
        assert_eq!(node.name, "judge");
        assert_eq!(node.role.as_deref(), Some("judge"));
    }

    #[test]
    fn workflow_edge_minimal() {
        let json = r#"{"from":"a","to":"b"}"#;
        let edge: RuntimeWorkflowEdge = serde_json::from_str(json).expect("deserialize");
        assert_eq!(edge.from, "a");
        assert_eq!(edge.to, "b");
        assert!(edge.condition.is_none());
        assert!(edge.transform.is_none());
    }

    #[test]
    fn workflow_config_optional_fields_omitted() {
        let config = RuntimeWorkflowConfig {
            workflow_type: WorkflowType::Sequential,
            nodes: vec![RuntimeWorkflowNode {
                name: "only".into(),
                agent_name: "agent-a".into(),
                role: None,
            }],
            edges: vec![],
            max_iterations: None,
            stop_pattern: None,
            rounds: None,
            layers: None,
        };
        let json = serde_json::to_string(&config).expect("serialize");
        assert!(!json.contains("max_iterations"));
        assert!(!json.contains("stop_pattern"));
        assert!(!json.contains("rounds"));
        assert!(!json.contains("layers"));
    }
}
