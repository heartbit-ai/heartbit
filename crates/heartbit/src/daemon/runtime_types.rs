use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::agent::guardrails::pii::PiiAction;
use crate::config::DispatchMode;

/// Provider type for runtime execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RuntimeProviderType {
    Anthropic,
    Openrouter,
}

/// MCP server configuration for runtime execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeMcpServer {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_header: Option<String>,
}

/// Provider configuration for runtime execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeProviderConfig {
    pub provider_type: RuntimeProviderType,
    pub api_key: String,
    pub model: String,
    #[serde(default)]
    pub prompt_caching: bool,
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
}

/// Execution response from runtime to cloud.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeResponse {
    pub result: String,
    pub usage: crate::llm::types::TokenUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
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
}
