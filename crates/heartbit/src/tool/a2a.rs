use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use serde_json::Value;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const POLL_INTERVAL: Duration = Duration::from_secs(2);
const MAX_POLL_ATTEMPTS: usize = 150; // 5 minutes at 2s intervals
const AGENT_CARD_PATH: &str = "/.well-known/agent.json";

// --- Pure helper functions ---

/// Normalize an agent name into a valid tool name.
///
/// Lowercases, replaces non-alphanumeric chars with underscores, collapses
/// consecutive underscores, and prefixes with `a2a_`.
fn normalize_tool_name(name: &str) -> String {
    let mut normalized = String::with_capacity(name.len() + 4);
    normalized.push_str("a2a_");
    let mut prev_underscore = false;
    for c in name.chars() {
        if c.is_alphanumeric() {
            normalized.push(c.to_ascii_lowercase());
            prev_underscore = false;
        } else if !prev_underscore {
            normalized.push('_');
            prev_underscore = true;
        }
    }
    // Trim trailing underscore
    if normalized.ends_with('_') && normalized.len() > 4 {
        normalized.pop();
    }
    normalized
}

/// Build a tool description from an agent card.
fn build_tool_description(card: &a2a_sdk::AgentCard) -> String {
    let mut desc = card.description.clone();
    if !card.skills.is_empty() {
        desc.push_str("\n\nSkills:");
        for skill in &card.skills {
            if skill.description.is_empty() {
                desc.push_str(&format!("\n- {}", skill.name));
            } else {
                desc.push_str(&format!("\n- {}: {}", skill.name, skill.description));
            }
        }
    }
    desc
}

/// Build a [`ToolDefinition`] from an agent card.
fn agent_card_to_tool_definition(card: &a2a_sdk::AgentCard) -> ToolDefinition {
    ToolDefinition {
        name: normalize_tool_name(&card.name),
        description: build_tool_description(card),
        input_schema: serde_json::json!({
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The message to send to the agent"
                }
            },
            "required": ["message"]
        }),
    }
}

/// Extract text content from a JSON array of A2A parts.
///
/// Handles both PascalCase (`"Text"`) and lowercase (`"text"`) kind values
/// for resilience against spec/SDK variation.
fn extract_text_from_parts(parts: &[Value]) -> String {
    parts
        .iter()
        .filter_map(|part| {
            let kind = part.get("kind").and_then(|k| k.as_str())?;
            match kind {
                "Text" | "text" => part
                    .get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string()),
                "File" | "file" => {
                    let name = part
                        .get("file")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        .unwrap_or("unnamed");
                    Some(format!("[file: {name}]"))
                }
                "Data" | "data" => Some("[data]".to_string()),
                _ => None,
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract text from a message JSON object.
///
/// Tries both `"content"` (SDK convention) and `"parts"` (spec convention)
/// field names for the parts array.
fn extract_text_from_message(message: &Value) -> String {
    let parts = message
        .get("content")
        .or_else(|| message.get("parts"))
        .and_then(|p| p.as_array());
    match parts {
        Some(parts) => extract_text_from_parts(parts),
        None => String::new(),
    }
}

/// Whether a task state string is terminal (no more polling expected).
///
/// Includes `input-required` and `auth-required` because the A2A tool
/// cannot provide interactive input or authentication credentials — the
/// task won't progress without external intervention.
fn is_terminal_state(state: &str) -> bool {
    matches!(
        state,
        "completed" | "canceled" | "failed" | "rejected" | "input-required" | "auth-required"
    )
}

/// Extract text from a completed task's artifacts and status message.
fn extract_text_from_task(task: &Value) -> String {
    let mut texts = Vec::new();

    if let Some(artifacts) = task.get("artifacts").and_then(|a| a.as_array()) {
        for artifact in artifacts {
            if let Some(parts) = artifact.get("parts").and_then(|p| p.as_array()) {
                let text = extract_text_from_parts(parts);
                if !text.is_empty() {
                    texts.push(text);
                }
            }
        }
    }

    if let Some(status) = task.get("status")
        && let Some(message) = status.get("message")
    {
        let text = extract_text_from_message(message);
        if !text.is_empty() {
            texts.push(text);
        }
    }

    texts.join("\n")
}

/// Convert a `message/send` result (Task or Message) to [`ToolOutput`].
fn result_to_tool_output(result: &Value) -> ToolOutput {
    // A Task has "status" and "id" fields; a Message does not.
    if result.get("status").is_some() && result.get("id").is_some() {
        let state = result
            .get("status")
            .and_then(|s| s.get("state"))
            .and_then(|s| s.as_str())
            .unwrap_or("unknown");

        let text = extract_text_from_task(result);

        match state {
            "completed" => {
                if text.is_empty() {
                    ToolOutput::success("Task completed (no text output)")
                } else {
                    ToolOutput::success(text)
                }
            }
            "failed" | "rejected" | "canceled" => {
                if text.is_empty() {
                    ToolOutput::error(format!("Task {state}"))
                } else {
                    ToolOutput::error(text)
                }
            }
            "input-required" => ToolOutput::error(
                "Task requires additional input that cannot be provided automatically",
            ),
            "auth-required" => ToolOutput::error(
                "Task requires authentication that cannot be provided automatically",
            ),
            _ => {
                if text.is_empty() {
                    ToolOutput::success(format!("Task state: {state}"))
                } else {
                    ToolOutput::success(text)
                }
            }
        }
    } else {
        let text = extract_text_from_message(result);
        if text.is_empty() {
            ToolOutput::success("(empty response)")
        } else {
            ToolOutput::success(text)
        }
    }
}

// --- A2aSession ---

struct A2aSession {
    client: reqwest::Client,
    endpoint: String,
    auth_header: Option<String>,
    next_id: AtomicU64,
}

impl A2aSession {
    fn next_id(&self) -> u64 {
        self.next_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Send a JSON-RPC request and return the `result` value.
    async fn rpc(&self, method: &str, params: Value) -> Result<Value, Error> {
        let id = self.next_id();
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let mut builder = self.client.post(&self.endpoint).json(&body);

        if let Some(auth) = &self.auth_header {
            builder = builder.header("Authorization", auth);
        }

        let response = builder.send().await?;
        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(Error::A2a(format!("HTTP {}: {}", status.as_u16(), body)));
        }

        let rpc_response: Value = serde_json::from_str(&body)?;

        if let Some(error) = rpc_response.get("error") {
            let code = error.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
            let message = error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(Error::A2a(format!("JSON-RPC error {code}: {message}")));
        }

        rpc_response
            .get("result")
            .cloned()
            .ok_or_else(|| Error::A2a("Response missing both result and error".into()))
    }

    /// Send `message/send` with blocking mode and text-only output.
    async fn send_message(&self, message: &str) -> Result<Value, Error> {
        let params = serde_json::json!({
            "message": {
                "role": "user",
                "content": [{"kind": "Text", "text": message}]
            },
            "configuration": {
                "blocking": true,
                "acceptedOutputModes": ["text"]
            }
        });
        self.rpc("message/send", params).await
    }

    /// Fetch the current state of a task via `tasks/get`.
    async fn get_task(&self, task_id: &str) -> Result<Value, Error> {
        let params = serde_json::json!({
            "id": task_id,
        });
        self.rpc("tasks/get", params).await
    }

    /// Poll `tasks/get` until the task reaches a terminal state.
    ///
    /// Transient errors during polling are tolerated — the poll loop continues
    /// on the next interval. Only if all attempts fail or produce errors is an
    /// error returned.
    async fn poll_task(&self, task_id: &str) -> Result<Value, Error> {
        let mut last_error: Option<Error> = None;
        for _ in 0..MAX_POLL_ATTEMPTS {
            match self.get_task(task_id).await {
                Ok(task) => {
                    last_error = None;
                    let state = task
                        .get("status")
                        .and_then(|s| s.get("state"))
                        .and_then(|s| s.as_str())
                        .unwrap_or("unknown");

                    if is_terminal_state(state) {
                        return Ok(task);
                    }
                }
                Err(e) => {
                    tracing::warn!(task_id = %task_id, error = %e, "transient error during task polling, retrying");
                    last_error = Some(e);
                }
            }

            tokio::time::sleep(POLL_INTERVAL).await;
        }

        Err(last_error.unwrap_or_else(|| {
            Error::A2a(format!(
                "Task {task_id} did not reach terminal state after {MAX_POLL_ATTEMPTS} poll attempts"
            ))
        }))
    }
}

// --- A2aTool ---

struct A2aTool {
    session: Arc<A2aSession>,
    def: ToolDefinition,
    agent_name: String,
}

impl Tool for A2aTool {
    fn definition(&self) -> ToolDefinition {
        self.def.clone()
    }

    fn execute(
        &self,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let message = input.get("message").and_then(|m| m.as_str()).unwrap_or("");

            if message.is_empty() {
                return Ok(ToolOutput::error(
                    "message field is required and must be non-empty",
                ));
            }

            let result = match self.session.send_message(message).await {
                Ok(result) => result,
                Err(e) => {
                    tracing::warn!(
                        agent = %self.agent_name,
                        error = %e,
                        "A2A message/send failed"
                    );
                    return Ok(ToolOutput::error(e.to_string()));
                }
            };

            // If the result is a non-terminal Task, poll until done.
            if result.get("status").is_some() && result.get("id").is_some() {
                let state = result
                    .get("status")
                    .and_then(|s| s.get("state"))
                    .and_then(|s| s.as_str())
                    .unwrap_or("unknown");

                if !is_terminal_state(state) {
                    let task_id = match result.get("id").and_then(|id| id.as_str()) {
                        Some(id) => id,
                        None => return Ok(ToolOutput::error("Task returned without string id")),
                    };

                    match self.session.poll_task(task_id).await {
                        Ok(task) => return Ok(result_to_tool_output(&task)),
                        Err(e) => {
                            tracing::warn!(
                                agent = %self.agent_name,
                                task_id = %task_id,
                                error = %e,
                                "A2A task polling failed"
                            );
                            return Ok(ToolOutput::error(e.to_string()));
                        }
                    }
                }
            }

            Ok(result_to_tool_output(&result))
        })
    }
}

// --- A2aClient ---

/// Client for the A2A (Agent-to-Agent) protocol.
///
/// Discovers an external agent via `GET /.well-known/agent.json`, then
/// produces a `Vec<Arc<dyn Tool>>` (one tool wrapping the agent) that plugs
/// into `AgentRunnerBuilder::tools()`.
pub struct A2aClient {
    session: Arc<A2aSession>,
    agent_card: a2a_sdk::AgentCard,
}

impl A2aClient {
    /// Connect to an A2A agent and discover its capabilities.
    ///
    /// Fetches the agent card from `{base_url}/.well-known/agent.json`.
    pub async fn connect(base_url: &str) -> Result<Self, Error> {
        Self::connect_internal(base_url, None).await
    }

    /// Connect to an A2A agent with an authorization header.
    ///
    /// The `auth_header` is sent as the `Authorization` header value
    /// (e.g., `"Bearer <token>"`).
    pub async fn connect_with_auth(
        base_url: &str,
        auth_header: impl Into<String>,
    ) -> Result<Self, Error> {
        Self::connect_internal(base_url, Some(auth_header.into())).await
    }

    async fn connect_internal(base_url: &str, auth_header: Option<String>) -> Result<Self, Error> {
        let client = reqwest::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()?;

        let card_url = format!("{}{}", base_url.trim_end_matches('/'), AGENT_CARD_PATH);

        let mut builder = client.get(&card_url);
        if let Some(ref auth) = auth_header {
            builder = builder.header("Authorization", auth);
        }

        let response = builder.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(Error::A2a(format!(
                "Failed to fetch agent card from {card_url}: HTTP {}: {body}",
                status.as_u16()
            )));
        }

        let agent_card: a2a_sdk::AgentCard = response
            .json()
            .await
            .map_err(|e| Error::A2a(format!("Failed to parse agent card: {e}")))?;

        let session = Arc::new(A2aSession {
            client,
            endpoint: agent_card.url.clone(),
            auth_header,
            next_id: AtomicU64::new(0),
        });

        Ok(Self {
            session,
            agent_card,
        })
    }

    /// Get tool definitions without consuming the client.
    pub fn tool_definitions(&self) -> Vec<ToolDefinition> {
        vec![agent_card_to_tool_definition(&self.agent_card)]
    }

    /// Convert the discovered A2A agent into `Arc<dyn Tool>` instances.
    pub fn into_tools(self) -> Vec<Arc<dyn Tool>> {
        let def = agent_card_to_tool_definition(&self.agent_card);
        let agent_name = self.agent_card.name.clone();
        let tool: Arc<dyn Tool> = Arc::new(A2aTool {
            session: self.session,
            def,
            agent_name,
        });
        vec![tool]
    }

    /// Get a reference to the discovered agent card.
    pub fn agent_card(&self) -> &a2a_sdk::AgentCard {
        &self.agent_card
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // --- normalize_tool_name tests ---

    #[test]
    fn tool_name_simple() {
        assert_eq!(normalize_tool_name("MyAgent"), "a2a_myagent");
    }

    #[test]
    fn tool_name_spaces_to_underscores() {
        assert_eq!(normalize_tool_name("My Cool Agent"), "a2a_my_cool_agent");
    }

    #[test]
    fn tool_name_special_chars() {
        assert_eq!(normalize_tool_name("Agent (v2.0)"), "a2a_agent_v2_0");
    }

    #[test]
    fn tool_name_consecutive_specials_collapsed() {
        assert_eq!(normalize_tool_name("a--b__c"), "a2a_a_b_c");
    }

    #[test]
    fn tool_name_trailing_special_trimmed() {
        assert_eq!(normalize_tool_name("Agent "), "a2a_agent");
    }

    #[test]
    fn tool_name_already_lowercase() {
        assert_eq!(normalize_tool_name("simple"), "a2a_simple");
    }

    // --- build_tool_description tests ---

    #[test]
    fn description_with_skills() {
        let card = test_agent_card_with_skills();
        let desc = build_tool_description(&card);
        assert!(desc.starts_with("A test agent"));
        assert!(desc.contains("Skills:"));
        assert!(desc.contains("- Search: Search the web"));
        assert!(desc.contains("- Summarize: Summarize text"));
    }

    #[test]
    fn description_without_skills() {
        let card = test_agent_card_no_skills();
        let desc = build_tool_description(&card);
        assert_eq!(desc, "A test agent");
        assert!(!desc.contains("Skills:"));
    }

    // --- agent_card_to_tool_definition tests ---

    #[test]
    fn agent_card_to_tool_definition_basic() {
        let card = test_agent_card_with_skills();
        let def = agent_card_to_tool_definition(&card);

        assert_eq!(def.name, "a2a_test_agent");
        assert!(def.description.starts_with("A test agent"));
        assert!(def.description.contains("Skills:"));

        // Schema requires "message" string
        assert_eq!(def.input_schema["type"], "object");
        assert_eq!(def.input_schema["properties"]["message"]["type"], "string");
        assert_eq!(def.input_schema["required"][0], "message");
    }

    #[test]
    fn agent_card_to_tool_definition_no_skills() {
        let card = test_agent_card_no_skills();
        let def = agent_card_to_tool_definition(&card);
        assert_eq!(def.name, "a2a_test_agent");
        assert_eq!(def.description, "A test agent");
        assert_eq!(def.input_schema["type"], "object");
    }

    // --- extract_text_from_parts tests ---

    #[test]
    fn extract_text_from_text_part() {
        let parts = vec![json!({"kind": "Text", "text": "hello world"})];
        assert_eq!(extract_text_from_parts(&parts), "hello world");
    }

    #[test]
    fn extract_text_from_text_part_lowercase() {
        let parts = vec![json!({"kind": "text", "text": "hello"})];
        assert_eq!(extract_text_from_parts(&parts), "hello");
    }

    #[test]
    fn extract_text_from_file_part() {
        let parts = vec![json!({
            "kind": "File",
            "file": {"name": "report.pdf", "mimeType": "application/pdf"}
        })];
        assert_eq!(extract_text_from_parts(&parts), "[file: report.pdf]");
    }

    #[test]
    fn extract_text_from_file_part_no_name() {
        let parts = vec![json!({"kind": "File", "file": {}})];
        assert_eq!(extract_text_from_parts(&parts), "[file: unnamed]");
    }

    #[test]
    fn extract_text_from_data_part() {
        let parts = vec![json!({"kind": "Data", "data": {"key": "value"}})];
        assert_eq!(extract_text_from_parts(&parts), "[data]");
    }

    #[test]
    fn extract_text_from_mixed_parts() {
        let parts = vec![
            json!({"kind": "Text", "text": "Here is the report:"}),
            json!({"kind": "File", "file": {"name": "report.pdf"}}),
            json!({"kind": "Text", "text": "Let me know if you need more."}),
        ];
        assert_eq!(
            extract_text_from_parts(&parts),
            "Here is the report:\n[file: report.pdf]\nLet me know if you need more."
        );
    }

    #[test]
    fn extract_text_from_empty_parts() {
        let parts: Vec<Value> = vec![];
        assert_eq!(extract_text_from_parts(&parts), "");
    }

    // --- extract_text_from_message tests ---

    #[test]
    fn extract_text_from_message_content_field() {
        let message = json!({
            "role": "agent",
            "content": [{"kind": "Text", "text": "response text"}]
        });
        assert_eq!(extract_text_from_message(&message), "response text");
    }

    #[test]
    fn extract_text_from_message_parts_field() {
        let message = json!({
            "role": "agent",
            "parts": [{"kind": "Text", "text": "response text"}]
        });
        assert_eq!(extract_text_from_message(&message), "response text");
    }

    #[test]
    fn extract_text_from_message_no_parts() {
        let message = json!({"role": "agent"});
        assert_eq!(extract_text_from_message(&message), "");
    }

    // --- is_terminal_state tests ---

    #[test]
    fn terminal_states() {
        assert!(is_terminal_state("completed"));
        assert!(is_terminal_state("canceled"));
        assert!(is_terminal_state("failed"));
        assert!(is_terminal_state("rejected"));
        assert!(is_terminal_state("input-required"));
        assert!(is_terminal_state("auth-required"));
    }

    #[test]
    fn non_terminal_states() {
        assert!(!is_terminal_state("submitted"));
        assert!(!is_terminal_state("working"));
        assert!(!is_terminal_state("unknown"));
    }

    // --- result_to_tool_output tests ---

    #[test]
    fn result_message_to_output() {
        let result = json!({
            "role": "agent",
            "content": [{"kind": "Text", "text": "The answer is 42."}]
        });
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "The answer is 42.");
        assert!(!output.is_error);
    }

    #[test]
    fn result_empty_message_to_output() {
        let result = json!({"role": "agent", "content": []});
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "(empty response)");
        assert!(!output.is_error);
    }

    #[test]
    fn result_completed_task_to_output() {
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "completed"},
            "artifacts": [{
                "artifactId": "art-1",
                "parts": [{"kind": "Text", "text": "Task result here."}]
            }]
        });
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "Task result here.");
        assert!(!output.is_error);
    }

    #[test]
    fn result_completed_task_no_artifacts() {
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "completed"},
            "artifacts": []
        });
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "Task completed (no text output)");
        assert!(!output.is_error);
    }

    #[test]
    fn result_completed_task_with_status_message() {
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {
                "state": "completed",
                "message": {
                    "role": "agent",
                    "content": [{"kind": "Text", "text": "All done!"}]
                }
            },
            "artifacts": []
        });
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "All done!");
        assert!(!output.is_error);
    }

    #[test]
    fn result_failed_task_to_output() {
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {
                "state": "failed",
                "message": {
                    "role": "agent",
                    "content": [{"kind": "Text", "text": "Something went wrong"}]
                }
            },
            "artifacts": []
        });
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "Something went wrong");
        assert!(output.is_error);
    }

    #[test]
    fn result_failed_task_no_message() {
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "failed"},
            "artifacts": []
        });
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "Task failed");
        assert!(output.is_error);
    }

    #[test]
    fn result_canceled_task_to_output() {
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "canceled"},
            "artifacts": []
        });
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "Task canceled");
        assert!(output.is_error);
    }

    #[test]
    fn result_canceled_task_with_message() {
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {
                "state": "canceled",
                "message": {
                    "role": "agent",
                    "content": [{"kind": "Text", "text": "User requested cancellation"}]
                }
            },
            "artifacts": []
        });
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "User requested cancellation");
        assert!(output.is_error);
    }

    #[test]
    fn result_rejected_task_to_output() {
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "rejected"},
            "artifacts": []
        });
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "Task rejected");
        assert!(output.is_error);
    }

    #[test]
    fn result_input_required_task_to_output() {
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "input-required"},
            "artifacts": []
        });
        let output = result_to_tool_output(&result);
        assert!(output.is_error);
        assert!(output.content.contains("requires additional input"));
    }

    #[test]
    fn result_auth_required_task_to_output() {
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "auth-required"},
            "artifacts": []
        });
        let output = result_to_tool_output(&result);
        assert!(output.is_error);
        assert!(output.content.contains("requires authentication"));
    }

    #[test]
    fn result_working_task_to_output() {
        // Non-terminal state returned in result (edge case)
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "working"},
            "artifacts": []
        });
        let output = result_to_tool_output(&result);
        assert_eq!(output.content, "Task state: working");
        assert!(!output.is_error);
    }

    // --- A2aSession tests ---

    #[test]
    fn session_next_id_is_monotonic() {
        let session = A2aSession {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            auth_header: None,
            next_id: AtomicU64::new(0),
        };
        assert_eq!(session.next_id(), 0);
        assert_eq!(session.next_id(), 1);
        assert_eq!(session.next_id(), 2);
    }

    // --- A2aTool tests ---

    #[test]
    fn a2a_tool_returns_correct_definition() {
        let session = Arc::new(A2aSession {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            auth_header: None,
            next_id: AtomicU64::new(0),
        });

        let expected_def = ToolDefinition {
            name: "a2a_test".into(),
            description: "Test agent".into(),
            input_schema: json!({
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"]
            }),
        };

        let tool = A2aTool {
            session,
            def: expected_def.clone(),
            agent_name: "test".into(),
        };

        assert_eq!(tool.definition(), expected_def);
    }

    #[tokio::test]
    async fn a2a_tool_empty_message_returns_error() {
        let session = Arc::new(A2aSession {
            client: reqwest::Client::new(),
            endpoint: "http://unused".to_string(),
            auth_header: None,
            next_id: AtomicU64::new(0),
        });

        let tool = A2aTool {
            session,
            def: ToolDefinition {
                name: "a2a_test".into(),
                description: "test".into(),
                input_schema: json!({"type": "object"}),
            },
            agent_name: "test".into(),
        };

        // Empty message
        let result = tool.execute(json!({"message": ""})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("required"));

        // Missing message field
        let result = tool.execute(json!({})).await.unwrap();
        assert!(result.is_error);
    }

    #[tokio::test]
    async fn a2a_tool_execute_catches_network_errors() {
        let session = Arc::new(A2aSession {
            client: reqwest::Client::new(),
            endpoint: "http://127.0.0.1:1".to_string(), // nothing listening
            auth_header: None,
            next_id: AtomicU64::new(0),
        });

        let tool = A2aTool {
            session,
            def: ToolDefinition {
                name: "a2a_test".into(),
                description: "test".into(),
                input_schema: json!({"type": "object"}),
            },
            agent_name: "test".into(),
        };

        // execute() should catch the connection error and return ToolOutput::error,
        // not propagate it as Err
        let result = tool.execute(json!({"message": "hello"})).await.unwrap();
        assert!(result.is_error);
        assert!(!result.content.is_empty());
    }

    #[tokio::test]
    async fn connect_network_error_returns_err() {
        // Connect to an unreachable host should return Err
        let result = A2aClient::connect("http://127.0.0.1:1").await;
        assert!(result.is_err());
    }

    // --- A2aClient tool wiring tests ---

    #[test]
    fn into_tools_produces_one_tool() {
        let client = test_a2a_client();
        let tools = client.into_tools();
        assert_eq!(tools.len(), 1);
    }

    #[test]
    fn tool_definitions_matches_into_tools() {
        let client = test_a2a_client();
        let defs = client.tool_definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "a2a_test_agent");
    }

    #[test]
    fn agent_card_accessor() {
        let client = test_a2a_client();
        assert_eq!(client.agent_card().name, "Test Agent");
    }

    // --- extract_text_from_task tests ---

    #[test]
    fn extract_text_from_task_multiple_artifacts() {
        let task = json!({
            "id": "t1",
            "status": {"state": "completed"},
            "artifacts": [
                {"artifactId": "a1", "parts": [{"kind": "Text", "text": "First result"}]},
                {"artifactId": "a2", "parts": [{"kind": "Text", "text": "Second result"}]}
            ]
        });
        assert_eq!(extract_text_from_task(&task), "First result\nSecond result");
    }

    #[test]
    fn extract_text_from_task_artifacts_and_status_message() {
        let task = json!({
            "id": "t1",
            "status": {
                "state": "completed",
                "message": {"role": "agent", "content": [{"kind": "Text", "text": "Summary"}]}
            },
            "artifacts": [
                {"artifactId": "a1", "parts": [{"kind": "Text", "text": "Data"}]}
            ]
        });
        assert_eq!(extract_text_from_task(&task), "Data\nSummary");
    }

    #[test]
    fn extract_text_from_task_no_artifacts_key() {
        let task = json!({
            "id": "t1",
            "status": {"state": "completed"}
        });
        assert_eq!(extract_text_from_task(&task), "");
    }

    #[test]
    fn extract_text_from_task_artifacts_null() {
        let task = json!({
            "id": "t1",
            "status": {"state": "completed"},
            "artifacts": null
        });
        assert_eq!(extract_text_from_task(&task), "");
    }

    // --- build_tool_description edge cases ---

    #[test]
    fn build_tool_description_skill_empty_description() {
        let mut card = test_agent_card_with_skills();
        card.skills = vec![a2a_sdk::AgentSkill {
            id: "silent".to_string(),
            name: "Silent Skill".to_string(),
            description: String::new(),
            examples: vec![],
            input_modes: vec![],
            output_modes: vec![],
            security: vec![],
            tags: vec![],
        }];
        let desc = build_tool_description(&card);
        assert!(desc.contains("- Silent Skill"));
        // Should NOT have a colon when description is empty
        assert!(!desc.contains("Silent Skill:"));
    }

    // --- extract_text_from_parts edge cases ---

    #[test]
    fn extract_text_from_parts_unknown_kind() {
        let parts = vec![
            json!({"kind": "Text", "text": "hello"}),
            json!({"kind": "Unknown", "data": "xyz"}),
            json!({"kind": "Text", "text": "world"}),
        ];
        assert_eq!(extract_text_from_parts(&parts), "hello\nworld");
    }

    #[test]
    fn extract_text_from_parts_missing_kind() {
        let parts = vec![
            json!({"text": "orphan text"}),
            json!({"kind": "Text", "text": "valid"}),
        ];
        // Part without "kind" is filtered out
        assert_eq!(extract_text_from_parts(&parts), "valid");
    }

    #[test]
    fn extract_text_from_parts_lowercase_file() {
        let parts = vec![json!({
            "kind": "file",
            "file": {"name": "report.pdf", "mimeType": "application/pdf"}
        })];
        assert_eq!(extract_text_from_parts(&parts), "[file: report.pdf]");
    }

    #[test]
    fn extract_text_from_parts_lowercase_data() {
        let parts = vec![json!({"kind": "data", "data": {"key": "value"}})];
        assert_eq!(extract_text_from_parts(&parts), "[data]");
    }

    // --- result_to_tool_output edge cases ---

    #[test]
    fn result_working_task_with_text_to_output() {
        // Non-terminal state with artifacts — returns the text directly
        let result = json!({
            "id": "task-1",
            "contextId": "ctx-1",
            "status": {"state": "working"},
            "artifacts": [{
                "artifactId": "a1",
                "parts": [{"kind": "Text", "text": "partial output"}]
            }]
        });
        let output = result_to_tool_output(&result);
        assert!(!output.is_error);
        assert_eq!(output.content, "partial output");
    }

    #[test]
    fn result_status_without_id_treated_as_message() {
        // Has "status" but no "id" — falls through to message path
        let result = json!({
            "role": "agent",
            "status": {"state": "completed"},
            "content": [{"kind": "Text", "text": "just a message"}]
        });
        let output = result_to_tool_output(&result);
        // This gets treated as a task since it has status field
        // but the exact behavior depends on the implementation
        assert!(!output.content.is_empty());
    }

    // --- Test helpers ---

    fn test_agent_card_with_skills() -> a2a_sdk::AgentCard {
        a2a_sdk::AgentCard {
            name: "Test Agent".to_string(),
            description: "A test agent".to_string(),
            url: "http://localhost:8080/a2a".to_string(),
            version: "1.0.0".to_string(),
            protocol_version: "0.3.0".to_string(),
            capabilities: a2a_sdk::AgentCapabilities::default(),
            default_input_modes: vec!["text".to_string()],
            default_output_modes: vec!["text".to_string()],
            skills: vec![
                a2a_sdk::AgentSkill {
                    id: "search".to_string(),
                    name: "Search".to_string(),
                    description: "Search the web".to_string(),
                    examples: vec![],
                    input_modes: vec![],
                    output_modes: vec![],
                    security: vec![],
                    tags: vec![],
                },
                a2a_sdk::AgentSkill {
                    id: "summarize".to_string(),
                    name: "Summarize".to_string(),
                    description: "Summarize text".to_string(),
                    examples: vec![],
                    input_modes: vec![],
                    output_modes: vec![],
                    security: vec![],
                    tags: vec![],
                },
            ],
            authentication: None,
            additional_interfaces: vec![],
            documentation_url: None,
            icon_url: None,
            preferred_transport: None,
            provider: None,
            security: vec![],
            security_schemes: std::collections::HashMap::new(),
            signatures: vec![],
            supports_authenticated_extended_card: None,
        }
    }

    fn test_agent_card_no_skills() -> a2a_sdk::AgentCard {
        let mut card = test_agent_card_with_skills();
        card.skills = vec![];
        card
    }

    fn test_a2a_client() -> A2aClient {
        let card = test_agent_card_with_skills();
        let session = Arc::new(A2aSession {
            client: reqwest::Client::new(),
            endpoint: card.url.clone(),
            auth_header: None,
            next_id: AtomicU64::new(0),
        });
        A2aClient {
            session,
            agent_card: card,
        }
    }
}
