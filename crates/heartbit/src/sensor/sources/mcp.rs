use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use rdkafka::producer::{FutureProducer, FutureRecord};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::config::McpServerEntry;
use crate::sensor::triage::{Priority, TriageDecision, TriageProcessor};
use crate::sensor::{Sensor, SensorEvent, SensorModality};
use crate::tool::Tool;
use crate::tool::builtins::floor_char_boundary;
use crate::tool::mcp::McpClient;

/// Generic MCP sensor that connects to any MCP server, polls a specific tool,
/// and maps results to `SensorEvent` values on Kafka.
///
/// This avoids building per-service sensors (Gmail, Calendar, Drive…) — the MCP
/// server handles authentication and API details, while this sensor handles
/// polling, JSON result mapping, dedup, and Kafka production.
pub struct McpSensor {
    name: String,
    server: McpServerEntry,
    tool_name: String,
    tool_args: Value,
    kafka_topic: String,
    modality: SensorModality,
    poll_interval: Duration,
    id_field: String,
    content_field: Option<String>,
    items_field: Option<String>,
    /// Optional enrichment tool to call for each new item (e.g., `gmail_get_message`).
    enrich_tool_name: Option<String>,
    /// Parameter name for the item ID when calling the enrichment tool (e.g., `messageId`).
    enrich_id_param: Option<String>,
    /// How long to remember seen IDs before evicting them.
    dedup_ttl: Duration,
    /// Directory for persisting dedup state across restarts.
    /// If `None`, dedup is in-memory only (lost on restart).
    state_dir: Option<PathBuf>,
}

impl McpSensor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        name: impl Into<String>,
        server: McpServerEntry,
        tool_name: impl Into<String>,
        tool_args: Value,
        kafka_topic: impl Into<String>,
        modality: SensorModality,
        poll_interval: Duration,
        id_field: impl Into<String>,
        content_field: Option<String>,
        items_field: Option<String>,
        enrich_tool_name: Option<String>,
        enrich_id_param: Option<String>,
        dedup_ttl: Duration,
        state_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            name: name.into(),
            server,
            tool_name: tool_name.into(),
            tool_args,
            kafka_topic: kafka_topic.into(),
            modality,
            poll_interval,
            id_field: id_field.into(),
            content_field,
            items_field,
            enrich_tool_name,
            enrich_id_param,
            dedup_ttl,
            state_dir,
        }
    }
}

impl Sensor for McpSensor {
    fn name(&self) -> &str {
        &self.name
    }

    fn modality(&self) -> SensorModality {
        self.modality
    }

    fn kafka_topic(&self) -> &str {
        &self.kafka_topic
    }

    fn run(
        &self,
        producer: FutureProducer,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            let mut poll_tool: Option<Arc<dyn Tool>> = None;
            // Enrichment tool uses a SEPARATE MCP connection to avoid session
            // conflicts with the poll tool (agentgateway may close sessions
            // between sequential tool calls on the same session).
            let mut enrich_tool: Option<Arc<dyn Tool>> = None;
            let mut enrich_failures: u32 = 0;
            // Dedup map: source_id → first-seen Unix timestamp (epoch seconds).
            // Persisted to disk so that restarts don't re-process old items.
            let mut seen: HashMap<String, i64> = load_seen_state(
                self.state_dir.as_deref(),
                &self.name,
                self.dedup_ttl.as_secs() as i64,
            );
            let dedup_ttl_secs = self.dedup_ttl.as_secs() as i64;
            let mut last_cleanup = Instant::now();

            loop {
                if cancel.is_cancelled() {
                    return Ok(());
                }

                // Ensure MCP connection is alive; reconnect on first run or after error.
                if poll_tool.is_none() {
                    match connect_mcp(&self.server).await {
                        Ok(tools) => match find_tool(&tools, &self.tool_name) {
                            Ok(t) => {
                                tracing::info!(
                                    sensor = %self.name,
                                    tool = %self.tool_name,
                                    server = %self.server.display_name(),
                                    "MCP connection established"
                                );
                                poll_tool = Some(t);
                            }
                            Err(e) => {
                                tracing::warn!(
                                    sensor = %self.name,
                                    error = %e,
                                    "tool not found on MCP server, retrying next tick"
                                );
                            }
                        },
                        Err(e) => {
                            tracing::warn!(
                                sensor = %self.name,
                                server = %self.server.display_name(),
                                error = %e,
                                "MCP connection failed, retrying next tick"
                            );
                        }
                    }
                }

                // Resolve enrichment tool on a SEPARATE MCP connection.
                // Independent from poll_tool to avoid shared session issues.
                if let Some(ref enrich_name) = self.enrich_tool_name
                    && enrich_tool.is_none()
                {
                    match connect_mcp(&self.server).await {
                        Ok(tools) => match find_tool(&tools, enrich_name) {
                            Ok(t) => {
                                tracing::info!(
                                    sensor = %self.name,
                                    tool = %enrich_name,
                                    "enrichment tool resolved (separate session)"
                                );
                                enrich_tool = Some(t);
                                enrich_failures = 0;
                            }
                            Err(e) => {
                                tracing::debug!(
                                    sensor = %self.name,
                                    error = %e,
                                    "enrichment tool not found, retrying next tick"
                                );
                            }
                        },
                        Err(e) => {
                            tracing::debug!(
                                sensor = %self.name,
                                error = %e,
                                "enrichment MCP connection failed, retrying next tick"
                            );
                        }
                    }
                }

                // Poll the tool if connected.
                if let Some(ref t) = poll_tool {
                    match t.execute(self.tool_args.clone()).await {
                        Ok(output) => {
                            if output.is_error {
                                tracing::warn!(
                                    sensor = %self.name,
                                    tool = %self.tool_name,
                                    error = %output.content,
                                    "MCP tool returned error, reconnecting"
                                );
                                poll_tool = None;
                                enrich_tool = None;
                            } else {
                                match parse_tool_result(
                                    &output.content,
                                    self.items_field.as_deref(),
                                ) {
                                    Ok(items) => {
                                        let mut produced = 0usize;
                                        for item in &items {
                                            let mut event = item_to_sensor_event(
                                                &self.name,
                                                item,
                                                &self.id_field,
                                                self.content_field.as_deref(),
                                                self.modality,
                                            );

                                            if seen.contains_key(&event.source_id) {
                                                continue;
                                            }
                                            seen.insert(
                                                event.source_id.clone(),
                                                Utc::now().timestamp(),
                                            );

                                            // Enrich new items before producing to Kafka.
                                            if let Some(ref et) = enrich_tool {
                                                let id_param =
                                                    self.enrich_id_param.as_deref().unwrap_or("id");
                                                let args = serde_json::json!({
                                                    id_param: event.source_id
                                                });
                                                match et.execute(args).await {
                                                    Ok(enrich_out) if !enrich_out.is_error => {
                                                        enrich_failures = 0;
                                                        if let Ok(enrichment) =
                                                            serde_json::from_str::<Value>(
                                                                &enrich_out.content,
                                                            )
                                                        {
                                                            merge_enrichment(
                                                                &mut event,
                                                                &enrichment,
                                                            );
                                                        }
                                                    }
                                                    Ok(enrich_out) => {
                                                        enrich_failures += 1;
                                                        tracing::debug!(
                                                            sensor = %self.name,
                                                            source_id = %event.source_id,
                                                            error = %enrich_out.content,
                                                            "enrichment tool returned error"
                                                        );
                                                    }
                                                    Err(e) => {
                                                        enrich_failures += 1;
                                                        tracing::debug!(
                                                            sensor = %self.name,
                                                            source_id = %event.source_id,
                                                            error = %e,
                                                            "enrichment tool call failed"
                                                        );
                                                    }
                                                }
                                                // Reset after 3 consecutive failures to force
                                                // re-resolution on next poll cycle.
                                                if enrich_failures >= 3 {
                                                    tracing::warn!(
                                                        sensor = %self.name,
                                                        failures = enrich_failures,
                                                        "enrichment tool reset after consecutive failures"
                                                    );
                                                    enrich_tool = None;
                                                    enrich_failures = 0;
                                                }
                                            }

                                            let payload = match serde_json::to_vec(&event) {
                                                Ok(p) => p,
                                                Err(e) => {
                                                    tracing::warn!(
                                                        sensor = %self.name,
                                                        error = %e,
                                                        "failed to serialize event"
                                                    );
                                                    continue;
                                                }
                                            };
                                            let key = format!("{}:{}", self.name, event.source_id);

                                            if let Err((e, _)) = producer
                                                .send(
                                                    FutureRecord::to(&self.kafka_topic)
                                                        .key(&key)
                                                        .payload(&payload),
                                                    rdkafka::util::Timeout::After(
                                                        Duration::from_secs(5),
                                                    ),
                                                )
                                                .await
                                            {
                                                tracing::warn!(
                                                    sensor = %self.name,
                                                    error = %e,
                                                    "failed to produce event to Kafka"
                                                );
                                            } else {
                                                produced += 1;
                                            }
                                        }
                                        if produced > 0 {
                                            tracing::info!(
                                                sensor = %self.name,
                                                count = produced,
                                                "produced new sensor events"
                                            );
                                            save_seen_state(
                                                self.state_dir.as_deref(),
                                                &self.name,
                                                &seen,
                                            );
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            sensor = %self.name,
                                            error = %e,
                                            "failed to parse tool result"
                                        );
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                sensor = %self.name,
                                tool = %self.tool_name,
                                error = %e,
                                "MCP tool execution failed, reconnecting"
                            );
                            poll_tool = None;
                            enrich_tool = None;
                        }
                    }
                }

                // Periodic cleanup of expired dedup entries (every 5 min).
                let now = Instant::now();
                if now.duration_since(last_cleanup) >= Duration::from_secs(300) {
                    let now_epoch = Utc::now().timestamp();
                    let before = seen.len();
                    seen.retain(|_, &mut first_seen| now_epoch - first_seen < dedup_ttl_secs);
                    if seen.len() < before {
                        tracing::debug!(
                            sensor = %self.name,
                            evicted = before - seen.len(),
                            remaining = seen.len(),
                            "dedup cache cleanup"
                        );
                        save_seen_state(self.state_dir.as_deref(), &self.name, &seen);
                    }
                    last_cleanup = now;
                }

                tokio::select! {
                    _ = cancel.cancelled() => return Ok(()),
                    _ = tokio::time::sleep(self.poll_interval) => {}
                }
            }
        })
    }
}

// ---------------------------------------------------------------------------
// Pure helper functions (testable without network)
// ---------------------------------------------------------------------------

/// Parse the text output of an MCP tool call into a list of JSON items.
///
/// - If the text is a JSON array, returns its elements.
/// - If the text is a JSON object:
///   - With `items_field` set: extracts the named array field.
///   - Without `items_field`: wraps the object in a single-element vec.
/// - Returns an error for non-JSON or missing fields.
pub(crate) fn parse_tool_result(
    text: &str,
    items_field: Option<&str>,
) -> Result<Vec<Value>, Error> {
    let value: Value = serde_json::from_str(text)
        .map_err(|e| Error::Sensor(format!("MCP tool result is not valid JSON: {e}")))?;

    match value {
        Value::Array(arr) => Ok(arr),
        Value::Object(_) => {
            if let Some(field) = items_field {
                match value.get(field) {
                    Some(Value::Array(arr)) => Ok(arr.clone()),
                    Some(_) => Err(Error::Sensor(format!(
                        "items_field '{field}' is not a JSON array"
                    ))),
                    None => Err(Error::Sensor(format!(
                        "items_field '{field}' not found in tool result"
                    ))),
                }
            } else {
                Ok(vec![value])
            }
        }
        _ => Err(Error::Sensor(
            "MCP tool result is neither a JSON array nor object".into(),
        )),
    }
}

/// Map a single JSON item from an MCP tool result to a `SensorEvent`.
///
/// - `id_field`: JSON key for the item's unique ID (falls back to content hash).
/// - `content_field`: If set, extract that field as the event content; otherwise
///   serialize the entire item as JSON.
/// - `modality`: The sensor modality for the event.
pub(crate) fn item_to_sensor_event(
    sensor_name: &str,
    item: &Value,
    id_field: &str,
    content_field: Option<&str>,
    modality: SensorModality,
) -> SensorEvent {
    let full_json = || serde_json::to_string(item).unwrap_or_default();
    let content = match content_field {
        Some(field) => match item.get(field) {
            Some(v) => v
                .as_str()
                .map(String::from)
                .unwrap_or_else(|| v.to_string()),
            None => full_json(), // Field not present → fall back to full JSON.
        },
        None => full_json(),
    };

    let source_id = item
        .get(id_field)
        .filter(|v| !v.is_null())
        .and_then(|v| v.as_str().map(String::from).or_else(|| Some(v.to_string())))
        .unwrap_or_else(|| SensorEvent::generate_id(&content, sensor_name));

    SensorEvent {
        id: SensorEvent::generate_id(&content, &source_id),
        sensor_name: sensor_name.to_string(),
        modality,
        observed_at: Utc::now(),
        content,
        source_id,
        metadata: Some(item.clone()),
        binary_ref: None,
        related_ids: vec![],
    }
}

/// Merge enrichment data into a sensor event's metadata.
///
/// Copies all top-level keys from `enrichment` into the event's metadata object.
/// Then flattens Gmail-style `payload.headers` into top-level keys (e.g.,
/// `"from"`, `"subject"`, `"date"`) so downstream triage processors can
/// access them without knowing the Gmail JSON structure.
pub(crate) fn merge_enrichment(event: &mut SensorEvent, enrichment: &Value) {
    let meta = event.metadata.get_or_insert_with(|| serde_json::json!({}));

    // Merge top-level fields from enrichment into metadata.
    if let (Some(meta_obj), Some(enrich_obj)) = (meta.as_object_mut(), enrichment.as_object()) {
        for (k, v) in enrich_obj {
            meta_obj.insert(k.clone(), v.clone());
        }
    }

    // Flatten Gmail-style payload.headers into top-level keys.
    flatten_gmail_headers(meta);
}

/// Flatten Gmail `payload.headers` array into top-level metadata keys.
///
/// Gmail's `get_message` returns headers as `payload.headers: [{name, value}, ...]`.
/// This extracts common headers (From, Subject, To, Date, etc.) as lowercase
/// top-level keys so triage processors can use simple `metadata["from"]` lookups.
pub(crate) fn flatten_gmail_headers(metadata: &mut Value) {
    let headers: Vec<(String, String)> = metadata
        .get("payload")
        .and_then(|p| p.get("headers"))
        .and_then(|h| h.as_array())
        .map(|headers| {
            headers
                .iter()
                .filter_map(|header| {
                    let name = header.get("name").and_then(|n| n.as_str())?;
                    let value = header.get("value").and_then(|v| v.as_str())?;
                    Some((name.to_lowercase(), value.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();

    if let Some(obj) = metadata.as_object_mut() {
        for (key, value) in headers {
            // Don't overwrite existing top-level keys (e.g., "id", "threadId").
            obj.entry(&key).or_insert(Value::String(value));
        }
    }
}

/// Connect to an MCP server and return all discovered tools.
///
/// Dispatches to Streamable HTTP or stdio transport based on the server entry type.
pub(crate) async fn connect_mcp(server: &McpServerEntry) -> Result<Vec<Arc<dyn Tool>>, Error> {
    let client = match server {
        McpServerEntry::Stdio { command, args, env } => {
            McpClient::connect_stdio(command, args, env).await?
        }
        _ => match server.auth_header() {
            Some(auth) => McpClient::connect_with_auth(server.url(), auth).await?,
            None => McpClient::connect(server.url()).await?,
        },
    };
    Ok(client.into_tools())
}

/// Find a tool by name from a list of MCP tools.
pub(crate) fn find_tool(tools: &[Arc<dyn Tool>], name: &str) -> Result<Arc<dyn Tool>, Error> {
    tools
        .iter()
        .find(|t| t.definition().name == name)
        .cloned()
        .ok_or_else(|| {
            let available: Vec<String> =
                tools.iter().map(|t| t.definition().name.clone()).collect();
            Error::Sensor(format!(
                "tool '{name}' not found on MCP server; available: {available:?}"
            ))
        })
}

// ---------------------------------------------------------------------------
// Triage processor for MCP sensors with configurable topic/modality
// ---------------------------------------------------------------------------

/// Generic triage processor for MCP sensor events.
///
/// Promotes all events with Normal priority — the user specifically configured
/// this MCP tool poll, so every result is presumed relevant. Downstream story
/// correlation and commands processing can further filter.
pub(crate) struct McpTriageProcessor {
    topic: String,
    modality: SensorModality,
}

impl McpTriageProcessor {
    pub fn new(topic: impl Into<String>, modality: SensorModality) -> Self {
        Self {
            topic: topic.into(),
            modality,
        }
    }
}

impl TriageProcessor for McpTriageProcessor {
    fn modality(&self) -> SensorModality {
        self.modality
    }

    fn source_topic(&self) -> &str {
        &self.topic
    }

    fn process(
        &self,
        event: &SensorEvent,
    ) -> Pin<Box<dyn Future<Output = Result<TriageDecision, Error>> + Send + '_>> {
        // Truncate content to build a reasonable summary (UTF-8 safe).
        let summary = if event.content.len() > 200 {
            let cut = floor_char_boundary(&event.content, 200);
            format!("{}…", &event.content[..cut])
        } else {
            event.content.clone()
        };

        let estimated_tokens = event.content.len() / 4 + 256;

        let decision = TriageDecision::Promote {
            priority: Priority::Normal,
            summary,
            extracted_entities: vec![],
            estimated_tokens,
            action_categories: vec![],
            action_hints: vec![],
            has_attachments: false,
            sender: None,
            subject: None,
            message_ref: None,
        };

        Box::pin(async move { Ok(decision) })
    }
}

// ---------------------------------------------------------------------------
// Persistent dedup state
// ---------------------------------------------------------------------------

/// File path for the sensor's persisted dedup state.
fn seen_state_path(state_dir: &std::path::Path, sensor_name: &str) -> PathBuf {
    state_dir.join(format!("{sensor_name}.seen.json"))
}

/// Load previously seen IDs from disk, filtering out entries older than `ttl_secs`.
///
/// Returns an empty map if the file doesn't exist or can't be parsed.
pub(crate) fn load_seen_state(
    state_dir: Option<&std::path::Path>,
    sensor_name: &str,
    ttl_secs: i64,
) -> HashMap<String, i64> {
    let Some(dir) = state_dir else {
        return HashMap::new();
    };
    let path = seen_state_path(dir, sensor_name);
    let data = match std::fs::read_to_string(&path) {
        Ok(d) => d,
        Err(_) => return HashMap::new(),
    };
    let mut map: HashMap<String, i64> = match serde_json::from_str(&data) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(
                sensor = sensor_name,
                path = %path.display(),
                error = %e,
                "failed to parse dedup state, starting fresh"
            );
            return HashMap::new();
        }
    };
    // Evict expired entries.
    let now = Utc::now().timestamp();
    map.retain(|_, &mut ts| now - ts < ttl_secs);
    tracing::info!(
        sensor = sensor_name,
        loaded = map.len(),
        "loaded persisted dedup state"
    );
    map
}

/// Save the current seen map to disk.
///
/// No-op if `state_dir` is `None`. Errors are logged but not propagated.
pub(crate) fn save_seen_state(
    state_dir: Option<&std::path::Path>,
    sensor_name: &str,
    seen: &HashMap<String, i64>,
) {
    let Some(dir) = state_dir else {
        return;
    };
    if let Err(e) = std::fs::create_dir_all(dir) {
        tracing::warn!(
            sensor = sensor_name,
            path = %dir.display(),
            error = %e,
            "failed to create dedup state directory"
        );
        return;
    }
    let path = seen_state_path(dir, sensor_name);
    match serde_json::to_string(seen) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&path, json) {
                tracing::warn!(
                    sensor = sensor_name,
                    path = %path.display(),
                    error = %e,
                    "failed to write dedup state"
                );
            }
        }
        Err(e) => {
            tracing::warn!(
                sensor = sensor_name,
                error = %e,
                "failed to serialize dedup state"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- McpSensor property tests ---

    #[test]
    fn mcp_sensor_properties() {
        let sensor = McpSensor::new(
            "gmail_inbox",
            McpServerEntry::Simple("http://localhost:3000/mcp".into()),
            "search_emails",
            serde_json::json!({"query": "is:unread"}),
            "hb.sensor.email",
            SensorModality::Text,
            Duration::from_secs(60),
            "messageId",
            Some("snippet".into()),
            None,
            None,
            None,
            Duration::from_secs(604800),
            None,
        );

        assert_eq!(sensor.name(), "gmail_inbox");
        assert_eq!(sensor.modality(), SensorModality::Text);
        assert_eq!(sensor.kafka_topic(), "hb.sensor.email");
    }

    #[test]
    fn mcp_sensor_structured_modality() {
        let sensor = McpSensor::new(
            "calendar_events",
            McpServerEntry::Simple("http://localhost:3000/mcp".into()),
            "list_calendar_events",
            serde_json::json!({}),
            "hb.sensor.calendar",
            SensorModality::Structured,
            Duration::from_secs(300),
            "eventId",
            Some("summary".into()),
            None,
            None,
            None,
            Duration::from_secs(604800),
            None,
        );

        assert_eq!(sensor.name(), "calendar_events");
        assert_eq!(sensor.modality(), SensorModality::Structured);
        assert_eq!(sensor.kafka_topic(), "hb.sensor.calendar");
    }

    // --- parse_tool_result tests ---

    #[test]
    fn parse_tool_result_json_array() {
        let text = r#"[{"id": "1", "subject": "Hello"}, {"id": "2", "subject": "World"}]"#;
        let items = parse_tool_result(text, None).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["id"], "1");
        assert_eq!(items[1]["id"], "2");
    }

    #[test]
    fn parse_tool_result_single_object() {
        let text = r#"{"id": "42", "title": "Meeting"}"#;
        let items = parse_tool_result(text, None).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["id"], "42");
    }

    #[test]
    fn parse_tool_result_with_items_field() {
        let text = r#"{"messages": [{"id": "a"}, {"id": "b"}], "total": 2}"#;
        let items = parse_tool_result(text, Some("messages")).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0]["id"], "a");
        assert_eq!(items[1]["id"], "b");
    }

    #[test]
    fn parse_tool_result_empty_array() {
        let text = "[]";
        let items = parse_tool_result(text, None).unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn parse_tool_result_items_field_missing() {
        let text = r#"{"data": [1, 2, 3]}"#;
        let err = parse_tool_result(text, Some("messages")).unwrap_err();
        assert!(err.to_string().contains("not found"), "error: {err}");
    }

    #[test]
    fn parse_tool_result_items_field_not_array() {
        let text = r#"{"messages": "not an array"}"#;
        let err = parse_tool_result(text, Some("messages")).unwrap_err();
        assert!(err.to_string().contains("not a JSON array"), "error: {err}");
    }

    #[test]
    fn parse_tool_result_not_json() {
        let text = "this is plain text, not JSON";
        let err = parse_tool_result(text, None).unwrap_err();
        assert!(err.to_string().contains("not valid JSON"), "error: {err}");
    }

    #[test]
    fn parse_tool_result_primitive_rejected() {
        let text = "42";
        let err = parse_tool_result(text, None).unwrap_err();
        assert!(
            err.to_string().contains("neither a JSON array nor object"),
            "error: {err}"
        );
    }

    // --- item_to_sensor_event tests ---

    #[test]
    fn item_to_sensor_event_full_json() {
        let item = serde_json::json!({"id": "msg-1", "from": "alice", "body": "Hi!"});
        let event = item_to_sensor_event("gmail", &item, "id", None, SensorModality::Text);

        // Content should be the full JSON serialization.
        assert!(event.content.contains("msg-1"));
        assert!(event.content.contains("alice"));
        assert_eq!(event.source_id, "msg-1");
        assert_eq!(event.sensor_name, "gmail");
        assert_eq!(event.modality, SensorModality::Text);
    }

    #[test]
    fn item_to_sensor_event_content_field() {
        let item = serde_json::json!({"id": "msg-2", "snippet": "Meeting at 3pm"});
        let event =
            item_to_sensor_event("gmail", &item, "id", Some("snippet"), SensorModality::Text);

        assert_eq!(event.content, "Meeting at 3pm");
        assert_eq!(event.source_id, "msg-2");
    }

    #[test]
    fn item_to_sensor_event_id_field() {
        let item = serde_json::json!({"messageId": "abc-123", "text": "hello"});
        let event = item_to_sensor_event("sensor", &item, "messageId", None, SensorModality::Text);

        assert_eq!(event.source_id, "abc-123");
    }

    #[test]
    fn item_to_sensor_event_missing_id_fallback() {
        let item = serde_json::json!({"text": "no id here"});
        let event = item_to_sensor_event("sensor", &item, "id", None, SensorModality::Text);

        // Should fall back to a hash-based ID (not empty).
        assert!(!event.source_id.is_empty());
        // The fallback uses SensorEvent::generate_id(content, sensor_name).
        let expected_source_id = SensorEvent::generate_id(&event.content, "sensor");
        assert_eq!(event.source_id, expected_source_id);
    }

    #[test]
    fn item_to_sensor_event_metadata() {
        let item = serde_json::json!({"id": "1", "extra": "data", "nested": {"a": 1}});
        let event = item_to_sensor_event("sensor", &item, "id", None, SensorModality::Structured);

        // Full item should be stored as metadata.
        let meta = event.metadata.unwrap();
        assert_eq!(meta["extra"], "data");
        assert_eq!(meta["nested"]["a"], 1);
    }

    #[test]
    fn item_to_sensor_event_deterministic_id() {
        let item = serde_json::json!({"id": "fixed", "content": "same"});
        let e1 = item_to_sensor_event("s", &item, "id", None, SensorModality::Text);
        let e2 = item_to_sensor_event("s", &item, "id", None, SensorModality::Text);

        assert_eq!(e1.id, e2.id, "same input should produce same event ID");
    }

    #[test]
    fn item_to_sensor_event_numeric_id() {
        let item = serde_json::json!({"id": 42, "text": "numeric"});
        let event = item_to_sensor_event("sensor", &item, "id", None, SensorModality::Text);

        // Numeric IDs should be stringified.
        assert_eq!(event.source_id, "42");
    }

    #[test]
    fn item_to_sensor_event_null_id_fallback() {
        let item = serde_json::json!({"id": null, "text": "null id"});
        let event = item_to_sensor_event("sensor", &item, "id", None, SensorModality::Text);

        // Null ID should fall back to hash, not "null" string.
        assert_ne!(event.source_id, "null");
        let expected = SensorEvent::generate_id(&event.content, "sensor");
        assert_eq!(event.source_id, expected);
    }

    #[test]
    fn item_to_sensor_event_content_field_missing_falls_back() {
        let item = serde_json::json!({"id": "1", "other": "data"});
        let event = item_to_sensor_event(
            "sensor",
            &item,
            "id",
            Some("nonexistent"),
            SensorModality::Text,
        );

        // Missing content_field should fall back to full JSON, not empty string.
        assert!(!event.content.is_empty());
        assert!(
            event.content.contains("other"),
            "should contain full JSON, got: {}",
            event.content
        );
    }

    #[test]
    fn item_to_sensor_event_content_field_non_string() {
        let item = serde_json::json!({"id": "1", "data": {"nested": true}});
        let event = item_to_sensor_event(
            "sensor",
            &item,
            "id",
            Some("data"),
            SensorModality::Structured,
        );

        // Non-string content fields should be JSON-serialized.
        assert!(event.content.contains("nested"));
    }

    // --- find_tool tests ---

    #[test]
    fn find_tool_not_found() {
        let tools: Vec<Arc<dyn Tool>> = vec![];
        let result = find_tool(&tools, "nonexistent");
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("not found"), "error: {err}");
    }

    // --- McpTriageProcessor tests ---

    #[tokio::test]
    async fn mcp_triage_promotes_all_events() {
        let processor = McpTriageProcessor::new("hb.sensor.calendar", SensorModality::Structured);
        let event = SensorEvent {
            id: "test-1".into(),
            sensor_name: "calendar".into(),
            modality: SensorModality::Structured,
            observed_at: Utc::now(),
            content: r#"{"summary": "Team standup"}"#.into(),
            source_id: "evt-123".into(),
            metadata: None,
            binary_ref: None,
            related_ids: vec![],
        };

        let decision = processor.process(&event).await.unwrap();
        match decision {
            TriageDecision::Promote { priority, .. } => {
                assert_eq!(priority, Priority::Normal);
            }
            _ => panic!("expected Promote, got {decision:?}"),
        }
    }

    #[test]
    fn mcp_triage_processor_properties() {
        let processor = McpTriageProcessor::new("hb.sensor.calendar", SensorModality::Structured);
        assert_eq!(processor.source_topic(), "hb.sensor.calendar");
        assert_eq!(processor.modality(), SensorModality::Structured);
    }

    #[tokio::test]
    async fn mcp_triage_summary_truncated() {
        let processor = McpTriageProcessor::new("hb.sensor.test", SensorModality::Text);
        let long_content = "A".repeat(500);
        let event = SensorEvent {
            id: "test-2".into(),
            sensor_name: "test".into(),
            modality: SensorModality::Text,
            observed_at: Utc::now(),
            content: long_content,
            source_id: "src-1".into(),
            metadata: None,
            binary_ref: None,
            related_ids: vec![],
        };

        let decision = processor.process(&event).await.unwrap();
        if let TriageDecision::Promote { summary, .. } = &decision {
            assert!(
                summary.len() <= 204, // 200 chars + "…" (3 bytes UTF-8)
                "summary should be truncated, got {} bytes",
                summary.len()
            );
        } else {
            panic!("expected Promote");
        }
    }

    #[tokio::test]
    async fn mcp_triage_summary_utf8_safe() {
        let processor = McpTriageProcessor::new("hb.sensor.test", SensorModality::Text);
        // Build a string with multi-byte chars that would panic if sliced naively at byte 200.
        // "é" is 2 bytes. 100 * "éa" = 100 * 3 bytes = 300 bytes, 200 chars.
        let content: String = "éa".repeat(100);
        assert_eq!(content.len(), 300); // 300 bytes
        let event = SensorEvent {
            id: "test-3".into(),
            sensor_name: "test".into(),
            modality: SensorModality::Text,
            observed_at: Utc::now(),
            content,
            source_id: "src-2".into(),
            metadata: None,
            binary_ref: None,
            related_ids: vec![],
        };

        // This must not panic.
        let decision = processor.process(&event).await.unwrap();
        assert!(matches!(decision, TriageDecision::Promote { .. }));
    }

    // --- merge_enrichment tests ---

    fn make_event_for_enrich(source_id: &str, metadata: Option<Value>) -> SensorEvent {
        SensorEvent {
            id: "test-enrich".into(),
            sensor_name: "gmail".into(),
            modality: SensorModality::Text,
            observed_at: Utc::now(),
            content: "email snippet".into(),
            source_id: source_id.into(),
            metadata,
            binary_ref: None,
            related_ids: vec![],
        }
    }

    #[test]
    fn merge_enrichment_adds_top_level_fields() {
        let mut event = make_event_for_enrich(
            "msg-1",
            Some(serde_json::json!({"id": "msg-1", "threadId": "msg-1"})),
        );
        let enrichment = serde_json::json!({
            "snippet": "Hello from Alice",
            "labelIds": ["INBOX", "UNREAD"],
        });
        merge_enrichment(&mut event, &enrichment);
        let meta = event.metadata.unwrap();
        assert_eq!(meta["snippet"], "Hello from Alice");
        assert!(meta["labelIds"].is_array());
        // Original fields preserved.
        assert_eq!(meta["id"], "msg-1");
    }

    #[test]
    fn merge_enrichment_creates_metadata_if_none() {
        let mut event = make_event_for_enrich("msg-2", None);
        let enrichment = serde_json::json!({"snippet": "test"});
        merge_enrichment(&mut event, &enrichment);
        assert!(event.metadata.is_some());
        assert_eq!(event.metadata.unwrap()["snippet"], "test");
    }

    #[test]
    fn merge_enrichment_flattens_gmail_headers() {
        let mut event = make_event_for_enrich(
            "msg-3",
            Some(serde_json::json!({"id": "msg-3", "threadId": "thread-1"})),
        );
        let enrichment = serde_json::json!({
            "payload": {
                "headers": [
                    {"name": "From", "value": "alice@example.com"},
                    {"name": "Subject", "value": "Meeting Tomorrow"},
                    {"name": "To", "value": "owner@example.com"},
                    {"name": "Date", "value": "Mon, 24 Feb 2026 10:00:00 +0100"},
                ],
                "parts": [
                    {"filename": "report.pdf", "mimeType": "application/pdf"}
                ]
            }
        });
        merge_enrichment(&mut event, &enrichment);
        let meta = event.metadata.unwrap();
        // Headers flattened to lowercase top-level keys.
        assert_eq!(meta["from"], "alice@example.com");
        assert_eq!(meta["subject"], "Meeting Tomorrow");
        assert_eq!(meta["to"], "owner@example.com");
        assert_eq!(meta["date"], "Mon, 24 Feb 2026 10:00:00 +0100");
        // Original fields preserved.
        assert_eq!(meta["id"], "msg-3");
        assert_eq!(meta["threadId"], "thread-1");
        // payload still accessible for attachment detection.
        assert!(meta["payload"]["parts"].is_array());
    }

    #[test]
    fn flatten_gmail_headers_no_overwrite() {
        // If a top-level key already exists, headers should NOT overwrite it.
        let mut meta = serde_json::json!({
            "id": "msg-4",
            "payload": {
                "headers": [
                    {"name": "id", "value": "should-not-overwrite"}
                ]
            }
        });
        flatten_gmail_headers(&mut meta);
        assert_eq!(
            meta["id"], "msg-4",
            "existing keys should not be overwritten"
        );
    }

    #[test]
    fn flatten_gmail_headers_no_payload() {
        let mut meta = serde_json::json!({"id": "msg-5"});
        flatten_gmail_headers(&mut meta);
        // Should be a no-op.
        assert_eq!(meta, serde_json::json!({"id": "msg-5"}));
    }

    #[test]
    fn flatten_gmail_headers_empty_headers_array() {
        let mut meta = serde_json::json!({
            "payload": {"headers": []}
        });
        flatten_gmail_headers(&mut meta);
        // No new keys added.
        assert!(meta.get("from").is_none());
    }

    #[test]
    fn mcp_sensor_with_enrichment_config() {
        let sensor = McpSensor::new(
            "gmail_inbox",
            McpServerEntry::Simple("http://localhost:4000/mcp".into()),
            "gmail_list_messages",
            serde_json::json!({"q": "is:unread"}),
            "hb.sensor.email",
            SensorModality::Text,
            Duration::from_secs(60),
            "id",
            Some("snippet".into()),
            Some("messages".into()),
            Some("gmail_get_message".into()),
            Some("messageId".into()),
            Duration::from_secs(604800),
            None,
        );

        assert_eq!(
            sensor.enrich_tool_name.as_deref(),
            Some("gmail_get_message")
        );
        assert_eq!(sensor.enrich_id_param.as_deref(), Some("messageId"));
    }

    // --- Dedup state persistence tests ---

    #[test]
    fn load_seen_state_none_dir() {
        let seen = load_seen_state(None, "test", 3600);
        assert!(seen.is_empty());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut seen = HashMap::new();
        let now = Utc::now().timestamp();
        seen.insert("msg-1".into(), now);
        seen.insert("msg-2".into(), now - 100);

        save_seen_state(Some(dir.path()), "gmail", &seen);
        let loaded = load_seen_state(Some(dir.path()), "gmail", 3600);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["msg-1"], now);
        assert_eq!(loaded["msg-2"], now - 100);
    }

    #[test]
    fn load_evicts_expired_entries() {
        let dir = tempfile::tempdir().unwrap();
        let mut seen = HashMap::new();
        let now = Utc::now().timestamp();
        seen.insert("fresh".into(), now - 10);
        seen.insert("expired".into(), now - 7200); // 2h old

        save_seen_state(Some(dir.path()), "sensor", &seen);
        // TTL of 1h → "expired" should be evicted.
        let loaded = load_seen_state(Some(dir.path()), "sensor", 3600);
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains_key("fresh"));
        assert!(!loaded.contains_key("expired"));
    }

    #[test]
    fn load_missing_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let loaded = load_seen_state(Some(dir.path()), "nonexistent", 3600);
        assert!(loaded.is_empty());
    }

    #[test]
    fn load_corrupt_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.seen.json");
        std::fs::write(&path, "not valid json{{{").unwrap();
        let loaded = load_seen_state(Some(dir.path()), "corrupt", 3600);
        assert!(loaded.is_empty());
    }

    #[test]
    fn save_creates_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("sub").join("dir");
        let mut seen = HashMap::new();
        seen.insert("id-1".into(), Utc::now().timestamp());
        save_seen_state(Some(&nested), "test", &seen);
        assert!(nested.join("test.seen.json").exists());
    }

    #[test]
    fn save_none_dir_is_noop() {
        // Should not panic or create anything.
        let seen = HashMap::new();
        save_seen_state(None, "test", &seen);
    }
}
