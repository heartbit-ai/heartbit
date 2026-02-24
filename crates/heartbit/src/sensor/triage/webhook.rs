use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::Error;
use crate::llm::DynLlmProvider;
use crate::llm::types::{CompletionRequest, ContentBlock, Message, Role};
use crate::sensor::triage::{Priority, TriageDecision, TriageProcessor};
use crate::sensor::{SensorEvent, SensorModality};

/// Webhook triage processor. Uses an SLM to classify incoming webhook
/// payloads and assign priority for downstream agent processing.
///
/// Processing pipeline:
/// 1. Parse event content as JSON (if possible).
/// 2. Detect webhook source from metadata (`source` field).
/// 3. Truncate content at 2000 bytes before sending to SLM.
/// 4. Use SLM to classify: `{ "category", "summary", "entities", "action_required" }`.
/// 5. Priority assignment:
///    - `action_required: true` -> High
///    - PR review request -> High
///    - Issue opened/updated -> Normal
///    - CI/CD events -> Low
///    - Other -> Low
/// 6. On SLM parse failure: fallback to Normal priority with content-based summary.
pub struct WebhookTriageProcessor {
    slm_provider: Arc<dyn DynLlmProvider>,
}

/// Maximum bytes of content to send to the SLM for classification.
const MAX_CONTENT_BYTES: usize = 2000;

/// System prompt for webhook triage classification.
const WEBHOOK_TRIAGE_SYSTEM_PROMPT: &str = "You are a webhook event triage classifier. \
    Analyze the webhook payload and classify it by category, urgency, and extract relevant entities. \
    Output valid JSON only, no markdown.";

impl WebhookTriageProcessor {
    /// Create a new webhook triage processor.
    pub fn new(slm_provider: Arc<dyn DynLlmProvider>) -> Self {
        Self { slm_provider }
    }
}

/// Extract the webhook source name from event metadata.
fn extract_source(event: &SensorEvent) -> &str {
    event
        .metadata
        .as_ref()
        .and_then(|m| m.get("source"))
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
}

/// Map SLM classification to priority level.
///
/// Rules:
/// - `action_required: true` -> High
/// - Category "pr_review" -> High
/// - Category "issue" -> Normal
/// - Category "ci" or "deployment" -> Low
/// - Everything else -> Low
fn classify_priority(category: &str, action_required: bool) -> Priority {
    if action_required {
        return Priority::High;
    }
    match category {
        "pr_review" | "pull_request_review" | "review_requested" => Priority::High,
        "issue" | "issue_comment" | "discussion" => Priority::Normal,
        "ci" | "deployment" | "build" | "workflow" | "status" => Priority::Low,
        _ => Priority::Low,
    }
}

/// Build the classification prompt for the SLM.
fn build_classification_prompt(source: &str, content: &str) -> String {
    format!(
        "Classify this webhook event from source \"{source}\".\n\
         Payload: {content}\n\n\
         Respond with JSON only:\n\
         {{\"category\": \"pr_review|issue|ci|deployment|notification|other\", \
         \"summary\": \"one sentence describing the event\", \
         \"entities\": [\"entity1\", \"entity2\"], \
         \"action_required\": true/false}}"
    )
}

/// Build a fallback summary when the SLM response cannot be parsed.
fn build_fallback_summary(source: &str, content: &str) -> String {
    let preview: String = content.chars().take(100).collect();
    format!("Webhook event from {source}: {preview}")
}

impl TriageProcessor for WebhookTriageProcessor {
    fn modality(&self) -> SensorModality {
        SensorModality::Structured
    }

    fn source_topic(&self) -> &str {
        "hb.sensor.webhook"
    }

    fn process(
        &self,
        event: &SensorEvent,
    ) -> Pin<Box<dyn Future<Output = Result<TriageDecision, Error>> + Send + '_>> {
        let content = event.content.clone();
        let source_id = event.source_id.clone();
        let source = extract_source(event).to_string();

        Box::pin(async move {
            // Truncate content before sending to SLM
            let truncated = if content.len() > MAX_CONTENT_BYTES {
                let boundary =
                    crate::tool::builtins::floor_char_boundary(&content, MAX_CONTENT_BYTES);
                &content[..boundary]
            } else {
                &content
            };

            let prompt = build_classification_prompt(&source, truncated);

            let request = CompletionRequest {
                system: WEBHOOK_TRIAGE_SYSTEM_PROMPT.into(),
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text { text: prompt }],
                }],
                max_tokens: 200,
                tools: vec![],
                tool_choice: None,
                reasoning_effort: None,
            };

            let response = self.slm_provider.complete(request).await.map_err(|e| {
                Error::Sensor(format!("SLM webhook triage failed for {source_id}: {e}"))
            })?;

            // Extract text response
            let text = response
                .content
                .iter()
                .find_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .unwrap_or("");

            // Parse SLM response with fallback
            let parsed: serde_json::Value = match serde_json::from_str(text) {
                Ok(v) => v,
                Err(_) => {
                    // Fallback: Normal priority with content-based summary
                    let summary = build_fallback_summary(&source, &content);
                    let estimated_tokens = content.len() / 4 + 256;

                    return Ok(TriageDecision::Promote {
                        priority: Priority::Normal,
                        summary,
                        extracted_entities: vec![],
                        estimated_tokens,
                    });
                }
            };

            let category = parsed["category"].as_str().unwrap_or("other");
            let action_required = parsed["action_required"].as_bool().unwrap_or(false);
            let priority = classify_priority(category, action_required);

            let summary = parsed["summary"]
                .as_str()
                .map(String::from)
                .unwrap_or_else(|| build_fallback_summary(&source, &content));

            let entities: Vec<String> = parsed["entities"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let estimated_tokens = content.len() / 4 + 256;

            Ok(TriageDecision::Promote {
                priority,
                summary,
                extracted_entities: entities,
                estimated_tokens,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::{CompletionResponse, StopReason, TokenUsage};

    // --- Mock provider ---

    struct MockProvider {
        response_json: String,
    }

    impl MockProvider {
        fn with_json(json: &str) -> Self {
            Self {
                response_json: json.into(),
            }
        }
    }

    impl DynLlmProvider for MockProvider {
        fn complete<'a>(
            &'a self,
            _req: CompletionRequest,
        ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>> {
            Box::pin(async {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: self.response_json.clone(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                })
            })
        }

        fn stream_complete<'a>(
            &'a self,
            _req: CompletionRequest,
            _on_text: &'a crate::llm::OnText,
        ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>> {
            Box::pin(async { Err(Error::Sensor("streaming not supported in mock".into())) })
        }

        fn model_name(&self) -> Option<&str> {
            Some("mock-slm")
        }
    }

    struct FailingProvider;

    impl DynLlmProvider for FailingProvider {
        fn complete<'a>(
            &'a self,
            _req: CompletionRequest,
        ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>> {
            Box::pin(async { Err(Error::Sensor("provider unavailable".into())) })
        }

        fn stream_complete<'a>(
            &'a self,
            _req: CompletionRequest,
            _on_text: &'a crate::llm::OnText,
        ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>> {
            Box::pin(async { Err(Error::Sensor("streaming not supported".into())) })
        }

        fn model_name(&self) -> Option<&str> {
            None
        }
    }

    /// Captures the prompt sent to the SLM for verification.
    struct CapturingProvider {
        response_json: String,
        captured: std::sync::Mutex<Option<String>>,
    }

    impl CapturingProvider {
        fn with_json(json: &str) -> Self {
            Self {
                response_json: json.into(),
                captured: std::sync::Mutex::new(None),
            }
        }

        fn captured_prompt(&self) -> Option<String> {
            self.captured
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
        }
    }

    impl DynLlmProvider for CapturingProvider {
        fn complete<'a>(
            &'a self,
            req: CompletionRequest,
        ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>> {
            // Capture the user message text
            let user_text = req
                .messages
                .first()
                .and_then(|m| {
                    m.content.iter().find_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.clone()),
                        _ => None,
                    })
                })
                .unwrap_or_default();
            if let Ok(mut guard) = self.captured.lock() {
                *guard = Some(user_text);
            }

            Box::pin(async {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text {
                        text: self.response_json.clone(),
                    }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                    model: None,
                })
            })
        }

        fn stream_complete<'a>(
            &'a self,
            _req: CompletionRequest,
            _on_text: &'a crate::llm::OnText,
        ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>> {
            Box::pin(async { Err(Error::Sensor("not supported".into())) })
        }

        fn model_name(&self) -> Option<&str> {
            Some("capturing-mock")
        }
    }

    // --- Helpers ---

    fn make_event(content: &str, source: &str) -> SensorEvent {
        SensorEvent {
            id: SensorEvent::generate_id(content, &format!("{source}:test")),
            sensor_name: source.into(),
            modality: SensorModality::Structured,
            observed_at: chrono::Utc::now(),
            content: content.into(),
            source_id: format!("{source}:test"),
            metadata: Some(serde_json::json!({
                "source": source,
                "content_type": "application/json",
            })),
            binary_ref: None,
            related_ids: vec![],
        }
    }

    fn pr_review_slm_json() -> String {
        serde_json::json!({
            "category": "pr_review",
            "summary": "PR review requested on feature branch",
            "entities": ["user/repo", "feature-branch"],
            "action_required": true
        })
        .to_string()
    }

    fn issue_slm_json() -> String {
        serde_json::json!({
            "category": "issue",
            "summary": "New issue opened: Bug in login flow",
            "entities": ["login", "authentication"],
            "action_required": false
        })
        .to_string()
    }

    fn ci_slm_json() -> String {
        serde_json::json!({
            "category": "ci",
            "summary": "CI pipeline passed for main branch",
            "entities": ["main", "pipeline"],
            "action_required": false
        })
        .to_string()
    }

    fn action_required_slm_json() -> String {
        serde_json::json!({
            "category": "notification",
            "summary": "Deployment approval needed for production",
            "entities": ["production", "deployment"],
            "action_required": true
        })
        .to_string()
    }

    fn other_slm_json() -> String {
        serde_json::json!({
            "category": "other",
            "summary": "Generic webhook event",
            "entities": [],
            "action_required": false
        })
        .to_string()
    }

    // --- Trait property tests ---

    #[test]
    fn webhook_triage_modality_is_structured() {
        let processor = WebhookTriageProcessor::new(Arc::new(MockProvider::with_json("{}")));
        assert_eq!(processor.modality(), SensorModality::Structured);
    }

    #[test]
    fn webhook_triage_source_topic() {
        let processor = WebhookTriageProcessor::new(Arc::new(MockProvider::with_json("{}")));
        assert_eq!(processor.source_topic(), "hb.sensor.webhook");
    }

    // --- classify_priority unit tests ---

    #[test]
    fn classify_priority_action_required_always_high() {
        assert_eq!(classify_priority("other", true), Priority::High);
        assert_eq!(classify_priority("ci", true), Priority::High);
        assert_eq!(classify_priority("issue", true), Priority::High);
    }

    #[test]
    fn classify_priority_pr_review_high() {
        assert_eq!(classify_priority("pr_review", false), Priority::High);
        assert_eq!(
            classify_priority("pull_request_review", false),
            Priority::High
        );
        assert_eq!(classify_priority("review_requested", false), Priority::High);
    }

    #[test]
    fn classify_priority_issue_normal() {
        assert_eq!(classify_priority("issue", false), Priority::Normal);
        assert_eq!(classify_priority("issue_comment", false), Priority::Normal);
        assert_eq!(classify_priority("discussion", false), Priority::Normal);
    }

    #[test]
    fn classify_priority_ci_low() {
        assert_eq!(classify_priority("ci", false), Priority::Low);
        assert_eq!(classify_priority("deployment", false), Priority::Low);
        assert_eq!(classify_priority("build", false), Priority::Low);
        assert_eq!(classify_priority("workflow", false), Priority::Low);
        assert_eq!(classify_priority("status", false), Priority::Low);
    }

    #[test]
    fn classify_priority_unknown_low() {
        assert_eq!(classify_priority("random_thing", false), Priority::Low);
    }

    // --- Helper function tests ---

    #[test]
    fn extract_source_from_metadata() {
        let event = make_event(r#"{"action":"opened"}"#, "github");
        assert_eq!(extract_source(&event), "github");
    }

    #[test]
    fn extract_source_no_metadata() {
        let mut event = make_event(r#"{"test":true}"#, "test");
        event.metadata = None;
        assert_eq!(extract_source(&event), "unknown");
    }

    #[test]
    fn build_classification_prompt_contains_source() {
        let prompt = build_classification_prompt("github", r#"{"action":"opened"}"#);
        assert!(
            prompt.contains("github"),
            "prompt should contain source: {prompt}"
        );
    }

    #[test]
    fn build_fallback_summary_contains_source() {
        let summary = build_fallback_summary("slack", "hello world");
        assert!(
            summary.contains("slack"),
            "summary should contain source: {summary}"
        );
        assert!(
            summary.contains("hello world"),
            "summary should contain content: {summary}"
        );
    }

    // --- Async process() tests ---

    #[tokio::test]
    async fn classify_pr_review_high_priority() {
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&pr_review_slm_json())));

        let event = make_event(
            r#"{"action":"review_requested","pull_request":{"number":42}}"#,
            "github",
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::High);
        }
    }

    #[tokio::test]
    async fn classify_issue_normal_priority() {
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&issue_slm_json())));

        let event = make_event(
            r#"{"action":"opened","issue":{"title":"Bug in login"}}"#,
            "github",
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Normal);
        }
    }

    #[tokio::test]
    async fn classify_ci_event_low_priority() {
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&ci_slm_json())));

        let event = make_event(
            r#"{"action":"completed","workflow_run":{"conclusion":"success"}}"#,
            "github",
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Low);
        }
    }

    #[tokio::test]
    async fn classify_action_required_high_priority() {
        let processor = WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(
            &action_required_slm_json(),
        )));

        let event = make_event(
            r#"{"type":"approval_needed","environment":"production"}"#,
            "custom_deploy",
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::High);
        }
    }

    #[tokio::test]
    async fn slm_parse_failure_fallback_normal() {
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json("not valid json {{")));

        let event = make_event(r#"{"action":"test"}"#, "github");

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote {
            priority, summary, ..
        } = &decision
        {
            assert_eq!(*priority, Priority::Normal);
            assert!(
                summary.contains("github"),
                "fallback summary should contain source: {summary}"
            );
        }
    }

    #[tokio::test]
    async fn slm_error_propagated() {
        let processor = WebhookTriageProcessor::new(Arc::new(FailingProvider));

        let event = make_event(r#"{"test":true}"#, "test");

        let result = processor.process(&event).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("SLM webhook triage failed"), "error: {err}");
    }

    #[tokio::test]
    async fn empty_content_handled() {
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&other_slm_json())));

        let event = make_event("", "empty_source");

        let decision = processor
            .process(&event)
            .await
            .expect("should not fail on empty content");
        assert!(decision.is_promote());
    }

    #[tokio::test]
    async fn entities_extracted() {
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&issue_slm_json())));

        let event = make_event(
            r#"{"action":"opened","issue":{"title":"Login bug"}}"#,
            "github",
        );

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            extracted_entities, ..
        } = &decision
        {
            assert_eq!(extracted_entities, &["login", "authentication"]);
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }

    #[tokio::test]
    async fn estimated_tokens_reasonable() {
        let content = "A".repeat(800);
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&other_slm_json())));

        let event = make_event(&content, "test");

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            estimated_tokens, ..
        } = &decision
        {
            assert_eq!(
                *estimated_tokens, 456,
                "800 bytes / 4 + 256 overhead = 456 tokens"
            );
        } else {
            panic!("expected Promote");
        }
    }

    #[tokio::test]
    async fn summary_contains_source_info() {
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&ci_slm_json())));

        let event = make_event(r#"{"status":"success"}"#, "github");

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote { summary, .. } = &decision {
            assert!(!summary.is_empty(), "summary should not be empty");
            // SLM-generated summary should contain meaningful text
            assert!(
                summary.contains("CI") || summary.contains("pipeline") || summary.contains("main"),
                "summary should describe the event: {summary}"
            );
        } else {
            panic!("expected Promote");
        }
    }

    #[tokio::test]
    async fn content_truncated_before_slm() {
        // Create a very long payload (> 2000 bytes)
        let long_content = "x".repeat(5000);
        let default_json = other_slm_json();
        let provider = Arc::new(CapturingProvider::with_json(&default_json));
        let processor = WebhookTriageProcessor::new(provider.clone());

        let event = make_event(&long_content, "test");

        let _decision = processor.process(&event).await.expect("process failed");

        // Verify the prompt sent to SLM was truncated
        let captured = provider
            .captured_prompt()
            .expect("should have captured prompt");
        // The prompt contains the truncated content (max 2000 bytes) plus
        // the prompt template text
        assert!(
            captured.len() < 5000 + 200,
            "prompt should be truncated, got {} chars",
            captured.len()
        );
        // The prompt should NOT contain the full 5000-char content
        assert!(
            !captured.contains(&"x".repeat(5000)),
            "prompt should not contain full content"
        );
    }

    #[tokio::test]
    async fn missing_metadata_handled() {
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&other_slm_json())));

        let mut event = make_event(r#"{"test":true}"#, "test");
        event.metadata = None;

        let decision = processor
            .process(&event)
            .await
            .expect("should handle missing metadata");
        assert!(decision.is_promote());
    }

    #[tokio::test]
    async fn other_category_gets_low_priority() {
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&other_slm_json())));

        let event = make_event(r#"{"type":"ping"}"#, "unknown_service");

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Low);
        }
    }

    // --- Serde roundtrip tests ---

    #[tokio::test]
    async fn triage_decision_promote_serde_roundtrip() {
        let processor =
            WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&pr_review_slm_json())));

        let event = make_event(r#"{"action":"review_requested"}"#, "github");

        let decision = processor.process(&event).await.expect("process");
        let json = serde_json::to_string(&decision).expect("serialize");
        let back: TriageDecision = serde_json::from_str(&json).expect("deserialize");
        assert!(back.is_promote());
        if let TriageDecision::Promote {
            priority,
            summary,
            extracted_entities,
            estimated_tokens,
        } = back
        {
            assert_eq!(priority, Priority::High);
            assert!(!summary.is_empty());
            assert!(!extracted_entities.is_empty());
            assert!(estimated_tokens > 0);
        }
    }

    #[tokio::test]
    async fn fallback_summary_on_missing_summary_field() {
        // SLM returns valid JSON but without a "summary" field
        let json = serde_json::json!({
            "category": "other",
            "entities": [],
            "action_required": false
        })
        .to_string();
        let processor = WebhookTriageProcessor::new(Arc::new(MockProvider::with_json(&json)));

        let event = make_event(r#"{"data":"test"}"#, "custom");

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote { summary, .. } = &decision {
            assert!(
                summary.contains("custom"),
                "fallback summary should contain source: {summary}"
            );
        } else {
            panic!("expected Promote");
        }
    }
}
