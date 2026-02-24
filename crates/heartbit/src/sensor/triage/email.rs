use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::Error;
use crate::llm::DynLlmProvider;
use crate::llm::types::{CompletionRequest, ContentBlock, Message, Role};
use crate::sensor::triage::{Priority, TriageDecision, TriageProcessor};
use crate::sensor::{SensorEvent, SensorModality};

/// Email triage processor. The most sophisticated triage — email is the
/// highest-priority direct-to-user input.
///
/// Processing pipeline:
/// 1. **Sender check**: known contacts get at least `High` priority; blocked
///    senders are immediately dropped.
/// 2. **Thread detection**: `in_reply_to` / `references` metadata fields
///    indicate a thread reply, boosting priority by one level.
/// 3. **SLM classification**: subject + body snippet sent to a small language
///    model for relevance, urgency, summary, and entity extraction.
/// 4. **Priority assignment**: Critical if known contact + urgent; High if
///    known contact OR urgent; Normal otherwise. Thread replies get +1.
pub struct EmailTriageProcessor {
    slm_provider: Arc<dyn DynLlmProvider>,
    known_contacts: HashSet<String>,
    blocked_senders: HashSet<String>,
}

impl EmailTriageProcessor {
    pub fn new(
        slm_provider: Arc<dyn DynLlmProvider>,
        known_contacts: Vec<String>,
        blocked_senders: Vec<String>,
    ) -> Self {
        Self {
            slm_provider,
            known_contacts: known_contacts.into_iter().collect(),
            blocked_senders: blocked_senders.into_iter().collect(),
        }
    }
}

/// Extract a metadata string field from an optional JSON value.
fn meta_str<'a>(metadata: Option<&'a serde_json::Value>, key: &str) -> Option<&'a str> {
    metadata.and_then(|m| m.get(key)).and_then(|v| v.as_str())
}

/// Returns `true` if the metadata contains `in_reply_to` or `references`,
/// indicating this email is part of an existing thread.
fn is_thread_reply(metadata: Option<&serde_json::Value>) -> bool {
    meta_str(metadata, "in_reply_to").is_some() || meta_str(metadata, "references").is_some()
}

/// Boost a priority by one level. `Critical` stays `Critical`.
fn boost_priority(p: Priority) -> Priority {
    match p {
        Priority::Low => Priority::Normal,
        Priority::Normal => Priority::High,
        Priority::High => Priority::Critical,
        Priority::Critical => Priority::Critical,
    }
}

/// Determine base priority from known-contact status and SLM urgency.
fn base_priority(is_known_contact: bool, urgency: &str) -> Priority {
    let is_urgent = urgency == "high" || urgency == "critical";
    match (is_known_contact, is_urgent) {
        (true, true) => Priority::Critical,
        (true, false) | (false, true) => Priority::High,
        (false, false) => Priority::Normal,
    }
}

impl TriageProcessor for EmailTriageProcessor {
    fn modality(&self) -> SensorModality {
        SensorModality::Text
    }

    fn source_topic(&self) -> &str {
        "hb.sensor.email"
    }

    fn process(
        &self,
        event: &SensorEvent,
    ) -> Pin<Box<dyn Future<Output = Result<TriageDecision, Error>> + Send + '_>> {
        let content = event.content.clone();
        let source_id = event.source_id.clone();
        let metadata = event.metadata.clone();

        Box::pin(async move {
            // Step 1: Sender check
            let sender = meta_str(metadata.as_ref(), "from").unwrap_or("");
            let sender_lower = sender.to_lowercase();

            if self
                .blocked_senders
                .iter()
                .any(|b| b.to_lowercase() == sender_lower)
                && !sender_lower.is_empty()
            {
                return Ok(TriageDecision::Drop {
                    reason: format!("blocked sender: {sender}"),
                });
            }

            let is_known = !sender_lower.is_empty()
                && self
                    .known_contacts
                    .iter()
                    .any(|c| c.to_lowercase() == sender_lower);

            // Step 2: Thread detection
            let is_thread = is_thread_reply(metadata.as_ref());

            // Step 3: SLM classification
            let subject = meta_str(metadata.as_ref(), "subject").unwrap_or("");
            let truncated = if content.len() > 2000 {
                let boundary = crate::tool::builtins::floor_char_boundary(&content, 2000);
                &content[..boundary]
            } else {
                &content
            };

            let prompt = format!(
                "Classify this email. Respond with JSON only.\n\
                 Subject: {subject}\n\
                 Body: {truncated}\n\n\
                 Respond with: {{\"relevant\": true/false, \"urgency\": \"high\"|\"normal\"|\"low\", \
                 \"summary\": \"one sentence\", \"entities\": [\"entity1\", \"entity2\"], \
                 \"category\": \"work|personal|marketing|notification|other\"}}"
            );

            let request = CompletionRequest {
                system: "You are an email triage classifier. Output valid JSON only, no markdown."
                    .into(),
                messages: vec![Message {
                    role: Role::User,
                    content: vec![ContentBlock::Text { text: prompt }],
                }],
                max_tokens: 200,
                tools: vec![],
                tool_choice: None,
                reasoning_effort: None,
            };

            let response =
                self.slm_provider.complete(request).await.map_err(|e| {
                    Error::Sensor(format!("SLM triage failed for {source_id}: {e}"))
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
            let parsed: serde_json::Value =
                serde_json::from_str(text).unwrap_or(serde_json::json!({
                    "relevant": true,
                    "urgency": "normal",
                    "summary": content.chars().take(100).collect::<String>(),
                    "entities": [],
                }));

            let is_relevant = parsed["relevant"].as_bool().unwrap_or(true);
            if !is_relevant && !is_known {
                return Ok(TriageDecision::Drop {
                    reason: "SLM classified as irrelevant".into(),
                });
            }

            let summary = parsed["summary"].as_str().unwrap_or("Email").to_string();

            let entities: Vec<String> = parsed["entities"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            let urgency = parsed["urgency"].as_str().unwrap_or("normal");

            // Step 4: Priority assignment
            let mut priority = base_priority(is_known, urgency);

            // Thread replies get +1 priority level
            if is_thread {
                priority = boost_priority(priority);
            }

            let estimated_tokens = content.len() / 4;

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

    fn make_event(content: &str, metadata: Option<serde_json::Value>) -> SensorEvent {
        SensorEvent {
            id: SensorEvent::generate_id(content, "test-email"),
            sensor_name: "work_email".into(),
            modality: SensorModality::Text,
            observed_at: chrono::Utc::now(),
            content: content.into(),
            source_id: "msg-id-001@example.com".into(),
            metadata,
            binary_ref: None,
            related_ids: vec![],
        }
    }

    fn default_slm_json() -> String {
        serde_json::json!({
            "relevant": true,
            "urgency": "normal",
            "summary": "Test email summary",
            "entities": ["Acme Corp"],
            "category": "work"
        })
        .to_string()
    }

    fn urgent_slm_json() -> String {
        serde_json::json!({
            "relevant": true,
            "urgency": "high",
            "summary": "Urgent request",
            "entities": ["billing", "deadline"],
            "category": "work"
        })
        .to_string()
    }

    fn irrelevant_slm_json() -> String {
        serde_json::json!({
            "relevant": false,
            "urgency": "low",
            "summary": "Marketing spam",
            "entities": [],
            "category": "marketing"
        })
        .to_string()
    }

    // --- Unit tests for helpers ---

    #[test]
    fn meta_str_extracts_field() {
        let meta = serde_json::json!({"from": "alice@example.com"});
        assert_eq!(meta_str(Some(&meta), "from"), Some("alice@example.com"));
    }

    #[test]
    fn meta_str_returns_none_for_missing_field() {
        let meta = serde_json::json!({"from": "alice@example.com"});
        assert_eq!(meta_str(Some(&meta), "subject"), None);
    }

    #[test]
    fn meta_str_returns_none_when_no_metadata() {
        assert_eq!(meta_str(None, "from"), None);
    }

    #[test]
    fn meta_str_returns_none_for_non_string() {
        let meta = serde_json::json!({"count": 42});
        assert_eq!(meta_str(Some(&meta), "count"), None);
    }

    #[test]
    fn is_thread_reply_with_in_reply_to() {
        let meta = serde_json::json!({"in_reply_to": "<msg-000@example.com>"});
        assert!(is_thread_reply(Some(&meta)));
    }

    #[test]
    fn is_thread_reply_with_references() {
        let meta = serde_json::json!({"references": "<msg-000@example.com>"});
        assert!(is_thread_reply(Some(&meta)));
    }

    #[test]
    fn is_thread_reply_without_thread_headers() {
        let meta = serde_json::json!({"from": "alice@example.com"});
        assert!(!is_thread_reply(Some(&meta)));
    }

    #[test]
    fn is_thread_reply_no_metadata() {
        assert!(!is_thread_reply(None));
    }

    #[test]
    fn boost_priority_low_to_normal() {
        assert_eq!(boost_priority(Priority::Low), Priority::Normal);
    }

    #[test]
    fn boost_priority_normal_to_high() {
        assert_eq!(boost_priority(Priority::Normal), Priority::High);
    }

    #[test]
    fn boost_priority_high_to_critical() {
        assert_eq!(boost_priority(Priority::High), Priority::Critical);
    }

    #[test]
    fn boost_priority_critical_stays_critical() {
        assert_eq!(boost_priority(Priority::Critical), Priority::Critical);
    }

    #[test]
    fn base_priority_known_and_urgent() {
        assert_eq!(base_priority(true, "high"), Priority::Critical);
    }

    #[test]
    fn base_priority_known_not_urgent() {
        assert_eq!(base_priority(true, "normal"), Priority::High);
    }

    #[test]
    fn base_priority_unknown_and_urgent() {
        assert_eq!(base_priority(false, "high"), Priority::High);
    }

    #[test]
    fn base_priority_unknown_not_urgent() {
        assert_eq!(base_priority(false, "normal"), Priority::Normal);
    }

    #[test]
    fn base_priority_critical_urgency_treated_as_urgent() {
        assert_eq!(base_priority(false, "critical"), Priority::High);
    }

    #[test]
    fn base_priority_low_urgency_treated_as_not_urgent() {
        assert_eq!(base_priority(false, "low"), Priority::Normal);
    }

    // --- Trait method tests ---

    #[test]
    fn email_triage_modality_is_text() {
        let processor =
            EmailTriageProcessor::new(Arc::new(MockProvider::with_json("{}")), vec![], vec![]);
        assert_eq!(processor.modality(), SensorModality::Text);
    }

    #[test]
    fn email_triage_source_topic() {
        let processor =
            EmailTriageProcessor::new(Arc::new(MockProvider::with_json("{}")), vec![], vec![]);
        assert_eq!(processor.source_topic(), "hb.sensor.email");
    }

    // --- Async process() tests ---

    #[tokio::test]
    async fn blocked_sender_is_dropped() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec![],
            vec!["spam@malicious.com".into()],
        );

        let event = make_event(
            "Buy cheap watches!",
            Some(serde_json::json!({"from": "spam@malicious.com"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_drop());
        if let TriageDecision::Drop { reason } = &decision {
            assert!(reason.contains("blocked sender"), "reason: {reason}");
        }
    }

    #[tokio::test]
    async fn blocked_sender_case_insensitive() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec![],
            vec!["SPAM@Malicious.com".into()],
        );

        let event = make_event(
            "Buy cheap watches!",
            Some(serde_json::json!({"from": "spam@malicious.com"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_drop());
    }

    #[tokio::test]
    async fn known_contact_gets_high_priority() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec!["alice@example.com".into()],
            vec![],
        );

        let event = make_event(
            "Project update attached.",
            Some(serde_json::json!({
                "from": "alice@example.com",
                "subject": "Project Update"
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert!(
                *priority >= Priority::High,
                "known contact should get at least High, got {priority}"
            );
        }
    }

    #[tokio::test]
    async fn known_contact_case_insensitive() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec!["Alice@Example.com".into()],
            vec![],
        );

        let event = make_event(
            "Hello",
            Some(serde_json::json!({"from": "alice@example.com"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert!(*priority >= Priority::High);
        }
    }

    #[tokio::test]
    async fn known_contact_urgent_gets_critical() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&urgent_slm_json())),
            vec!["boss@company.com".into()],
            vec![],
        );

        let event = make_event(
            "Need your response ASAP on the contract.",
            Some(serde_json::json!({
                "from": "boss@company.com",
                "subject": "URGENT: Contract Review"
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Critical);
        }
    }

    #[tokio::test]
    async fn unknown_sender_normal_urgency_gets_normal() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec![],
            vec![],
        );

        let event = make_event(
            "General inquiry about your services.",
            Some(serde_json::json!({
                "from": "stranger@unknown.com",
                "subject": "Inquiry"
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Normal);
        }
    }

    #[tokio::test]
    async fn unknown_sender_urgent_gets_high() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&urgent_slm_json())),
            vec![],
            vec![],
        );

        let event = make_event(
            "Server is down! Immediate attention required.",
            Some(serde_json::json!({
                "from": "monitoring@infra.com",
                "subject": "ALERT: Server Down"
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::High);
        }
    }

    #[tokio::test]
    async fn thread_reply_boosts_priority() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec![],
            vec![],
        );

        // Unknown sender + normal urgency = Normal, but thread reply boosts to High
        let event = make_event(
            "Re: Following up on our conversation.",
            Some(serde_json::json!({
                "from": "colleague@other.com",
                "subject": "Re: Meeting Notes",
                "in_reply_to": "<msg-000@other.com>"
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(
                *priority,
                Priority::High,
                "thread reply should boost Normal to High"
            );
        }
    }

    #[tokio::test]
    async fn thread_reply_with_references_field() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec![],
            vec![],
        );

        let event = make_event(
            "Additional context for the thread.",
            Some(serde_json::json!({
                "from": "someone@example.com",
                "references": "<msg-000@example.com> <msg-001@example.com>"
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(
                *priority,
                Priority::High,
                "references-based thread should also boost"
            );
        }
    }

    #[tokio::test]
    async fn known_contact_thread_reply_boosts_high_to_critical() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec!["alice@example.com".into()],
            vec![],
        );

        // Known contact + normal urgency = High, thread reply boosts to Critical
        let event = make_event(
            "Re: Thanks for the update.",
            Some(serde_json::json!({
                "from": "alice@example.com",
                "subject": "Re: Project",
                "in_reply_to": "<msg-prev@example.com>"
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Critical);
        }
    }

    #[tokio::test]
    async fn missing_metadata_falls_back_gracefully() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec![],
            vec![],
        );

        let event = make_event("Some email body with no metadata.", None);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(
                *priority,
                Priority::Normal,
                "no metadata means unknown sender + no thread = Normal"
            );
        }
    }

    #[tokio::test]
    async fn missing_from_field_treated_as_unknown() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec!["alice@example.com".into()],
            vec![],
        );

        let event = make_event(
            "Email without from field.",
            Some(serde_json::json!({"subject": "No Sender"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Normal);
        }
    }

    #[tokio::test]
    async fn irrelevant_unknown_sender_dropped() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&irrelevant_slm_json())),
            vec![],
            vec![],
        );

        let event = make_event(
            "50% off all items this weekend only!",
            Some(serde_json::json!({
                "from": "deals@marketing.com",
                "subject": "SALE"
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_drop());
    }

    #[tokio::test]
    async fn irrelevant_known_contact_still_promoted() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&irrelevant_slm_json())),
            vec!["alice@example.com".into()],
            vec![],
        );

        let event = make_event(
            "Forwarding this newsletter.",
            Some(serde_json::json!({"from": "alice@example.com"})),
        );

        // Known contact should still be promoted even if SLM says irrelevant
        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_promote(),
            "known contact should be promoted regardless of SLM relevance"
        );
    }

    #[tokio::test]
    async fn entities_extracted_from_slm_response() {
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec![],
            vec![],
        );

        let event = make_event(
            "Email about Acme Corp partnership.",
            Some(serde_json::json!({"from": "partner@acme.com"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            extracted_entities, ..
        } = &decision
        {
            assert_eq!(extracted_entities, &["Acme Corp"]);
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }

    #[tokio::test]
    async fn estimated_tokens_calculated() {
        let content = "A".repeat(400);
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec![],
            vec![],
        );

        let event = make_event(
            &content,
            Some(serde_json::json!({"from": "user@example.com"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            estimated_tokens, ..
        } = &decision
        {
            assert_eq!(*estimated_tokens, 100, "400 bytes / 4 = 100 tokens");
        } else {
            panic!("expected Promote");
        }
    }

    #[tokio::test]
    async fn empty_sender_in_blocked_list_not_blocked() {
        // Edge case: empty from field should not match a blocked sender
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec![],
            vec!["spam@bad.com".into()],
        );

        let event = make_event(
            "Email with no from.",
            Some(serde_json::json!({"subject": "Hello"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_promote(),
            "missing from should not match blocked list"
        );
    }

    #[tokio::test]
    async fn constructor_deduplicates_contacts() {
        // Duplicate contacts in input should not cause double-matching issues
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec!["alice@example.com".into(), "alice@example.com".into()],
            vec![],
        );
        // HashSet deduplication means only one entry
        assert_eq!(processor.known_contacts.len(), 1);
    }

    #[tokio::test]
    async fn process_slm_returns_invalid_json_fallback() {
        // SLM returns invalid JSON → fallback defaults: relevant=true, urgency=normal
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json("not valid json {{")),
            vec![],
            vec![],
        );

        let event = make_event(
            "Important email body content here.",
            Some(serde_json::json!({"from": "user@example.com", "subject": "Test"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_promote(),
            "invalid JSON fallback defaults to relevant=true"
        );
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(
                *priority,
                Priority::Normal,
                "fallback urgency=normal + unknown sender = Normal"
            );
        }
    }

    #[tokio::test]
    async fn process_long_content_truncated() {
        // Content > 2000 bytes should be truncated but still process successfully
        let content = "X".repeat(5000);
        let processor = EmailTriageProcessor::new(
            Arc::new(MockProvider::with_json(&default_slm_json())),
            vec![],
            vec![],
        );

        let event = make_event(
            &content,
            Some(serde_json::json!({"from": "user@example.com"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_promote(),
            "long content should still process after truncation"
        );
        if let TriageDecision::Promote {
            estimated_tokens, ..
        } = &decision
        {
            assert_eq!(*estimated_tokens, 5000 / 4, "tokens based on full content");
        }
    }

    #[tokio::test]
    async fn process_summary_defaults_when_missing() {
        // SLM JSON with no "summary" field → defaults to "Email"
        let json = serde_json::json!({
            "relevant": true,
            "urgency": "normal",
            "entities": ["test"],
            "category": "work"
        })
        .to_string();

        let processor =
            EmailTriageProcessor::new(Arc::new(MockProvider::with_json(&json)), vec![], vec![]);

        let event = make_event(
            "Email without summary in SLM response.",
            Some(serde_json::json!({"from": "user@example.com"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote { summary, .. } = &decision {
            assert_eq!(
                summary, "Email",
                "missing summary should default to 'Email'"
            );
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }
}
