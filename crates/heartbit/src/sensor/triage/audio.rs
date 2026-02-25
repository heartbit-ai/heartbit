use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::Error;
use crate::llm::DynLlmProvider;
use crate::llm::types::{CompletionRequest, ContentBlock, Message, Role};
use crate::sensor::triage::{Priority, TriageDecision, TriageProcessor};
use crate::sensor::{SensorEvent, SensorModality};

/// Audio triage processor.
///
/// Classifies audio sensor events using an SLM. Currently works on file
/// metadata (filename, size, extension) since whisper-rs transcription is not
/// yet integrated. Once transcription is available, the SLM will classify
/// the transcript text instead.
///
/// Classification categories:
/// - `voice_note` -- personal voice memo or message -> High
/// - `meeting` -- meeting recording -> Normal
/// - `podcast` -- podcast episode -> Low
/// - `music` -- music file -> Drop
/// - `other` -- unrecognized audio -> Low
///
/// Known contacts: if the metadata contains a `speaker` field matching a
/// known contact, the priority is boosted by one level.
pub struct AudioTriageProcessor {
    slm_provider: Arc<dyn DynLlmProvider>,
    known_contacts: HashSet<String>,
}

impl AudioTriageProcessor {
    pub fn new(slm_provider: Arc<dyn DynLlmProvider>, known_contacts: Vec<String>) -> Self {
        Self {
            slm_provider,
            known_contacts: known_contacts.into_iter().collect(),
        }
    }
}

/// Map an audio category string to a base priority.
fn category_to_priority(category: &str) -> Option<Priority> {
    match category {
        "voice_note" => Some(Priority::High),
        "meeting" => Some(Priority::Normal),
        "podcast" | "other" => Some(Priority::Low),
        "music" => None, // Drop
        _ => Some(Priority::Low),
    }
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

/// Extract a metadata string field from an optional JSON value.
fn meta_str<'a>(metadata: Option<&'a serde_json::Value>, key: &str) -> Option<&'a str> {
    metadata.and_then(|m| m.get(key)).and_then(|v| v.as_str())
}

/// Extract filename from event metadata or content for fallback summary.
fn extract_filename(event: &SensorEvent) -> String {
    meta_str(event.metadata.as_ref(), "filename")
        .map(String::from)
        .unwrap_or_else(|| {
            // Try to extract from content: "Audio: filename.mp3 (..."
            event
                .content
                .strip_prefix("Audio: ")
                .and_then(|rest| rest.split(" (").next())
                .map(String::from)
                .unwrap_or_else(|| "unknown audio".into())
        })
}

impl TriageProcessor for AudioTriageProcessor {
    fn modality(&self) -> SensorModality {
        SensorModality::Audio
    }

    fn source_topic(&self) -> &str {
        "hb.sensor.audio"
    }

    fn process(
        &self,
        event: &SensorEvent,
    ) -> Pin<Box<dyn Future<Output = Result<TriageDecision, Error>> + Send + '_>> {
        let content = event.content.clone();
        let source_id = event.source_id.clone();
        let metadata = event.metadata.clone();
        let filename = extract_filename(event);

        Box::pin(async move {
            // Build classification prompt from available metadata
            let truncated = if content.len() > 2000 {
                let boundary = crate::tool::builtins::floor_char_boundary(&content, 2000);
                &content[..boundary]
            } else {
                &content
            };

            let meta_summary = if let Some(ref meta) = metadata {
                format!(
                    "Filename: {}\nExtension: {}\nSize: {} bytes",
                    meta.get("filename")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown"),
                    meta.get("extension")
                        .and_then(|v| v.as_str())
                        .unwrap_or("unknown"),
                    meta.get("size_bytes").and_then(|v| v.as_u64()).unwrap_or(0),
                )
            } else {
                String::new()
            };

            let prompt = format!(
                "Classify this audio file. Respond with JSON only.\n\
                 Content: {truncated}\n\
                 {meta_summary}\n\n\
                 Respond with: {{\"category\": \"voice_note|meeting|podcast|music|other\", \
                 \"summary\": \"one sentence describing the audio\", \
                 \"entities\": [\"entity1\", \"entity2\"]}}"
            );

            let request = CompletionRequest {
                system: "You are an audio triage classifier. Classify audio files by their \
                         metadata and content. Output valid JSON only, no markdown."
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
                    "category": "other",
                    "summary": format!("Audio file: {filename}"),
                    "entities": [],
                }));

            let category = parsed["category"].as_str().unwrap_or("other");

            // Check for music -> Drop
            let base_priority = match category_to_priority(category) {
                Some(p) => p,
                None => {
                    return Ok(TriageDecision::Drop {
                        reason: format!("music file: {filename}"),
                    });
                }
            };

            let summary = parsed["summary"]
                .as_str()
                .unwrap_or("Audio file")
                .to_string();

            let entities: Vec<String> = parsed["entities"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            // Check for known contacts in speaker metadata
            let speaker = meta_str(metadata.as_ref(), "speaker").unwrap_or("");
            let speaker_lower = speaker.to_lowercase();
            let is_known_contact = !speaker_lower.is_empty()
                && self
                    .known_contacts
                    .iter()
                    .any(|c| c.to_lowercase() == speaker_lower);

            let priority = if is_known_contact {
                boost_priority(base_priority)
            } else {
                base_priority
            };

            let estimated_tokens = content.len() / 4 + 256;

            Ok(TriageDecision::Promote {
                priority,
                summary,
                extracted_entities: entities,
                estimated_tokens,
                action_categories: vec![],
                action_hints: vec![],
                has_attachments: false,
                sender: None,
                subject: None,
                message_ref: None,
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

    struct FailingProvider;

    impl DynLlmProvider for FailingProvider {
        fn complete<'a>(
            &'a self,
            _req: CompletionRequest,
        ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>> {
            Box::pin(async { Err(Error::Sensor("model unavailable".into())) })
        }

        fn stream_complete<'a>(
            &'a self,
            _req: CompletionRequest,
            _on_text: &'a crate::llm::OnText,
        ) -> Pin<Box<dyn Future<Output = Result<CompletionResponse, Error>> + Send + 'a>> {
            Box::pin(async { Err(Error::Sensor("not supported".into())) })
        }

        fn model_name(&self) -> Option<&str> {
            Some("failing-mock")
        }
    }

    fn make_event(content: &str, metadata: Option<serde_json::Value>) -> SensorEvent {
        SensorEvent {
            id: SensorEvent::generate_id(content, "test-audio"),
            sensor_name: "mic".into(),
            modality: SensorModality::Audio,
            observed_at: chrono::Utc::now(),
            content: content.into(),
            source_id: "/tmp/audio/test.mp3".into(),
            metadata,
            binary_ref: Some("/tmp/audio/test.mp3".into()),
            related_ids: vec![],
        }
    }

    fn voice_note_json() -> String {
        serde_json::json!({
            "category": "voice_note",
            "summary": "Personal voice memo about grocery list",
            "entities": ["grocery", "shopping"]
        })
        .to_string()
    }

    fn meeting_json() -> String {
        serde_json::json!({
            "category": "meeting",
            "summary": "Weekly team standup recording",
            "entities": ["standup", "team"]
        })
        .to_string()
    }

    fn podcast_json() -> String {
        serde_json::json!({
            "category": "podcast",
            "summary": "Tech podcast episode about Rust",
            "entities": ["Rust", "programming"]
        })
        .to_string()
    }

    fn music_json() -> String {
        serde_json::json!({
            "category": "music",
            "summary": "Pop song",
            "entities": []
        })
        .to_string()
    }

    fn other_json() -> String {
        serde_json::json!({
            "category": "other",
            "summary": "Unrecognized audio clip",
            "entities": []
        })
        .to_string()
    }

    // --- Trait property tests ---

    #[test]
    fn audio_triage_modality() {
        let processor = AudioTriageProcessor::new(Arc::new(MockProvider::with_json("{}")), vec![]);
        assert_eq!(processor.modality(), SensorModality::Audio);
    }

    #[test]
    fn audio_triage_source_topic() {
        let processor = AudioTriageProcessor::new(Arc::new(MockProvider::with_json("{}")), vec![]);
        assert_eq!(processor.source_topic(), "hb.sensor.audio");
    }

    // --- Helper function tests ---

    #[test]
    fn category_to_priority_voice_note() {
        assert_eq!(category_to_priority("voice_note"), Some(Priority::High));
    }

    #[test]
    fn category_to_priority_meeting() {
        assert_eq!(category_to_priority("meeting"), Some(Priority::Normal));
    }

    #[test]
    fn category_to_priority_podcast() {
        assert_eq!(category_to_priority("podcast"), Some(Priority::Low));
    }

    #[test]
    fn category_to_priority_music_returns_none() {
        assert_eq!(category_to_priority("music"), None);
    }

    #[test]
    fn category_to_priority_other() {
        assert_eq!(category_to_priority("other"), Some(Priority::Low));
    }

    #[test]
    fn category_to_priority_unknown_falls_to_low() {
        assert_eq!(category_to_priority("unknown_thing"), Some(Priority::Low));
    }

    #[test]
    fn boost_priority_low_to_normal() {
        assert_eq!(boost_priority(Priority::Low), Priority::Normal);
    }

    #[test]
    fn boost_priority_critical_stays_critical() {
        assert_eq!(boost_priority(Priority::Critical), Priority::Critical);
    }

    #[test]
    fn extract_filename_from_metadata() {
        let event = make_event(
            "Audio: notes.mp3 (1000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({"filename": "notes.mp3"})),
        );
        assert_eq!(extract_filename(&event), "notes.mp3");
    }

    #[test]
    fn extract_filename_from_content_fallback() {
        let event = make_event(
            "Audio: recording.wav (5000 bytes, 2026-02-22T10:00:00Z)",
            None,
        );
        assert_eq!(extract_filename(&event), "recording.wav");
    }

    #[test]
    fn extract_filename_unknown_fallback() {
        let event = make_event("some random content", None);
        assert_eq!(extract_filename(&event), "unknown audio");
    }

    // --- Async process() tests ---

    #[tokio::test]
    async fn classify_voice_note_high_priority() {
        let processor = AudioTriageProcessor::new(
            Arc::new(MockProvider::with_json(&voice_note_json())),
            vec![],
        );

        let event = make_event(
            "Audio: voice_memo.m4a (24000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "voice_memo.m4a",
                "extension": "m4a",
                "size_bytes": 24000
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::High);
        }
    }

    #[tokio::test]
    async fn classify_meeting_normal_priority() {
        let processor =
            AudioTriageProcessor::new(Arc::new(MockProvider::with_json(&meeting_json())), vec![]);

        let event = make_event(
            "Audio: standup.mp3 (5000000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "standup.mp3",
                "extension": "mp3",
                "size_bytes": 5000000
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Normal);
        }
    }

    #[tokio::test]
    async fn classify_podcast_low_priority() {
        let processor =
            AudioTriageProcessor::new(Arc::new(MockProvider::with_json(&podcast_json())), vec![]);

        let event = make_event(
            "Audio: episode42.ogg (80000000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "episode42.ogg",
                "extension": "ogg",
                "size_bytes": 80000000
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Low);
        }
    }

    #[tokio::test]
    async fn classify_music_dropped() {
        let processor =
            AudioTriageProcessor::new(Arc::new(MockProvider::with_json(&music_json())), vec![]);

        let event = make_event(
            "Audio: song.flac (30000000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "song.flac",
                "extension": "flac",
                "size_bytes": 30000000
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_drop());
        if let TriageDecision::Drop { reason } = &decision {
            assert!(reason.contains("music"), "reason: {reason}");
            assert!(reason.contains("song.flac"), "reason: {reason}");
        }
    }

    #[tokio::test]
    async fn classify_other_low_priority() {
        let processor =
            AudioTriageProcessor::new(Arc::new(MockProvider::with_json(&other_json())), vec![]);

        let event = make_event(
            "Audio: unknown.wav (1000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "unknown.wav",
                "extension": "wav",
                "size_bytes": 1000
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Low);
        }
    }

    #[tokio::test]
    async fn slm_parse_failure_fallback_to_normal() {
        // Return invalid JSON so the SLM response cannot be parsed
        let processor = AudioTriageProcessor::new(
            Arc::new(MockProvider::with_json("not valid json at all")),
            vec![],
        );

        let event = make_event(
            "Audio: ambiguous.mp3 (5000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "ambiguous.mp3",
                "extension": "mp3",
                "size_bytes": 5000
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        // Fallback: category="other" -> Low priority
        assert!(decision.is_promote());
        if let TriageDecision::Promote {
            priority, summary, ..
        } = &decision
        {
            assert_eq!(*priority, Priority::Low);
            assert!(
                summary.contains("ambiguous.mp3"),
                "summary should contain filename: {summary}"
            );
        }
    }

    #[tokio::test]
    async fn slm_error_returns_error() {
        let processor = AudioTriageProcessor::new(Arc::new(FailingProvider), vec![]);

        let event = make_event("Audio: test.wav (100 bytes, 2026-02-22T10:00:00Z)", None);

        let result = processor.process(&event).await;
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("SLM triage failed"), "error: {err_msg}");
    }

    #[tokio::test]
    async fn empty_content_handled() {
        let processor =
            AudioTriageProcessor::new(Arc::new(MockProvider::with_json(&other_json())), vec![]);

        let event = make_event("", None);

        let decision = processor.process(&event).await.expect("process failed");
        // Should not panic, should produce a valid decision
        assert!(decision.is_promote() || decision.is_drop());
    }

    #[tokio::test]
    async fn entities_extracted() {
        let processor =
            AudioTriageProcessor::new(Arc::new(MockProvider::with_json(&meeting_json())), vec![]);

        let event = make_event(
            "Audio: standup.mp3 (5000000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "standup.mp3",
                "extension": "mp3",
                "size_bytes": 5000000
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            extracted_entities, ..
        } = &decision
        {
            assert_eq!(extracted_entities, &["standup", "team"]);
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }

    #[tokio::test]
    async fn estimated_tokens_reasonable() {
        let content = "A".repeat(800);
        let processor =
            AudioTriageProcessor::new(Arc::new(MockProvider::with_json(&meeting_json())), vec![]);

        let event = make_event(
            &content,
            Some(
                serde_json::json!({"filename": "test.mp3", "extension": "mp3", "size_bytes": 100}),
            ),
        );

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
    async fn summary_contains_filename_on_fallback() {
        // Return JSON with missing summary field
        let processor = AudioTriageProcessor::new(
            Arc::new(MockProvider::with_json(
                r#"{"category": "meeting", "entities": []}"#,
            )),
            vec![],
        );

        let event = make_event(
            "Audio: weekly_sync.mp3 (5000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({"filename": "weekly_sync.mp3"})),
        );

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote { summary, .. } = &decision {
            // When summary is missing from SLM response, the default "Audio file" is used
            assert!(!summary.is_empty(), "summary should not be empty");
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }

    #[tokio::test]
    async fn known_contact_boosts_priority() {
        let processor = AudioTriageProcessor::new(
            Arc::new(MockProvider::with_json(&meeting_json())),
            vec!["Alice".into()],
        );

        let event = make_event(
            "Audio: standup.mp3 (5000000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "standup.mp3",
                "extension": "mp3",
                "size_bytes": 5000000,
                "speaker": "alice"
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            // meeting = Normal, known contact boost = High
            assert_eq!(
                *priority,
                Priority::High,
                "known contact should boost Normal to High"
            );
        }
    }

    #[tokio::test]
    async fn known_contact_case_insensitive() {
        let processor = AudioTriageProcessor::new(
            Arc::new(MockProvider::with_json(&podcast_json())),
            vec!["Bob Smith".into()],
        );

        let event = make_event(
            "Audio: interview.mp3 (10000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "interview.mp3",
                "speaker": "bob smith"
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            // podcast = Low, known contact boost = Normal
            assert_eq!(*priority, Priority::Normal);
        }
    }

    #[tokio::test]
    async fn no_speaker_no_boost() {
        let processor = AudioTriageProcessor::new(
            Arc::new(MockProvider::with_json(&meeting_json())),
            vec!["Alice".into()],
        );

        let event = make_event(
            "Audio: standup.mp3 (5000000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "standup.mp3",
                "extension": "mp3",
                "size_bytes": 5000000
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(
                *priority,
                Priority::Normal,
                "without speaker metadata, no boost should occur"
            );
        }
    }

    // --- Serde roundtrip tests ---

    #[tokio::test]
    async fn triage_decision_promote_serde_roundtrip() {
        let processor = AudioTriageProcessor::new(
            Arc::new(MockProvider::with_json(&voice_note_json())),
            vec![],
        );

        let event = make_event(
            "Audio: memo.m4a (1000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({"filename": "memo.m4a"})),
        );

        let decision = processor.process(&event).await.expect("process");
        let json = serde_json::to_string(&decision).expect("serialize");
        let back: TriageDecision = serde_json::from_str(&json).expect("deserialize");
        assert!(back.is_promote());
    }

    #[tokio::test]
    async fn triage_decision_drop_serde_roundtrip() {
        let processor =
            AudioTriageProcessor::new(Arc::new(MockProvider::with_json(&music_json())), vec![]);

        let event = make_event(
            "Audio: pop_song.mp3 (4000000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({"filename": "pop_song.mp3"})),
        );

        let decision = processor.process(&event).await.expect("process");
        let json = serde_json::to_string(&decision).expect("serialize");
        let back: TriageDecision = serde_json::from_str(&json).expect("deserialize");
        assert!(back.is_drop());
    }

    #[test]
    fn constructor_deduplicates_contacts() {
        let processor = AudioTriageProcessor::new(
            Arc::new(MockProvider::with_json("{}")),
            vec!["Alice".into(), "Alice".into(), "Bob".into()],
        );
        assert_eq!(processor.known_contacts.len(), 2);
    }

    #[tokio::test]
    async fn process_slm_returns_empty_text() {
        // SLM returns empty string → JSON parse fails → fallback category="other" → Low
        let processor = AudioTriageProcessor::new(Arc::new(MockProvider::with_json("")), vec![]);

        let event = make_event(
            "Audio: recording.wav (5000 bytes, 2026-02-22T10:00:00Z)",
            Some(serde_json::json!({
                "filename": "recording.wav",
                "extension": "wav",
                "size_bytes": 5000
            })),
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote {
            priority, summary, ..
        } = &decision
        {
            assert_eq!(*priority, Priority::Low, "fallback category=other → Low");
            assert!(
                summary.contains("recording.wav"),
                "fallback summary should contain filename: {summary}"
            );
        }
    }

    #[tokio::test]
    async fn process_content_truncated_before_slm() {
        // Content > 2000 bytes should be truncated but SLM still called
        let content = "Transcribed audio text. ".repeat(200); // ~4800 bytes
        let processor =
            AudioTriageProcessor::new(Arc::new(MockProvider::with_json(&meeting_json())), vec![]);

        let event = make_event(
            &content,
            Some(serde_json::json!({
                "filename": "long_meeting.mp3",
                "extension": "mp3",
                "size_bytes": 50000000
            })),
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
            // estimated_tokens = content.len() / 4 + 256
            assert_eq!(
                *estimated_tokens,
                content.len() / 4 + 256,
                "tokens based on full content + 256 overhead"
            );
        }
    }
}
