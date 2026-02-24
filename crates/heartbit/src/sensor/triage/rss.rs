use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::Error;
use crate::llm::DynLlmProvider;
use crate::llm::types::{CompletionRequest, Message, Role};
use crate::sensor::triage::{Priority, TriageDecision, TriageProcessor};
use crate::sensor::{SensorEvent, SensorModality};

/// RSS triage processor. Uses keyword pre-filtering and SLM classification
/// to decide whether to promote, drop, or dead-letter RSS feed items.
pub struct RssTriageProcessor {
    slm_provider: Arc<dyn DynLlmProvider>,
    interest_keywords: Vec<String>,
}

impl RssTriageProcessor {
    pub fn new(slm_provider: Arc<dyn DynLlmProvider>, interest_keywords: Vec<String>) -> Self {
        Self {
            slm_provider,
            interest_keywords,
        }
    }
}

/// Simple BM25-inspired keyword relevance score.
///
/// Counts keyword occurrences in text, weighted by inverse document frequency
/// (approximated as 1.0 for all keywords since we don't have corpus stats).
fn keyword_relevance(text: &str, keywords: &[String]) -> f64 {
    if keywords.is_empty() {
        return 0.0;
    }
    let lower = text.to_lowercase();
    let matches: usize = keywords
        .iter()
        .filter(|kw| lower.contains(&kw.to_lowercase()))
        .count();
    matches as f64 / keywords.len() as f64
}

impl TriageProcessor for RssTriageProcessor {
    fn modality(&self) -> SensorModality {
        SensorModality::Text
    }

    fn source_topic(&self) -> &str {
        "hb.sensor.rss"
    }

    fn process(
        &self,
        event: &SensorEvent,
    ) -> Pin<Box<dyn Future<Output = Result<TriageDecision, Error>> + Send + '_>> {
        let content = event.content.clone();
        let source_id = event.source_id.clone();

        Box::pin(async move {
            // Step 1: Keyword pre-filter (zero-cost, no SLM)
            let relevance = keyword_relevance(&content, &self.interest_keywords);
            if relevance < 0.01 && !self.interest_keywords.is_empty() {
                return Ok(TriageDecision::Drop {
                    reason: "no keyword match".into(),
                });
            }

            // Step 2: SLM classification
            let truncated = if content.len() > 2000 {
                let boundary = crate::tool::builtins::floor_char_boundary(&content, 2000);
                &content[..boundary]
            } else {
                &content
            };

            let prompt = format!(
                "Classify this RSS article. Respond with JSON only.\n\
                 Article: {truncated}\n\n\
                 Respond with: {{\"relevant\": true/false, \"summary\": \"one sentence\", \
                 \"entities\": [\"entity1\", \"entity2\"], \"category\": \"tech|news|other\"}}"
            );

            let request = CompletionRequest {
                system: "You are a triage classifier. Output valid JSON only, no markdown.".into(),
                messages: vec![Message {
                    role: Role::User,
                    content: vec![crate::llm::types::ContentBlock::Text { text: prompt }],
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
                    crate::llm::types::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .unwrap_or("");

            // Parse SLM response
            let parsed: serde_json::Value =
                serde_json::from_str(text).unwrap_or(serde_json::json!({
                    "relevant": relevance > 0.3,
                    "summary": content.chars().take(100).collect::<String>(),
                    "entities": [],
                }));

            let is_relevant = parsed["relevant"].as_bool().unwrap_or(false);
            if !is_relevant {
                return Ok(TriageDecision::Drop {
                    reason: "SLM classified as irrelevant".into(),
                });
            }

            let summary = parsed["summary"]
                .as_str()
                .unwrap_or("RSS article")
                .to_string();

            let entities: Vec<String> = parsed["entities"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();

            // RSS items are always Normal or Low priority
            let priority = if relevance > 0.5 {
                Priority::Normal
            } else {
                Priority::Low
            };

            let estimated_tokens = content.len() / 4; // rough estimate

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
    use crate::llm::types::{CompletionResponse, ContentBlock, StopReason, TokenUsage};

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

    fn make_event(content: &str, source_id: &str) -> SensorEvent {
        SensorEvent {
            id: SensorEvent::generate_id(content, source_id),
            sensor_name: "rss_feed".into(),
            modality: SensorModality::Text,
            observed_at: chrono::Utc::now(),
            content: content.into(),
            source_id: source_id.into(),
            metadata: None,
            binary_ref: None,
            related_ids: vec![],
        }
    }

    fn relevant_slm_json() -> String {
        serde_json::json!({
            "relevant": true,
            "summary": "Article about Rust programming",
            "entities": ["Rust", "programming"],
            "category": "tech"
        })
        .to_string()
    }

    fn irrelevant_slm_json() -> String {
        serde_json::json!({
            "relevant": false,
            "summary": "Unrelated article",
            "entities": [],
            "category": "other"
        })
        .to_string()
    }

    // --- Sync helper tests ---

    #[test]
    fn keyword_relevance_no_keywords() {
        let score = keyword_relevance("any text here", &[]);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn keyword_relevance_all_match() {
        let keywords = vec!["rust".into(), "ai".into()];
        let score = keyword_relevance("Rust and AI are great", &keywords);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn keyword_relevance_partial_match() {
        let keywords = vec!["rust".into(), "python".into(), "go".into()];
        let score = keyword_relevance("I love Rust programming", &keywords);
        assert!((score - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn keyword_relevance_no_match() {
        let keywords = vec!["blockchain".into()];
        let score = keyword_relevance("This is about cooking recipes", &keywords);
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn keyword_relevance_case_insensitive() {
        let keywords = vec!["RUST".into()];
        let score = keyword_relevance("rust is great", &keywords);
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn rss_triage_modality_is_text() {
        let processor = RssTriageProcessor::new(Arc::new(MockProvider::with_json("{}")), vec![]);
        assert_eq!(processor.modality(), SensorModality::Text);
    }

    #[test]
    fn rss_triage_source_topic() {
        let processor = RssTriageProcessor::new(Arc::new(MockProvider::with_json("{}")), vec![]);
        assert_eq!(processor.source_topic(), "hb.sensor.rss");
    }

    // --- Async process() tests ---

    #[tokio::test]
    async fn process_no_keyword_match_drops() {
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&relevant_slm_json())),
            vec!["blockchain".into()],
        );

        let event = make_event("This article is about cooking recipes", "feed/cooking-101");

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_drop());
        if let TriageDecision::Drop { reason } = &decision {
            assert!(reason.contains("no keyword match"), "reason: {reason}");
        }
    }

    #[tokio::test]
    async fn process_keyword_match_promotes_with_slm_relevant() {
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&relevant_slm_json())),
            vec!["rust".into()],
        );

        let event = make_event(
            "New Rust compiler release with major performance improvements",
            "feed/rust-release",
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_promote(),
            "keyword match + SLM relevant should promote"
        );
    }

    #[tokio::test]
    async fn process_keyword_match_but_slm_irrelevant_drops() {
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&irrelevant_slm_json())),
            vec!["rust".into()],
        );

        let event = make_event(
            "The rust belt economy is facing challenges",
            "feed/economy-news",
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_drop(),
            "keyword match but SLM irrelevant should drop"
        );
        if let TriageDecision::Drop { reason } = &decision {
            assert!(
                reason.contains("SLM classified as irrelevant"),
                "reason: {reason}"
            );
        }
    }

    #[tokio::test]
    async fn process_empty_keywords_always_sends_to_slm() {
        // Empty keyword list → relevance=0.0, but the pre-filter is skipped
        // because interest_keywords.is_empty() is true. SLM decides.
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&relevant_slm_json())),
            vec![],
        );

        let event = make_event("Any article content at all", "feed/random-article");

        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_promote(),
            "empty keywords should delegate to SLM, which says relevant"
        );
        if let TriageDecision::Promote { priority, .. } = &decision {
            // relevance=0.0 < 0.5 → Low
            assert_eq!(*priority, Priority::Low);
        }
    }

    #[tokio::test]
    async fn process_high_relevance_gets_normal_priority() {
        // Two keywords, both match → relevance=1.0 > 0.5 → Normal
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&relevant_slm_json())),
            vec!["rust".into(), "ai".into()],
        );

        let event = make_event(
            "Rust and AI are transforming the tech landscape",
            "feed/tech-trends",
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(
                *priority,
                Priority::Normal,
                "relevance > 0.5 should give Normal"
            );
        }
    }

    #[tokio::test]
    async fn process_low_relevance_gets_low_priority() {
        // Three keywords, one match → relevance=1/3 ≈ 0.33 < 0.5 → Low
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&relevant_slm_json())),
            vec!["rust".into(), "python".into(), "go".into()],
        );

        let event = make_event(
            "Rust tutorial for beginners covers ownership and borrowing",
            "feed/rust-tutorial",
        );

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Low, "relevance < 0.5 should give Low");
        }
    }

    #[tokio::test]
    async fn process_extracts_entities_from_slm() {
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&relevant_slm_json())),
            vec!["rust".into()],
        );

        let event = make_event("Rust programming article", "feed/rust-entities");

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            extracted_entities, ..
        } = &decision
        {
            assert_eq!(extracted_entities, &["Rust", "programming"]);
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }

    #[tokio::test]
    async fn process_extracts_summary_from_slm() {
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&relevant_slm_json())),
            vec!["rust".into()],
        );

        let event = make_event("Rust programming article", "feed/rust-summary");

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote { summary, .. } = &decision {
            assert_eq!(summary, "Article about Rust programming");
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }

    #[tokio::test]
    async fn process_estimated_tokens_calculated() {
        let content = "A".repeat(400);
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&relevant_slm_json())),
            vec![],
        );

        let event = make_event(&content, "feed/token-test");

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
    async fn process_long_content_truncated_before_slm() {
        // Content > 2000 bytes should be truncated but SLM still called
        let content = "Rust ".repeat(1000); // 5000 bytes
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&relevant_slm_json())),
            vec!["rust".into()],
        );

        let event = make_event(&content, "feed/long-article");

        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_promote(),
            "long content should still be processed after truncation"
        );
        if let TriageDecision::Promote {
            estimated_tokens, ..
        } = &decision
        {
            // estimated_tokens is based on full content, not truncated
            assert_eq!(
                *estimated_tokens,
                content.len() / 4,
                "tokens should be based on full content length"
            );
        }
    }

    #[tokio::test]
    async fn process_slm_returns_invalid_json_uses_fallback() {
        // SLM returns invalid JSON → fallback: relevant = relevance > 0.3
        // With keywords=["rust"], content contains "rust" → relevance=1.0 > 0.3 → relevant=true
        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json("not valid json at all")),
            vec!["rust".into()],
        );

        let event = make_event("Rust article with invalid SLM output", "feed/bad-json");

        let decision = processor.process(&event).await.expect("process failed");
        assert!(
            decision.is_promote(),
            "fallback with high relevance should promote"
        );
    }

    #[tokio::test]
    async fn process_slm_returns_empty_entities() {
        let json = serde_json::json!({
            "relevant": true,
            "summary": "Short article",
            "entities": [],
            "category": "tech"
        })
        .to_string();

        let processor = RssTriageProcessor::new(
            Arc::new(MockProvider::with_json(&json)),
            vec!["tech".into()],
        );

        let event = make_event("A tech article with no entities", "feed/no-entities");

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            extracted_entities, ..
        } = &decision
        {
            assert!(
                extracted_entities.is_empty(),
                "entities should be empty vec"
            );
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }
}
