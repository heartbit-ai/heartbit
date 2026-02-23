use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::Error;
use crate::llm::DynLlmProvider;
use crate::llm::types::{CompletionRequest, ContentBlock, Message, Role};
use crate::sensor::triage::{Priority, TriageDecision, TriageProcessor};
use crate::sensor::{SensorEvent, SensorModality};

/// Image triage processor. Uses an SLM to classify images based on their
/// metadata (filename, extension, size) and assign priority.
///
/// Processing pipeline:
/// 1. Extract metadata from the event (filename, extension, size).
/// 2. Send metadata to SLM for classification into category.
/// 3. Map category to priority: document/invoice -> High, screenshot -> Normal,
///    photo -> Normal, other -> Low.
/// 4. Extract entities from the SLM response.
/// 5. On parse failure: fallback to Normal priority with basic summary.
pub struct ImageTriageProcessor {
    slm_provider: Arc<dyn DynLlmProvider>,
}

impl ImageTriageProcessor {
    pub fn new(slm_provider: Arc<dyn DynLlmProvider>) -> Self {
        Self { slm_provider }
    }
}

/// Extract a metadata string field from an optional JSON value.
fn meta_str<'a>(metadata: Option<&'a serde_json::Value>, key: &str) -> Option<&'a str> {
    metadata.and_then(|m| m.get(key)).and_then(|v| v.as_str())
}

/// Map image category to priority level.
fn category_to_priority(category: &str) -> Priority {
    match category {
        "document" | "invoice" | "receipt" => Priority::High,
        "screenshot" | "photo" => Priority::Normal,
        _ => Priority::Low,
    }
}

/// Build summary text from metadata, with fallback to content.
fn build_fallback_summary(event: &SensorEvent) -> String {
    let filename = meta_str(event.metadata.as_ref(), "filename").unwrap_or("unknown");
    let extension = meta_str(event.metadata.as_ref(), "extension").unwrap_or("unknown");
    format!("Image file: {filename} ({extension})")
}

/// Build the SLM classification prompt from event metadata and content.
fn build_classification_prompt(event: &SensorEvent) -> String {
    let filename = meta_str(event.metadata.as_ref(), "filename").unwrap_or("unknown");
    let extension = meta_str(event.metadata.as_ref(), "extension").unwrap_or("unknown");
    let size = event
        .metadata
        .as_ref()
        .and_then(|m| m.get("size_bytes"))
        .and_then(|v| v.as_u64())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    format!(
        "Classify this image based on its metadata.\n\
         Filename: {filename}\n\
         Extension: {extension}\n\
         Size: {size} bytes\n\
         Content description: {}\n\n\
         Respond with JSON only:\n\
         {{\"category\": \"document|photo|screenshot|diagram|invoice|receipt|other\", \
         \"summary\": \"one sentence description\", \
         \"entities\": [\"entity1\", \"entity2\"], \
         \"is_document\": true/false}}",
        event.content
    )
}

/// System prompt for the image classification SLM.
const IMAGE_TRIAGE_SYSTEM_PROMPT: &str = "You are an image triage classifier. Based on file metadata (name, extension, size), \
     classify the image into a category and extract relevant entities. \
     Output valid JSON only, no markdown.";

impl TriageProcessor for ImageTriageProcessor {
    fn modality(&self) -> SensorModality {
        SensorModality::Image
    }

    fn source_topic(&self) -> &str {
        "hb.sensor.image"
    }

    fn process(
        &self,
        event: &SensorEvent,
    ) -> Pin<Box<dyn Future<Output = Result<TriageDecision, Error>> + Send + '_>> {
        let content = event.content.clone();
        let source_id = event.source_id.clone();
        let metadata = event.metadata.clone();
        let prompt = build_classification_prompt(event);
        let fallback_summary = build_fallback_summary(event);

        Box::pin(async move {
            let request = CompletionRequest {
                system: IMAGE_TRIAGE_SYSTEM_PROMPT.into(),
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
                Error::Sensor(format!("SLM image triage failed for {source_id}: {e}"))
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
                    // Fallback: use metadata-based classification
                    let filename = meta_str(metadata.as_ref(), "filename").unwrap_or("");
                    let extension = meta_str(metadata.as_ref(), "extension").unwrap_or("");
                    let estimated_tokens = content.len() / 4 + 256; // base overhead for image

                    return Ok(TriageDecision::Promote {
                        priority: extension_heuristic_priority(filename, extension),
                        summary: fallback_summary,
                        extracted_entities: vec![],
                        estimated_tokens,
                    });
                }
            };

            let category = parsed["category"].as_str().unwrap_or("other");
            let priority = category_to_priority(category);

            let summary = parsed["summary"]
                .as_str()
                .unwrap_or(&fallback_summary)
                .to_string();

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

/// Heuristic priority when SLM response cannot be parsed.
/// Uses filename keywords to guess category.
fn extension_heuristic_priority(filename: &str, _extension: &str) -> Priority {
    let lower = filename.to_lowercase();
    if lower.contains("invoice")
        || lower.contains("receipt")
        || lower.contains("document")
        || lower.contains("contract")
        || lower.contains("scan")
    {
        Priority::High
    } else {
        // Photos, screenshots, and unknown files all get Normal priority
        // when no SLM classification is available.
        Priority::Normal
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

    // --- Helpers ---

    fn make_image_event(filename: &str, extension: &str, size_bytes: u64) -> SensorEvent {
        let content = format!("Image: {filename} ({size_bytes} bytes, 2026-02-22T10:00:00Z)");
        let source_id = format!("/inbox/{filename}");
        SensorEvent {
            id: SensorEvent::generate_id(&content, &source_id),
            sensor_name: "scanner".into(),
            modality: SensorModality::Image,
            observed_at: chrono::Utc::now(),
            content,
            source_id: source_id.clone(),
            metadata: Some(serde_json::json!({
                "filename": filename,
                "extension": extension,
                "size_bytes": size_bytes,
                "modified_at": "2026-02-22T10:00:00+00:00",
            })),
            binary_ref: Some(source_id),
            related_ids: vec![],
        }
    }

    fn document_slm_json() -> String {
        serde_json::json!({
            "category": "document",
            "summary": "Scanned invoice document",
            "entities": ["Acme Corp", "billing"],
            "is_document": true
        })
        .to_string()
    }

    fn photo_slm_json() -> String {
        serde_json::json!({
            "category": "photo",
            "summary": "Outdoor landscape photo",
            "entities": ["nature"],
            "is_document": false
        })
        .to_string()
    }

    fn screenshot_slm_json() -> String {
        serde_json::json!({
            "category": "screenshot",
            "summary": "Desktop screenshot showing browser",
            "entities": ["browser", "code"],
            "is_document": false
        })
        .to_string()
    }

    fn other_slm_json() -> String {
        serde_json::json!({
            "category": "other",
            "summary": "Unclassified image",
            "entities": [],
            "is_document": false
        })
        .to_string()
    }

    fn diagram_slm_json() -> String {
        serde_json::json!({
            "category": "diagram",
            "summary": "Architecture diagram",
            "entities": ["microservices", "kafka"],
            "is_document": false
        })
        .to_string()
    }

    fn invoice_slm_json() -> String {
        serde_json::json!({
            "category": "invoice",
            "summary": "Invoice from supplier",
            "entities": ["supplier", "$1234.56"],
            "is_document": true
        })
        .to_string()
    }

    // --- Trait method tests ---

    #[test]
    fn image_triage_modality_is_image() {
        let processor = ImageTriageProcessor::new(Arc::new(MockProvider::with_json("{}")));
        assert_eq!(processor.modality(), SensorModality::Image);
    }

    #[test]
    fn image_triage_source_topic() {
        let processor = ImageTriageProcessor::new(Arc::new(MockProvider::with_json("{}")));
        assert_eq!(processor.source_topic(), "hb.sensor.image");
    }

    // --- category_to_priority unit tests ---

    #[test]
    fn category_document_is_high() {
        assert_eq!(category_to_priority("document"), Priority::High);
    }

    #[test]
    fn category_invoice_is_high() {
        assert_eq!(category_to_priority("invoice"), Priority::High);
    }

    #[test]
    fn category_receipt_is_high() {
        assert_eq!(category_to_priority("receipt"), Priority::High);
    }

    #[test]
    fn category_photo_is_normal() {
        assert_eq!(category_to_priority("photo"), Priority::Normal);
    }

    #[test]
    fn category_screenshot_is_normal() {
        assert_eq!(category_to_priority("screenshot"), Priority::Normal);
    }

    #[test]
    fn category_other_is_low() {
        assert_eq!(category_to_priority("other"), Priority::Low);
    }

    #[test]
    fn category_unknown_is_low() {
        assert_eq!(category_to_priority("wallpaper"), Priority::Low);
    }

    // --- extension_heuristic_priority tests ---

    #[test]
    fn heuristic_invoice_filename_is_high() {
        assert_eq!(
            extension_heuristic_priority("invoice_2026.jpg", "jpg"),
            Priority::High
        );
    }

    #[test]
    fn heuristic_receipt_filename_is_high() {
        assert_eq!(
            extension_heuristic_priority("receipt_grocery.png", "png"),
            Priority::High
        );
    }

    #[test]
    fn heuristic_screenshot_filename_is_normal() {
        assert_eq!(
            extension_heuristic_priority("screenshot_2026.png", "png"),
            Priority::Normal
        );
    }

    #[test]
    fn heuristic_generic_filename_is_normal() {
        assert_eq!(
            extension_heuristic_priority("IMG_20260222.jpg", "jpg"),
            Priority::Normal
        );
    }

    // --- build_fallback_summary tests ---

    #[test]
    fn build_fallback_summary_includes_filename() {
        let event = make_image_event("photo.jpg", "jpg", 1000);
        let summary = build_fallback_summary(&event);
        assert!(summary.contains("photo.jpg"), "summary: {summary}");
    }

    #[test]
    fn build_fallback_summary_includes_extension() {
        let event = make_image_event("chart.png", "png", 5000);
        let summary = build_fallback_summary(&event);
        assert!(summary.contains("png"), "summary: {summary}");
    }

    #[test]
    fn build_fallback_summary_no_metadata() {
        let mut event = make_image_event("test.jpg", "jpg", 100);
        event.metadata = None;
        let summary = build_fallback_summary(&event);
        assert!(summary.contains("unknown"), "summary: {summary}");
    }

    // --- build_classification_prompt tests ---

    #[test]
    fn classification_prompt_contains_filename() {
        let event = make_image_event("invoice.pdf.jpg", "jpg", 2048);
        let prompt = build_classification_prompt(&event);
        assert!(prompt.contains("invoice.pdf.jpg"), "prompt: {prompt}");
    }

    #[test]
    fn classification_prompt_contains_size() {
        let event = make_image_event("photo.png", "png", 99999);
        let prompt = build_classification_prompt(&event);
        assert!(prompt.contains("99999"), "prompt: {prompt}");
    }

    // --- Async process() tests ---

    #[tokio::test]
    async fn classify_document_image_high_priority() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&document_slm_json())));
        let event = make_image_event("invoice_scan.jpg", "jpg", 2_000_000);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::High);
        }
    }

    #[tokio::test]
    async fn classify_invoice_category_high_priority() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&invoice_slm_json())));
        let event = make_image_event("scan001.jpg", "jpg", 1_500_000);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::High);
        }
    }

    #[tokio::test]
    async fn classify_photo_normal_priority() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&photo_slm_json())));
        let event = make_image_event("IMG_20260222.jpg", "jpg", 5_000_000);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Normal);
        }
    }

    #[tokio::test]
    async fn classify_screenshot_normal_priority() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&screenshot_slm_json())));
        let event = make_image_event("screenshot_2026.png", "png", 800_000);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Normal);
        }
    }

    #[tokio::test]
    async fn classify_diagram_low_priority() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&diagram_slm_json())));
        let event = make_image_event("architecture.svg", "svg", 12_000);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Low);
        }
    }

    #[tokio::test]
    async fn classify_other_low_priority() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&other_slm_json())));
        let event = make_image_event("random.bmp", "bmp", 300_000);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(*priority, Priority::Low);
        }
    }

    #[tokio::test]
    async fn slm_parse_failure_fallback_to_normal() {
        // SLM returns invalid JSON
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json("not valid json {{")));
        let event = make_image_event("photo.jpg", "jpg", 1_000_000);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote {
            priority, summary, ..
        } = &decision
        {
            assert_eq!(*priority, Priority::Normal);
            assert!(
                summary.contains("photo.jpg"),
                "fallback summary should contain filename: {summary}"
            );
        }
    }

    #[tokio::test]
    async fn slm_parse_failure_invoice_filename_heuristic_high() {
        // SLM returns invalid JSON but filename contains "invoice"
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json("garbled output")));
        let event = make_image_event("invoice_march.jpg", "jpg", 2_000_000);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(
                *priority,
                Priority::High,
                "invoice in filename should trigger High via heuristic"
            );
        }
    }

    #[tokio::test]
    async fn slm_error_returns_sensor_error() {
        let processor = ImageTriageProcessor::new(Arc::new(FailingProvider));
        let event = make_image_event("photo.jpg", "jpg", 100);

        let result = processor.process(&event).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("SLM image triage failed"),
            "error: {err}"
        );
    }

    #[tokio::test]
    async fn entities_extracted_from_response() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&document_slm_json())));
        let event = make_image_event("invoice.jpg", "jpg", 500_000);

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            extracted_entities, ..
        } = &decision
        {
            assert_eq!(extracted_entities, &["Acme Corp", "billing"]);
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }

    #[tokio::test]
    async fn estimated_tokens_reasonable() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&photo_slm_json())));
        // Content is ~60 chars -> 60/4 + 256 = 271
        let event = make_image_event("landscape.jpg", "jpg", 3_000_000);

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote {
            estimated_tokens, ..
        } = &decision
        {
            assert!(
                *estimated_tokens > 0,
                "tokens should be positive, got {estimated_tokens}"
            );
            assert!(
                *estimated_tokens < 10_000,
                "tokens should be reasonable, got {estimated_tokens}"
            );
        } else {
            panic!("expected Promote");
        }
    }

    #[tokio::test]
    async fn summary_contains_meaningful_text() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&document_slm_json())));
        let event = make_image_event("invoice.jpg", "jpg", 500_000);

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote { summary, .. } = &decision {
            assert!(!summary.is_empty(), "summary should not be empty");
            assert!(
                summary.contains("invoice") || summary.contains("Scanned"),
                "summary should be descriptive: {summary}"
            );
        } else {
            panic!("expected Promote");
        }
    }

    #[tokio::test]
    async fn empty_content_handled_gracefully() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&photo_slm_json())));
        let mut event = make_image_event("empty.png", "png", 0);
        event.content = String::new();

        let decision = processor
            .process(&event)
            .await
            .expect("process should not fail");
        assert!(decision.is_promote());
    }

    #[tokio::test]
    async fn missing_metadata_handled_gracefully() {
        let processor =
            ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&photo_slm_json())));
        let mut event = make_image_event("test.jpg", "jpg", 1000);
        event.metadata = None;

        let decision = processor
            .process(&event)
            .await
            .expect("process should not fail");
        assert!(decision.is_promote());
    }

    #[tokio::test]
    async fn missing_category_defaults_to_low() {
        let json = serde_json::json!({
            "summary": "Some image",
            "entities": ["thing"],
            "is_document": false
        })
        .to_string();
        let processor = ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&json)));
        let event = make_image_event("test.jpg", "jpg", 1000);

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote { priority, .. } = &decision {
            assert_eq!(
                *priority,
                Priority::Low,
                "missing category should default to 'other' -> Low"
            );
        } else {
            panic!("expected Promote");
        }
    }

    // --- Serde roundtrip for decisions ---

    #[test]
    fn triage_decision_promote_serde_roundtrip() {
        let decision = TriageDecision::Promote {
            priority: Priority::High,
            summary: "Scanned document".into(),
            extracted_entities: vec!["invoice".into()],
            estimated_tokens: 300,
        };
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
            assert_eq!(summary, "Scanned document");
            assert_eq!(extracted_entities, vec!["invoice"]);
            assert_eq!(estimated_tokens, 300);
        }
    }

    #[test]
    fn triage_decision_drop_serde_roundtrip() {
        let decision = TriageDecision::Drop {
            reason: "duplicate image".into(),
        };
        let json = serde_json::to_string(&decision).expect("serialize");
        let back: TriageDecision = serde_json::from_str(&json).expect("deserialize");
        assert!(back.is_drop());
    }

    #[test]
    fn triage_decision_dead_letter_serde_roundtrip() {
        let decision = TriageDecision::DeadLetter {
            error: "corrupted file".into(),
        };
        let json = serde_json::to_string(&decision).expect("serialize");
        let back: TriageDecision = serde_json::from_str(&json).expect("deserialize");
        assert!(back.is_dead_letter());
    }

    // --- meta_str tests ---

    #[test]
    fn meta_str_extracts_field() {
        let meta = serde_json::json!({"filename": "test.jpg"});
        assert_eq!(meta_str(Some(&meta), "filename"), Some("test.jpg"));
    }

    #[test]
    fn meta_str_returns_none_for_missing() {
        let meta = serde_json::json!({"filename": "test.jpg"});
        assert_eq!(meta_str(Some(&meta), "extension"), None);
    }

    #[test]
    fn meta_str_returns_none_when_no_metadata() {
        assert_eq!(meta_str(None, "filename"), None);
    }

    #[tokio::test]
    async fn process_slm_returns_empty_text() {
        // SLM returns empty string → JSON parse fails → fallback heuristic
        let processor = ImageTriageProcessor::new(Arc::new(MockProvider::with_json("")));
        let event = make_image_event("vacation.jpg", "jpg", 3_000_000);

        let decision = processor.process(&event).await.expect("process failed");
        assert!(decision.is_promote());
        if let TriageDecision::Promote {
            priority, summary, ..
        } = &decision
        {
            // "vacation.jpg" doesn't contain invoice/receipt/document/contract/scan → Normal
            assert_eq!(*priority, Priority::Normal);
            assert!(
                summary.contains("vacation.jpg"),
                "fallback summary should contain filename: {summary}"
            );
        }
    }

    #[tokio::test]
    async fn process_no_metadata_uses_fallback_summary() {
        // SLM returns valid JSON but no "summary" field, metadata is None
        // → fallback_summary = "Image file: unknown (unknown)"
        let json = serde_json::json!({
            "category": "photo",
            "entities": ["nature"],
            "is_document": false
        })
        .to_string();
        let processor = ImageTriageProcessor::new(Arc::new(MockProvider::with_json(&json)));
        let mut event = make_image_event("test.jpg", "jpg", 1000);
        event.metadata = None;

        let decision = processor.process(&event).await.expect("process failed");
        if let TriageDecision::Promote { summary, .. } = &decision {
            assert_eq!(
                summary, "Image file: unknown (unknown)",
                "no metadata + missing summary should use fallback: {summary}"
            );
        } else {
            panic!("expected Promote, got {decision:?}");
        }
    }
}
