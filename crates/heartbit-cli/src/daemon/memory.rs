/// Stop words excluded from keyword extraction for institutional memory entries.
pub(crate) const STOP_WORDS: &[&str] = &[
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall", "can", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "as", "into", "through", "about", "and",
    "but", "or", "nor", "not", "so", "yet", "both", "either", "neither", "each", "every", "it",
    "its", "this", "that", "these", "those", "i", "we", "you", "he", "she", "they", "my", "our",
    "your", "his", "her", "their", "me", "us", "him", "them",
];

/// Extract keywords from a source tag and result text for institutional memory.
pub(crate) fn extract_keywords(source: &str, result: &str) -> Vec<String> {
    let mut keywords = Vec::new();

    // Add source components (e.g. "sensor:rss" -> ["sensor", "rss"])
    for part in source.split(':') {
        let lower = part.to_lowercase();
        if !lower.is_empty() && !STOP_WORDS.contains(&lower.as_str()) {
            keywords.push(lower);
        }
    }

    // Extract significant words from first 200 chars of result
    let prefix = if result.len() > 200 {
        &result[..result.floor_char_boundary(200)]
    } else {
        result
    };
    for word in prefix.split_whitespace().take(30) {
        let clean: String = word
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '-')
            .collect();
        let lower = clean.to_lowercase();
        if lower.len() >= 3 && !STOP_WORDS.contains(&lower.as_str()) && !keywords.contains(&lower) {
            keywords.push(lower);
        }
    }

    keywords
}

/// Build a `MemoryEntry` for persisting a task result to the institutional namespace.
pub(crate) fn build_institutional_entry(
    task_id: &uuid::Uuid,
    source: &str,
    result: &str,
    story_id: Option<&str>,
    user_id: Option<&str>,
    tenant_id: Option<&str>,
) -> heartbit::MemoryEntry {
    // Truncate long results to 2000 chars
    let content = if result.len() > 2000 {
        let boundary = result.floor_char_boundary(2000);
        &result[..boundary]
    } else {
        result
    };

    let mut tags = vec![source.to_string()];
    if let Some(sid) = story_id {
        tags.push(sid.to_string());
    }

    let story_label = story_id.unwrap_or("no story");
    let summary = format!("Task result from {source} ({story_label})");

    heartbit::MemoryEntry {
        id: format!("institutional:{task_id}"),
        agent: "institutional".into(),
        content: content.to_string(),
        category: "task_result".into(),
        tags,
        created_at: chrono::Utc::now(),
        last_accessed: chrono::Utc::now(),
        access_count: 0,
        importance: 7,
        memory_type: heartbit::MemoryType::Episodic,
        keywords: extract_keywords(source, result),
        summary: Some(summary),
        strength: 1.0,
        related_ids: Vec::new(),
        source_ids: Vec::new(),
        embedding: None,
        confidentiality: heartbit::Confidentiality::Internal,
        author_user_id: user_id.map(String::from),
        author_tenant_id: tenant_id.map(String::from),
    }
}

#[cfg(test)]
mod institutional_memory_tests {
    use std::sync::Arc;

    use super::*;

    #[test]
    fn extract_keywords_from_source() {
        let kw = extract_keywords("sensor:rss", "");
        assert!(kw.contains(&"sensor".to_string()));
        assert!(kw.contains(&"rss".to_string()));
    }

    #[test]
    fn extract_keywords_from_result() {
        let kw = extract_keywords(
            "api",
            "Claude 4.6 released with improved reasoning capabilities",
        );
        assert!(kw.contains(&"api".to_string()));
        assert!(kw.contains(&"claude".to_string()));
        assert!(kw.contains(&"released".to_string()));
        assert!(kw.contains(&"reasoning".to_string()));
        // Stop words should be excluded
        assert!(!kw.contains(&"with".to_string()));
    }

    #[test]
    fn extract_keywords_deduplicates() {
        let kw = extract_keywords("rss", "rss feed from rss source");
        let rss_count = kw.iter().filter(|w| *w == "rss").count();
        assert_eq!(rss_count, 1, "keywords should be deduplicated");
    }

    #[test]
    fn extract_keywords_filters_short_words() {
        let kw = extract_keywords("api", "an AI ML tool");
        // "an" is a stop word; "ai" and "ml" are < 3 chars
        assert!(!kw.contains(&"an".to_string()));
        assert!(!kw.contains(&"ai".to_string()));
        assert!(!kw.contains(&"ml".to_string()));
    }

    #[test]
    fn extract_keywords_handles_double_colons() {
        // "sensor::rss" splits into ["sensor", "", "rss"]
        let kw = extract_keywords("sensor::rss", "");
        assert!(kw.contains(&"sensor".to_string()));
        assert!(kw.contains(&"rss".to_string()));
        assert!(
            !kw.contains(&String::new()),
            "empty parts should be filtered"
        );
    }

    #[test]
    fn extract_keywords_empty_inputs() {
        let kw = extract_keywords("", "");
        assert!(kw.is_empty());
    }

    #[test]
    fn extract_keywords_unicode_source() {
        let kw = extract_keywords("sensor:cafe\u{301}", "");
        assert!(kw.contains(&"sensor".to_string()));
        assert!(kw.contains(&"cafe\u{301}".to_string()));
    }

    #[test]
    fn build_institutional_entry_empty_result_produces_empty_content() {
        let id = uuid::Uuid::new_v4();
        let entry = build_institutional_entry(&id, "api", "", None, None, None);
        assert!(entry.content.is_empty());
        // The on_complete guard (result.is_empty()) prevents this from being stored,
        // but verify the builder doesn't panic on empty input.
    }

    #[test]
    fn build_institutional_entry_basic() {
        let id = uuid::Uuid::nil();
        let entry =
            build_institutional_entry(&id, "sensor:rss", "Some analysis result", None, None, None);

        assert_eq!(entry.id, format!("institutional:{}", uuid::Uuid::nil()));
        assert_eq!(entry.agent, "institutional");
        assert_eq!(entry.content, "Some analysis result");
        assert_eq!(entry.category, "task_result");
        assert_eq!(entry.importance, 7);
        assert_eq!(entry.strength, 1.0);
        assert_eq!(entry.memory_type, heartbit::MemoryType::Episodic);
        assert_eq!(entry.confidentiality, heartbit::Confidentiality::Internal);
        assert!(entry.tags.contains(&"sensor:rss".to_string()));
        assert!(entry.summary.as_ref().unwrap().contains("sensor:rss"));
        assert!(entry.summary.as_ref().unwrap().contains("no story"));
    }

    #[test]
    fn build_institutional_entry_with_story_id() {
        let id = uuid::Uuid::new_v4();
        let entry = build_institutional_entry(
            &id,
            "sensor:rss",
            "Result text",
            Some("story-abc"),
            None,
            None,
        );

        assert!(entry.tags.contains(&"story-abc".to_string()));
        assert!(entry.summary.as_ref().unwrap().contains("story-abc"));
    }

    #[test]
    fn build_institutional_entry_truncates_long_result() {
        let long_result = "x".repeat(5000);
        let id = uuid::Uuid::new_v4();
        let entry = build_institutional_entry(&id, "api", &long_result, None, None, None);

        assert!(
            entry.content.len() <= 2000,
            "content should be truncated to 2000 chars, got {}",
            entry.content.len()
        );
    }

    #[test]
    fn build_institutional_entry_idempotent_id() {
        let id = uuid::Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").unwrap();
        let entry1 = build_institutional_entry(&id, "api", "result", None, None, None);
        let entry2 = build_institutional_entry(&id, "api", "result", None, None, None);

        assert_eq!(
            entry1.id, entry2.id,
            "same task ID should produce same entry ID"
        );
        assert_eq!(
            entry1.id,
            "institutional:550e8400-e29b-41d4-a716-446655440000"
        );
    }

    #[tokio::test]
    async fn on_complete_persists_to_institutional_memory() {
        let store = Arc::new(heartbit::InMemoryStore::new());

        let memory: Arc<dyn heartbit::Memory> = store.clone();
        let entry = build_institutional_entry(
            &uuid::Uuid::nil(),
            "sensor:rss",
            "Analysis: Claude 4.6 has new features",
            Some("story-123"),
            None,
            None,
        );

        let scope = heartbit::TenantScope::default();
        memory.store(&scope, entry).await.unwrap();

        // Recall from institutional namespace
        let query = heartbit::MemoryQuery {
            text: Some("Claude features".to_string()),
            agent_prefix: Some("institutional".to_string()),
            limit: 5,
            ..Default::default()
        };
        let results = memory.recall(&scope, query).await.unwrap();
        assert!(!results.is_empty(), "should find institutional memory");
        assert!(results[0].content.contains("Claude 4.6"));
        assert_eq!(results[0].agent, "institutional");
    }
}
