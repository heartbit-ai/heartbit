//! Memory lifecycle: store, recall, update, prune.
//!
//! Drives the [`InMemoryStore`] directly — no LLM call, no API key needed.
//!
//! `cargo run -p heartbit-core --example memory`

use chrono::{Duration, Utc};

use heartbit_core::memory::Confidentiality;
use heartbit_core::{InMemoryStore, Memory, MemoryEntry, MemoryQuery, MemoryType};

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    let store = InMemoryStore::new();

    // 1. Store an episodic entry.
    let entry = MemoryEntry {
        id: "m1".into(),
        agent: "assistant".into(),
        content: "The user prefers terse responses without preamble.".into(),
        category: "preference".into(),
        tags: vec!["style".into()],
        created_at: Utc::now(),
        last_accessed: Utc::now(),
        access_count: 0,
        importance: 7,
        memory_type: MemoryType::Episodic,
        keywords: vec!["terse".into(), "preference".into()],
        summary: Some("User wants short answers.".into()),
        strength: 1.0,
        related_ids: vec![],
        source_ids: vec![],
        embedding: None,
        confidentiality: Confidentiality::Internal,
        author_user_id: None,
        author_tenant_id: None,
    };
    store.store(entry).await?;

    // 2. Recall by keyword. Composite scoring orders the result set by
    //    recency + importance + relevance + strength.
    let hits = store
        .recall(MemoryQuery {
            agent: Some("assistant".into()),
            text: Some("preference".into()),
            limit: 5,
            ..MemoryQuery::default()
        })
        .await?;
    println!("recalled {} entry/entries", hits.len());

    // 3. Update the content in place. The entry id is preserved.
    store
        .update("m1", "User prefers terse, formal responses.".into())
        .await?;

    // 4. Prune anything older than 1 hour with strength below 0.1. The
    //    fresh entry above is far above that threshold, so this is a
    //    no-op here — but it's the call you'd schedule periodically in a
    //    long-running deployment.
    let pruned = store.prune(0.1, Duration::hours(1), None).await?;
    println!("pruned {pruned} weak entries");

    Ok(())
}
