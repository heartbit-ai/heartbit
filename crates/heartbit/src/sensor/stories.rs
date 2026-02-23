use std::collections::{HashMap, HashSet};
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::sensor::triage::Priority;

/// The type of subject a story tracks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubjectType {
    /// A person (colleague, contact, family member).
    Person,
    /// A project or workstream.
    Project,
    /// A recurring topic or theme.
    Topic,
    /// An incident or urgent situation.
    Incident,
    /// A routine/habitual pattern.
    Routine,
}

/// Lifecycle status of a story.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoryStatus {
    /// Receiving events, actively tracked.
    Active,
    /// No recent events — may be winding down.
    Stale,
    /// Concluded or resolved.
    Resolved,
    /// Moved to long-term storage.
    Archived,
}

/// An event appended to a story's timeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoryEvent {
    /// The sensor event ID that triggered this story event.
    pub event_id: String,
    /// Name of the sensor that produced the event.
    pub sensor_name: String,
    /// SLM-generated summary of the event.
    pub summary: String,
    /// When this event was added to the story.
    pub added_at: DateTime<Utc>,
}

/// A story is a correlated thread of sensor events about a subject.
///
/// Stories are the central data structure for the perception layer.
/// They aggregate related events across sensors and time windows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Story {
    /// Unique story identifier.
    pub id: String,
    /// Human-readable subject line.
    pub subject: String,
    /// Type of subject this story tracks.
    pub subject_type: SubjectType,
    /// Current priority (highest of any contributing event).
    pub priority: Priority,
    /// Lifecycle status.
    pub status: StoryStatus,
    /// Timeline of contributing events.
    pub events: Vec<StoryEvent>,
    /// Union of extracted entities from all contributing events.
    pub entities: HashSet<String>,
    /// When the story was first created.
    pub created_at: DateTime<Utc>,
    /// When the story was last updated.
    pub updated_at: DateTime<Utc>,
    /// Optional parent story for hierarchical correlation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_story_id: Option<String>,
}

/// Correlates sensor events into stories based on entity overlap, thread links,
/// and temporal proximity.
pub struct StoryCorrelator {
    stories: HashMap<String, Story>,
    /// Maps sensor event IDs to story IDs for thread-based correlation.
    event_to_story: HashMap<String, String>,
    correlation_window: Duration,
    next_id: u64,
}

impl StoryCorrelator {
    /// Create a new correlator with the given temporal window.
    ///
    /// Events are correlated to a story only if the story was updated
    /// within `correlation_window` of the current time.
    pub fn new(correlation_window: Duration) -> Self {
        Self {
            stories: HashMap::new(),
            event_to_story: HashMap::new(),
            correlation_window,
            next_id: 1,
        }
    }

    /// Correlate an event into an existing story or create a new one.
    ///
    /// Returns the story ID (existing or newly created).
    ///
    /// Matching logic (in priority order):
    /// 1. **Thread match**: If `related_ids` reference events already in a story,
    ///    join that story (email threading via In-Reply-To/References).
    /// 2. **Entity overlap**: Find active stories within the correlation window
    ///    that share entities. Pick the best match (most overlap).
    /// 3. **New story**: If no match, create a new story.
    ///
    /// Person-based correlation works through entities: if the sender's email
    /// address is included as an entity, it will match stories about that person.
    pub fn correlate(
        &mut self,
        event_id: &str,
        sensor_name: &str,
        summary: &str,
        entities: &HashSet<String>,
        priority: Priority,
    ) -> String {
        self.correlate_with_links(event_id, sensor_name, summary, entities, priority, &[])
    }

    /// Like [`correlate`](Self::correlate) but with explicit thread links.
    ///
    /// `related_ids` are event IDs from the same thread (e.g., extracted from
    /// email In-Reply-To and References headers). If any related event is
    /// already tracked in a story, this event joins that story.
    pub fn correlate_with_links(
        &mut self,
        event_id: &str,
        sensor_name: &str,
        summary: &str,
        entities: &HashSet<String>,
        priority: Priority,
        related_ids: &[String],
    ) -> String {
        let now = Utc::now();
        let window_cutoff = now
            - chrono::Duration::from_std(self.correlation_window).unwrap_or(chrono::Duration::MAX);

        // Step 1: Thread-based match — check if any related event is in a story.
        let thread_match = related_ids
            .iter()
            .find_map(|rid| self.event_to_story.get(rid))
            .and_then(|sid| {
                let story = self.stories.get(sid)?;
                if story.status == StoryStatus::Active {
                    Some(sid.clone())
                } else {
                    None
                }
            });

        // Step 2: Entity overlap match (only if no thread match).
        let best_match = thread_match.or_else(|| {
            self.stories
                .iter()
                .filter(|(_, story)| story.status == StoryStatus::Active)
                .filter(|(_, story)| story.updated_at >= window_cutoff)
                .filter_map(|(id, story)| {
                    let overlap = story.entities.intersection(entities).count();
                    if overlap > 0 {
                        Some((id.clone(), overlap))
                    } else {
                        None
                    }
                })
                .max_by_key(|(_, overlap)| *overlap)
                .map(|(id, _)| id)
        });

        if let Some(story_id) = best_match {
            let story = self
                .stories
                .get_mut(&story_id)
                .expect("story exists — just found it in iterator");
            story.events.push(StoryEvent {
                event_id: event_id.into(),
                sensor_name: sensor_name.into(),
                summary: summary.into(),
                added_at: now,
            });
            story.entities = story.entities.union(entities).cloned().collect();
            if priority > story.priority {
                story.priority = priority;
            }
            story.updated_at = now;
            self.event_to_story
                .insert(event_id.to_string(), story_id.clone());
            story_id
        } else {
            let story_id = format!("story-{}", self.next_id);
            self.next_id += 1;

            // Infer subject type from entities and sensor name.
            let subject_type = infer_subject_type(sensor_name, entities);

            let story = Story {
                id: story_id.clone(),
                subject: summary.into(),
                subject_type,
                priority,
                status: StoryStatus::Active,
                events: vec![StoryEvent {
                    event_id: event_id.into(),
                    sensor_name: sensor_name.into(),
                    summary: summary.into(),
                    added_at: now,
                }],
                entities: entities.clone(),
                created_at: now,
                updated_at: now,
                parent_story_id: None,
            };
            self.stories.insert(story_id.clone(), story);
            self.event_to_story
                .insert(event_id.to_string(), story_id.clone());
            story_id
        }
    }

    /// Look up a story by ID.
    pub fn get_story(&self, id: &str) -> Option<&Story> {
        self.stories.get(id)
    }

    /// Return all stories with `Active` status.
    pub fn active_stories(&self) -> Vec<&Story> {
        self.stories
            .values()
            .filter(|s| s.status == StoryStatus::Active)
            .collect()
    }

    /// Mark stories with no events within `stale_threshold` as `Stale`,
    /// and prune the `event_to_story` index for non-active stories.
    pub fn mark_stale(&mut self, stale_threshold: Duration) {
        let cutoff = Utc::now()
            - chrono::Duration::from_std(stale_threshold).unwrap_or(chrono::Duration::MAX);
        for story in self.stories.values_mut() {
            if story.status == StoryStatus::Active && story.updated_at < cutoff {
                story.status = StoryStatus::Stale;
            }
        }
        // Prune event_to_story entries whose stories are no longer active.
        // This prevents unbounded growth of the index in long-running daemons.
        self.event_to_story.retain(|_, story_id| {
            self.stories
                .get(story_id)
                .is_some_and(|s| s.status == StoryStatus::Active)
        });
    }
}

/// Infer `SubjectType` from sensor name and entities.
///
/// Heuristic: email sensors with entity containing `@` → `Person`;
/// sensors with "incident" or critical priority in entities → `Incident`;
/// otherwise `Topic`.
fn infer_subject_type(sensor_name: &str, entities: &HashSet<String>) -> SubjectType {
    let has_email_entity = entities.iter().any(|e| e.contains('@'));
    if (sensor_name.contains("email") || sensor_name.contains("jmap")) && has_email_entity {
        return SubjectType::Person;
    }
    SubjectType::Topic
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subject_type_serde_roundtrip() {
        for st in [
            SubjectType::Person,
            SubjectType::Project,
            SubjectType::Topic,
            SubjectType::Incident,
            SubjectType::Routine,
        ] {
            let json = serde_json::to_string(&st).unwrap();
            let back: SubjectType = serde_json::from_str(&json).unwrap();
            assert_eq!(back, st);
        }
    }

    #[test]
    fn story_status_serde_roundtrip() {
        for ss in [
            StoryStatus::Active,
            StoryStatus::Stale,
            StoryStatus::Resolved,
            StoryStatus::Archived,
        ] {
            let json = serde_json::to_string(&ss).unwrap();
            let back: StoryStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, ss);
        }
    }

    #[test]
    fn story_serde_roundtrip() {
        let story = Story {
            id: "story-1".into(),
            subject: "Project update".into(),
            subject_type: SubjectType::Project,
            priority: Priority::Normal,
            status: StoryStatus::Active,
            events: vec![StoryEvent {
                event_id: "evt-1".into(),
                sensor_name: "work_email".into(),
                summary: "Email about project".into(),
                added_at: Utc::now(),
            }],
            entities: HashSet::from(["Alice".into(), "ProjectX".into()]),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            parent_story_id: None,
        };
        let json = serde_json::to_string(&story).unwrap();
        let back: Story = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "story-1");
        assert_eq!(back.subject_type, SubjectType::Project);
        assert_eq!(back.priority, Priority::Normal);
        assert_eq!(back.status, StoryStatus::Active);
        assert_eq!(back.events.len(), 1);
        assert!(back.parent_story_id.is_none());
    }

    #[test]
    fn story_parent_story_id_omitted_when_none() {
        let story = Story {
            id: "s1".into(),
            subject: "s".into(),
            subject_type: SubjectType::Topic,
            priority: Priority::Low,
            status: StoryStatus::Active,
            events: vec![],
            entities: HashSet::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
            parent_story_id: None,
        };
        let json = serde_json::to_string(&story).unwrap();
        assert!(!json.contains("parent_story_id"));
    }

    #[test]
    fn correlate_creates_new_story() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["Alice".into()]);
        let story_id = correlator.correlate(
            "evt-1",
            "email",
            "Meeting with Alice",
            &entities,
            Priority::Normal,
        );

        let story = correlator.get_story(&story_id).unwrap();
        assert_eq!(story.events.len(), 1);
        assert_eq!(story.events[0].event_id, "evt-1");
        assert_eq!(story.priority, Priority::Normal);
        assert!(story.entities.contains("Alice"));
    }

    #[test]
    fn correlate_merges_into_existing_story() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities1 = HashSet::from(["Alice".into(), "ProjectX".into()]);
        let id1 = correlator.correlate(
            "evt-1",
            "email",
            "Project kickoff",
            &entities1,
            Priority::Normal,
        );

        let entities2 = HashSet::from(["Alice".into(), "Budget".into()]);
        let id2 = correlator.correlate(
            "evt-2",
            "slack",
            "Budget discussion with Alice",
            &entities2,
            Priority::Normal,
        );

        // Should merge into the same story because of "Alice" overlap.
        assert_eq!(id1, id2);
        let story = correlator.get_story(&id1).unwrap();
        assert_eq!(story.events.len(), 2);
        // Entities should be the union.
        assert!(story.entities.contains("Alice"));
        assert!(story.entities.contains("ProjectX"));
        assert!(story.entities.contains("Budget"));
    }

    #[test]
    fn correlate_respects_temporal_window() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(1));
        let entities = HashSet::from(["Alice".into()]);
        let id1 =
            correlator.correlate("evt-1", "email", "First email", &entities, Priority::Normal);

        // Manually backdate the story's updated_at to simulate time passing.
        correlator.stories.get_mut(&id1).unwrap().updated_at =
            Utc::now() - chrono::Duration::seconds(10);

        let id2 = correlator.correlate(
            "evt-2",
            "email",
            "Second email",
            &entities,
            Priority::Normal,
        );

        // Should create a new story because the first is outside the window.
        assert_ne!(id1, id2);
    }

    #[test]
    fn correlate_bumps_priority() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["Server".into()]);
        let id1 =
            correlator.correlate("evt-1", "monitor", "CPU spike", &entities, Priority::Normal);
        let id2 = correlator.correlate(
            "evt-2",
            "monitor",
            "Server down!",
            &entities,
            Priority::Critical,
        );

        assert_eq!(id1, id2);
        let story = correlator.get_story(&id1).unwrap();
        assert_eq!(story.priority, Priority::Critical);
    }

    #[test]
    fn correlate_does_not_lower_priority() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["Server".into()]);
        correlator.correlate(
            "evt-1",
            "monitor",
            "Server down!",
            &entities,
            Priority::Critical,
        );
        let id = correlator.correlate("evt-2", "monitor", "Recovering", &entities, Priority::Low);

        let story = correlator.get_story(&id).unwrap();
        assert_eq!(story.priority, Priority::Critical);
    }

    #[test]
    fn correlate_no_overlap_creates_separate_stories() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities1 = HashSet::from(["Alice".into()]);
        let entities2 = HashSet::from(["Bob".into()]);
        let id1 = correlator.correlate(
            "evt-1",
            "email",
            "Alice's email",
            &entities1,
            Priority::Normal,
        );
        let id2 = correlator.correlate(
            "evt-2",
            "email",
            "Bob's email",
            &entities2,
            Priority::Normal,
        );

        assert_ne!(id1, id2);
    }

    #[test]
    fn mark_stale_marks_old_stories() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["Alice".into()]);
        let id = correlator.correlate("evt-1", "email", "Old email", &entities, Priority::Normal);

        // Backdate the story.
        correlator.stories.get_mut(&id).unwrap().updated_at =
            Utc::now() - chrono::Duration::hours(2);

        correlator.mark_stale(Duration::from_secs(3600));
        let story = correlator.get_story(&id).unwrap();
        assert_eq!(story.status, StoryStatus::Stale);
    }

    #[test]
    fn mark_stale_does_not_affect_recent_stories() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["Alice".into()]);
        let id = correlator.correlate(
            "evt-1",
            "email",
            "Recent email",
            &entities,
            Priority::Normal,
        );

        correlator.mark_stale(Duration::from_secs(3600));
        let story = correlator.get_story(&id).unwrap();
        assert_eq!(story.status, StoryStatus::Active);
    }

    #[test]
    fn active_stories_filters_correctly() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let e1 = HashSet::from(["Alice".into()]);
        let e2 = HashSet::from(["Bob".into()]);
        correlator.correlate("evt-1", "email", "Alice story", &e1, Priority::Normal);
        let id2 = correlator.correlate("evt-2", "email", "Bob story", &e2, Priority::Normal);

        // Manually mark one as stale.
        correlator.stories.get_mut(&id2).unwrap().status = StoryStatus::Stale;

        let active = correlator.active_stories();
        assert_eq!(active.len(), 1);
        assert!(active[0].entities.contains("Alice"));
    }

    #[test]
    fn mark_stale_ignores_non_active_stories() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["Alice".into()]);
        let id = correlator.correlate("evt-1", "email", "Old email", &entities, Priority::Normal);

        // Mark as resolved, then backdate.
        correlator.stories.get_mut(&id).unwrap().status = StoryStatus::Resolved;
        correlator.stories.get_mut(&id).unwrap().updated_at =
            Utc::now() - chrono::Duration::hours(2);

        correlator.mark_stale(Duration::from_secs(3600));
        let story = correlator.get_story(&id).unwrap();
        // Should still be Resolved, not Stale.
        assert_eq!(story.status, StoryStatus::Resolved);
    }

    // --- Thread-based correlation tests ---

    #[test]
    fn correlate_with_links_joins_thread() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities1 = HashSet::from(["alice@example.com".into()]);
        let id1 = correlator.correlate_with_links(
            "msg-001",
            "work_email",
            "Original email",
            &entities1,
            Priority::Normal,
            &[],
        );

        // Reply references the original message.
        let entities2 = HashSet::from(["bob@example.com".into()]);
        let id2 = correlator.correlate_with_links(
            "msg-002",
            "work_email",
            "Re: Original email",
            &entities2,
            Priority::Normal,
            &["msg-001".into()],
        );

        // Thread link should correlate to the same story even with different entities.
        assert_eq!(id1, id2);
        let story = correlator.get_story(&id1).unwrap();
        assert_eq!(story.events.len(), 2);
        // Both entity sets should be merged.
        assert!(story.entities.contains("alice@example.com"));
        assert!(story.entities.contains("bob@example.com"));
    }

    #[test]
    fn correlate_with_links_thread_takes_precedence_over_entities() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        // Story A: about Alice
        let entities_a = HashSet::from(["Alice".into()]);
        let id_a = correlator.correlate_with_links(
            "evt-a",
            "email",
            "Meeting with Alice",
            &entities_a,
            Priority::Normal,
            &[],
        );

        // Story B: about Bob (different, no overlap with A)
        let entities_b = HashSet::from(["Bob".into()]);
        let id_b = correlator.correlate_with_links(
            "evt-b",
            "email",
            "Meeting with Bob",
            &entities_b,
            Priority::Normal,
            &[],
        );
        assert_ne!(id_a, id_b);

        // New event has "Alice" entity (would match story A) but thread-links to story B.
        let entities_c = HashSet::from(["Alice".into()]);
        let id_c = correlator.correlate_with_links(
            "evt-c",
            "email",
            "Re: Meeting with Bob (cc Alice)",
            &entities_c,
            Priority::Normal,
            &["evt-b".into()],
        );

        // Thread link wins — joins story B, not A.
        assert_eq!(id_c, id_b);
    }

    #[test]
    fn correlate_with_links_unknown_references_fall_through() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["Alice".into()]);
        // References an event that doesn't exist yet.
        let id = correlator.correlate_with_links(
            "msg-003",
            "email",
            "Reply to unknown",
            &entities,
            Priority::Normal,
            &["nonexistent-id".into()],
        );

        // Should create a new story (no match found).
        let story = correlator.get_story(&id).unwrap();
        assert_eq!(story.events.len(), 1);
    }

    #[test]
    fn correlate_with_links_ignores_stale_story_threads() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["Alice".into()]);
        let id1 = correlator.correlate_with_links(
            "msg-001",
            "email",
            "Old thread",
            &entities,
            Priority::Normal,
            &[],
        );

        // Mark story as stale.
        correlator.stories.get_mut(&id1).unwrap().status = StoryStatus::Stale;

        // Reply references old thread in stale story.
        let id2 = correlator.correlate_with_links(
            "msg-002",
            "email",
            "Re: Old thread",
            &entities,
            Priority::Normal,
            &["msg-001".into()],
        );

        // Thread link to stale story should be ignored — creates new story.
        assert_ne!(id1, id2);
    }

    // --- Person-based correlation tests ---

    #[test]
    fn email_sensor_with_email_entity_creates_person_story() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["alice@acme.com".into(), "billing".into()]);
        let id = correlator.correlate_with_links(
            "msg-001",
            "work_email",
            "Invoice from Alice",
            &entities,
            Priority::Normal,
            &[],
        );

        let story = correlator.get_story(&id).unwrap();
        assert_eq!(story.subject_type, SubjectType::Person);
    }

    #[test]
    fn non_email_sensor_creates_topic_story() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["rust".into(), "ai".into()]);
        let id = correlator.correlate(
            "evt-1",
            "tech_rss",
            "Rust AI article",
            &entities,
            Priority::Low,
        );

        let story = correlator.get_story(&id).unwrap();
        assert_eq!(story.subject_type, SubjectType::Topic);
    }

    #[test]
    fn email_sensor_without_email_entity_creates_topic() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["billing".into(), "invoices".into()]);
        let id = correlator.correlate_with_links(
            "msg-001",
            "work_email",
            "General billing update",
            &entities,
            Priority::Normal,
            &[],
        );

        let story = correlator.get_story(&id).unwrap();
        assert_eq!(story.subject_type, SubjectType::Topic);
    }

    // --- Infer subject type tests ---

    #[test]
    fn infer_subject_type_email_with_address() {
        let entities = HashSet::from(["alice@example.com".into()]);
        assert_eq!(
            infer_subject_type("work_email", &entities),
            SubjectType::Person
        );
    }

    #[test]
    fn infer_subject_type_jmap_with_address() {
        let entities = HashSet::from(["bob@acme.com".into()]);
        assert_eq!(
            infer_subject_type("jmap_inbox", &entities),
            SubjectType::Person
        );
    }

    #[test]
    fn infer_subject_type_rss_always_topic() {
        let entities = HashSet::from(["alice@example.com".into()]);
        assert_eq!(
            infer_subject_type("tech_rss", &entities),
            SubjectType::Topic
        );
    }

    #[test]
    fn infer_subject_type_email_no_address() {
        let entities = HashSet::from(["billing".into()]);
        assert_eq!(
            infer_subject_type("work_email", &entities),
            SubjectType::Topic
        );
    }

    #[test]
    fn event_to_story_index_populated() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["Alice".into()]);
        let id = correlator.correlate("evt-1", "email", "Test", &entities, Priority::Normal);
        assert_eq!(correlator.event_to_story.get("evt-1"), Some(&id));
    }

    #[test]
    fn mark_stale_prunes_event_to_story_index() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let entities = HashSet::from(["Alice".into()]);
        let id = correlator.correlate("evt-1", "email", "Old email", &entities, Priority::Normal);

        // Verify the index has the entry.
        assert_eq!(correlator.event_to_story.len(), 1);

        // Backdate the story so mark_stale transitions it.
        correlator.stories.get_mut(&id).unwrap().updated_at =
            Utc::now() - chrono::Duration::hours(2);

        correlator.mark_stale(Duration::from_secs(3600));
        assert_eq!(
            correlator.get_story(&id).unwrap().status,
            StoryStatus::Stale
        );
        // Index entry should be pruned for non-active stories.
        assert!(correlator.event_to_story.is_empty());
    }

    #[test]
    fn mark_stale_keeps_active_story_index_entries() {
        let mut correlator = StoryCorrelator::new(Duration::from_secs(3600));
        let e1 = HashSet::from(["Alice".into()]);
        let e2 = HashSet::from(["Bob".into()]);

        // Create two stories: one old (will go stale), one recent (stays active).
        let id_old = correlator.correlate("evt-old", "email", "Old email", &e1, Priority::Normal);
        let _id_new = correlator.correlate("evt-new", "email", "New email", &e2, Priority::Normal);

        assert_eq!(correlator.event_to_story.len(), 2);

        // Only backdate the old story.
        correlator.stories.get_mut(&id_old).unwrap().updated_at =
            Utc::now() - chrono::Duration::hours(2);

        correlator.mark_stale(Duration::from_secs(3600));
        // Only the active story's event should remain in the index.
        assert_eq!(correlator.event_to_story.len(), 1);
        assert!(correlator.event_to_story.contains_key("evt-new"));
        assert!(!correlator.event_to_story.contains_key("evt-old"));
    }
}
