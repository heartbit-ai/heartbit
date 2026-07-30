//! Experience-stage memory: trajectory store + skill distillation.
//!
//! The 2026 agent-memory arc is Storage → Reflection → **Experience** (surveys
//! arXiv 2605.06716): the frontier is *abstracting* successful trajectories into
//! reusable procedures the agent can recall when it meets a similar task — the
//! self-improvement flywheel (Voyager-style skill acquisition). heartbit already
//! has storage + reflection (memory consolidation, `/learn` lessons) but skills
//! are authored by hand; this module adds the automatic experience loop.
//!
//! [`TrajectoryStore`] records `(task, actions, outcome)` from completed runs;
//! [`TrajectoryStore::skill_hint`] recalls the most-similar **successful** past
//! trajectory, distilled into an injectable procedure ([`distill_procedure`]) the
//! caller can prepend to a new task's prompt as procedural memory.

use std::sync::RwLock;

/// One recorded run: what was asked, the ordered actions taken, and the outcome.
#[derive(Debug, Clone)]
pub struct Trajectory {
    /// The task the agent was given.
    pub task: String,
    /// Ordered actions taken (tool names or short step descriptions).
    pub actions: Vec<String>,
    /// Whether the run succeeded (only successful trajectories are recalled).
    pub success: bool,
    /// The final result/answer (for context in the distilled procedure).
    pub result: String,
}

/// Distill a (typically successful) trajectory into a reusable procedure the
/// caller can inject as procedural memory before a similar task.
pub fn distill_procedure(t: &Trajectory) -> String {
    let steps = t
        .actions
        .iter()
        .enumerate()
        .map(|(i, a)| format!("  {}. {a}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Learned procedure — a task like \"{}\" was solved before with these steps:\n{steps}",
        t.task
    )
}

/// Very common words that carry no task signal — filtered before similarity so
/// "write **a** scraper" and "bake **a** cake" aren't judged similar by "a".
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "to", "for", "of", "and", "or", "in", "on", "at", "with", "by", "from", "as",
    "is", "are", "be", "it", "this", "that", "my", "me", "i", "you", "do", "please", "can", "will",
];

/// Lowercase alphanumeric word tokens (stopwords removed), de-duplicated, for
/// similarity scoring.
fn tokens(s: &str) -> std::collections::HashSet<String> {
    s.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .trim_matches(|c: char| !c.is_alphanumeric())
                .to_string()
        })
        .filter(|w| !w.is_empty() && !STOPWORDS.contains(&w.as_str()))
        .collect()
}

/// Jaccard similarity of two token sets, in `[0, 1]`.
fn jaccard(a: &std::collections::HashSet<String>, b: &std::collections::HashSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count() as f64;
    let union = a.union(b).count() as f64;
    inter / union
}

/// An in-memory store of completed trajectories, capacity-bounded (drop-oldest).
/// Recall is keyword-similarity over the task text, successful trajectories only.
pub struct TrajectoryStore {
    trajectories: RwLock<Vec<Trajectory>>,
    capacity: usize,
}

impl TrajectoryStore {
    /// Store up to `capacity` trajectories (oldest evicted past the cap).
    pub fn new(capacity: usize) -> Self {
        Self {
            trajectories: RwLock::new(Vec::new()),
            capacity: capacity.max(1),
        }
    }

    /// Record a completed run.
    pub fn record(&self, trajectory: Trajectory) {
        let mut g = self
            .trajectories
            .write()
            .expect("trajectory store poisoned");
        g.push(trajectory);
        // Drop-oldest past the cap.
        let overflow = g.len().saturating_sub(self.capacity);
        if overflow > 0 {
            g.drain(0..overflow);
        }
    }

    /// Number of stored trajectories.
    pub fn len(&self) -> usize {
        self.trajectories.read().expect("poisoned").len()
    }

    /// True when no trajectories are stored.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The `k` most task-similar **successful** trajectories, ranked best-first.
    /// Only matches with non-zero similarity are returned.
    pub fn recall_similar(&self, task: &str, k: usize) -> Vec<Trajectory> {
        let query = tokens(task);
        let g = self.trajectories.read().expect("poisoned");
        let mut scored: Vec<(f64, &Trajectory)> = g
            .iter()
            .filter(|t| t.success)
            .map(|t| (jaccard(&query, &tokens(&t.task)), t))
            .filter(|(s, _)| *s > 0.0)
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(k).map(|(_, t)| t.clone()).collect()
    }

    /// The single most-similar successful procedure, distilled and ready to inject
    /// as procedural memory, or `None` when nothing relevant was learned yet.
    pub fn skill_hint(&self, task: &str) -> Option<String> {
        self.recall_similar(task, 1).first().map(distill_procedure)
    }
}

/// Run `runner` on `task` with the experience loop closed: prime the task with
/// any learned [`skill_hint`](TrajectoryStore::skill_hint) from `store`
/// (procedural memory from a similar past run), then record the resulting
/// trajectory back into `store`. Each call can learn from prior successful ones —
/// the self-improvement flywheel as a single runtime entry point.
///
/// A run is recorded as successful unless it errored or its goal judge returned
/// `Some(false)`.
pub async fn run_with_experience<P>(
    runner: &super::AgentRunner<P>,
    store: &TrajectoryStore,
    task: &str,
) -> Result<super::AgentOutput, crate::error::Error>
where
    P: crate::llm::LlmProvider + 'static,
{
    // Prime with a learned procedure for a similar past task, if any.
    let primed = match store.skill_hint(task) {
        Some(hint) => format!("{hint}\n\n---\nNow do this task:\n{task}"),
        None => task.to_string(),
    };
    let output = runner.execute(&primed).await;
    // Record the trajectory (using the ORIGINAL task for future similarity).
    let trajectory = match &output {
        Ok(o) => Trajectory {
            task: task.to_string(),
            actions: o
                .tool_call_results
                .iter()
                .map(|r| r.tool_name.clone())
                .collect(),
            success: o.goal_met != Some(false),
            result: o.result.clone(),
        },
        Err(_) => Trajectory {
            task: task.to_string(),
            actions: Vec::new(),
            success: false,
            result: String::new(),
        },
    };
    store.record(trajectory);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj(task: &str, actions: &[&str], success: bool) -> Trajectory {
        Trajectory {
            task: task.into(),
            actions: actions.iter().map(|s| s.to_string()).collect(),
            success,
            result: "ok".into(),
        }
    }

    #[test]
    fn distill_numbers_the_steps() {
        let t = traj("deploy the service", &["build", "test", "ship"], true);
        let p = distill_procedure(&t);
        assert!(p.contains("1. build"));
        assert!(p.contains("3. ship"));
        assert!(p.contains("deploy the service"));
    }

    #[test]
    fn recall_returns_most_similar_successful() {
        let store = TrajectoryStore::new(100);
        store.record(traj(
            "write a python web scraper",
            &["plan", "code", "run"],
            true,
        ));
        store.record(traj("bake a chocolate cake", &["mix", "bake"], true));
        let hits = store.recall_similar("write a python scraper for a site", 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].task.contains("scraper"));
    }

    #[test]
    fn recall_ignores_failed_trajectories() {
        let store = TrajectoryStore::new(100);
        store.record(traj("write a python web scraper", &["plan", "code"], false));
        let hits = store.recall_similar("write a python scraper", 5);
        assert!(hits.is_empty(), "failed trajectories must not be recalled");
    }

    #[test]
    fn skill_hint_distills_top_match() {
        let store = TrajectoryStore::new(100);
        store.record(traj(
            "summarize a long PDF document",
            &["read", "chunk", "summarize"],
            true,
        ));
        let hint = store.skill_hint("summarize a PDF document for me").unwrap();
        assert!(hint.contains("Learned procedure"));
        assert!(hint.contains("chunk"));
    }

    #[test]
    fn skill_hint_none_when_unrelated() {
        let store = TrajectoryStore::new(100);
        store.record(traj("bake a cake", &["mix"], true));
        assert!(store.skill_hint("debug a rust compiler error").is_none());
    }

    #[tokio::test]
    async fn run_with_experience_records_then_primes_next_run() {
        use crate::agent::test_helpers::{MockProvider, make_agent};
        use std::sync::Arc;

        let store = TrajectoryStore::new(100);

        // First run: no prior experience → task sent verbatim, then recorded.
        let p1 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "scraped the site",
            5,
            5,
        )]));
        let runner1 = make_agent(Arc::clone(&p1), "a");
        run_with_experience(&runner1, &store, "scrape a website for prices")
            .await
            .unwrap();
        assert_eq!(store.len(), 1);
        // The first task was NOT primed (no prior experience). Scope the guard so
        // it drops before the next await.
        let user1 = {
            let req1 = p1.captured_requests.lock().unwrap();
            req1[0].messages[0]
                .content
                .iter()
                .find_map(|b| match b {
                    crate::llm::types::ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .unwrap()
        };
        assert!(!user1.contains("Learned procedure"));

        // Second, similar run: the learned procedure primes the task.
        let p2 = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            "ok", 5, 5,
        )]));
        let runner2 = make_agent(Arc::clone(&p2), "b");
        run_with_experience(&runner2, &store, "scrape a website for product data")
            .await
            .unwrap();
        let req2 = p2.captured_requests.lock().unwrap();
        let user2 = req2[0].messages[0]
            .content
            .iter()
            .find_map(|b| match b {
                crate::llm::types::ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .unwrap();
        assert!(
            user2.contains("Learned procedure"),
            "the second similar run must be primed with the learned procedure: {user2}"
        );
        assert_eq!(store.len(), 2);
    }

    // ── Frontier invariant #6 (no cross-session leak) ──
    // A trajectory recorded in one session's store is NEVER recalled through
    // another store. Recall is scoped to the instance, so a second session cannot
    // read the first session's task text, actions or results. This is a standing
    // regression guard: making the store a process-global static would break it.
    #[test]
    fn trajectories_never_leak_across_stores() {
        let session_a = TrajectoryStore::new(100);
        let session_b = TrajectoryStore::new(100);

        session_a.record(Trajectory {
            task: "deploy the acme billing service".into(),
            actions: vec!["read_secret".into(), "kubectl_apply".into()],
            success: true,
            result: "CONFIDENTIAL-A: deployed with token sk-live-a1b2".into(),
        });

        assert_eq!(session_a.len(), 1);
        assert_eq!(session_b.len(), 0, "a fresh session starts empty");

        // The other session recalls NOTHING for the very same task text…
        assert!(
            session_b
                .recall_similar("deploy the acme billing service", 5)
                .is_empty(),
            "session B must not see session A's trajectory"
        );
        // …and gets no distilled procedure either (so nothing leaks into a prompt).
        assert!(
            session_b
                .skill_hint("deploy the acme billing service")
                .is_none(),
            "no cross-session skill hint may be produced"
        );
        // Sanity: the owning session DOES recall it (the test would pass vacuously
        // if recall were simply broken).
        let hint = session_a
            .skill_hint("deploy the acme billing service")
            .expect("the owning session recalls its own trajectory");
        assert!(hint.contains("kubectl_apply"));
    }

    #[test]
    fn capacity_evicts_oldest() {
        let store = TrajectoryStore::new(2);
        // Distinct content words so similarity doesn't cross-match.
        store.record(traj("alpha gizmo", &["a"], true));
        store.record(traj("beta widget", &["b"], true));
        store.record(traj("gamma sprocket", &["c"], true));
        assert_eq!(store.len(), 2);
        // "alpha gizmo" was evicted (oldest).
        assert!(store.recall_similar("alpha gizmo", 5).is_empty());
        assert_eq!(store.recall_similar("gamma sprocket", 5).len(), 1);
    }
}
