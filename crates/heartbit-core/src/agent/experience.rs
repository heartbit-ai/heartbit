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
