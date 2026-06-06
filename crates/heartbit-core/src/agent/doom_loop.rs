use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use crate::llm::types::ToolCall;

/// Tracks consecutive identical tool-call turns to detect doom loops.
///
/// Each turn's tool calls are hashed as a sorted set of `(name, input_json)` pairs.
/// When N consecutive turns produce the same hash, the tracker signals a doom loop.
pub(super) struct DoomLoopTracker {
    /// Hash of the previous turn's tool calls, and its consecutive count.
    last_hash: Option<u64>,
    count: u32,
    /// Tracks fuzzy matches (same tool names, different inputs).
    last_names_hash: Option<u64>,
    fuzzy_count: u32,
}

impl DoomLoopTracker {
    pub(super) fn new() -> Self {
        Self {
            last_hash: None,
            count: 0,
            last_names_hash: None,
            fuzzy_count: 0,
        }
    }

    /// Consecutive exact-match count.
    pub(super) fn count(&self) -> u32 {
        self.count
    }

    /// Consecutive fuzzy-match count (same tool names, different inputs).
    pub(super) fn fuzzy_count(&self) -> u32 {
        self.fuzzy_count
    }

    /// Hash a set of tool calls for the current turn. Tool calls are sorted by
    /// name so that ordering differences don't produce different hashes.
    fn hash_tool_calls(calls: &[ToolCall]) -> u64 {
        let mut sorted: Vec<(String, String)> = calls
            .iter()
            .map(|tc| (tc.name.clone(), tc.input.to_string()))
            .collect();
        sorted.sort();
        let mut hasher = DefaultHasher::new();
        for (name, input) in &sorted {
            name.hash(&mut hasher);
            input.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Hash only the tool names (sorted) for fuzzy matching.
    fn hash_tool_names(calls: &[ToolCall]) -> u64 {
        let mut names: Vec<&str> = calls.iter().map(|tc| tc.name.as_str()).collect();
        names.sort();
        let mut hasher = DefaultHasher::new();
        for name in &names {
            name.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Record the current turn's tool calls. Returns `(exact_match, fuzzy_match)`.
    /// `exact_match` is true if exact same calls repeated >= threshold times.
    /// `fuzzy_match` is true if same tool names (different inputs) repeated >= fuzzy_threshold times.
    pub(super) fn record(
        &mut self,
        calls: &[ToolCall],
        threshold: u32,
        fuzzy_threshold: Option<u32>,
    ) -> (bool, bool) {
        let hash = Self::hash_tool_calls(calls);
        match self.last_hash {
            Some(prev) if prev == hash => {
                self.count += 1;
            }
            _ => {
                self.last_hash = Some(hash);
                self.count = 1;
            }
        }

        // Fuzzy tracking: same tool names, possibly different inputs
        if let Some(_ft) = fuzzy_threshold {
            let names_hash = Self::hash_tool_names(calls);
            match self.last_names_hash {
                Some(prev) if prev == names_hash => {
                    // Only count fuzzy if NOT an exact match (avoid double-counting)
                    if self.count < threshold {
                        self.fuzzy_count += 1;
                    }
                }
                _ => {
                    self.last_names_hash = Some(names_hash);
                    self.fuzzy_count = 1;
                }
            }
        }

        let exact = self.count >= threshold;
        // Fuzzy only fires when exact does not (avoid double-counting)
        let fuzzy = !exact && fuzzy_threshold.is_some_and(|ft| self.fuzzy_count >= ft);
        (exact, fuzzy)
    }

    /// Inform the tracker of the executed batch's outcome. A batch where every
    /// call succeeded resets the FUZZY state: consecutive same-tool batches
    /// that keep succeeding are normal sequential work (e.g. writing N files
    /// looks like N consecutive `[write]` batches — live /analyze finding),
    /// not a loop. The exact-match counter is deliberately left untouched:
    /// byte-identical repeats are suspicious even when they "succeed" (true
    /// no-progress loops often do).
    pub(super) fn note_batch_outcome(&mut self, any_error: bool) {
        if !any_error {
            self.last_names_hash = None;
            self.fuzzy_count = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(name: &str, input: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "tc".into(),
            name: name.into(),
            input,
        }
    }

    // Live /analyze finding: the fuzzy counter climbed across SUCCESSFUL
    // writes — sequential same-tool batches (writing N files) are normal
    // work, not a loop. A clean batch must reset the fuzzy state.
    #[test]
    fn fuzzy_resets_after_a_clean_batch() {
        let mut t = DoomLoopTracker::new();
        for i in 0..10 {
            let batch = [call("write", json!({"file_path": format!("f{i}.md")}))];
            let (exact, fuzzy) = t.record(&batch, 3, Some(3));
            assert!(!exact && !fuzzy, "clean sequential writes must never trip");
            t.note_batch_outcome(false); // batch succeeded
        }
        assert_eq!(t.fuzzy_count(), 0, "clean batches reset the fuzzy count");
    }

    #[test]
    fn fuzzy_still_fires_on_consecutive_erroring_batches() {
        let mut t = DoomLoopTracker::new();
        let mut fired = false;
        for i in 0..3 {
            let batch = [call("write", json!({"attempt": i}))];
            let (_, fuzzy) = t.record(&batch, 10, Some(3));
            fired = fuzzy;
            t.note_batch_outcome(true); // batch errored
        }
        assert!(
            fired,
            "3 consecutive erroring same-name batches must trip fuzzy"
        );
    }

    #[test]
    fn exact_counter_is_not_reset_by_clean_batches() {
        // Byte-identical consecutive batches stay suspicious even when they
        // succeed (true no-progress loops often "succeed"); only the fuzzy
        // counter is outcome-aware.
        let mut t = DoomLoopTracker::new();
        let mut fired = false;
        for _ in 0..3 {
            let batch = [call("bash", json!({"command": "echo same"}))];
            let (exact, _) = t.record(&batch, 3, Some(99));
            fired = exact;
            t.note_batch_outcome(false);
        }
        assert!(
            fired,
            "identical batches must still trip exact at threshold"
        );
    }
}
