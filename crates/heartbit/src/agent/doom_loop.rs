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
}
