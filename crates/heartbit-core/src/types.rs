use serde::{Deserialize, Serialize};

/// Dispatch mode for orchestrator delegation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DispatchMode {
    /// All delegated tasks run in parallel via JoinSet (default).
    Parallel,
    /// One task at a time. Schema constrains `maxItems: 1` on delegate_task.
    Sequential,
}

/// Trust classification for the sender of an external message.
///
/// Resolved deterministically from config lists — never LLM-based.
/// Ordered from least to most trusted; `PartialOrd`/`Ord` follow declaration order.
///
/// Defined in core types so it's always available for TOML/JSON deserialization
/// even when the `sensor` feature is disabled. Re-exported from
/// `heartbit::config` and `heartbit::sensor::triage::context`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum TrustLevel {
    /// Explicitly blocked sender. Zero action permitted.
    Quarantined,
    /// No prior relationship. Read-only access.
    #[default]
    Unknown,
    /// Recognized but not privileged.
    Known,
    /// In the priority senders list. May trigger replies (with approval).
    Verified,
    /// The system owner. Full access.
    Owner,
}

impl TrustLevel {
    /// Resolve trust level from sender email against config lists.
    ///
    /// Priority: Owner > Blocked(Quarantined) > Priority(Verified) > Unknown.
    /// Matching is case-insensitive.
    pub fn resolve(
        sender: Option<&str>,
        owner_emails: &[String],
        priority_senders: &[String],
        blocked_senders: &[String],
    ) -> Self {
        let sender = match sender {
            Some(s) if !s.trim().is_empty() => s.trim(),
            _ => return TrustLevel::Unknown,
        };
        let lower = sender.to_lowercase();

        if owner_emails
            .iter()
            .any(|e| e.trim().to_lowercase() == lower)
        {
            return TrustLevel::Owner;
        }
        if blocked_senders
            .iter()
            .any(|e| e.trim().to_lowercase() == lower)
        {
            return TrustLevel::Quarantined;
        }
        if priority_senders
            .iter()
            .any(|e| e.trim().to_lowercase() == lower)
        {
            return TrustLevel::Verified;
        }
        TrustLevel::Unknown
    }
}

impl std::fmt::Display for TrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrustLevel::Quarantined => write!(f, "quarantined"),
            TrustLevel::Unknown => write!(f, "unknown"),
            TrustLevel::Known => write!(f, "known"),
            TrustLevel::Verified => write!(f, "verified"),
            TrustLevel::Owner => write!(f, "owner"),
        }
    }
}

/// Token usage statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Tokens used to create a new cache entry (Anthropic prompt caching).
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Tokens read from an existing cache entry (Anthropic prompt caching).
    #[serde(default)]
    pub cache_read_input_tokens: u32,
    /// Tokens consumed by the model's internal reasoning/thinking.
    #[serde(default)]
    pub reasoning_tokens: u32,
}

impl TokenUsage {
    /// Total tokens consumed (input + output) as `u64`.
    pub fn total(&self) -> u64 {
        self.input_tokens as u64 + self.output_tokens as u64
    }
}

impl std::ops::AddAssign for TokenUsage {
    fn add_assign(&mut self, rhs: Self) {
        self.input_tokens += rhs.input_tokens;
        self.output_tokens += rhs.output_tokens;
        self.cache_creation_input_tokens += rhs.cache_creation_input_tokens;
        self.cache_read_input_tokens += rhs.cache_read_input_tokens;
        self.reasoning_tokens += rhs.reasoning_tokens;
    }
}

/// Configuration for dynamic agent spawning via `spawn_agent`.
///
/// Controls security boundaries: which tools spawned agents may use,
/// how many can be created, and their token budgets.
///
/// Defined in core types so the agent core can reference it without
/// pulling in the umbrella's full config module. Re-exported from
/// `heartbit::config`.
#[derive(Debug, Clone, Deserialize)]
pub struct SpawnConfig {
    /// Maximum number of agents that can be spawned per orchestrator run.
    #[serde(default = "default_max_spawned_agents")]
    pub max_spawned_agents: u32,
    /// Allowlist of tool names that spawned agents may use.
    /// Only builtin tools from this list are available; unknown names
    /// are rejected at build time.
    #[serde(default)]
    pub tool_allowlist: Vec<String>,
    /// Maximum turns per spawned agent.
    #[serde(default = "default_spawn_max_turns")]
    pub max_turns: usize,
    /// Maximum tokens per LLM call for spawned agents.
    #[serde(default = "default_spawn_max_tokens")]
    pub max_tokens: u32,
    /// Cumulative token budget across ALL spawned agents in a single run.
    #[serde(default = "default_max_total_tokens")]
    pub max_total_tokens: u64,
}

fn default_max_spawned_agents() -> u32 {
    3
}

fn default_spawn_max_turns() -> usize {
    15
}

fn default_spawn_max_tokens() -> u32 {
    4096
}

fn default_max_total_tokens() -> u64 {
    50_000
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_usage_default() {
        let usage = TokenUsage::default();
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.total(), 0);
    }

    #[test]
    fn token_usage_total() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            reasoning_tokens: 0,
        };
        assert_eq!(usage.total(), 150);
    }

    #[test]
    fn token_usage_add_assign() {
        let mut usage1 = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 10,
            cache_read_input_tokens: 5,
            reasoning_tokens: 0,
        };
        let usage2 = TokenUsage {
            input_tokens: 30,
            output_tokens: 20,
            cache_creation_input_tokens: 2,
            cache_read_input_tokens: 1,
            reasoning_tokens: 15,
        };
        usage1 += usage2;
        assert_eq!(usage1.input_tokens, 130);
        assert_eq!(usage1.output_tokens, 70);
        assert_eq!(usage1.cache_creation_input_tokens, 12);
        assert_eq!(usage1.cache_read_input_tokens, 6);
        assert_eq!(usage1.reasoning_tokens, 15);
    }
}
