use serde::{Deserialize, Serialize};

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
