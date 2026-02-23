use crate::llm::types::TokenUsage;

/// Estimate the cost in USD for a given model and token usage.
///
/// Returns `None` for unknown models. Pricing is based on Anthropic's published
/// per-million-token rates:
/// - Cache reads are 90% cheaper than standard input tokens
/// - Cache writes are 25% more expensive than standard input tokens
/// - Reasoning tokens are priced at the output token rate
pub fn estimate_cost(model: &str, usage: &TokenUsage) -> Option<f64> {
    let (input_per_m, output_per_m) = model_pricing(model)?;
    let input_cost = (usage.input_tokens as f64 / 1_000_000.0) * input_per_m;
    let output_cost = (usage.output_tokens as f64 / 1_000_000.0) * output_per_m;
    let cache_read_cost = (usage.cache_read_input_tokens as f64 / 1_000_000.0) * input_per_m * 0.1;
    let cache_write_cost =
        (usage.cache_creation_input_tokens as f64 / 1_000_000.0) * input_per_m * 1.25;
    let reasoning_cost = (usage.reasoning_tokens as f64 / 1_000_000.0) * output_per_m;
    Some(input_cost + output_cost + cache_read_cost + cache_write_cost + reasoning_cost)
}

/// Return (input_per_million, output_per_million) pricing for a known model.
fn model_pricing(model: &str) -> Option<(f64, f64)> {
    match model {
        // Claude 4.6 generation
        "claude-sonnet-4-6-20250610" => Some((3.0, 15.0)),
        "claude-opus-4-6-20250610" => Some((5.0, 25.0)),
        // Claude 4.5 generation
        "claude-sonnet-4-5-20250514" => Some((3.0, 15.0)),
        "claude-opus-4-5-20250514" => Some((5.0, 25.0)),
        // Claude 4 generation
        "claude-sonnet-4-20250514" => Some((3.0, 15.0)),
        "claude-opus-4-20250514" | "claude-opus-4-1-20250414" => Some((15.0, 75.0)),
        "claude-haiku-4-5-20251001" => Some((1.0, 5.0)),
        // Claude 3.5 generation
        "claude-3-5-sonnet-20241022" | "claude-3-5-sonnet-20240620" => Some((3.0, 15.0)),
        "claude-3-5-haiku-20241022" => Some((0.80, 4.0)),
        // Claude 3 generation
        "claude-3-opus-20240229" => Some((15.0, 75.0)),
        "claude-3-sonnet-20240229" => Some((3.0, 15.0)),
        "claude-3-haiku-20240307" => Some((0.25, 1.25)),
        // OpenRouter model aliases
        "anthropic/claude-sonnet-4.6" => Some((3.0, 15.0)),
        "anthropic/claude-opus-4.6" => Some((5.0, 25.0)),
        "anthropic/claude-sonnet-4.5" => Some((3.0, 15.0)),
        "anthropic/claude-opus-4.5" => Some((5.0, 25.0)),
        "anthropic/claude-sonnet-4" => Some((3.0, 15.0)),
        "anthropic/claude-opus-4" => Some((15.0, 75.0)),
        "anthropic/claude-haiku-4" => Some((1.0, 5.0)),
        "anthropic/claude-3.5-sonnet" | "anthropic/claude-3.5-sonnet:beta" => Some((3.0, 15.0)),
        "anthropic/claude-3.5-haiku" | "anthropic/claude-3.5-haiku:beta" => Some((0.80, 4.0)),
        "anthropic/claude-3-opus" | "anthropic/claude-3-opus:beta" => Some((15.0, 75.0)),
        "anthropic/claude-3-haiku" | "anthropic/claude-3-haiku:beta" => Some((0.25, 1.25)),
        // Qwen models (OpenRouter)
        "qwen/qwen3.5-plus-02-15" => Some((0.40, 2.40)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sonnet_pricing() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = estimate_cost("claude-sonnet-4-20250514", &usage).unwrap();
        // $3/M input + $15/M output = $18
        assert!((cost - 18.0).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn opus_pricing() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = estimate_cost("claude-opus-4-20250514", &usage).unwrap();
        // $15/M input + $75/M output = $90
        assert!((cost - 90.0).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn haiku_pricing() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = estimate_cost("claude-haiku-4-5-20251001", &usage).unwrap();
        // $1/M input + $5/M output = $6
        assert!((cost - 6.0).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn unknown_model_returns_none() {
        let usage = TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            ..Default::default()
        };
        assert!(estimate_cost("gpt-4o", &usage).is_none());
        assert!(estimate_cost("unknown-model", &usage).is_none());
    }

    #[test]
    fn zero_usage_returns_zero() {
        let usage = TokenUsage::default();
        let cost = estimate_cost("claude-sonnet-4-20250514", &usage).unwrap();
        assert!((cost - 0.0).abs() < f64::EPSILON, "cost: {cost}");
    }

    #[test]
    fn cache_read_tokens_priced_at_10_percent() {
        let usage = TokenUsage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: 1_000_000,
            cache_creation_input_tokens: 0,
            reasoning_tokens: 0,
        };
        let cost = estimate_cost("claude-sonnet-4-20250514", &usage).unwrap();
        // 1M cache reads * $3/M * 0.1 = $0.30
        assert!((cost - 0.30).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn cache_write_tokens_priced_at_125_percent() {
        let usage = TokenUsage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 1_000_000,
            reasoning_tokens: 0,
        };
        let cost = estimate_cost("claude-sonnet-4-20250514", &usage).unwrap();
        // 1M cache writes * $3/M * 1.25 = $3.75
        assert!((cost - 3.75).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn mixed_usage_accumulates_correctly() {
        let usage = TokenUsage {
            input_tokens: 500_000,
            output_tokens: 100_000,
            cache_read_input_tokens: 200_000,
            cache_creation_input_tokens: 50_000,
            reasoning_tokens: 0,
        };
        let cost = estimate_cost("claude-sonnet-4-20250514", &usage).unwrap();
        // input: 0.5M * $3 = $1.50
        // output: 0.1M * $15 = $1.50
        // cache_read: 0.2M * $3 * 0.1 = $0.06
        // cache_write: 0.05M * $3 * 1.25 = $0.1875
        let expected = 1.50 + 1.50 + 0.06 + 0.1875;
        assert!(
            (cost - expected).abs() < 0.001,
            "cost: {cost}, expected: {expected}"
        );
    }

    #[test]
    fn openrouter_model_aliases() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 0,
            ..Default::default()
        };
        let cost_native = estimate_cost("claude-sonnet-4-20250514", &usage).unwrap();
        let cost_or = estimate_cost("anthropic/claude-sonnet-4", &usage).unwrap();
        assert!((cost_native - cost_or).abs() < f64::EPSILON);
    }

    #[test]
    fn claude_3_5_sonnet_pricing() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = estimate_cost("claude-3-5-sonnet-20241022", &usage).unwrap();
        assert!((cost - 18.0).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn claude_3_haiku_pricing() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = estimate_cost("claude-3-haiku-20240307", &usage).unwrap();
        // $0.25/M input + $1.25/M output = $1.50
        assert!((cost - 1.50).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn opus_4_5_pricing() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = estimate_cost("claude-opus-4-5-20250514", &usage).unwrap();
        // $5/M input + $25/M output = $30
        assert!((cost - 30.0).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn opus_4_6_pricing() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = estimate_cost("claude-opus-4-6-20250610", &usage).unwrap();
        // $5/M input + $25/M output = $30
        assert!((cost - 30.0).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn qwen_pricing() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = estimate_cost("qwen/qwen3.5-plus-02-15", &usage).unwrap();
        // $0.40/M input + $2.40/M output = $2.80
        assert!((cost - 2.80).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn reasoning_tokens_priced_at_output_rate() {
        let usage = TokenUsage {
            input_tokens: 0,
            output_tokens: 0,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
            reasoning_tokens: 1_000_000,
        };
        let cost = estimate_cost("claude-sonnet-4-20250514", &usage).unwrap();
        // 1M reasoning tokens * $15/M (output rate) = $15
        assert!((cost - 15.0).abs() < 0.001, "cost: {cost}");
    }

    #[test]
    fn mixed_usage_with_reasoning_accumulates_correctly() {
        let usage = TokenUsage {
            input_tokens: 500_000,
            output_tokens: 100_000,
            cache_read_input_tokens: 0,
            cache_creation_input_tokens: 0,
            reasoning_tokens: 200_000,
        };
        let cost = estimate_cost("claude-sonnet-4-20250514", &usage).unwrap();
        // input: 0.5M * $3 = $1.50
        // output: 0.1M * $15 = $1.50
        // reasoning: 0.2M * $15 = $3.00
        let expected = 1.50 + 1.50 + 3.00;
        assert!(
            (cost - expected).abs() < 0.001,
            "cost: {cost}, expected: {expected}"
        );
    }

    #[test]
    fn openrouter_claude_3_5_aliases() {
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 0,
            ..Default::default()
        };
        let native = estimate_cost("claude-3-5-sonnet-20241022", &usage).unwrap();
        let or = estimate_cost("anthropic/claude-3.5-sonnet", &usage).unwrap();
        assert!((native - or).abs() < f64::EPSILON);

        let or_beta = estimate_cost("anthropic/claude-3.5-sonnet:beta", &usage).unwrap();
        assert!((native - or_beta).abs() < f64::EPSILON);
    }
}
