//! Adaptive thinking-budget router.
//!
//! SOTA pattern (Anthropic adaptive+effort, Route-to-Reason, adaptive
//! multi-turn budgets): allocate `enable_thinking` + reasoning effort +
//! `max_tokens` from **structural** task complexity — not keyword lists alone.
//!
//! Qwen3 on vLLM is mostly a boolean (`chat_template_kwargs.enable_thinking`);
//! effort tiers still matter for OpenRouter / DashScope and for our max_tokens
//! ceiling. Trivial chat must land on `enable_thinking=false` (live probe
//! 2026-09-22: `reasoning.effort=none` alone still over-thinks).

use super::types::ReasoningEffort;

/// Discrete thinking allocation tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingTier {
    /// Greetings / one-word chat — thinking off.
    Off,
    /// Short factual Q&A — thinking off (Qwen) / minimal elsewhere.
    Low,
    /// Normal single-domain work.
    Medium,
    /// Hard multi-step / agentic / TB2-class.
    High,
}

impl ThinkingTier {
    /// Stable label for traces.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }
}

/// Resolved per-request thinking budget.
#[derive(Debug, Clone, PartialEq)]
pub struct ThinkingBudget {
    /// Discrete tier chosen by the scorer.
    pub tier: ThinkingTier,
    /// Whether the provider should emit chain-of-thought (Qwen
    /// `enable_thinking`).
    pub enable_thinking: bool,
    /// Effort level mapped onto `CompletionRequest.reasoning_effort`.
    pub effort: ReasoningEffort,
    /// Cap applied to this request (≤ ceiling).
    pub max_tokens: u32,
    /// Human-readable why — emitted on traces.
    pub reason: String,
    /// Complexity score in `[0, 1]` after clamps.
    pub score: f32,
}

/// Inputs for budget resolution. All fields are optional signals; missing
/// signals bias toward the cheaper tier (safe for chat).
#[derive(Debug, Clone)]
pub struct ThinkingBudgetInput<'a> {
    /// Latest user prompt / task text.
    pub prompt: &'a str,
    /// Request-intent mode label (`answer`/`execute`/`study`/`clarify`) when
    /// the L0 router already ran.
    pub request_mode: Option<&'a str>,
    /// How many tools the agent can call (agentic signal).
    pub tool_count: usize,
    /// Profile / runner ceiling — never exceeded.
    pub max_tokens_ceiling: u32,
}

impl<'a> ThinkingBudgetInput<'a> {
    /// Build an input with defaults (no mode, no tools, 8192 ceiling).
    pub fn new(prompt: &'a str) -> Self {
        Self {
            prompt,
            request_mode: None,
            tool_count: 0,
            max_tokens_ceiling: 8192,
        }
    }

    /// Attach a request-intent mode label.
    pub fn with_mode(mut self, mode: Option<&'a str>) -> Self {
        self.request_mode = mode;
        self
    }

    /// Attach the agent tool inventory size.
    pub fn with_tool_count(mut self, n: usize) -> Self {
        self.tool_count = n;
        self
    }

    /// Cap max_tokens at this ceiling (floored at 256).
    pub fn with_ceiling(mut self, ceiling: u32) -> Self {
        self.max_tokens_ceiling = ceiling.max(256);
        self
    }
}

/// Structural + mode features used by the scorer (exposed for tests / traces).
#[derive(Debug, Clone, PartialEq)]
pub struct ThinkingSignals {
    /// Character length of the trimmed prompt.
    pub char_len: usize,
    /// Whitespace-separated token count.
    pub word_count: usize,
    /// Prompt contains a fenced code block.
    pub has_code_fence: bool,
    /// Prompt mentions a filesystem- or extension-like path.
    pub has_path_like: bool,
    /// Count of numbered / phrase step markers.
    pub step_markers: usize,
    /// Approximate sentence count.
    pub sentence_count: usize,
    /// Whole prompt is a greeting / ack.
    pub greeting_only: bool,
    /// Secondary hard-task lexicon hits (capped contribution).
    pub hard_lexical_hits: usize,
    /// Score delta from request mode.
    pub mode_boost: f32,
    /// Score delta from tool inventory.
    pub tool_boost: f32,
}

/// Resolve a thinking budget for one user request.
pub fn resolve_thinking_budget(input: &ThinkingBudgetInput<'_>) -> ThinkingBudget {
    let signals = extract_signals(input);
    let (score, reason_bits) = score_signals(&signals, input);
    let tier = tier_from_score(score, signals.greeting_only);
    budget_for_tier(tier, input.max_tokens_ceiling, score, &reason_bits)
}

fn extract_signals(input: &ThinkingBudgetInput<'_>) -> ThinkingSignals {
    let prompt = input.prompt.trim();
    let lower = prompt.to_lowercase();
    let words: Vec<&str> = prompt.split_whitespace().collect();
    let word_count = words.len();
    let char_len = prompt.chars().count();

    let has_code_fence = prompt.contains("```") || prompt.contains("~~~");
    let has_path_like = words.iter().any(|w| {
        let t = w.trim_matches(|c: char| {
            !c.is_ascii_alphanumeric() && c != '/' && c != '.' && c != '_' && c != '-'
        });
        t.contains('/')
            || t.ends_with(".rs")
            || t.ends_with(".py")
            || t.ends_with(".js")
            || t.ends_with(".ts")
            || t.ends_with(".tsx")
            || t.ends_with(".go")
            || t.ends_with(".toml")
            || t.ends_with(".json")
            || t.ends_with(".sh")
            || t.ends_with(".yml")
            || t.ends_with(".yaml")
    });

    let step_markers = count_step_markers(&lower, &words);
    let sentence_count = prompt
        .chars()
        .filter(|c| matches!(c, '.' | '!' | '?' | '。' | '！' | '？'))
        .count()
        .max(if prompt.is_empty() { 0 } else { 1 });

    let greeting_only = is_greeting_only(prompt, &words);
    let hard_lexical_hits = count_hard_lexical(&lower);

    let mode_boost = match input.request_mode.map(|m| m.trim().to_ascii_lowercase()) {
        Some(ref m) if m == "answer" => -0.28,
        Some(ref m) if m == "clarify" => -0.12,
        Some(ref m) if m == "study" => 0.22,
        Some(ref m) if m == "execute" => 0.12,
        _ => 0.0,
    };

    let tool_boost = if input.tool_count >= 8 {
        0.28
    } else if input.tool_count >= 3 {
        0.16
    } else if input.tool_count >= 1 {
        0.06
    } else {
        0.0
    };

    ThinkingSignals {
        char_len,
        word_count,
        has_code_fence,
        has_path_like,
        step_markers,
        sentence_count,
        greeting_only,
        hard_lexical_hits,
        mode_boost,
        tool_boost,
    }
}

fn score_signals(
    signals: &ThinkingSignals,
    _input: &ThinkingBudgetInput<'_>,
) -> (f32, Vec<&'static str>) {
    let mut score: f32 = 0.0;
    let mut reasons: Vec<&'static str> = Vec::new();

    if signals.greeting_only {
        return (0.0, vec!["greeting_only"]);
    }

    // Length bands — structural, language-agnostic.
    if signals.char_len > 2000 {
        score += 0.35;
        reasons.push("very_long");
    } else if signals.char_len > 800 {
        score += 0.25;
        reasons.push("long");
    } else if signals.char_len > 200 {
        score += 0.15;
        reasons.push("medium_length");
    } else if signals.char_len > 40 {
        score += 0.05;
        reasons.push("short");
    } else {
        reasons.push("tiny");
    }

    if signals.word_count > 80 {
        score += 0.10;
        reasons.push("many_words");
    }

    if signals.has_code_fence {
        score += 0.25;
        reasons.push("code_fence");
    }
    if signals.has_path_like {
        score += 0.15;
        reasons.push("path_like");
    }
    if signals.step_markers >= 2 {
        score += 0.22;
        reasons.push("multi_step");
    } else if signals.step_markers == 1 {
        score += 0.08;
        reasons.push("one_step");
    }
    if signals.sentence_count >= 4 {
        score += 0.10;
        reasons.push("multi_sentence");
    }

    // Lexical boosts are secondary — never the sole path to High without
    // structure/mode/tools (capped contribution).
    if signals.hard_lexical_hits > 0 {
        let boost = (0.08 * signals.hard_lexical_hits as f32).min(0.18);
        score += boost;
        reasons.push("hard_lexicon");
    }

    score += signals.mode_boost;
    if signals.mode_boost > 0.0 {
        reasons.push("mode_up");
    } else if signals.mode_boost < 0.0 {
        reasons.push("mode_down");
    }

    score += signals.tool_boost;
    if signals.tool_boost > 0.0 {
        reasons.push("agentic_tools");
    }

    (score.clamp(0.0, 1.0), reasons)
}

fn tier_from_score(score: f32, greeting_only: bool) -> ThinkingTier {
    if greeting_only || score < 0.18 {
        ThinkingTier::Off
    } else if score < 0.38 {
        ThinkingTier::Low
    } else if score < 0.62 {
        ThinkingTier::Medium
    } else {
        ThinkingTier::High
    }
}

fn budget_for_tier(
    tier: ThinkingTier,
    ceiling: u32,
    score: f32,
    reason_bits: &[&str],
) -> ThinkingBudget {
    let ceiling = ceiling.max(256);
    let (enable_thinking, effort, target) = match tier {
        ThinkingTier::Off => (false, ReasoningEffort::None, 512),
        ThinkingTier::Low => (false, ReasoningEffort::Low, 2048),
        ThinkingTier::Medium => (true, ReasoningEffort::Medium, 8192),
        ThinkingTier::High => (true, ReasoningEffort::High, 16384),
    };
    let max_tokens = target.min(ceiling).max(256.min(ceiling));
    let reason = format!(
        "tier={} score={:.2} [{}]",
        tier.label(),
        score,
        reason_bits.join(",")
    );
    ThinkingBudget {
        tier,
        enable_thinking,
        effort,
        max_tokens,
        reason,
        score,
    }
}

fn is_greeting_only(prompt: &str, words: &[&str]) -> bool {
    if words.is_empty() || words.len() > 4 {
        return false;
    }
    // Punctuation-stripped whole prompt.
    let cleaned: String = prompt
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '\'')
        .collect::<String>()
        .to_lowercase();
    let cleaned = cleaned.trim();
    const GREETINGS: &[&str] = &[
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "bonjour",
        "salut",
        "coucou",
        "bonsoir",
        "hola",
        "thanks",
        "thank you",
        "merci",
        "ok",
        "okay",
        "ping",
        "pong",
        "test",
        "hello there",
        "hi there",
        "hey there",
        "good morning",
        "good evening",
        "ça va",
        "ca va",
    ];
    GREETINGS.contains(&cleaned)
}

fn count_step_markers(lower: &str, words: &[&str]) -> usize {
    const PHRASES: &[&str] = &[
        "first,",
        "second,",
        "third,",
        "then,",
        "finally,",
        "next,",
        "after that",
        "step 1",
        "step 2",
        "step 3",
        "etape 1",
        "étape 1",
        "puis,",
        "ensuite,",
    ];
    let phrase_hits = PHRASES.iter().filter(|p| lower.contains(**p)).count();
    let numbered = words
        .iter()
        .filter(|w| {
            let t = w.trim_end_matches([';', ',', ':']);
            if let Some(prefix) = t.strip_suffix('.') {
                return !prefix.is_empty() && prefix.chars().all(|c| c.is_ascii_digit());
            }
            if t.starts_with('(') && t.ends_with(')') {
                let inner = &t[1..t.len() - 1];
                return !inner.is_empty() && inner.chars().all(|c| c.is_ascii_digit());
            }
            false
        })
        .count();
    phrase_hits + numbered
}

fn count_hard_lexical(lower: &str) -> usize {
    // Secondary signal only — French + English agentic verbs.
    const HITS: &[&str] = &[
        "implement",
        "implémente",
        "implemente",
        "refactor",
        "debug",
        "fix failing",
        "migrate",
        "reproduce",
        "benchmark",
        "optimize",
        "optimise",
        "prove that",
        "write a patch",
        "multi-step",
        "end-to-end",
        "terminal-bench",
        "tb2",
        "orchestr",
        "configure nginx",
        "openssl",
        "failing test",
        "stack trace",
        "race condition",
    ];
    HITS.iter().filter(|h| lower.contains(**h)).count()
}

/// Map a resolved budget onto OpenAI-compat / vLLM thinking kwargs.
///
/// Returns `Some(enable_thinking)` when the caller set an explicit effort;
/// `None` means leave the provider default alone.
pub fn enable_thinking_for_effort(effort: Option<ReasoningEffort>) -> Option<bool> {
    match effort {
        Some(ReasoningEffort::None) | Some(ReasoningEffort::Low) => Some(false),
        Some(ReasoningEffort::Medium) | Some(ReasoningEffort::High) => Some(true),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hello_is_off_no_thinking() {
        let b = resolve_thinking_budget(&ThinkingBudgetInput::new("hello"));
        assert_eq!(b.tier, ThinkingTier::Off);
        assert!(!b.enable_thinking);
        assert_eq!(b.effort, ReasoningEffort::None);
        assert!(b.max_tokens <= 512);
        assert!(b.reason.contains("greeting_only"));
    }

    #[test]
    fn french_salut_is_off() {
        let b = resolve_thinking_budget(&ThinkingBudgetInput::new("salut"));
        assert_eq!(b.tier, ThinkingTier::Off);
        assert!(!b.enable_thinking);
    }

    #[test]
    fn short_answer_mode_stays_low_or_off() {
        let b = resolve_thinking_budget(
            &ThinkingBudgetInput::new("What is the capital of France?").with_mode(Some("answer")),
        );
        assert!(
            matches!(b.tier, ThinkingTier::Off | ThinkingTier::Low),
            "got {:?} score={}",
            b.tier,
            b.score
        );
        assert!(!b.enable_thinking);
    }

    #[test]
    fn hard_agentic_tb2_style_is_high() {
        let prompt = r#"
Fix the failing nginx + openssl setup in /app/config/nginx.conf.
1. Reproduce the TLS handshake error from the stack trace.
2. Patch the certificate chain and reload nginx.
3. Verify with openssl s_client and leave a short report.
```bash
openssl s_client -connect localhost:443
```
"#;
        let b = resolve_thinking_budget(
            &ThinkingBudgetInput::new(prompt)
                .with_mode(Some("execute"))
                .with_tool_count(12)
                .with_ceiling(32768),
        );
        assert_eq!(
            b.tier,
            ThinkingTier::High,
            "score={} reason={}",
            b.score,
            b.reason
        );
        assert!(b.enable_thinking);
        assert_eq!(b.effort, ReasoningEffort::High);
        assert!(b.max_tokens >= 8192);
    }

    #[test]
    fn medium_coding_without_tools_is_medium() {
        let prompt = "Refactor this Rust function to return Result and add unit tests:\n```rust\nfn add(a:i32,b:i32)->i32{a+b}\n```";
        let b = resolve_thinking_budget(
            &ThinkingBudgetInput::new(prompt)
                .with_mode(Some("execute"))
                .with_ceiling(8192),
        );
        assert!(
            matches!(b.tier, ThinkingTier::Medium | ThinkingTier::High),
            "got {:?} {}",
            b.tier,
            b.reason
        );
        assert!(b.enable_thinking);
    }

    #[test]
    fn ceiling_is_respected() {
        let b = resolve_thinking_budget(
            &ThinkingBudgetInput::new(
                "implement a multi-step end-to-end migration with benchmarks",
            )
            .with_mode(Some("execute"))
            .with_tool_count(10)
            .with_ceiling(4096),
        );
        assert!(b.max_tokens <= 4096);
    }

    #[test]
    fn keyword_alone_does_not_force_high_on_tiny_prompt() {
        // "debug" alone without structure/tools/mode → not High.
        let b = resolve_thinking_budget(&ThinkingBudgetInput::new("debug this"));
        assert_ne!(
            b.tier,
            ThinkingTier::High,
            "naive keyword must not alone force High: {}",
            b.reason
        );
    }

    #[test]
    fn study_mode_boosts_toward_thinking() {
        let short = resolve_thinking_budget(
            &ThinkingBudgetInput::new(
                "Compare two approaches for caching tokens in the agent loop.",
            )
            .with_mode(Some("study")),
        );
        let answer = resolve_thinking_budget(
            &ThinkingBudgetInput::new(
                "Compare two approaches for caching tokens in the agent loop.",
            )
            .with_mode(Some("answer")),
        );
        assert!(
            short.score > answer.score,
            "study {} vs answer {}",
            short.score,
            answer.score
        );
    }

    #[test]
    fn enable_thinking_for_effort_mapping() {
        assert_eq!(
            enable_thinking_for_effort(Some(ReasoningEffort::None)),
            Some(false)
        );
        assert_eq!(
            enable_thinking_for_effort(Some(ReasoningEffort::Low)),
            Some(false)
        );
        assert_eq!(
            enable_thinking_for_effort(Some(ReasoningEffort::Medium)),
            Some(true)
        );
        assert_eq!(
            enable_thinking_for_effort(Some(ReasoningEffort::High)),
            Some(true)
        );
        assert_eq!(enable_thinking_for_effort(None), None);
    }
}
