//! PII detection guardrail.
//!
//! Scans LLM responses and tool outputs for personally identifiable
//! information (email, phone, SSN, credit card) and can redact, warn, or deny.

use std::future::Future;
use std::pin::Pin;
use std::sync::LazyLock;

use regex::Regex;

use crate::agent::guardrail::{GuardAction, Guardrail, GuardrailMeta};
use crate::error::Error;
use crate::llm::types::{CompletionResponse, ContentBlock, ToolCall};
use crate::tool::ToolOutput;

// Static compiled regexes for built-in PII detectors.
static EMAIL_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap());
static PHONE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}").unwrap());
static SSN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap());
static CC_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{1,7}\b").unwrap());

/// What to do when PII is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiiAction {
    /// Replace matched PII with `[REDACTED:type]` (recommended default).
    Redact,
    /// Issue a warning but allow the content through unmodified.
    Warn,
    /// Deny the operation entirely.
    Deny,
}

/// Built-in and custom PII detector types.
#[derive(Debug, Clone)]
pub enum PiiDetector {
    Email,
    Phone,
    Ssn,
    CreditCard,
    Custom { name: String, pattern: Regex },
}

impl PiiDetector {
    fn label(&self) -> &str {
        match self {
            PiiDetector::Email => "email",
            PiiDetector::Phone => "phone",
            PiiDetector::Ssn => "ssn",
            PiiDetector::CreditCard => "credit_card",
            PiiDetector::Custom { name, .. } => name,
        }
    }

    /// Find all matches in text, returning (start, end, label) tuples.
    fn find_matches(&self, text: &str) -> Vec<(usize, usize, String)> {
        let re: &Regex = match self {
            PiiDetector::Email => &EMAIL_RE,
            PiiDetector::Phone => &PHONE_RE,
            PiiDetector::Ssn => &SSN_RE,
            PiiDetector::CreditCard => &CC_RE,
            PiiDetector::Custom { pattern, .. } => pattern,
        };
        let label = self.label().to_string();
        re.find_iter(text)
            .filter(|m| {
                // Luhn validation for credit cards
                if matches!(self, PiiDetector::CreditCard) {
                    let digits: String =
                        m.as_str().chars().filter(|c| c.is_ascii_digit()).collect();
                    return luhn_check(&digits);
                }
                true
            })
            .map(|m| (m.start(), m.end(), label.clone()))
            .collect()
    }
}

/// PII detection guardrail.
pub struct PiiGuardrail {
    detectors: Vec<PiiDetector>,
    action: PiiAction,
}

impl PiiGuardrail {
    pub fn new(detectors: Vec<PiiDetector>, action: PiiAction) -> Self {
        Self { detectors, action }
    }

    /// Create with all built-in detectors (email, phone, SSN, credit card).
    pub fn all_builtin(action: PiiAction) -> Self {
        Self::new(
            vec![
                PiiDetector::Email,
                PiiDetector::Phone,
                PiiDetector::Ssn,
                PiiDetector::CreditCard,
            ],
            action,
        )
    }

    /// Scan text for PII, returning all matches sorted by position.
    fn scan(&self, text: &str) -> Vec<(usize, usize, String)> {
        let mut matches = Vec::new();
        for det in &self.detectors {
            matches.extend(det.find_matches(text));
        }
        matches.sort_by_key(|m| m.0);
        matches
    }

    /// Redact PII in text, replacing matches with `[REDACTED:type]`.
    fn redact(&self, text: &str) -> String {
        let matches = self.scan(text);
        if matches.is_empty() {
            return text.to_string();
        }
        let mut result = String::with_capacity(text.len());
        let mut last = 0;
        for (start, end, label) in &matches {
            if *start < last {
                continue; // Skip overlapping matches
            }
            result.push_str(&text[last..*start]);
            result.push_str(&format!("[REDACTED:{label}]"));
            last = *end;
        }
        result.push_str(&text[last..]);
        result
    }

    /// Determine the guard action for detected PII matches.
    ///
    /// In `Redact` mode, returns `Warn` — actual redaction is done inline in
    /// `post_tool` (which has `&mut ToolOutput`). For `post_llm` (immutable
    /// response), the `Warn` ensures the PII detection is logged.
    fn handle_pii(&self, matches: &[(usize, usize, String)]) -> GuardAction {
        if matches.is_empty() {
            return GuardAction::Allow;
        }
        let labels: Vec<&str> = matches.iter().map(|(_, _, l)| l.as_str()).collect();
        let reason = format!("PII detected: {}", labels.join(", "));
        match self.action {
            PiiAction::Redact => GuardAction::warn(reason),
            PiiAction::Warn => GuardAction::warn(reason),
            PiiAction::Deny => GuardAction::deny(reason),
        }
    }
}

impl GuardrailMeta for PiiGuardrail {
    fn name(&self) -> &str {
        "pii"
    }
}

impl Guardrail for PiiGuardrail {
    fn post_llm(
        &self,
        response: &CompletionResponse,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        // Scan LLM response text for PII (synchronous — no captured refs needed).
        // In Redact mode, handle_pii returns Warn since the response is
        // immutable (actual redaction only works in post_tool via &mut ToolOutput).
        let mut all_matches = Vec::new();
        for block in &response.content {
            if let ContentBlock::Text { text } = block {
                all_matches.extend(self.scan(text));
            }
        }
        let action = self.handle_pii(&all_matches);
        Box::pin(async move { Ok(action) })
    }

    fn post_tool(
        &self,
        _call: &ToolCall,
        output: &mut ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        if output.is_error {
            return Box::pin(async { Ok(()) });
        }
        let matches = self.scan(&output.content);
        if matches.is_empty() {
            return Box::pin(async { Ok(()) });
        }
        match self.action {
            PiiAction::Redact => {
                output.content = self.redact(&output.content);
                Box::pin(async { Ok(()) })
            }
            PiiAction::Warn => Box::pin(async { Ok(()) }),
            PiiAction::Deny => {
                let labels: Vec<&str> = matches.iter().map(|(_, _, l)| l.as_str()).collect();
                let reason = format!("PII detected in tool output: {}", labels.join(", "));
                Box::pin(async move { Err(Error::Guardrail(reason)) })
            }
        }
    }
}

/// Luhn algorithm for credit card validation.
fn luhn_check(digits: &str) -> bool {
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    let mut sum = 0u32;
    let mut double = false;
    for c in digits.chars().rev() {
        let Some(d) = c.to_digit(10) else {
            return false;
        };
        let val = if double {
            let doubled = d * 2;
            if doubled > 9 { doubled - 9 } else { doubled }
        } else {
            d
        };
        sum += val;
        double = !double;
    }
    sum.is_multiple_of(10)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_guard(action: PiiAction) -> PiiGuardrail {
        PiiGuardrail::all_builtin(action)
    }

    #[test]
    fn detects_email_address() {
        let g = make_guard(PiiAction::Deny);
        let matches = g.scan("Contact john@example.com for details");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].2, "email");
    }

    #[test]
    fn detects_phone_number() {
        let g = make_guard(PiiAction::Deny);
        let matches = g.scan("Call me at (555) 123-4567");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].2, "phone");
    }

    #[test]
    fn detects_ssn() {
        let g = make_guard(PiiAction::Deny);
        let matches = g.scan("SSN: 123-45-6789");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].2, "ssn");
    }

    #[test]
    fn detects_credit_card_with_luhn() {
        let g = make_guard(PiiAction::Deny);
        // 4532015112830366 is a valid Luhn number
        let matches = g.scan("Card: 4532 0151 1283 0366");
        assert_eq!(matches.len(), 1, "matches: {matches:?}");
        assert_eq!(matches[0].2, "credit_card");
    }

    #[test]
    fn rejects_invalid_luhn() {
        let g = make_guard(PiiAction::Deny);
        let matches = g.scan("Card: 1234 5678 9012 3456");
        assert!(
            matches.iter().all(|m| m.2 != "credit_card"),
            "should reject invalid Luhn: {matches:?}"
        );
    }

    #[test]
    fn redact_mode_replaces_pii() {
        let g = make_guard(PiiAction::Redact);
        let result = g.redact("Email: john@example.com, SSN: 123-45-6789");
        assert!(result.contains("[REDACTED:email]"), "result: {result}");
        assert!(result.contains("[REDACTED:ssn]"), "result: {result}");
        assert!(!result.contains("john@example.com"), "result: {result}");
        assert!(!result.contains("123-45-6789"), "result: {result}");
    }

    #[test]
    fn redact_mode_warns_on_detection() {
        // handle_pii returns Warn for Redact mode (actual redaction happens
        // inline in post_tool; post_llm can't mutate so it warns).
        let g = make_guard(PiiAction::Redact);
        let scan_matches = g.scan("Contact john@example.com");
        let action = g.handle_pii(&scan_matches);
        assert!(matches!(action, GuardAction::Warn { .. }));
    }

    #[test]
    fn warn_mode_warns() {
        let g = make_guard(PiiAction::Warn);
        let scan_matches = g.scan("Contact john@example.com");
        let action = g.handle_pii(&scan_matches);
        assert!(matches!(action, GuardAction::Warn { .. }));
    }

    #[test]
    fn deny_mode_denies() {
        let g = make_guard(PiiAction::Deny);
        let scan_matches = g.scan("Contact john@example.com");
        let action = g.handle_pii(&scan_matches);
        assert!(action.is_denied());
    }

    #[test]
    fn custom_detector_works() {
        let custom = PiiDetector::Custom {
            name: "api_key".into(),
            pattern: Regex::new(r"tok-[a-zA-Z0-9]{32,}").unwrap(),
        };
        let g = PiiGuardrail::new(vec![custom], PiiAction::Deny);
        let matches = g.scan("Key: tok-AAAAAAAAAAAABBBBBBBBBBBBCCCCCCCCCCCC");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].2, "api_key");
    }

    #[test]
    fn no_false_positive_on_clean_text() {
        let g = make_guard(PiiAction::Deny);
        let matches = g.scan("The weather in Paris is nice today. Temperature: 22C.");
        assert!(matches.is_empty(), "false positives: {matches:?}");
    }

    #[test]
    fn multiple_pii_types_in_one_string() {
        let g = make_guard(PiiAction::Redact);
        let text = "Name: John, email: john@example.com, SSN: 123-45-6789, phone: (555) 123-4567";
        let matches = g.scan(text);
        let labels: Vec<&str> = matches.iter().map(|m| m.2.as_str()).collect();
        assert!(labels.contains(&"email"), "labels: {labels:?}");
        assert!(labels.contains(&"ssn"), "labels: {labels:?}");
        assert!(labels.contains(&"phone"), "labels: {labels:?}");
    }

    #[tokio::test]
    async fn post_tool_redacts_pii() {
        let g = make_guard(PiiAction::Redact);
        let call = ToolCall {
            id: "c1".into(),
            name: "test".into(),
            input: serde_json::json!({}),
        };
        let mut output = ToolOutput::success("Email: john@example.com".to_string());
        g.post_tool(&call, &mut output).await.unwrap();
        assert!(output.content.contains("[REDACTED:email]"));
        assert!(!output.content.contains("john@example.com"));
    }

    #[tokio::test]
    async fn post_tool_deny_returns_error() {
        let g = make_guard(PiiAction::Deny);
        let call = ToolCall {
            id: "c1".into(),
            name: "test".into(),
            input: serde_json::json!({}),
        };
        let mut output = ToolOutput::success("SSN: 123-45-6789".to_string());
        let result = g.post_tool(&call, &mut output).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn post_tool_skips_error_outputs() {
        let g = make_guard(PiiAction::Deny);
        let call = ToolCall {
            id: "c1".into(),
            name: "test".into(),
            input: serde_json::json!({}),
        };
        let mut output = ToolOutput::error("john@example.com");
        let result = g.post_tool(&call, &mut output).await;
        assert!(result.is_ok());
    }

    #[test]
    fn luhn_valid() {
        assert!(luhn_check("4532015112830366")); // 16 digits, valid Visa
        assert!(luhn_check("5425233430109903")); // 16 digits, valid Mastercard
    }

    #[test]
    fn luhn_invalid() {
        assert!(!luhn_check("1234567890123456"));
        assert!(!luhn_check("123")); // too short
    }

    #[tokio::test]
    async fn post_llm_redact_mode_warns_on_pii() {
        use crate::llm::types::{StopReason, TokenUsage};

        let g = make_guard(PiiAction::Redact);
        let response = CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Contact john@example.com for help".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        };
        let action = g.post_llm(&response).await.unwrap();
        // Can't redact immutable response — should warn instead of silently allowing
        assert!(
            matches!(action, GuardAction::Warn { .. }),
            "expected Warn for PII in LLM response with Redact mode, got: {action:?}"
        );
    }

    #[tokio::test]
    async fn post_llm_deny_mode_denies_on_pii() {
        use crate::llm::types::{StopReason, TokenUsage};

        let g = make_guard(PiiAction::Deny);
        let response = CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "SSN: 123-45-6789".into(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::default(),
            model: None,
        };
        let action = g.post_llm(&response).await.unwrap();
        assert!(action.is_denied());
    }

    #[test]
    fn meta_name() {
        let g = make_guard(PiiAction::Redact);
        assert_eq!(g.name(), "pii");
    }
}
