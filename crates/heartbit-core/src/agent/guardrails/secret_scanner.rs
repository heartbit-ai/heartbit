//! Secret scanning guardrail.
//!
//! Scans LLM responses and tool outputs for leaked secrets (API keys,
//! tokens, private keys, connection strings) and can redact or deny.

use std::future::Future;
use std::pin::Pin;
use std::sync::LazyLock;

use regex::Regex;

use crate::agent::guardrail::{GuardAction, Guardrail};
use crate::error::Error;
use crate::llm::types::{CompletionResponse, ContentBlock, ToolCall};
use crate::tool::ToolOutput;

/// What to do when a secret is detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretAction {
    /// Replace secrets with `[REDACTED:label]` (recommended default).
    Redact,
    /// Block entire output containing a secret.
    Deny,
}

/// A compiled pattern for detecting a specific type of secret.
#[derive(Debug, Clone)]
pub struct SecretPattern {
    label: String,
    regex: Regex,
}

// Built-in patterns (compiled once via LazyLock).
static AWS_KEY_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"AKIA[0-9A-Z]{16}").unwrap());
static GENERIC_API_KEY_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['"]?[a-zA-Z0-9_\-]{20,}"#,
    )
    .unwrap()
});
static BEARER_TOKEN_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"Bearer\s+[a-zA-Z0-9_\-\.~+/]{20,}=*").unwrap());
static JWT_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"eyJ[a-zA-Z0-9_\-]{10,}\.eyJ[a-zA-Z0-9_\-]{10,}\.[a-zA-Z0-9_\-+/=]+").unwrap()
});
static PRIVATE_KEY_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"-----BEGIN[A-Z ]*PRIVATE KEY-----").unwrap());
static CONNECTION_STRING_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(postgres|mysql|mongodb|redis)://[^\s]{10,}").unwrap());

fn builtin_patterns() -> Vec<SecretPattern> {
    vec![
        SecretPattern {
            label: "aws_key".into(),
            regex: AWS_KEY_RE.clone(),
        },
        SecretPattern {
            label: "api_key".into(),
            regex: GENERIC_API_KEY_RE.clone(),
        },
        SecretPattern {
            label: "bearer_token".into(),
            regex: BEARER_TOKEN_RE.clone(),
        },
        SecretPattern {
            label: "jwt".into(),
            regex: JWT_RE.clone(),
        },
        SecretPattern {
            label: "private_key".into(),
            regex: PRIVATE_KEY_RE.clone(),
        },
        SecretPattern {
            label: "connection_string".into(),
            regex: CONNECTION_STRING_RE.clone(),
        },
    ]
}

/// Scan text for secrets, returning the redacted text and labels of found secrets.
fn scan_and_redact(text: &str, patterns: &[SecretPattern]) -> (String, Vec<String>) {
    // Collect all matches first, then replace from end to start to preserve positions.
    let mut matches: Vec<(usize, usize, String)> = Vec::new();
    for pattern in patterns {
        for m in pattern.regex.find_iter(text) {
            matches.push((m.start(), m.end(), pattern.label.clone()));
        }
    }
    // Sort by position descending to replace from end.
    matches.sort_by_key(|m| std::cmp::Reverse(m.0));
    matches.dedup_by(|a, b| a.0 == b.0 && a.1 == b.1);

    let mut result = text.to_string();
    let mut found = Vec::new();
    for (start, end, label) in &matches {
        let replacement = format!("[REDACTED:{label}]");
        result.replace_range(*start..*end, &replacement);
        if !found.contains(label) {
            found.push(label.clone());
        }
    }
    (result, found)
}

/// Guardrail that scans for leaked secrets in LLM responses and tool outputs.
pub struct SecretScannerGuardrail {
    patterns: Vec<SecretPattern>,
    action: SecretAction,
}

impl SecretScannerGuardrail {
    /// Create a builder for configuring the secret scanner.
    pub fn builder() -> SecretScannerGuardrailBuilder {
        SecretScannerGuardrailBuilder {
            patterns: builtin_patterns(),
            action: SecretAction::Redact,
        }
    }
}

/// Builder for [`SecretScannerGuardrail`].
pub struct SecretScannerGuardrailBuilder {
    patterns: Vec<SecretPattern>,
    action: SecretAction,
}

impl SecretScannerGuardrailBuilder {
    /// Set the action to take when a secret is detected.
    pub fn action(mut self, action: SecretAction) -> Self {
        self.action = action;
        self
    }

    /// Add a custom pattern to detect.
    pub fn custom_pattern(mut self, label: impl Into<String>, regex: Regex) -> Self {
        self.patterns.push(SecretPattern {
            label: label.into(),
            regex,
        });
        self
    }

    /// Build the guardrail.
    pub fn build(self) -> SecretScannerGuardrail {
        SecretScannerGuardrail {
            patterns: self.patterns,
            action: self.action,
        }
    }
}

impl Guardrail for SecretScannerGuardrail {
    fn name(&self) -> &str {
        "secret_scanner"
    }

    fn post_llm(
        &self,
        response: &mut CompletionResponse,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        let text = response
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" ");

        let (_, labels) = scan_and_redact(&text, &self.patterns);

        Box::pin(async move {
            if labels.is_empty() {
                Ok(GuardAction::Allow)
            } else {
                Ok(GuardAction::deny(format!(
                    "Secret scanner: detected {} in response. Output blocked to prevent leakage.",
                    labels.join(", ")
                )))
            }
        })
    }

    fn post_tool(
        &self,
        _call: &ToolCall,
        output: &mut ToolOutput,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        if output.is_error {
            return Box::pin(async { Ok(()) });
        }

        let (redacted, labels) = scan_and_redact(&output.content, &self.patterns);
        if !labels.is_empty() {
            match self.action {
                SecretAction::Redact => {
                    output.content = redacted;
                }
                SecretAction::Deny => {
                    output.content = format!(
                        "[BLOCKED] Tool output contained secrets ({}). Output suppressed.",
                        labels.join(", ")
                    );
                    output.is_error = true;
                }
            }
        }

        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn detects_aws_key() {
        let (redacted, labels) =
            scan_and_redact("My key is AKIAIOSFODNN7EXAMPLE ok", &builtin_patterns());
        assert!(redacted.contains("[REDACTED:aws_key]"));
        assert!(!redacted.contains("AKIAIOSFODNN7EXAMPLE"));
        assert!(labels.contains(&"aws_key".to_string()));
    }

    #[test]
    fn detects_jwt() {
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"; // gitleaks:allow (test data)
        let (redacted, labels) = scan_and_redact(&format!("Token: {jwt}"), &builtin_patterns());
        assert!(redacted.contains("[REDACTED:jwt]"));
        assert!(!redacted.contains("eyJhbGci"));
        assert!(labels.contains(&"jwt".to_string()));
    }

    #[test]
    fn detects_private_key() {
        let text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow...";
        let (_, labels) = scan_and_redact(text, &builtin_patterns());
        assert!(!labels.is_empty());
    }

    #[test]
    fn detects_bearer_token() {
        let text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9abcdef";
        let (_, labels) = scan_and_redact(text, &builtin_patterns());
        assert!(!labels.is_empty());
    }

    #[test]
    fn detects_connection_string() {
        let text = "DATABASE_URL=postgres://user:pass@host:5432/dbname";
        let (redacted, labels) = scan_and_redact(text, &builtin_patterns());
        assert!(redacted.contains("[REDACTED:connection_string]"));
        assert!(labels.contains(&"connection_string".to_string()));
    }

    #[test]
    fn detects_generic_api_key() {
        let text = "api_key = sk-proj-abcdefghijklmnopqrstuvwx";
        let (_, labels) = scan_and_redact(text, &builtin_patterns());
        assert!(!labels.is_empty());
    }

    #[test]
    fn clean_text_passes() {
        let text = "Hello world, this is a normal response with no secrets.";
        let (_, labels) = scan_and_redact(text, &builtin_patterns());
        assert!(labels.is_empty());
    }

    #[test]
    fn custom_pattern_works() {
        let mut patterns = builtin_patterns();
        patterns.push(SecretPattern {
            label: "custom_token".into(),
            regex: Regex::new(r"xoxb-[0-9A-Za-z-]{20,}").unwrap(),
        });
        let text = "Slack token: xoxb-FAKE0000000-TESTDATAONLY12345"; // gitleaks:allow (test data)
        let (redacted, labels) = scan_and_redact(text, &patterns);
        assert!(redacted.contains("[REDACTED:custom_token]"));
        assert!(labels.contains(&"custom_token".to_string()));
    }

    #[tokio::test]
    async fn post_tool_redacts_secrets() {
        let guard = SecretScannerGuardrail::builder().build();
        let call = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            input: json!({}),
        };
        let mut output = ToolOutput::success("Found key AKIAIOSFODNN7EXAMPLE in env");
        guard.post_tool(&call, &mut output).await.unwrap();
        assert!(output.content.contains("[REDACTED:aws_key]"));
        assert!(!output.content.contains("AKIAIOSFODNN7EXAMPLE"));
    }

    #[tokio::test]
    async fn post_tool_denies_with_deny_action() {
        let guard = SecretScannerGuardrail::builder()
            .action(SecretAction::Deny)
            .build();
        let call = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            input: json!({}),
        };
        let mut output = ToolOutput::success("Found key AKIAIOSFODNN7EXAMPLE");
        guard.post_tool(&call, &mut output).await.unwrap();
        assert!(output.is_error);
        assert!(output.content.contains("BLOCKED"));
    }

    #[tokio::test]
    async fn post_tool_skips_error_outputs() {
        let guard = SecretScannerGuardrail::builder()
            .action(SecretAction::Deny)
            .build();
        let call = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            input: json!({}),
        };
        let mut output = ToolOutput::error("AKIAIOSFODNN7EXAMPLE");
        let result = guard.post_tool(&call, &mut output).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn post_llm_denies_on_secret() {
        let guard = SecretScannerGuardrail::builder().build();
        let mut response = CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Here is the key: AKIAIOSFODNN7EXAMPLE".into(),
            }],
            stop_reason: crate::llm::types::StopReason::EndTurn,
            reasoning: None,
            usage: crate::llm::types::TokenUsage::default(),
            model: None,
        };
        let action = guard.post_llm(&mut response).await.unwrap();
        assert!(action.is_denied());
    }

    #[tokio::test]
    async fn post_llm_allows_clean_response() {
        let guard = SecretScannerGuardrail::builder().build();
        let mut response = CompletionResponse {
            content: vec![ContentBlock::Text {
                text: "Hello, how can I help?".into(),
            }],
            stop_reason: crate::llm::types::StopReason::EndTurn,
            reasoning: None,
            usage: crate::llm::types::TokenUsage::default(),
            model: None,
        };
        let action = guard.post_llm(&mut response).await.unwrap();
        assert!(!action.is_denied());
    }

    #[test]
    fn guardrail_name() {
        let guard = SecretScannerGuardrail::builder().build();
        assert_eq!(guard.name(), "secret_scanner");
    }
}
