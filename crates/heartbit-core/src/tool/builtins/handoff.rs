//! The `handoff` tool: distill the CURRENT session into a purpose-tailored brief.
//!
//! Unlike the agent-to-agent [`HandoffTool`](crate::tool::handoff::HandoffTool)
//! (which routes live conversation control to a peer agent), this tool produces
//! a *human-carried, cross-session* Markdown brief that seeds a DIFFERENT
//! session. It mirrors the [`AdvisorTool`](crate::tool::advisor::AdvisorTool)
//! seam: it takes an `Arc<BoxedProvider>` at construction, reads the full
//! conversation from [`ExecutionContext::transcript`](crate::ExecutionContext)
//! (snapshotted by the runner at tool-dispatch time), and makes ONE LLM call.
//!
//! The brief is tailored to a mandatory `purpose` argument — without it a
//! handoff is just a worse compaction — and is written to a disposable location
//! OUTSIDE the workspace (the `handoff_dir` supplied at construction). The
//! distillation prompt enforces pointers-over-duplication and secret redaction.

use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::error::Error;
use crate::llm::types::{CompletionRequest, Message, ToolDefinition};
use crate::llm::{BoxedProvider, LlmProvider};
use crate::tool::{Tool, ToolOutput};

/// Upper bound on the generated brief.
const HANDOFF_MAX_TOKENS: u32 = 2048;

/// Maximum length of the purpose slug in the output filename.
const SLUG_MAX_LEN: usize = 40;

/// The handoff brief's operating contract — the canonical session-bridge rules.
const HANDOFF_SYSTEM_PROMPT: &str = "\
You write a HANDOFF BRIEF: a concise Markdown document so a FRESH agent — with \
NO access to this session — can continue the work deliberately. The full \
conversation transcript so far follows; distill it for the stated PURPOSE.\n\
\n\
Structure the brief with these sections, in order:\n\
- Purpose — the WHY: what the next session is for (tailor everything to this).\n\
- Goal — what 'done' looks like for that purpose.\n\
- State / Progress — where things actually stand right now.\n\
- What worked / What didn't — the load-bearing lessons, briefly.\n\
- Pointers to artifacts — file paths, issue/PR numbers, diffs, commands. Point \
to them; NEVER duplicate their content. Brevity IS the mechanism.\n\
- Suggested next steps — the concrete first moves for the fresh agent.\n\
\n\
Hard rules:\n\
- REDACT secrets. Replace any API key, token, password, or PII with \
[redacted]. A brief on disk is a leak surface.\n\
- Be concise: pointers over prose. A short brief beats a long transcript.\n\
- Write only the brief, in Markdown. No preamble, no meta-commentary.";

/// Distill the current session into a purpose-tailored handoff brief on disk.
///
/// Construct over the provider that should generate the brief (a capable model)
/// and the directory the brief is written to (supplied by the host — core has
/// no config-dir dependency, so the location is a constructor parameter and is
/// therefore fully testable).
pub struct SessionHandoffTool {
    provider: Arc<BoxedProvider>,
    handoff_dir: PathBuf,
}

impl SessionHandoffTool {
    /// Build over the brief-writing provider and the directory to write into.
    pub fn new(provider: Arc<BoxedProvider>, handoff_dir: PathBuf) -> Self {
        Self {
            provider,
            handoff_dir,
        }
    }
}

/// Slugify a purpose into a filename-safe fragment: lowercase alphanumerics,
/// runs of anything else collapsed to single dashes, trimmed, length-bounded.
fn slugify(purpose: &str) -> String {
    let mut slug = String::with_capacity(purpose.len().min(SLUG_MAX_LEN));
    let mut prev_dash = false;
    for ch in purpose.chars() {
        if ch.is_ascii_alphanumeric() {
            slug.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash {
            slug.push('-');
            prev_dash = true;
        }
        if slug.len() >= SLUG_MAX_LEN {
            break;
        }
    }
    let trimmed = slug.trim_matches('-');
    if trimmed.is_empty() {
        "handoff".to_string()
    } else {
        trimmed.to_string()
    }
}

impl Tool for SessionHandoffTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "handoff".into(),
            description: "Distill the CURRENT session into a purpose-tailored Markdown brief that \
                          seeds a DIFFERENT session. Your FULL conversation transcript is forwarded \
                          automatically. Use this to hand work off to a fresh agent without dragging \
                          a stale transcript along: the brief carries the WHY, the goal, current \
                          state, what worked/didn't, pointers to artifacts (paths/issues/diffs — \
                          never their content), and next steps. The 'purpose' argument is MANDATORY \
                          — the brief is tailored to it. Returns the written file path and a short \
                          summary."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "purpose": {
                        "type": "string",
                        "description": "What the next session will be used for — the brief is tailored to this."
                    }
                },
                "required": ["purpose"]
            }),
        }
    }

    fn execute(
        &self,
        ctx: &crate::ExecutionContext,
        input: Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        let transcript = ctx.transcript.clone();
        let purpose = input
            .get("purpose")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .trim()
            .to_string();
        Box::pin(async move {
            // Purpose is the load-bearing design rule: without it a handoff is a
            // worse compaction. Reject up front (tests call execute() directly,
            // bypassing schema validation, so guard here too).
            if purpose.is_empty() {
                return Ok(ToolOutput::error(
                    "handoff requires a non-empty 'purpose': the brief is tailored to what the \
                     next session will be used for. Without it a handoff is just a worse \
                     compaction."
                        .to_string(),
                ));
            }

            let Some(messages) = transcript else {
                return Ok(ToolOutput::error(
                    "handoff unavailable: this runner does not forward the conversation \
                     transcript to tools"
                        .to_string(),
                ));
            };

            let rendered = crate::agent::context::messages_to_text(&messages);
            let request = CompletionRequest {
                system: HANDOFF_SYSTEM_PROMPT.to_string(),
                messages: vec![Message::user(format!(
                    "=== PURPOSE FOR THE NEXT SESSION ===\n{purpose}\n\
                     === END PURPOSE ===\n\n\
                     === TRANSCRIPT (oldest first) ===\n{rendered}\n=== END TRANSCRIPT ===\n\
                     Write the handoff brief now, tailored to the PURPOSE above."
                ))],
                tools: vec![],
                max_tokens: HANDOFF_MAX_TOKENS,
                tool_choice: None,
                reasoning_effort: None,
            };

            let brief = match LlmProvider::complete(self.provider.as_ref(), request).await {
                Ok(resp) => {
                    let text: String = resp
                        .content
                        .iter()
                        .filter_map(|b| match b {
                            crate::llm::types::ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect();
                    if text.trim().is_empty() {
                        return Ok(ToolOutput::error(
                            "handoff: the model returned an empty brief".to_string(),
                        ));
                    }
                    text
                }
                // Fail open with an honest error — handoff must never kill a run.
                Err(e) => return Ok(ToolOutput::error(format!("handoff unavailable: {e}"))),
            };

            // Write the brief to <handoff_dir>/<yyyy-mm-dd-HHMMSS>-<slug>.md.
            if let Err(e) = std::fs::create_dir_all(&self.handoff_dir) {
                return Ok(ToolOutput::error(format!(
                    "handoff: could not create handoff directory {}: {e}",
                    self.handoff_dir.display()
                )));
            }
            let stamp = chrono::Utc::now().format("%Y-%m-%d-%H%M%S");
            let filename = format!("{stamp}-{}.md", slugify(&purpose));
            let path = self.handoff_dir.join(&filename);
            if let Err(e) = std::fs::write(&path, &brief) {
                return Ok(ToolOutput::error(format!(
                    "handoff: could not write brief to {}: {e}",
                    path.display()
                )));
            }

            // Summary = the first few non-empty lines of the brief.
            let summary: String = brief
                .lines()
                .filter(|l| !l.trim().is_empty())
                .take(3)
                .collect::<Vec<_>>()
                .join("\n");
            Ok(ToolOutput::success(format!(
                "Handoff brief written to {}\n\n{summary}",
                path.display()
            )))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutionContext;
    use crate::agent::test_helpers::MockProvider;

    /// A transcript carrying a planted fake secret to exercise the redaction
    /// contract: the SYSTEM PROMPT must instruct the model to redact, and the
    /// stub returns a clean brief (redaction is the LLM's job).
    fn ctx_with_secret_transcript() -> ExecutionContext {
        ExecutionContext {
            transcript: Some(Arc::new(vec![
                Message::user("build the picker UI"),
                Message::user("here is my key sk-test-12345 for the API"),
            ])),
            ..ExecutionContext::default()
        }
    }

    fn tool_with(brief: &str, dir: PathBuf) -> (SessionHandoffTool, Arc<MockProvider>) {
        let mock = Arc::new(MockProvider::new(vec![MockProvider::text_response(
            brief, 50, 80,
        )]));
        let tool =
            SessionHandoffTool::new(Arc::new(BoxedProvider::from_arc(Arc::clone(&mock))), dir);
        (tool, mock)
    }

    #[tokio::test]
    async fn handoff_requires_purpose() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (tool, mock) = tool_with("unused", tmp.path().to_path_buf());

        // Missing purpose.
        let out = tool
            .execute(&ctx_with_secret_transcript(), json!({}))
            .await
            .expect("tool ok");
        assert!(out.is_error, "missing purpose must error: {}", out.content);
        assert!(
            out.content.contains("purpose"),
            "error must name the missing field: {}",
            out.content
        );

        // Empty / whitespace purpose.
        let out = tool
            .execute(&ctx_with_secret_transcript(), json!({"purpose": "   "}))
            .await
            .expect("tool ok");
        assert!(out.is_error, "empty purpose must error: {}", out.content);

        // The LLM is never called when purpose is absent.
        assert_eq!(
            mock.captured_requests.lock().expect("lock").len(),
            0,
            "no LLM call without a purpose"
        );
        // Nothing was written.
        let entries: Vec<_> = std::fs::read_dir(tmp.path()).expect("read dir").collect();
        assert!(entries.is_empty(), "no file should be written");
    }

    #[tokio::test]
    async fn handoff_writes_purpose_tailored_brief() {
        let tmp = tempfile::tempdir().expect("tempdir");
        // The stub returns a clean brief WITHOUT the secret — redaction is the
        // model's job, so we assert the SYSTEM PROMPT carries the instruction
        // and that the written file does not contain the planted secret.
        let clean_brief = "# Purpose\nPrototype the picker UI for the next session.\n\n\
                           # Pointers\nSee crates/heartbit-tui/src/picker.rs.";
        let (tool, mock) = tool_with(clean_brief, tmp.path().to_path_buf());

        let out = tool
            .execute(
                &ctx_with_secret_transcript(),
                json!({"purpose": "prototype the picker UI"}),
            )
            .await
            .expect("tool ok");
        assert!(!out.is_error, "{}", out.content);

        // Exactly one file landed under the handoff dir, name carries the slug.
        let mut files: Vec<_> = std::fs::read_dir(tmp.path())
            .expect("read dir")
            .map(|e| e.expect("entry").file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(files.len(), 1, "one brief written: {files:?}");
        let name = files.pop().expect("one file");
        assert!(name.ends_with(".md"), "markdown extension: {name}");
        assert!(
            name.contains("prototype-the-picker-ui"),
            "filename carries the purpose slug: {name}"
        );

        // Content carries the purpose section and NOT the planted secret.
        let path = tmp.path().join(&name);
        let written = std::fs::read_to_string(&path).expect("read brief");
        assert!(written.contains("Purpose"), "brief has a Purpose section");
        assert!(
            !written.contains("sk-test-12345"),
            "brief must not contain the planted secret: {written}"
        );

        // The redaction contract is in the SYSTEM PROMPT sent to the provider,
        // and the purpose was forwarded in the user message.
        let reqs = mock.captured_requests.lock().expect("lock");
        assert_eq!(reqs.len(), 1, "one LLM call");
        let req = &reqs[0];
        assert!(
            req.system.contains("REDACT") && req.system.contains("[redacted]"),
            "system prompt must instruct redaction: {}",
            req.system
        );
        assert!(
            req.system.contains("Pointers") && req.system.contains("NEVER duplicate"),
            "system prompt must enforce pointers-over-duplication: {}",
            req.system
        );
        let user: String = req
            .messages
            .iter()
            .flat_map(|m| m.content.iter())
            .filter_map(|b| match b {
                crate::llm::types::ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            user.contains("prototype the picker UI"),
            "purpose forwarded in the user message: {user}"
        );
        assert!(
            user.contains("build the picker UI"),
            "transcript forwarded in the user message: {user}"
        );
    }

    #[tokio::test]
    async fn handoff_returns_path_and_summary() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let brief = "# Purpose\nContinue the migration.\n\n# Goal\nGreen test suite.";
        let (tool, _mock) = tool_with(brief, tmp.path().to_path_buf());

        let out = tool
            .execute(
                &ctx_with_secret_transcript(),
                json!({"purpose": "continue the migration"}),
            )
            .await
            .expect("tool ok");
        assert!(!out.is_error, "{}", out.content);

        // The returned content carries the written path...
        let path = tmp.path().join(format!(
            "{}",
            std::fs::read_dir(tmp.path())
                .expect("read dir")
                .next()
                .expect("one file")
                .expect("entry")
                .file_name()
                .to_string_lossy()
        ));
        assert!(
            out.content.contains(&path.display().to_string()),
            "output names the written path: {}",
            out.content
        );
        // ...and a short summary (the first lines of the brief).
        assert!(
            out.content.contains("Continue the migration."),
            "output carries a summary of the brief: {}",
            out.content
        );
    }

    #[tokio::test]
    async fn handoff_without_a_transcript_is_honest() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (tool, _mock) = tool_with("unused", tmp.path().to_path_buf());
        let out = tool
            .execute(&ExecutionContext::default(), json!({"purpose": "anything"}))
            .await
            .expect("tool ok");
        assert!(out.is_error);
        assert!(out.content.contains("transcript"), "{}", out.content);
    }

    #[test]
    fn definition_encodes_the_mandatory_purpose() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (tool, _mock) = tool_with("unused", tmp.path().to_path_buf());
        let def = tool.definition();
        assert_eq!(def.name, "handoff");
        assert!(def.description.contains("MANDATORY"));
        let required = def
            .input_schema
            .get("required")
            .and_then(|r| r.as_array())
            .expect("required array");
        assert!(
            required.iter().any(|v| v == "purpose"),
            "purpose is required: {:?}",
            required
        );
    }

    #[test]
    fn slugify_is_filename_safe() {
        assert_eq!(
            slugify("Prototype the Picker UI!"),
            "prototype-the-picker-ui"
        );
        assert_eq!(slugify("   "), "handoff");
        assert_eq!(slugify(""), "handoff");
        // Length-bounded.
        let long = "a".repeat(100);
        assert!(slugify(&long).len() <= SLUG_MAX_LEN);
        // No leading/trailing dashes.
        let s = slugify("!!!hello world!!!");
        assert!(!s.starts_with('-') && !s.ends_with('-'), "got: {s}");
    }
}
