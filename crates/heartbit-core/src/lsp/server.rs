use std::collections::HashMap;
use std::path::{Path, PathBuf};

use tokio::process::Command;

use super::client::JsonRpcClient;
use super::language::LanguageConfig;
use super::types::{Diagnostic, RawDiagnostic};

/// An active LSP server process with JSON-RPC communication.
pub(super) struct LspServer {
    client: JsonRpcClient,
    /// Per-file version counters (LSP requires strictly increasing versions).
    file_versions: HashMap<PathBuf, i32>,
    /// Diagnostics version baseline recorded before each `didOpen`/`didChange`.
    /// `pull_diagnostics` waits for a version *newer* than this.
    baseline_versions: HashMap<PathBuf, u64>,
    /// Workspace root used during initialization.
    _root_uri: String,
    /// Child process handle — dropped when server is dropped (kills process).
    _child: tokio::process::Child,
}

impl LspServer {
    /// Spawn and initialize an LSP server.
    ///
    /// Sends `initialize` + `initialized` before returning.
    /// Returns `Err` if the server fails to start or initialize.
    pub async fn spawn(config: &LanguageConfig, workspace_root: &Path) -> Result<Self, String> {
        let mut child = Command::new(config.command)
            .args(config.args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("failed to spawn {}: {e}", config.command))?;

        let stdin = child.stdin.take().ok_or("failed to capture stdin")?;
        let stdout = child.stdout.take().ok_or("failed to capture stdout")?;

        let client = JsonRpcClient::new(stdin, stdout);

        let root_uri = format!("file://{}", workspace_root.display());

        // Send initialize request
        let init_params = serde_json::json!({
            "processId": std::process::id(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "publishDiagnostics": {
                        "relatedInformation": false
                    }
                }
            }
        });

        let _init_result = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            client.request("initialize", init_params),
        )
        .await
        .map_err(|_| "initialize timed out".to_string())?
        .map_err(|e| format!("initialize failed: {e}"))?;

        // Send initialized notification
        client
            .notify("initialized", serde_json::json!({}))
            .await
            .map_err(|e| format!("initialized notification failed: {e}"))?;

        Ok(Self {
            client,
            file_versions: HashMap::new(),
            baseline_versions: HashMap::new(),
            _root_uri: root_uri,
            _child: child,
        })
    }

    /// Notify the server that a file was changed (opened or modified).
    ///
    /// If this is the first time we're notifying about this file, sends
    /// `textDocument/didOpen`. Otherwise sends `textDocument/didChange`.
    /// Records the current diagnostics version before sending so that
    /// `pull_diagnostics` can wait for a *newer* notification.
    pub async fn notify_file_changed(&mut self, path: &Path, content: &str) -> Result<(), String> {
        let uri = format!("file://{}", path.display());
        let lang_id = super::language::detect_language(path).unwrap_or("plaintext");

        // Record the current diagnostics version *before* the notification.
        // pull_diagnostics will wait for version > this baseline.
        let baseline = self.client.diagnostics_version_for(&uri).await;
        self.baseline_versions.insert(path.to_path_buf(), baseline);

        let version = self.file_versions.entry(path.to_path_buf()).or_insert(0);

        if *version == 0 {
            // First time: didOpen (version starts at 1)
            *version = 1;
            self.client
                .notify(
                    "textDocument/didOpen",
                    serde_json::json!({
                        "textDocument": {
                            "uri": uri,
                            "languageId": lang_id,
                            "version": *version,
                            "text": content,
                        }
                    }),
                )
                .await?;
        } else {
            // Subsequent: didChange (full sync, strictly increasing version)
            *version += 1;
            self.client
                .notify(
                    "textDocument/didChange",
                    serde_json::json!({
                        "textDocument": {
                            "uri": uri,
                            "version": *version,
                        },
                        "contentChanges": [{
                            "text": content,
                        }]
                    }),
                )
                .await?;
        }
        Ok(())
    }

    /// Request diagnostics for a file.
    ///
    /// Tries the pull model (`textDocument/diagnostic`, LSP 3.17) first.
    /// If the server doesn't support it (e.g., rust-analyzer), falls back to
    /// cached `publishDiagnostics` notifications (push model).
    ///
    /// For the push model, uses version-based tracking: waits for notifications
    /// *newer* than the baseline recorded in `notify_file_changed`. If a
    /// notification arrives with empty diagnostics (e.g., syntax-only pass),
    /// records the new version and waits again for type-checking results.
    /// Total push-model timeout: 30 seconds.
    pub async fn pull_diagnostics(&self, path: &Path) -> Vec<Diagnostic> {
        let uri = format!("file://{}", path.display());
        let baseline = self.baseline_versions.get(path).copied().unwrap_or(0);

        // Small debounce to let the server start processing the file change
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;

        // Try pull model first (LSP 3.17 textDocument/diagnostic)
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(3),
            self.client.request(
                "textDocument/diagnostic",
                serde_json::json!({
                    "textDocument": { "uri": uri }
                }),
            ),
        )
        .await;

        match &result {
            Ok(Ok(value)) => {
                let diagnostics = parse_diagnostic_response(value);
                if !diagnostics.is_empty() {
                    tracing::debug!(
                        uri = %uri,
                        count = diagnostics.len(),
                        "pull model returned diagnostics"
                    );
                    return diagnostics;
                }
                tracing::debug!(uri = %uri, "pull model returned empty, trying push model");
            }
            Ok(Err(e)) => {
                tracing::debug!(uri = %uri, error = %e, "pull model request failed");
            }
            Err(_) => {
                tracing::debug!(uri = %uri, "pull model timed out");
            }
        }

        // Fall back to push model (publishDiagnostics notifications).
        // rust-analyzer sends notifications in phases:
        //   1. Syntax-only pass (fast, usually empty diagnostics)
        //   2. Type-checking pass (slow on cold start, has real errors)
        //
        // We loop: wait for the *next* notification after our baseline,
        // check if it has diagnostics, and if empty keep waiting for the
        // next one. Total timeout: 30s to handle cold-start workspace loading.
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
        let mut current_version = baseline;

        tracing::debug!(
            uri = %uri,
            baseline = baseline,
            "waiting for publishDiagnostics (up to 30s)"
        );

        loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                tracing::debug!(uri = %uri, "push model timed out — no diagnostics");
                return Vec::new();
            }

            if let Some(params) = self
                .client
                .wait_for_published_diagnostics(&uri, current_version, remaining)
                .await
            {
                let diagnostics = parse_diagnostic_response(&params);
                if !diagnostics.is_empty() {
                    tracing::debug!(
                        uri = %uri,
                        count = diagnostics.len(),
                        "push model returned diagnostics"
                    );
                    return diagnostics;
                }

                // Empty notification — update version and wait for the next one.
                // This handles the syntax-only → type-checking progression.
                let new_version = self.client.diagnostics_version_for(&uri).await;
                tracing::debug!(
                    uri = %uri,
                    version = new_version,
                    "push model returned empty, waiting for update"
                );
                current_version = new_version;
            } else {
                tracing::debug!(uri = %uri, "push model timed out — no diagnostics");
                return Vec::new();
            }
        }
    }
}

/// Parse diagnostics from a `textDocument/diagnostic` response.
fn parse_diagnostic_response(value: &serde_json::Value) -> Vec<Diagnostic> {
    // The response can be DocumentDiagnosticReport which has an `items` array
    if let Some(items) = value.get("items").and_then(|v| v.as_array()) {
        items
            .iter()
            .filter_map(|item| {
                serde_json::from_value::<RawDiagnostic>(item.clone())
                    .ok()
                    .map(|r| r.into_diagnostic())
            })
            .collect()
    } else if let Some(diagnostics) = value.get("diagnostics").and_then(|v| v.as_array()) {
        // publishDiagnostics format
        diagnostics
            .iter()
            .filter_map(|item| {
                serde_json::from_value::<RawDiagnostic>(item.clone())
                    .ok()
                    .map(|r| r.into_diagnostic())
            })
            .collect()
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_diagnostic_response_items_format() {
        let value = json!({
            "kind": "full",
            "items": [
                {
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 5}
                    },
                    "severity": 1,
                    "message": "syntax error"
                }
            ]
        });
        let diagnostics = parse_diagnostic_response(&value);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].message, "syntax error");
    }

    #[test]
    fn parse_diagnostic_response_diagnostics_format() {
        let value = json!({
            "diagnostics": [
                {
                    "range": {
                        "start": {"line": 5, "character": 0},
                        "end": {"line": 5, "character": 3}
                    },
                    "severity": 2,
                    "message": "unused variable"
                }
            ]
        });
        let diagnostics = parse_diagnostic_response(&value);
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].message, "unused variable");
    }

    #[test]
    fn parse_diagnostic_response_empty() {
        let value = json!({});
        let diagnostics = parse_diagnostic_response(&value);
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn parse_diagnostic_response_null_items() {
        let value = json!({"items": null});
        let diagnostics = parse_diagnostic_response(&value);
        assert!(diagnostics.is_empty());
    }
}
