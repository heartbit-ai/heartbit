use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout};
use tokio::sync::{Mutex, Notify, oneshot};

/// A versioned diagnostics entry. Each `publishDiagnostics` increments the version.
#[derive(Debug, Clone)]
pub(super) struct DiagnosticsEntry {
    pub version: u64,
    pub params: serde_json::Value,
}

/// JSON-RPC 2.0 client communicating over stdio with Content-Length framing.
pub(super) struct JsonRpcClient {
    stdin: Mutex<ChildStdin>,
    pending: Arc<Mutex<HashMap<i64, oneshot::Sender<serde_json::Value>>>>,
    next_id: AtomicI64,
    /// Cache of `publishDiagnostics` notifications, keyed by URI.
    /// Each entry has a version counter that increments on every notification.
    published_diagnostics: Arc<Mutex<HashMap<String, DiagnosticsEntry>>>,
    /// Global version counter — incremented on every `publishDiagnostics`.
    /// Kept alive here; the reader task owns a clone.
    _diagnostics_version: Arc<AtomicU64>,
    /// Signals when a new `publishDiagnostics` notification arrives.
    diagnostics_notify: Arc<Notify>,
}

impl JsonRpcClient {
    /// Create a new client and spawn a background reader task.
    pub fn new(stdin: ChildStdin, stdout: ChildStdout) -> Self {
        let pending: Arc<Mutex<HashMap<i64, oneshot::Sender<serde_json::Value>>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let published_diagnostics: Arc<Mutex<HashMap<String, DiagnosticsEntry>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let diagnostics_version = Arc::new(AtomicU64::new(0));
        let diagnostics_notify = Arc::new(Notify::new());

        // Spawn reader task
        let pending_clone = Arc::clone(&pending);
        let diag_clone = Arc::clone(&published_diagnostics);
        let version_clone = Arc::clone(&diagnostics_version);
        let notify_clone = Arc::clone(&diagnostics_notify);
        tokio::spawn(async move {
            if let Err(e) = Self::read_loop(
                stdout,
                pending_clone,
                diag_clone,
                version_clone,
                notify_clone,
            )
            .await
            {
                tracing::debug!(error = %e, "LSP JSON-RPC reader exited");
            }
        });

        Self {
            stdin: Mutex::new(stdin),
            pending,
            next_id: AtomicI64::new(1),
            published_diagnostics,
            _diagnostics_version: diagnostics_version,
            diagnostics_notify,
        }
    }

    /// Send a request and wait for the response.
    pub async fn request(
        &self,
        method: &str,
        params: serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let message = serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let (tx, rx) = oneshot::channel();
        {
            let mut pending = self.pending.lock().await;
            pending.insert(id, tx);
        }

        self.send_message(&message).await?;

        rx.await.map_err(|_| "response channel closed".to_string())
    }

    /// Send a notification (no response expected).
    pub async fn notify(&self, method: &str, params: serde_json::Value) -> Result<(), String> {
        let message = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        self.send_message(&message).await
    }

    /// Get the current diagnostics version for a URI.
    ///
    /// Returns 0 if no notification has been received for this URI.
    pub async fn diagnostics_version_for(&self, uri: &str) -> u64 {
        let cache = self.published_diagnostics.lock().await;
        cache.get(uri).map_or(0, |e| e.version)
    }

    /// Wait for a `publishDiagnostics` notification for a URI with version > `after_version`.
    ///
    /// Returns the notification params if received, `None` on timeout.
    pub async fn wait_for_published_diagnostics(
        &self,
        uri: &str,
        after_version: u64,
        timeout: Duration,
    ) -> Option<serde_json::Value> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            {
                let cache = self.published_diagnostics.lock().await;
                if let Some(entry) = cache.get(uri)
                    && entry.version > after_version
                {
                    return Some(entry.params.clone());
                }
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                return None;
            }
            let _ = tokio::time::timeout(remaining, self.diagnostics_notify.notified()).await;
        }
    }

    async fn send_message(&self, message: &serde_json::Value) -> Result<(), String> {
        let body = serde_json::to_string(message).map_err(|e| e.to_string())?;
        let header = format!("Content-Length: {}\r\n\r\n", body.len());

        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(header.as_bytes())
            .await
            .map_err(|e| format!("failed to write header: {e}"))?;
        stdin
            .write_all(body.as_bytes())
            .await
            .map_err(|e| format!("failed to write body: {e}"))?;
        stdin
            .flush()
            .await
            .map_err(|e| format!("failed to flush: {e}"))?;
        Ok(())
    }

    /// Background reader loop: parse Content-Length framed messages and dispatch.
    async fn read_loop(
        stdout: ChildStdout,
        pending: Arc<Mutex<HashMap<i64, oneshot::Sender<serde_json::Value>>>>,
        published_diagnostics: Arc<Mutex<HashMap<String, DiagnosticsEntry>>>,
        diagnostics_version: Arc<AtomicU64>,
        diagnostics_notify: Arc<Notify>,
    ) -> Result<(), String> {
        let mut reader = BufReader::new(stdout);
        let mut header_buf = String::new();

        loop {
            // Parse headers
            let content_length = loop {
                header_buf.clear();
                let n = reader
                    .read_line(&mut header_buf)
                    .await
                    .map_err(|e| format!("read header: {e}"))?;
                if n == 0 {
                    return Err("EOF reading headers".into());
                }
                let trimmed = header_buf.trim();
                if trimmed.is_empty() {
                    // Empty line = end of headers, but we need content-length
                    // This shouldn't happen before we've seen Content-Length
                    continue;
                }
                if let Some(len_str) = trimmed.strip_prefix("Content-Length:") {
                    let len: usize = len_str
                        .trim()
                        .parse()
                        .map_err(|e| format!("invalid Content-Length: {e}"))?;
                    // Read the blank line after headers
                    header_buf.clear();
                    reader
                        .read_line(&mut header_buf)
                        .await
                        .map_err(|e| format!("read blank line: {e}"))?;
                    break len;
                }
                // Skip other headers (e.g., Content-Type)
            };

            // Read body
            let mut body = vec![0u8; content_length];
            reader
                .read_exact(&mut body)
                .await
                .map_err(|e| format!("read body: {e}"))?;

            let msg: serde_json::Value =
                serde_json::from_slice(&body).map_err(|e| format!("parse JSON: {e}"))?;

            // Dispatch response (has "id") vs notification (no "id")
            if let Some(id) = msg.get("id").and_then(|v| v.as_i64()) {
                let mut pending = pending.lock().await;
                if let Some(tx) = pending.remove(&id) {
                    // Send the result (or error)
                    let result = if let Some(result) = msg.get("result") {
                        result.clone()
                    } else if let Some(error) = msg.get("error") {
                        error.clone()
                    } else {
                        serde_json::Value::Null
                    };
                    let _ = tx.send(result);
                }
            } else if let Some(method) = msg.get("method").and_then(|v| v.as_str()) {
                // Handle server notifications
                if method == "textDocument/publishDiagnostics"
                    && let Some(params) = msg.get("params")
                    && let Some(uri) = params.get("uri").and_then(|v| v.as_str())
                {
                    let version = diagnostics_version.fetch_add(1, Ordering::Relaxed) + 1;
                    let mut cache = published_diagnostics.lock().await;
                    cache.insert(
                        uri.to_string(),
                        DiagnosticsEntry {
                            version,
                            params: params.clone(),
                        },
                    );
                    diagnostics_notify.notify_waiters();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encode a JSON-RPC message with Content-Length framing.
    fn encode_message(body: &str) -> Vec<u8> {
        format!("Content-Length: {}\r\n\r\n{}", body.len(), body).into_bytes()
    }

    /// Parse Content-Length from a header string.
    fn parse_content_length(header: &str) -> Option<usize> {
        let trimmed = header.trim();
        trimmed
            .strip_prefix("Content-Length:")
            .and_then(|s| s.trim().parse().ok())
    }

    #[test]
    fn encode_message_format() {
        let body = r#"{"jsonrpc":"2.0","id":1,"method":"test"}"#;
        let encoded = encode_message(body);
        let s = String::from_utf8(encoded).unwrap();
        assert!(s.starts_with("Content-Length: 40\r\n\r\n"));
        assert!(s.ends_with(body));
    }

    #[test]
    fn parse_content_length_valid() {
        assert_eq!(parse_content_length("Content-Length: 42"), Some(42));
        assert_eq!(parse_content_length("Content-Length:42"), Some(42));
        assert_eq!(parse_content_length("  Content-Length: 100  "), Some(100));
    }

    #[test]
    fn parse_content_length_invalid() {
        assert_eq!(parse_content_length("Content-Type: application/json"), None);
        assert_eq!(parse_content_length(""), None);
        assert_eq!(parse_content_length("Content-Length: abc"), None);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let body = r#"{"jsonrpc":"2.0","method":"initialized","params":{}}"#;
        let encoded = encode_message(body);
        let s = String::from_utf8(encoded).unwrap();

        // Extract content-length from encoded
        let header_end = s.find("\r\n\r\n").unwrap();
        let header = &s[..header_end];
        let len = parse_content_length(header).unwrap();
        let decoded_body = &s[header_end + 4..];
        assert_eq!(decoded_body.len(), len);
        assert_eq!(decoded_body, body);
    }

    #[test]
    fn encode_empty_body() {
        let encoded = encode_message("");
        let s = String::from_utf8(encoded).unwrap();
        assert_eq!(s, "Content-Length: 0\r\n\r\n");
    }

    #[test]
    fn encode_unicode_body() {
        let body = r#"{"message":"hello 世界"}"#;
        let encoded = encode_message(body);
        let s = String::from_utf8(encoded).unwrap();
        // Content-Length is in bytes
        let expected_len = body.len();
        assert!(s.starts_with(&format!("Content-Length: {expected_len}\r\n\r\n")));
    }

    #[tokio::test]
    async fn diagnostics_version_tracking() {
        let cache: Arc<Mutex<HashMap<String, DiagnosticsEntry>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let notify = Arc::new(Notify::new());

        // No entry → version 0
        {
            let c = cache.lock().await;
            assert_eq!(c.get("file:///test.rs").map_or(0, |e| e.version), 0);
        }

        // Insert entry → version 1
        {
            let mut c = cache.lock().await;
            c.insert(
                "file:///test.rs".to_string(),
                DiagnosticsEntry {
                    version: 1,
                    params: serde_json::json!({"diagnostics": []}),
                },
            );
        }
        {
            let c = cache.lock().await;
            assert_eq!(c.get("file:///test.rs").unwrap().version, 1);
        }

        // Update entry → version 2
        {
            let mut c = cache.lock().await;
            c.insert(
                "file:///test.rs".to_string(),
                DiagnosticsEntry {
                    version: 2,
                    params: serde_json::json!({"diagnostics": [{"message": "error"}]}),
                },
            );
            notify.notify_waiters();
        }
        {
            let c = cache.lock().await;
            assert_eq!(c.get("file:///test.rs").unwrap().version, 2);
        }
    }

    #[tokio::test]
    async fn wait_for_diagnostics_after_version() {
        let cache: Arc<Mutex<HashMap<String, DiagnosticsEntry>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let notify = Arc::new(Notify::new());

        // Insert initial entry (version 1, empty diagnostics)
        {
            let mut c = cache.lock().await;
            c.insert(
                "file:///test.rs".to_string(),
                DiagnosticsEntry {
                    version: 1,
                    params: serde_json::json!({"diagnostics": []}),
                },
            );
        }

        let cache_clone = Arc::clone(&cache);
        let notify_clone = Arc::clone(&notify);

        // Spawn task that will update to version 2 with diagnostics after 50ms
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let mut c = cache_clone.lock().await;
            c.insert(
                "file:///test.rs".to_string(),
                DiagnosticsEntry {
                    version: 2,
                    params: serde_json::json!({
                        "diagnostics": [{"range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}, "severity": 1, "message": "type error"}]
                    }),
                },
            );
            notify_clone.notify_waiters();
        });

        // Wait for version > 1 (should wake after ~50ms)
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        let result = loop {
            {
                let c = cache.lock().await;
                if let Some(entry) = c.get("file:///test.rs") {
                    if entry.version > 1 {
                        break Some(entry.params.clone());
                    }
                }
            }
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break None;
            }
            let _ = tokio::time::timeout(remaining, notify.notified()).await;
        };

        assert!(result.is_some());
        let value = result.unwrap();
        let diags = value.get("diagnostics").unwrap().as_array().unwrap();
        assert_eq!(diags.len(), 1);
    }
}
