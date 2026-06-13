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
        let pending_drain = Arc::clone(&pending);
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
            // R4/A3: on reader exit (EOF, fatal framing error, server crash),
            // drop every pending response sender so in-flight `request()` callers
            // fail fast with "response channel closed" instead of hanging until
            // their own timeout, and the map retains no stale senders.
            pending_drain.lock().await.clear();
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
            // B6: register the notification waiter BEFORE reading the cache.
            // `Notify::notify_waiters()` stores no permit, so a notification
            // firing in the gap between the cache check and the await would be
            // lost — stalling diagnostics that already arrived until the full
            // remaining timeout elapses. `Notified::enable()` registers the
            // waiter eagerly (a bare `notified()` only registers on first poll).
            let notified = self.diagnostics_notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();

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
            let _ = tokio::time::timeout(remaining, notified.as_mut()).await;
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
    ///
    /// Generic over the byte source (`ChildStdout` in production) so the framing
    /// and resync behaviour can be unit-tested against in-memory inputs.
    async fn read_loop<R: tokio::io::AsyncRead + Unpin>(
        stdout: R,
        pending: Arc<Mutex<HashMap<i64, oneshot::Sender<serde_json::Value>>>>,
        published_diagnostics: Arc<Mutex<HashMap<String, DiagnosticsEntry>>>,
        diagnostics_version: Arc<AtomicU64>,
        diagnostics_notify: Arc<Notify>,
    ) -> Result<(), String> {
        let mut reader = BufReader::new(stdout);
        let mut header_buf = String::new();

        loop {
            // Parse the header block. B5: the LSP base protocol does NOT
            // guarantee `Content-Length` is the last header (a `Content-Type`
            // may follow it). Read header lines until the blank terminator,
            // capturing Content-Length wherever it appears, instead of assuming
            // exactly one more line follows it.
            let content_length = {
                let mut len: Option<usize> = None;
                loop {
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
                        // Blank line terminates the header block.
                        break;
                    }
                    if let Some(len_str) = trimmed.strip_prefix("Content-Length:") {
                        let parsed: usize = len_str
                            .trim()
                            .parse()
                            .map_err(|e| format!("invalid Content-Length: {e}"))?;
                        // SECURITY (F-LSP-1): cap Content-Length at 64 MiB. A
                        // hostile or compromised LSP server could send
                        // `Content-Length: 99999999999999`; without this cap, the
                        // `vec![0u8; len]` below would attempt a multi-TB
                        // allocation and OOM the agent.
                        const LSP_MAX_BODY_BYTES: usize = 64 * 1024 * 1024;
                        if parsed > LSP_MAX_BODY_BYTES {
                            return Err(format!(
                                "LSP Content-Length {parsed} exceeds cap of {LSP_MAX_BODY_BYTES} bytes (F-LSP-1)"
                            ));
                        }
                        len = Some(parsed);
                    }
                    // Any other header (e.g. Content-Type) is ignored; keep
                    // reading until the blank line.
                }
                match len {
                    Some(l) => l,
                    None => return Err("header block ended without Content-Length".into()),
                }
            };

            // Read body
            let mut body = vec![0u8; content_length];
            reader
                .read_exact(&mut body)
                .await
                .map_err(|e| format!("read body: {e}"))?;

            // B4: a single malformed body must NOT kill the reader task —
            // doing so silently disables ALL diagnostics for the rest of the
            // session (every later request resolves only via its own timeout).
            // The framing was valid (we read exactly `content_length` bytes), so
            // the stream is still aligned: skip this frame and resync on the next.
            let msg: serde_json::Value = match serde_json::from_slice(&body) {
                Ok(v) => v,
                Err(e) => {
                    tracing::debug!(error = %e, "LSP: skipping malformed JSON-RPC frame");
                    continue;
                }
            };

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

    /// Drive `read_loop` over a finite in-memory byte source (returns `Err` on
    /// EOF after consuming all frames — expected) and return the resulting
    /// diagnostics cache.
    async fn run_read_loop_to_eof(input: Vec<u8>) -> HashMap<String, DiagnosticsEntry> {
        let pending = Arc::new(Mutex::new(HashMap::new()));
        let diag = Arc::new(Mutex::new(HashMap::new()));
        let version = Arc::new(AtomicU64::new(0));
        let notify = Arc::new(Notify::new());
        let _ = JsonRpcClient::read_loop(
            &input[..],
            Arc::clone(&pending),
            Arc::clone(&diag),
            Arc::clone(&version),
            Arc::clone(&notify),
        )
        .await;
        let guard = diag.lock().await;
        guard.clone()
    }

    fn publish_diagnostics_frame(uri: &str) -> Vec<u8> {
        let body = format!(
            r#"{{"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{{"uri":"{uri}","diagnostics":[]}}}}"#
        );
        encode_message(&body)
    }

    #[tokio::test]
    async fn read_loop_handles_content_type_header_after_content_length() {
        // B5: Content-Length is not guaranteed to be the last header. A
        // Content-Type following it must not desync the body offset.
        let body = r#"{"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{"uri":"file:///x.rs","diagnostics":[]}}"#;
        let framed = format!(
            "Content-Length: {}\r\nContent-Type: application/vscode-jsonrpc; charset=utf-8\r\n\r\n{}",
            body.len(),
            body
        );
        let cache = run_read_loop_to_eof(framed.into_bytes()).await;
        assert!(
            cache.contains_key("file:///x.rs"),
            "a Content-Type header after Content-Length desynced the reader"
        );
    }

    #[tokio::test]
    async fn read_loop_skips_malformed_frame_and_continues() {
        // B4: one malformed body must not kill the reader for the whole session.
        let bad = "not valid json";
        let mut framed = encode_message(bad);
        framed.extend_from_slice(&publish_diagnostics_frame("file:///y.rs"));
        let cache = run_read_loop_to_eof(framed).await;
        assert!(
            cache.contains_key("file:///y.rs"),
            "a malformed frame killed the reader and dropped the next valid frame"
        );
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
                if let Some(entry) = c.get("file:///test.rs")
                    && entry.version > 1
                {
                    break Some(entry.params.clone());
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
