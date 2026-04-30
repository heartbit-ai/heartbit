mod client;
mod language;
mod server;
mod types;

pub use language::{LanguageConfig, detect_language, find_server_config, is_file_modifying_tool};
pub use types::{Diagnostic, DiagnosticSeverity, format_diagnostics};

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use server::LspServer;

/// Manages LSP servers lazily — one per language.
///
/// When a file is changed, the manager detects the language, spawns a server
/// if needed, notifies it, then pulls diagnostics. Servers that fail to start
/// are marked broken and not retried for the session.
pub struct LspManager {
    servers: tokio::sync::Mutex<HashMap<String, LspServer>>,
    broken: Mutex<HashSet<String>>,
    workspace_root: PathBuf,
}

impl LspManager {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            servers: tokio::sync::Mutex::new(HashMap::new()),
            broken: Mutex::new(HashSet::new()),
            workspace_root,
        }
    }

    /// Notify that a file was changed (written/edited/patched).
    ///
    /// Reads the file content, spawns the language server lazily, sends
    /// didOpen/didChange, waits a debounce period, then pulls diagnostics.
    ///
    /// Returns empty if the language is unsupported, the server is broken,
    /// or the file cannot be read.
    pub async fn notify_file_changed(&self, path: &Path) -> Vec<Diagnostic> {
        let lang_id = match detect_language(path) {
            Some(id) => id,
            None => return Vec::new(),
        };

        // Check broken set (std::sync::Mutex — never held across .await)
        {
            let broken = self.broken.lock().expect("broken lock poisoned");
            if broken.contains(lang_id) {
                return Vec::new();
            }
        }

        // Read the file content
        let content = match tokio::fs::read_to_string(path).await {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!(path = %path.display(), error = %e, "failed to read file for LSP");
                return Vec::new();
            }
        };

        let mut servers = self.servers.lock().await;

        // Spawn server lazily
        if !servers.contains_key(lang_id) {
            let config = match find_server_config(lang_id) {
                Some(c) => c,
                None => return Vec::new(),
            };
            tracing::debug!(
                lang = %lang_id,
                workspace = %self.workspace_root.display(),
                "spawning LSP server"
            );
            match LspServer::spawn(config, &self.workspace_root).await {
                Ok(srv) => {
                    tracing::debug!(lang = %lang_id, "LSP server initialized");
                    servers.insert(lang_id.to_string(), srv);
                }
                Err(e) => {
                    tracing::warn!(lang = %lang_id, error = %e, "LSP server failed to start, marking broken");
                    self.broken
                        .lock()
                        .expect("broken lock poisoned")
                        .insert(lang_id.to_string());
                    return Vec::new();
                }
            }
        }

        let srv = servers.get_mut(lang_id).expect("server just inserted");

        // Notify the server
        if let Err(e) = srv.notify_file_changed(path, &content).await {
            tracing::debug!(error = %e, "failed to notify LSP server of file change");
            return Vec::new();
        }

        // Pull diagnostics (handles debounce and push/pull fallback internally)
        srv.pull_diagnostics(path).await
    }

    /// Get diagnostics for a file on demand (without notifying a change).
    pub async fn diagnostics(&self, path: &Path) -> Vec<Diagnostic> {
        let lang_id = match detect_language(path) {
            Some(id) => id,
            None => return Vec::new(),
        };

        let servers = self.servers.lock().await;
        match servers.get(lang_id) {
            Some(srv) => srv.pull_diagnostics(path).await,
            None => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn lsp_manager_new_creates_empty() {
        let mgr = LspManager::new(PathBuf::from("/tmp/test"));
        assert!(mgr.broken.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn notify_unsupported_language_returns_empty() {
        let mgr = LspManager::new(PathBuf::from("/tmp"));
        let diagnostics = mgr.notify_file_changed(Path::new("/tmp/README.md")).await;
        assert!(diagnostics.is_empty());
    }

    #[tokio::test]
    async fn diagnostics_without_server_returns_empty() {
        let mgr = LspManager::new(PathBuf::from("/tmp"));
        let diagnostics = mgr.diagnostics(Path::new("/tmp/test.rs")).await;
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn broken_server_not_retried() {
        let mgr = LspManager::new(PathBuf::from("/tmp"));
        mgr.broken.lock().unwrap().insert("rust".to_string());
        // Verify it's in the broken set
        assert!(mgr.broken.lock().unwrap().contains("rust"));
    }

    #[tokio::test]
    async fn notify_broken_language_returns_empty() {
        let mgr = LspManager::new(PathBuf::from("/tmp"));
        mgr.broken.lock().unwrap().insert("rust".to_string());
        let diagnostics = mgr.notify_file_changed(Path::new("/tmp/test.rs")).await;
        assert!(diagnostics.is_empty());
    }

    #[tokio::test]
    async fn notify_nonexistent_file_returns_empty() {
        let mgr = LspManager::new(PathBuf::from("/tmp"));
        let diagnostics = mgr
            .notify_file_changed(Path::new("/tmp/does_not_exist_12345.rs"))
            .await;
        assert!(diagnostics.is_empty());
    }
}
