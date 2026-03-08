use serde::Deserialize;

/// Memory configuration for the orchestrator.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MemoryConfig {
    /// In-memory store (for development/testing).
    InMemory,
    /// PostgreSQL-backed store.
    Postgres {
        database_url: String,
        /// Optional embedding configuration for hybrid retrieval.
        #[serde(default)]
        embedding: Option<EmbeddingConfig>,
    },
}

/// Configuration for embedding generation (optional).
///
/// When configured under `[memory.embedding]`, enables hybrid retrieval
/// (BM25 + vector cosine) for improved recall quality.
#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    /// Provider name: "openai", "local", or "none" (default).
    #[serde(default = "default_embedding_provider")]
    pub provider: String,
    /// Model name for the embedding API.
    #[serde(default = "default_embedding_model")]
    pub model: String,
    /// Environment variable name containing the API key.
    #[serde(default = "default_embedding_api_key_env")]
    pub api_key_env: String,
    /// Base URL for the embedding API (optional, defaults to OpenAI).
    pub base_url: Option<String>,
    /// Embedding vector dimension (auto-detected from model if omitted).
    pub dimension: Option<usize>,
    /// Model cache directory for local embedding provider (optional).
    pub cache_dir: Option<String>,
}

fn default_embedding_provider() -> String {
    "none".into()
}

fn default_embedding_model() -> String {
    "text-embedding-3-small".into()
}

fn default_embedding_api_key_env() -> String {
    "OPENAI_API_KEY".into()
}

/// Knowledge base configuration for document retrieval.
#[derive(Debug, Deserialize)]
pub struct KnowledgeConfig {
    /// Maximum byte length per chunk.
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// Number of overlapping bytes between consecutive chunks.
    #[serde(default = "default_chunk_overlap")]
    pub chunk_overlap: usize,
    /// Document sources to index.
    #[serde(default)]
    pub sources: Vec<KnowledgeSourceConfig>,
}

fn default_chunk_size() -> usize {
    1000
}

fn default_chunk_overlap() -> usize {
    200
}

/// A single knowledge source to index.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum KnowledgeSourceConfig {
    /// A single file path.
    File { path: String },
    /// A glob pattern matching multiple files.
    Glob { pattern: String },
    /// A URL to fetch and index.
    Url { url: String },
}

/// Workspace configuration for agent home directories.
///
/// When configured, each agent gets a persistent home directory where it
/// can freely create and organize files. File tools resolve relative paths
/// against this directory.
#[derive(Debug, Deserialize)]
pub struct WorkspaceConfig {
    /// Workspace root directory. All agents share this single directory.
    /// Defaults to `~/.heartbit/workspaces` if not specified.
    #[serde(default = "default_workspace_root")]
    pub root: String,
    /// Environment variable allowlist for bash subprocesses.
    /// Empty = use `DAEMON_ENV_ALLOWLIST` defaults.
    #[serde(default)]
    pub env_allowlist: Vec<String>,
    /// Additional file path patterns to deny (e.g., `*.pem`, `secrets/`).
    #[serde(default)]
    pub protected_paths: Vec<String>,
    /// Enable Landlock filesystem sandbox (Linux only, requires `sandbox` feature).
    #[serde(default)]
    pub sandbox: bool,
    /// Additional paths the sandbox should allow reading (beyond system defaults).
    #[serde(default)]
    pub sandbox_read_paths: Vec<String>,
}

fn default_workspace_root() -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    format!("{home}/.heartbit/workspaces")
}

/// Restate server connection settings.
#[derive(Debug, Deserialize)]
pub struct RestateConfig {
    pub endpoint: String,
}

/// OpenTelemetry configuration.
#[derive(Debug, Deserialize)]
pub struct TelemetryConfig {
    /// OTLP endpoint (e.g., "http://localhost:4317")
    pub otlp_endpoint: String,
    /// Service name reported to the collector
    #[serde(default = "default_service_name")]
    pub service_name: String,
    /// Observability verbosity mode: "production", "analysis", or "debug".
    /// Overridden by `HEARTBIT_OBSERVABILITY` env var.
    #[serde(default)]
    pub observability_mode: Option<String>,
}

fn default_service_name() -> String {
    "heartbit".into()
}

/// LSP integration configuration.
///
/// When present, language servers are spawned lazily after file-modifying tools
/// and diagnostics are appended to tool output.
#[derive(Debug, Deserialize)]
pub struct LspConfig {
    /// Whether LSP integration is enabled. Default: `true` when the section is present.
    #[serde(default = "default_lsp_enabled")]
    pub enabled: bool,
}

fn default_lsp_enabled() -> bool {
    true
}
