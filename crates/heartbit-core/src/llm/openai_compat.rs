//! OpenAI-compatible LLM provider for any API that follows the OpenAI chat completions format.

use reqwest::Client;
use reqwest::redirect::Policy;
use std::time::Duration;

use crate::error::Error;
use crate::llm::LlmProvider;
use crate::llm::types::{CompletionRequest, CompletionResponse};

/// Build a hardened reqwest Client.
///
/// SECURITY (F-LLM-1, F-LLM-2, F-LLM-8): when `enforce_https` is true, the
/// client refuses non-HTTPS URLs entirely — protecting custom auth headers
/// like `api-key` (Azure-style) which reqwest does NOT strip on cross-host
/// redirect. When `false` (used only for `AuthStyle::None`, i.e. local
/// providers like Ollama/vLLM), HTTP is allowed.
fn build_secure_client(enforce_https: bool) -> Result<Client, Error> {
    let mut builder = Client::builder()
        .redirect(Policy::none())
        .connect_timeout(Duration::from_secs(10))
        .timeout(Duration::from_secs(120));
    if enforce_https {
        builder = builder.https_only(true);
    }
    builder.build().map_err(Error::from)
}

/// Authentication style for OpenAI-compatible API endpoints.
#[derive(Debug, Clone)]
pub enum AuthStyle {
    /// Standard `Authorization: Bearer <key>` header (default for most providers).
    Bearer,
    /// Custom header name for the API key (e.g., Azure uses `api-key`).
    ApiKeyHeader(&'static str),
    /// No authentication (local models like Ollama, vLLM, LM Studio).
    None,
}

/// Generalized OpenAI-compatible LLM provider.
///
/// Works with any endpoint that speaks the OpenAI chat completions format:
/// OpenRouter, Groq, DeepSeek, Together, Mistral, Fireworks, Ollama, vLLM, etc.
pub struct OpenAiCompatProvider {
    client: Client,
    api_key: String,
    model: String,
    base_url: String,
    auth_style: AuthStyle,
}

impl OpenAiCompatProvider {
    /// Create a new provider with custom base URL and auth style.
    ///
    /// **Security**: when `auth_style != AuthStyle::None`, the client enforces
    /// HTTPS — a non-HTTPS `base_url` will fail at request time. This protects
    /// API keys (`Authorization: Bearer ...` and `AuthStyle::ApiKeyHeader`)
    /// from being sent in clear over HTTP. For local providers like Ollama
    /// or vLLM that need plain HTTP, use `AuthStyle::None` (or the
    /// [`OpenAiCompatProvider::local`] convenience constructor).
    pub fn new(
        api_key: impl Into<String>,
        model: impl Into<String>,
        base_url: impl Into<String>,
        auth_style: AuthStyle,
    ) -> Self {
        let enforce_https = !matches!(auth_style, AuthStyle::None);
        Self {
            client: build_secure_client(enforce_https)
                .expect("failed to build hardened HTTPS client for OpenAiCompatProvider"),
            api_key: api_key.into(),
            model: model.into(),
            base_url: base_url.into(),
            auth_style,
        }
    }

    /// Convenience constructor for OpenRouter.
    pub fn openrouter(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self::new(
            api_key,
            model,
            "https://openrouter.ai/api/v1",
            AuthStyle::Bearer,
        )
    }

    /// Convenience constructor for local providers (Ollama, vLLM, LM Studio) that need no auth.
    pub fn local(model: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new("", model, base_url, AuthStyle::None)
    }

    fn completions_url(&self) -> String {
        format!("{}/chat/completions", self.base_url.trim_end_matches('/'))
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.auth_style {
            AuthStyle::Bearer => req.header("Authorization", format!("Bearer {}", self.api_key)),
            AuthStyle::ApiKeyHeader(header_name) => req.header(*header_name, &self.api_key),
            AuthStyle::None => req,
        }
    }
}

impl LlmProvider for OpenAiCompatProvider {
    fn model_name(&self) -> Option<&str> {
        Some(&self.model)
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        let body = super::openrouter::build_openai_request(&self.model, &request)?;

        let req = self
            .client
            .post(self.completions_url())
            .header("Content-Type", "application/json")
            .json(&body);
        let response = self.apply_auth(req).send().await?;

        if !response.status().is_success() {
            return Err(super::api_error_from_response(response).await);
        }

        let api_response: super::openrouter::OpenAiResponse = response.json().await?;
        super::openrouter::into_completion_response(api_response)
    }

    async fn stream_complete(
        &self,
        request: CompletionRequest,
        on_text: &crate::llm::OnText,
    ) -> Result<CompletionResponse, Error> {
        fn noop(_: &str) {}
        self.stream_inner(request, on_text, &noop).await
    }

    async fn stream_complete_with_reasoning(
        &self,
        request: CompletionRequest,
        on_text: &crate::llm::OnText,
        on_reasoning: &crate::llm::OnReasoning,
    ) -> Result<CompletionResponse, Error> {
        self.stream_inner(request, on_text, on_reasoning).await
    }
}

impl OpenAiCompatProvider {
    /// Shared streaming body for both text-only and reasoning-aware streaming.
    async fn stream_inner(
        &self,
        request: CompletionRequest,
        on_text: &crate::llm::OnText,
        on_reasoning: &crate::llm::OnReasoning,
    ) -> Result<CompletionResponse, Error> {
        let mut body = super::openrouter::build_openai_request(&self.model, &request)?;
        body["stream"] = serde_json::json!(true);
        body["stream_options"] = serde_json::json!({"include_usage": true});

        let req = self
            .client
            .post(self.completions_url())
            .header("Content-Type", "application/json")
            .json(&body);
        let response = self.apply_auth(req).send().await?;

        if !response.status().is_success() {
            return Err(super::api_error_from_response(response).await);
        }

        super::openrouter::parse_openai_stream_with_reasoning(
            response.bytes_stream(),
            on_text,
            on_reasoning,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openrouter_convenience_constructor() {
        let p = OpenAiCompatProvider::openrouter("key", "model");
        assert_eq!(p.base_url, "https://openrouter.ai/api/v1");
        assert!(matches!(p.auth_style, AuthStyle::Bearer));
        assert_eq!(p.model, "model");
    }

    #[test]
    fn local_convenience_constructor() {
        let p = OpenAiCompatProvider::local("llama3", "http://localhost:11434/v1");
        assert!(matches!(p.auth_style, AuthStyle::None));
        assert_eq!(p.api_key, "");
        assert_eq!(p.model, "llama3");
    }

    #[test]
    fn completions_url_strips_trailing_slash() {
        let p = OpenAiCompatProvider::new("k", "m", "http://example.com/v1/", AuthStyle::Bearer);
        assert_eq!(
            p.completions_url(),
            "http://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn completions_url_no_trailing_slash() {
        let p = OpenAiCompatProvider::new("k", "m", "http://example.com/v1", AuthStyle::Bearer);
        assert_eq!(
            p.completions_url(),
            "http://example.com/v1/chat/completions"
        );
    }

    #[test]
    fn apply_auth_bearer() {
        let p = OpenAiCompatProvider::new("my-key", "m", "http://x", AuthStyle::Bearer);
        let client = Client::new();
        let req = client.get("http://example.com");
        let req = p.apply_auth(req).build().expect("build request");
        let auth = req.headers().get("Authorization").expect("auth header");
        assert_eq!(auth.to_str().expect("header value"), "Bearer my-key");
    }

    #[test]
    fn apply_auth_api_key_header() {
        let p = OpenAiCompatProvider::new(
            "azure-key",
            "m",
            "http://x",
            AuthStyle::ApiKeyHeader("api-key"),
        );
        let client = Client::new();
        let req = client.get("http://example.com");
        let req = p.apply_auth(req).build().expect("build request");
        let key = req.headers().get("api-key").expect("api-key header");
        assert_eq!(key.to_str().expect("header value"), "azure-key");
    }

    #[test]
    fn apply_auth_none() {
        let p = OpenAiCompatProvider::new("ignored", "m", "http://x", AuthStyle::None);
        let client = Client::new();
        let req = client.get("http://example.com");
        let req = p.apply_auth(req).build().expect("build request");
        assert!(req.headers().get("Authorization").is_none());
        assert!(req.headers().get("api-key").is_none());
    }

    #[test]
    fn is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<OpenAiCompatProvider>();
    }
}
