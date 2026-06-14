//! OpenRouter model list — fetch the live catalog and filter it for the picker.
//! The `/api/v1/models` endpoint is public (no API key), so nothing secret is
//! sent. Parsing is pure and unit-tested on a fixture; only `fetch` does I/O.

use serde::Deserialize;

/// One model from OpenRouter's catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub context: Option<u64>,
}

/// Parse the `{ "data": [ { "id", "name", "context_length" }, … ] }` payload.
pub fn parse_models(json: &str) -> anyhow::Result<Vec<ModelEntry>> {
    #[derive(Deserialize)]
    struct Resp {
        data: Vec<Raw>,
    }
    #[derive(Deserialize)]
    struct Raw {
        id: String,
        #[serde(default)]
        name: Option<String>,
        #[serde(default)]
        context_length: Option<u64>,
    }
    let resp: Resp = serde_json::from_str(json)?;
    Ok(resp
        .data
        .into_iter()
        .map(|m| ModelEntry {
            name: m.name.unwrap_or_else(|| m.id.clone()),
            id: m.id,
            context: m.context_length,
        })
        .collect())
}

/// Indices (into `models`) whose id or name contains `query` (case-insensitive).
/// Empty query → all, in catalog order.
pub fn filter_models(models: &[ModelEntry], query: &str) -> Vec<usize> {
    let q = query.trim().to_lowercase();
    models
        .iter()
        .enumerate()
        .filter(|(_, m)| {
            q.is_empty() || m.id.to_lowercase().contains(&q) || m.name.to_lowercase().contains(&q)
        })
        .map(|(i, _)| i)
        .collect()
}

/// Fetch the live OpenRouter model catalog. No API key is sent (public endpoint).
pub async fn fetch_openrouter_models() -> anyhow::Result<Vec<ModelEntry>> {
    let body = reqwest::Client::new()
        .get("https://openrouter.ai/api/v1/models")
        .header("User-Agent", "heartbit-tui")
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    parse_models(&body)
}

/// Fetch the model list from an OpenAI-compatible endpoint's `/models` (e.g. the
/// ChatGPT-subscription Codex proxy, a local model server, or the real OpenAI
/// API). `base_url` is the `/v1` base (e.g. `http://127.0.0.1:10531/v1`); the
/// response is the same `{ "data": [ { "id" }, … ] }` shape `parse_models`
/// already handles (Codex models carry no name/context → name defaults to id).
/// Sends `HEARTBIT_OPENAI_API_KEY` as a bearer token when set (real HTTPS APIs);
/// a localhost proxy needs none.
pub async fn fetch_openai_models(base_url: &str) -> anyhow::Result<Vec<ModelEntry>> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let mut req = reqwest::Client::new()
        .get(&url)
        .header("User-Agent", "heartbit-tui");
    if let Ok(key) = std::env::var("HEARTBIT_OPENAI_API_KEY")
        && !key.trim().is_empty()
    {
        req = req.bearer_auth(key.trim());
    }
    let body = req.send().await?.error_for_status()?.text().await?;
    parse_models(&body)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"{
        "data": [
            {"id": "qwen/qwen3-235b-a22b-2507", "name": "Qwen3 235B", "context_length": 32768},
            {"id": "anthropic/claude-sonnet-4", "name": "Claude Sonnet 4", "context_length": 200000},
            {"id": "openai/gpt-4o", "name": "GPT-4o"}
        ]
    }"#;

    #[test]
    fn parse_models_extracts_id_name_context() {
        let m = parse_models(FIXTURE).unwrap();
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].id, "qwen/qwen3-235b-a22b-2507");
        assert_eq!(m[0].name, "Qwen3 235B");
        assert_eq!(m[0].context, Some(32768));
        assert_eq!(m[2].id, "openai/gpt-4o");
        assert_eq!(m[2].context, None, "missing context_length → None");
    }

    #[test]
    fn parse_models_defaults_name_to_id() {
        let m = parse_models(r#"{"data":[{"id":"x/y"}]}"#).unwrap();
        assert_eq!(m[0].id, "x/y");
        assert_eq!(m[0].name, "x/y");
    }

    #[test]
    fn parse_models_errors_on_garbage() {
        assert!(parse_models("not json").is_err());
    }

    #[test]
    fn parse_models_handles_openai_compat_codex_proxy_shape() {
        // The Codex proxy's /v1/models payload: id-only entries, no name/context.
        let proxy = r#"{"object":"list","data":[
            {"id":"gpt-5.5","object":"model","created":0,"owned_by":"codex-oauth"},
            {"id":"gpt-5.4-mini","object":"model","created":0,"owned_by":"codex-oauth"}
        ]}"#;
        let m = parse_models(proxy).unwrap();
        assert_eq!(m.len(), 2);
        assert_eq!(m[0].id, "gpt-5.5");
        assert_eq!(m[0].name, "gpt-5.5", "no name field → name defaults to id");
        assert_eq!(m[0].context, None);
    }

    #[test]
    fn filter_models_is_case_insensitive_substring() {
        let m = parse_models(FIXTURE).unwrap();
        assert_eq!(filter_models(&m, "claude"), vec![1]);
        assert_eq!(filter_models(&m, "GPT"), vec![2], "matches the name too");
        assert_eq!(filter_models(&m, "anthropic/"), vec![1]);
        assert_eq!(filter_models(&m, ""), vec![0, 1, 2], "empty → all in order");
        assert!(filter_models(&m, "nope").is_empty());
    }
}
