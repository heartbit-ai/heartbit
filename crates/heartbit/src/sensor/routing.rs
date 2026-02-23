use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::llm::DynLlmProvider;

/// Tier of model to use for a given processing step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    /// Local SLM (e.g., Ollama, llama.cpp) — lowest latency, zero API cost.
    Local,
    /// Cloud-hosted lightweight model (e.g., Haiku, GPT-4o-mini).
    CloudLight,
    /// Cloud-hosted frontier model (e.g., Opus, GPT-4o).
    CloudFrontier,
    /// Vision-capable model for image/document analysis.
    Vision,
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelTier::Local => write!(f, "local"),
            ModelTier::CloudLight => write!(f, "cloud_light"),
            ModelTier::CloudFrontier => write!(f, "cloud_frontier"),
            ModelTier::Vision => write!(f, "vision"),
        }
    }
}

/// Routes processing steps to the appropriate model tier and provider.
///
/// The router holds one provider per tier and selects the cheapest/fastest
/// model that can handle each processing step.
pub struct ModelRouter {
    local_provider: Option<Arc<dyn DynLlmProvider>>,
    light_provider: Arc<dyn DynLlmProvider>,
    frontier_provider: Arc<dyn DynLlmProvider>,
    vision_provider: Option<Arc<dyn DynLlmProvider>>,
}

impl ModelRouter {
    /// Create a new model router.
    ///
    /// `light_provider` and `frontier_provider` are required.
    /// `local_provider` and `vision_provider` are optional upgrades.
    pub fn new(
        local_provider: Option<Arc<dyn DynLlmProvider>>,
        light_provider: Arc<dyn DynLlmProvider>,
        frontier_provider: Arc<dyn DynLlmProvider>,
        vision_provider: Option<Arc<dyn DynLlmProvider>>,
    ) -> Self {
        Self {
            local_provider,
            light_provider,
            frontier_provider,
            vision_provider,
        }
    }

    /// Route triage processing — prefer local SLM, fall back to cloud light.
    pub fn route_triage(&self) -> (ModelTier, Arc<dyn DynLlmProvider>) {
        if let Some(local) = &self.local_provider {
            (ModelTier::Local, Arc::clone(local))
        } else {
            (ModelTier::CloudLight, Arc::clone(&self.light_provider))
        }
    }

    /// Route summarization — same preference as triage.
    pub fn route_summarize(&self) -> (ModelTier, Arc<dyn DynLlmProvider>) {
        if let Some(local) = &self.local_provider {
            (ModelTier::Local, Arc::clone(local))
        } else {
            (ModelTier::CloudLight, Arc::clone(&self.light_provider))
        }
    }

    /// Route reasoning/analysis — always uses frontier model.
    pub fn route_reason(&self) -> (ModelTier, Arc<dyn DynLlmProvider>) {
        (
            ModelTier::CloudFrontier,
            Arc::clone(&self.frontier_provider),
        )
    }

    /// Route vision processing — returns `None` if no vision provider configured.
    pub fn route_vision(&self) -> Option<(ModelTier, Arc<dyn DynLlmProvider>)> {
        self.vision_provider
            .as_ref()
            .map(|v| (ModelTier::Vision, Arc::clone(v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_tier_serde_roundtrip() {
        for tier in [
            ModelTier::Local,
            ModelTier::CloudLight,
            ModelTier::CloudFrontier,
            ModelTier::Vision,
        ] {
            let json = serde_json::to_string(&tier).unwrap();
            let back: ModelTier = serde_json::from_str(&json).unwrap();
            assert_eq!(back, tier);
        }
    }

    #[test]
    fn model_tier_snake_case() {
        assert_eq!(
            serde_json::to_string(&ModelTier::Local).unwrap(),
            r#""local""#
        );
        assert_eq!(
            serde_json::to_string(&ModelTier::CloudLight).unwrap(),
            r#""cloud_light""#
        );
        assert_eq!(
            serde_json::to_string(&ModelTier::CloudFrontier).unwrap(),
            r#""cloud_frontier""#
        );
        assert_eq!(
            serde_json::to_string(&ModelTier::Vision).unwrap(),
            r#""vision""#
        );
    }

    #[test]
    fn model_tier_display() {
        assert_eq!(ModelTier::Local.to_string(), "local");
        assert_eq!(ModelTier::CloudLight.to_string(), "cloud_light");
        assert_eq!(ModelTier::CloudFrontier.to_string(), "cloud_frontier");
        assert_eq!(ModelTier::Vision.to_string(), "vision");
    }

    #[test]
    fn model_tier_deserialize_from_string() {
        let tier: ModelTier = serde_json::from_str(r#""cloud_frontier""#).unwrap();
        assert_eq!(tier, ModelTier::CloudFrontier);
    }

    // --- ModelRouter tests ---

    use crate::Error;
    use crate::llm::types::{
        CompletionRequest, CompletionResponse, ContentBlock, StopReason, TokenUsage,
    };

    struct MockProvider(&'static str);

    impl DynLlmProvider for MockProvider {
        fn complete<'a>(
            &'a self,
            _req: CompletionRequest,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<CompletionResponse, Error>> + Send + 'a>,
        > {
            Box::pin(async {
                Ok(CompletionResponse {
                    content: vec![ContentBlock::Text { text: "ok".into() }],
                    stop_reason: StopReason::EndTurn,
                    usage: TokenUsage::default(),
                })
            })
        }

        fn stream_complete<'a>(
            &'a self,
            _req: CompletionRequest,
            _on_text: &'a crate::llm::OnText,
        ) -> std::pin::Pin<
            Box<dyn std::future::Future<Output = Result<CompletionResponse, Error>> + Send + 'a>,
        > {
            Box::pin(async { Err(Error::Sensor("not supported".into())) })
        }

        fn model_name(&self) -> Option<&str> {
            Some(self.0)
        }
    }

    #[test]
    fn route_triage_prefers_local() {
        let router = ModelRouter::new(
            Some(Arc::new(MockProvider("local"))),
            Arc::new(MockProvider("light")),
            Arc::new(MockProvider("frontier")),
            None,
        );
        let (tier, provider) = router.route_triage();
        assert_eq!(tier, ModelTier::Local);
        assert_eq!(provider.model_name(), Some("local"));
    }

    #[test]
    fn route_triage_falls_back_to_light() {
        let router = ModelRouter::new(
            None,
            Arc::new(MockProvider("light")),
            Arc::new(MockProvider("frontier")),
            None,
        );
        let (tier, provider) = router.route_triage();
        assert_eq!(tier, ModelTier::CloudLight);
        assert_eq!(provider.model_name(), Some("light"));
    }

    #[test]
    fn route_summarize_prefers_local() {
        let router = ModelRouter::new(
            Some(Arc::new(MockProvider("local"))),
            Arc::new(MockProvider("light")),
            Arc::new(MockProvider("frontier")),
            None,
        );
        let (tier, _) = router.route_summarize();
        assert_eq!(tier, ModelTier::Local);
    }

    #[test]
    fn route_summarize_falls_back_to_light() {
        let router = ModelRouter::new(
            None,
            Arc::new(MockProvider("light")),
            Arc::new(MockProvider("frontier")),
            None,
        );
        let (tier, _) = router.route_summarize();
        assert_eq!(tier, ModelTier::CloudLight);
    }

    #[test]
    fn route_reason_always_frontier() {
        let router = ModelRouter::new(
            Some(Arc::new(MockProvider("local"))),
            Arc::new(MockProvider("light")),
            Arc::new(MockProvider("frontier")),
            Some(Arc::new(MockProvider("vision"))),
        );
        let (tier, provider) = router.route_reason();
        assert_eq!(tier, ModelTier::CloudFrontier);
        assert_eq!(provider.model_name(), Some("frontier"));
    }

    #[test]
    fn route_vision_returns_some_when_configured() {
        let router = ModelRouter::new(
            None,
            Arc::new(MockProvider("light")),
            Arc::new(MockProvider("frontier")),
            Some(Arc::new(MockProvider("vision"))),
        );
        let result = router.route_vision();
        assert!(result.is_some());
        let (tier, provider) = result.unwrap();
        assert_eq!(tier, ModelTier::Vision);
        assert_eq!(provider.model_name(), Some("vision"));
    }

    #[test]
    fn route_vision_returns_none_when_not_configured() {
        let router = ModelRouter::new(
            None,
            Arc::new(MockProvider("light")),
            Arc::new(MockProvider("frontier")),
            None,
        );
        assert!(router.route_vision().is_none());
    }
}
