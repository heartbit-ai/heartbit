//! MAGUS-inspired multi-agent perception scaffold.
//!
//! Maps each `SensorModality` to a dedicated perceiver configuration that
//! defines how raw sensor data should be interpreted before reaching the
//! orchestrator's reasoning loop.
//!
//! ## MAGUS principles
//!
//! **Modality isolation:** Each sensor modality gets its own perceiver with a
//! specialized system prompt and model tier. Text sensors use lightweight models,
//! image sensors use vision-capable models, etc. This prevents cross-modality
//! interference and allows independent scaling.
//!
//! **Structured output:** Perceivers produce typed JSON (via `output_schema`)
//! rather than free-form text. This enables reliable downstream parsing and
//! consistent story correlation.
//!
//! **Attention-based fusion (TODO):** When a story spans multiple modalities
//! (e.g., an email with an attached image), a fusion layer should attend to
//! each perceiver's output and produce a unified representation. Requires:
//! - Cross-attention mechanism between perceiver outputs.
//! - Priority weighting based on modality relevance to the story.
//! - Token budget allocation across perceivers.
//!
//! **Confidence propagation (TODO):** Each perceiver should emit a confidence
//! score alongside its structured output. Low-confidence perceptions trigger
//! escalation to a higher-tier model or human review. Requires:
//! - Calibrated confidence scores from each perceiver.
//! - Threshold-based escalation rules per modality.
//! - Feedback loop to improve calibration over time.

pub mod perceivers;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::sensor::SensorModality;
use crate::sensor::routing::ModelTier;

/// Configuration for a modality-specific perceiver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceiverConfig {
    /// System prompt guiding the perceiver's interpretation.
    pub system_prompt: String,
    /// Model tier to use for this perceiver.
    pub model_tier: ModelTier,
    /// Optional JSON schema for structured output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_schema: Option<Value>,
    /// Tool names available to this perceiver (e.g., "web_search" for link resolution).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
}

/// Registry mapping each sensor modality to its perceiver configuration.
pub struct PerceiverRegistry {
    perceivers: HashMap<SensorModality, PerceiverConfig>,
}

impl Default for PerceiverRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl PerceiverRegistry {
    /// Create a default registry with baseline perceiver configs per modality.
    pub fn new() -> Self {
        let mut perceivers = HashMap::new();

        perceivers.insert(
            SensorModality::Text,
            PerceiverConfig {
                system_prompt: "Extract key entities, sentiment, and action items from the text. \
                    Output structured JSON with fields: entities, sentiment, action_items, summary."
                    .into(),
                model_tier: ModelTier::CloudLight,
                output_schema: None,
                tools: vec![],
            },
        );

        perceivers.insert(
            SensorModality::Image,
            PerceiverConfig {
                system_prompt: "Describe the image content, identify objects, text (OCR), \
                    and any actionable information. Output structured JSON."
                    .into(),
                model_tier: ModelTier::Vision,
                output_schema: None,
                tools: vec![],
            },
        );

        perceivers.insert(
            SensorModality::Audio,
            PerceiverConfig {
                system_prompt: "Analyze the audio transcription. Extract speakers, topics, \
                    decisions, and action items. Output structured JSON."
                    .into(),
                model_tier: ModelTier::CloudLight,
                output_schema: None,
                tools: vec![],
            },
        );

        perceivers.insert(
            SensorModality::Structured,
            PerceiverConfig {
                system_prompt: "Interpret the structured data. Identify anomalies, trends, \
                    and noteworthy values. Output structured JSON with analysis."
                    .into(),
                model_tier: ModelTier::Local,
                output_schema: None,
                tools: vec![],
            },
        );

        Self { perceivers }
    }

    /// Look up the perceiver config for a given modality.
    pub fn get(&self, modality: &SensorModality) -> Option<&PerceiverConfig> {
        self.perceivers.get(modality)
    }

    /// Return perceiver configs for only the modalities present in the input list.
    ///
    /// Useful for story correlation: given the modalities in a story, retrieve
    /// only the relevant perceivers.
    pub fn modalities_for_story(
        &self,
        modalities: &[SensorModality],
    ) -> Vec<(SensorModality, &PerceiverConfig)> {
        modalities
            .iter()
            .filter_map(|m| self.perceivers.get(m).map(|config| (*m, config)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_registry_has_all_modalities() {
        let registry = PerceiverRegistry::new();
        assert!(registry.get(&SensorModality::Text).is_some());
        assert!(registry.get(&SensorModality::Image).is_some());
        assert!(registry.get(&SensorModality::Audio).is_some());
        assert!(registry.get(&SensorModality::Structured).is_some());
    }

    #[test]
    fn get_returns_correct_config() {
        let registry = PerceiverRegistry::new();

        let text_config = registry.get(&SensorModality::Text).unwrap();
        assert_eq!(text_config.model_tier, ModelTier::CloudLight);

        let image_config = registry.get(&SensorModality::Image).unwrap();
        assert_eq!(image_config.model_tier, ModelTier::Vision);

        let audio_config = registry.get(&SensorModality::Audio).unwrap();
        assert_eq!(audio_config.model_tier, ModelTier::CloudLight);

        let structured_config = registry.get(&SensorModality::Structured).unwrap();
        assert_eq!(structured_config.model_tier, ModelTier::Local);
    }

    #[test]
    fn modalities_for_story_filters_correctly() {
        let registry = PerceiverRegistry::new();

        let result = registry.modalities_for_story(&[SensorModality::Text, SensorModality::Image]);
        assert_eq!(result.len(), 2);

        let modalities: Vec<SensorModality> = result.iter().map(|(m, _)| *m).collect();
        assert!(modalities.contains(&SensorModality::Text));
        assert!(modalities.contains(&SensorModality::Image));
    }

    #[test]
    fn modalities_for_story_empty_input() {
        let registry = PerceiverRegistry::new();
        let result = registry.modalities_for_story(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn modalities_for_story_all_modalities() {
        let registry = PerceiverRegistry::new();
        let all = [
            SensorModality::Text,
            SensorModality::Image,
            SensorModality::Audio,
            SensorModality::Structured,
        ];
        let result = registry.modalities_for_story(&all);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn perceiver_config_serde_roundtrip() {
        let config = PerceiverConfig {
            system_prompt: "Analyze text".into(),
            model_tier: ModelTier::CloudLight,
            output_schema: Some(serde_json::json!({"type": "object"})),
            tools: vec!["web_search".into()],
        };
        let json = serde_json::to_string(&config).unwrap();
        let back: PerceiverConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.system_prompt, "Analyze text");
        assert_eq!(back.model_tier, ModelTier::CloudLight);
        assert!(back.output_schema.is_some());
        assert_eq!(back.tools, vec!["web_search"]);
    }

    #[test]
    fn perceiver_config_optional_fields_omitted() {
        let config = PerceiverConfig {
            system_prompt: "test".into(),
            model_tier: ModelTier::Local,
            output_schema: None,
            tools: vec![],
        };
        let json = serde_json::to_string(&config).unwrap();
        assert!(!json.contains("output_schema"));
        assert!(!json.contains("tools"));
    }
}
