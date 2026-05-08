//! Blend recipe — declares how N writer profiles combine into one, with an
//! optional `PartialStyleProfile` of user-provided overrides applied last.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::voice::error::VoiceError;
use crate::voice::style::{
    EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, OpeningPattern,
    SentenceLengthTarget, SpecificityTarget, ThreadRhythm,
};

fn default_version() -> u32 {
    1
}

/// Recipe for blending N writer profiles into a single `StyleProfile`.
///
/// The blend algorithm itself lives in P1.2d; this type just declares the
/// inputs (writer references with weights) and any user-provided overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlendRecipe {
    /// Schema version. Currently must be 1.
    #[serde(default = "default_version")]
    pub version: u32,
    /// 1..=10 entries; weights sum to 1.0.
    pub blend: Vec<BlendEntry>,
    /// Optional user overrides applied after the blend.
    #[serde(default)]
    pub overrides: PartialStyleProfile,
}

/// One writer's contribution to a blend.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BlendEntry {
    /// Corpus key for this writer. Resolved against the corpus store in P1.2b.
    pub writer: String,
    /// Weight in the blend; in [0, 1]. All weights in a recipe sum to 1.0.
    pub weight: f64,
}

/// Every `StyleProfile` field as `Option<_>`. Used for blend overrides where
/// only a subset of fields is provided.
///
/// Constraints (sums, ranges) only apply after merge — this partial doesn't
/// validate on its own.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PartialStyleProfile {
    /// Override for `sentence_length_target`.
    #[serde(default)]
    pub sentence_length_target: Option<SentenceLengthTarget>,
    /// Override for `sentence_length_distribution`.
    #[serde(default)]
    pub sentence_length_distribution: Option<[u8; 4]>,
    /// Override for `fragment_frequency`.
    #[serde(default)]
    pub fragment_frequency: Option<FragmentFrequency>,
    /// Override for `opening_patterns`.
    #[serde(default)]
    pub opening_patterns: Option<Vec<OpeningPattern>>,
    /// Override for `opening_pattern_weights`.
    #[serde(default)]
    pub opening_pattern_weights: Option<Vec<f64>>,
    /// Override for `formatting`.
    #[serde(default)]
    pub formatting: Option<Formatting>,
    /// Override for `emoji_policy`.
    #[serde(default)]
    pub emoji_policy: Option<EmojiPolicy>,
    /// Override for `hashtag_policy`.
    #[serde(default)]
    pub hashtag_policy: Option<HashtagPolicy>,
    /// Override for `specificity_target`.
    #[serde(default)]
    pub specificity_target: Option<SpecificityTarget>,
    /// Override for `voice_traits`.
    #[serde(default)]
    pub voice_traits: Option<Vec<String>>,
    /// Override for `ai_tells_to_avoid`.
    #[serde(default)]
    pub ai_tells_to_avoid: Option<Vec<String>>,
    /// Override for `thread_rhythm`.
    #[serde(default)]
    pub thread_rhythm: Option<ThreadRhythm>,
    /// Override for `thread_max_length`.
    #[serde(default)]
    pub thread_max_length: Option<u32>,
    /// Override for `thread_opener_must_hook`.
    #[serde(default)]
    pub thread_opener_must_hook: Option<bool>,
    /// Override for `topical_obsessions`.
    #[serde(default)]
    pub topical_obsessions: Option<Vec<String>>,
    /// Override for `topical_avoidances`.
    #[serde(default)]
    pub topical_avoidances: Option<Vec<String>>,
}

impl BlendRecipe {
    /// Parse a `BlendRecipe` from a TOML string and validate it.
    pub fn from_toml(s: &str) -> Result<Self, VoiceError> {
        let parsed: Self = toml::from_str(s)?;
        parsed.validate()?;
        Ok(parsed)
    }

    /// Validation rules:
    /// - `version` must equal 1
    /// - `blend.len()` in 1..=10
    /// - `blend` weights each in [0, 1] and sum to 1.0 (epsilon 1e-6)
    /// - each `writer` non-empty after trim, unique within the recipe
    pub fn validate(&self) -> Result<(), VoiceError> {
        if self.version != 1 {
            return Err(VoiceError::UnsupportedVersion(self.version));
        }

        if !(1..=10).contains(&self.blend.len()) {
            return Err(VoiceError::Validation(format!(
                "blend must contain 1..=10 entries (got {})",
                self.blend.len()
            )));
        }

        let weights_sum: f64 = self.blend.iter().map(|e| e.weight).sum();
        if (weights_sum - 1.0).abs() > 1e-6 {
            return Err(VoiceError::Validation(format!(
                "blend weights must sum to 1.0 (got {weights_sum})"
            )));
        }

        for entry in &self.blend {
            if !(0.0..=1.0).contains(&entry.weight) {
                return Err(VoiceError::Validation(format!(
                    "each blend weight must be in [0, 1] (got {} for '{}')",
                    entry.weight, entry.writer
                )));
            }
            if entry.writer.trim().is_empty() {
                return Err(VoiceError::Validation(
                    "blend.writer must not be empty".into(),
                ));
            }
        }

        let mut seen: HashSet<&str> = HashSet::new();
        for entry in &self.blend {
            if !seen.insert(entry.writer.as_str()) {
                return Err(VoiceError::Validation(format!(
                    "duplicate writer in blend: '{}'",
                    entry.writer
                )));
            }
        }

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────
// P1.2d — blend algorithm
// ─────────────────────────────────────────────────────────────────────────

/// Errors raised by [`blend_profiles`].
#[derive(Debug, thiserror::Error)]
pub enum BlendError {
    /// The recipe references a writer handle that is not in the profiles map.
    #[error("missing profile for writer '{0}'")]
    MissingProfile(String),

    /// Result of merge + override application failed
    /// [`crate::voice::StyleProfile::validate`]. `inner` is the underlying
    /// validation error. Most commonly this surfaces when a user-supplied
    /// `recipe.overrides` violates a parallel-array length, sum-to-100 /
    /// sum-to-1.0, or range invariant.
    #[error("merged profile failed validation: {inner}")]
    PostMergeValidation {
        /// The underlying validation error from `StyleProfile::validate`.
        #[source]
        inner: VoiceError,
    },
}

/// Hare quota (largest-remainder method): distribute `total` units across
/// 4 buckets according to f64 weights, producing 4 integer values that
/// sum to exactly `total`.
///
/// Algorithm:
/// 1. Floor each weighted f64 to u32, accumulate `assigned`.
/// 2. residual = total - assigned (in 0..4).
/// 3. Sort buckets by (frac DESC, declaration index ASC).
/// 4. Add 1 to each of the first `residual` buckets in that order.
#[allow(dead_code)] // wired in by Task 3's blend_profiles
fn distribute_largest_remainder(weighted: [f64; 4], total: u32) -> [u8; 4] {
    let floors: [u32; 4] = [
        weighted[0] as u32,
        weighted[1] as u32,
        weighted[2] as u32,
        weighted[3] as u32,
    ];
    let assigned: u32 = floors[0] + floors[1] + floors[2] + floors[3];
    let residual = total.saturating_sub(assigned);

    let mut fracs: [(usize, f64); 4] = [(0, 0.0); 4];
    for i in 0..4 {
        fracs[i] = (i, weighted[i] - floors[i] as f64);
    }
    // Sort: largest fractional remainder first; declaration-order tiebreak.
    fracs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let mut out: [u8; 4] = [0; 4];
    for i in 0..4 {
        out[i] = floors[i].min(255) as u8;
    }
    let extras = (residual as usize).min(4);
    for (idx, _frac) in fracs.iter().take(extras) {
        out[*idx] = out[*idx].saturating_add(1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::style::EmDashPolicy;

    /// AI/tech blend (matches the umbrella heartbit-ghost spec §2.3 example).
    const VALID_BLEND: &str = r#"
version = 1

[[blend]]
writer = "karpathy"
weight = 0.30

[[blend]]
writer = "eladgil"
weight = 0.20

[[blend]]
writer = "swyx"
weight = 0.20

[[blend]]
writer = "naval"
weight = 0.15

[[blend]]
writer = "sama"
weight = 0.15
"#;

    #[test]
    fn five_writer_happy_path_parses_and_validates() {
        let recipe = BlendRecipe::from_toml(VALID_BLEND).expect("parses");
        assert_eq!(recipe.version, 1);
        assert_eq!(recipe.blend.len(), 5);
        assert_eq!(recipe.blend[0].writer, "karpathy");
        assert!((recipe.blend.iter().map(|e| e.weight).sum::<f64>() - 1.0).abs() < 1e-6);
        assert_eq!(recipe.overrides, PartialStyleProfile::default());
    }

    #[test]
    fn empty_blend_rejected() {
        let toml = "version = 1\nblend = []\n";
        let err = BlendRecipe::from_toml(toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("1..=10"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn eleven_entries_rejected() {
        let mut entries = String::from("version = 1\n");
        for i in 0..11 {
            entries.push_str(&format!(
                "[[blend]]\nwriter = \"w{i}\"\nweight = {}\n\n",
                1.0 / 11.0
            ));
        }
        let err = BlendRecipe::from_toml(&entries).unwrap_err();
        match err {
            VoiceError::Validation(msg) => assert!(msg.contains("1..=10")),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn weights_sum_0_99_rejected() {
        let toml = VALID_BLEND.replace("weight = 0.30", "weight = 0.29");
        let err = BlendRecipe::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("blend weights"));
                assert!(msg.contains("1.0"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn negative_weight_rejected() {
        // Weights sum to 1.0 (so the sum-rule passes) but one is outside [0, 1],
        // forcing the per-weight range check at validate() to fire.
        let toml = r#"
version = 1

[[blend]]
writer = "a"
weight = 1.5

[[blend]]
writer = "b"
weight = -0.5
"#;
        let err = BlendRecipe::from_toml(toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(
                    msg.contains("[0, 1]"),
                    "expected per-weight range error, got: {msg}"
                );
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_writer_rejected() {
        let toml = r#"
version = 1

[[blend]]
writer = "karpathy"
weight = 0.5

[[blend]]
writer = "karpathy"
weight = 0.5
"#;
        let err = BlendRecipe::from_toml(toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => {
                assert!(msg.contains("duplicate"));
                assert!(msg.contains("karpathy"));
            }
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn empty_writer_name_rejected() {
        let toml = r#"
version = 1

[[blend]]
writer = ""
weight = 1.0
"#;
        let err = BlendRecipe::from_toml(toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => assert!(msg.contains("writer must not be empty")),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn whitespace_only_writer_name_rejected() {
        let toml = r#"
version = 1

[[blend]]
writer = "   "
weight = 1.0
"#;
        let err = BlendRecipe::from_toml(toml).unwrap_err();
        match err {
            VoiceError::Validation(msg) => assert!(msg.contains("writer must not be empty")),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn version_999_rejected() {
        let toml = VALID_BLEND.replace("version = 1", "version = 999");
        let err = BlendRecipe::from_toml(&toml).unwrap_err();
        match err {
            VoiceError::UnsupportedVersion(v) => assert_eq!(v, 999),
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn version_absent_defaults_to_1() {
        let toml = VALID_BLEND.replace("version = 1\n", "");
        let recipe = BlendRecipe::from_toml(&toml).expect("parses");
        assert_eq!(recipe.version, 1);
    }

    #[test]
    fn partial_style_profile_default_is_all_none() {
        let p = PartialStyleProfile::default();
        assert!(p.sentence_length_target.is_none());
        assert!(p.formatting.is_none());
        assert!(p.thread_max_length.is_none());
        assert!(p.voice_traits.is_none());
    }

    #[test]
    fn partial_overrides_accept_subset_of_fields() {
        let toml = r#"
version = 1

[[blend]]
writer = "karpathy"
weight = 1.0

[overrides]
thread_max_length = 7

[overrides.formatting]
lowercase = true
periods = "always"
em_dashes = "forbidden"
quotation_marks = "double"
line_breaks = "single"
"#;
        let recipe = BlendRecipe::from_toml(toml).expect("parses");
        assert_eq!(recipe.overrides.thread_max_length, Some(7));
        let formatting = recipe.overrides.formatting.expect("set");
        assert!(formatting.lowercase);
        assert_eq!(formatting.em_dashes, EmDashPolicy::Forbidden);
        // Other fields remain None.
        assert!(recipe.overrides.voice_traits.is_none());
        assert!(recipe.overrides.thread_rhythm.is_none());
    }

    // ─────────────────────────────────────────────────────────────────────
    // P1.2d — BlendError display
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn missing_profile_error_includes_writer_handle() {
        let e = BlendError::MissingProfile("karpathy".to_string());
        let s = format!("{e}");
        assert!(s.contains("karpathy"), "got: {s}");
        assert!(s.starts_with("missing profile"), "got: {s}");
    }

    #[test]
    fn post_merge_validation_error_renders_inner_message() {
        let inner = VoiceError::Validation("thread_max_length must be 1..=25 (got 30)".to_string());
        let e = BlendError::PostMergeValidation { inner };
        let s = format!("{e}");
        assert!(
            s.starts_with("merged profile failed validation:"),
            "got: {s}"
        );
        assert!(s.contains("thread_max_length"), "got: {s}");
        assert!(s.contains("got 30"), "got: {s}");
    }

    #[test]
    fn post_merge_validation_inner_is_reachable() {
        let inner = VoiceError::Validation("test message".to_string());
        let e = BlendError::PostMergeValidation { inner };
        if let BlendError::PostMergeValidation { inner: i } = &e {
            assert!(matches!(i, VoiceError::Validation(_)));
        } else {
            panic!("not PostMergeValidation");
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // P1.2d — distribute_largest_remainder (Hare quota)
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn distribute_largest_remainder_already_summing_to_total_is_identity() {
        // 0.6 × [40, 30, 20, 10] + 0.4 × [20, 30, 30, 20] = [32.0, 30.0, 24.0, 14.0]
        // Floors sum to 100, residual is 0, no redistribution needed.
        let result = distribute_largest_remainder([32.0, 30.0, 24.0, 14.0], 100);
        assert_eq!(result, [32, 30, 24, 14]);
        assert_eq!(result.iter().map(|&x| x as u32).sum::<u32>(), 100);
    }

    #[test]
    fn distribute_largest_remainder_residual_uses_declaration_order_tiebreak() {
        // Weighted: [33.5, 33.0, 33.0, 0.5] — floors [33, 33, 33, 0] sum 99,
        // residual 1. Fractional parts: 0.5, 0.0, 0.0, 0.5 — buckets 0 and 3
        // tie at 0.5. Declaration-order tiebreak: bucket 0 wins, gets the +1.
        let result = distribute_largest_remainder([33.5, 33.0, 33.0, 0.5], 100);
        assert_eq!(result, [34, 33, 33, 0]);
        assert_eq!(result.iter().map(|&x| x as u32).sum::<u32>(), 100);
    }
}
