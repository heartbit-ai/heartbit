//! Blend recipe — declares how N writer profiles combine into one, with an
//! optional `PartialStyleProfile` of user-provided overrides applied last.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::voice::error::VoiceError;
use crate::voice::style::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
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

/// Vote across enum variants in `order` (declaration order). Returns the
/// variant with the highest summed weight; ties resolved by `order`.
///
/// Each profile contributes its `pick(profile)` choice with weight
/// `recipe.blend[i].weight`. Earlier-declared variants win ties.
fn weighted_vote_categorical<E: Copy + PartialEq>(
    blend: &[BlendEntry],
    profiles: &HashMap<String, StyleProfile>,
    pick: impl Fn(&StyleProfile) -> E,
    order: &[E],
) -> Result<E, BlendError> {
    let mut weights: Vec<f64> = vec![0.0; order.len()];
    for entry in blend {
        let profile = profiles
            .get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        let chosen = pick(profile);
        if let Some(idx) = order.iter().position(|v| *v == chosen) {
            weights[idx] += entry.weight;
        }
    }
    // Seed with order[0]; strict-> for declaration-order tiebreak.
    let mut best_idx = 0usize;
    let mut best_w = weights[0];
    for (i, &w) in weights.iter().enumerate().skip(1) {
        if w > best_w {
            best_w = w;
            best_idx = i;
        }
    }
    Ok(order[best_idx])
}

/// Declaration-order index of an `OpeningPattern` variant (0..7).
/// Used as a sort tiebreak in [`merge_opening_patterns`].
///
/// `OpeningPattern` is `#[non_exhaustive]` for downstream crates, but
/// within this crate the variants are exhaustive — so we list every
/// variant explicitly. Future variants must be added here (compiler
/// enforces it).
fn opening_pattern_decl_index(p: &OpeningPattern) -> usize {
    match p {
        OpeningPattern::ClaimFirst => 0,
        OpeningPattern::NumberFirst => 1,
        OpeningPattern::SceneFirst => 2,
        OpeningPattern::QuestionFirst => 3,
        OpeningPattern::AphoristicFirst => 4,
        OpeningPattern::AnecdoteFirst => 5,
        OpeningPattern::ContrarianFirst => 6,
    }
}

/// Merge per-profile (opening_patterns, opening_pattern_weights) parallel
/// arrays into a single normalized parallel pair. Output is sorted by
/// (weight DESC, declaration-order ASC).
fn merge_opening_patterns(
    blend: &[BlendEntry],
    profiles: &HashMap<String, StyleProfile>,
) -> Result<(Vec<OpeningPattern>, Vec<f64>), BlendError> {
    let mut acc: Vec<(OpeningPattern, f64)> = Vec::new();
    for entry in blend {
        let profile = profiles
            .get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        for (pat, w) in profile
            .opening_patterns
            .iter()
            .zip(profile.opening_pattern_weights.iter())
        {
            match acc.iter_mut().find(|(p, _)| p == pat) {
                Some(slot) => slot.1 += entry.weight * w,
                None => acc.push((*pat, entry.weight * w)),
            }
        }
    }
    // Defensive normalization against f64 drift.
    let total: f64 = acc.iter().map(|(_, w)| *w).sum();
    if total > 0.0 && (total - 1.0).abs() > 1e-9 {
        for (_, w) in acc.iter_mut() {
            *w /= total;
        }
    }
    // Sort by (weight DESC, declaration-order ASC).
    acc.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(opening_pattern_decl_index(&a.0).cmp(&opening_pattern_decl_index(&b.0)))
    });
    let (patterns, weights) = acc.into_iter().unzip();
    Ok((patterns, weights))
}

/// Union + dedup over `recipe.blend` iteration order. First occurrence
/// (across the recipe.blend iteration) wins position. Case-sensitive.
fn union_dedup_strings(
    blend: &[BlendEntry],
    profiles: &HashMap<String, StyleProfile>,
    pick: impl Fn(&StyleProfile) -> &[String],
) -> Result<Vec<String>, BlendError> {
    let mut out: Vec<String> = Vec::new();
    for entry in blend {
        let profile = profiles
            .get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        for s in pick(profile) {
            if !out.iter().any(|existing| existing == s) {
                out.push(s.clone());
            }
        }
    }
    Ok(out)
}

/// Apply [`PartialStyleProfile`] overrides to a merged `StyleProfile`.
/// For each `Some(value)` field, replace the merged value unconditionally.
fn apply_overrides(merged: &mut StyleProfile, overrides: &PartialStyleProfile) {
    if let Some(v) = overrides.sentence_length_target {
        merged.sentence_length_target = v;
    }
    if let Some(v) = overrides.sentence_length_distribution {
        merged.sentence_length_distribution = v;
    }
    if let Some(v) = overrides.fragment_frequency {
        merged.fragment_frequency = v;
    }
    if let Some(v) = overrides.opening_patterns.as_ref() {
        merged.opening_patterns = v.clone();
    }
    if let Some(v) = overrides.opening_pattern_weights.as_ref() {
        merged.opening_pattern_weights = v.clone();
    }
    if let Some(v) = overrides.formatting.as_ref() {
        merged.formatting = v.clone();
    }
    if let Some(v) = overrides.emoji_policy {
        merged.emoji_policy = v;
    }
    if let Some(v) = overrides.hashtag_policy {
        merged.hashtag_policy = v;
    }
    if let Some(v) = overrides.specificity_target {
        merged.specificity_target = v;
    }
    if let Some(v) = overrides.voice_traits.as_ref() {
        merged.voice_traits = v.clone();
    }
    if let Some(v) = overrides.ai_tells_to_avoid.as_ref() {
        merged.ai_tells_to_avoid = v.clone();
    }
    if let Some(v) = overrides.thread_rhythm {
        merged.thread_rhythm = v;
    }
    if let Some(v) = overrides.thread_max_length {
        merged.thread_max_length = v;
    }
    if let Some(v) = overrides.thread_opener_must_hook {
        merged.thread_opener_must_hook = v;
    }
    if let Some(v) = overrides.topical_obsessions.as_ref() {
        merged.topical_obsessions = v.clone();
    }
    if let Some(v) = overrides.topical_avoidances.as_ref() {
        merged.topical_avoidances = v.clone();
    }
}

/// Apply a [`BlendRecipe`] to a map of writer-handle → [`StyleProfile`],
/// producing one merged, validated [`StyleProfile`].
///
/// Per umbrella spec §2.3:
/// - Numeric fields → weighted average (Hare quota for the
///   `sentence_length_distribution` array)
/// - Categorical (enum + bool) fields → weighted vote, declaration-order
///   tiebreak (or `false` for booleans)
/// - List-of-strings fields → union, deduped, stable insertion order
/// - `opening_patterns` + `opening_pattern_weights` → weighted accumulator
///   per pattern, normalized to sum 1.0
/// - `recipe.overrides` (a [`PartialStyleProfile`]) → replaces blended
///   values where `Some(_)`, applied last
/// - Result is validated via [`StyleProfile::validate`] before returning
///
/// `recipe.validate()` is **NOT** called by this function; callers should
/// validate the recipe before passing it in (or rely on `from_toml`'s
/// internal validation).
pub fn blend_profiles(
    recipe: &BlendRecipe,
    profiles: &HashMap<String, StyleProfile>,
) -> Result<StyleProfile, BlendError> {
    let blend = &recipe.blend;

    // sentence_length_distribution: weighted f64 average + Hare quota.
    let mut weighted_dist: [f64; 4] = [0.0; 4];
    for entry in blend {
        let profile = profiles
            .get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        for (i, slot) in weighted_dist.iter_mut().enumerate() {
            *slot += entry.weight * profile.sentence_length_distribution[i] as f64;
        }
    }
    let sentence_length_distribution = distribute_largest_remainder(weighted_dist, 100);

    // thread_max_length: weighted f64 average → round → clamp.
    let mut weighted_thread_max: f64 = 0.0;
    for entry in blend {
        let profile = profiles
            .get(&entry.writer)
            .ok_or_else(|| BlendError::MissingProfile(entry.writer.clone()))?;
        weighted_thread_max += entry.weight * profile.thread_max_length as f64;
    }
    let thread_max_length: u32 = (weighted_thread_max.round() as u32).clamp(1, 25);

    // opening_patterns + opening_pattern_weights (parallel arrays).
    let (opening_patterns, opening_pattern_weights) = merge_opening_patterns(blend, profiles)?;

    // Categorical votes — explicit declaration-order slices for each enum.
    let sentence_length_target = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.sentence_length_target,
        &[
            SentenceLengthTarget::Short,
            SentenceLengthTarget::Mixed,
            SentenceLengthTarget::Long,
        ],
    )?;
    let fragment_frequency = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.fragment_frequency,
        &[
            FragmentFrequency::Rare,
            FragmentFrequency::Occasional,
            FragmentFrequency::Common,
        ],
    )?;
    let emoji_policy = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.emoji_policy,
        &[
            EmojiPolicy::Never,
            EmojiPolicy::RarePunchlineOnly,
            EmojiPolicy::Occasional,
            EmojiPolicy::Frequent,
        ],
    )?;
    let hashtag_policy = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.hashtag_policy,
        &[
            HashtagPolicy::Never,
            HashtagPolicy::Rare,
            HashtagPolicy::TopicRelevant,
            HashtagPolicy::Always,
        ],
    )?;
    let specificity_target = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.specificity_target,
        &[
            SpecificityTarget::Low,
            SpecificityTarget::Medium,
            SpecificityTarget::High,
        ],
    )?;
    let thread_rhythm = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.thread_rhythm,
        &[
            ThreadRhythm::Linear,
            ThreadRhythm::ListThenPayoff,
            ThreadRhythm::PunchlineCallbacks,
        ],
    )?;

    // Formatting sub-struct: per-field independent votes.
    let formatting = Formatting {
        lowercase: weighted_vote_categorical(
            blend,
            profiles,
            |p| p.formatting.lowercase,
            &[false, true], // tiebreak: false wins
        )?,
        periods: weighted_vote_categorical(
            blend,
            profiles,
            |p| p.formatting.periods,
            &[
                PeriodsPolicy::Always,
                PeriodsPolicy::Optional,
                PeriodsPolicy::Rare,
            ],
        )?,
        em_dashes: weighted_vote_categorical(
            blend,
            profiles,
            |p| p.formatting.em_dashes,
            &[
                EmDashPolicy::Preferred,
                EmDashPolicy::Ok,
                EmDashPolicy::Forbidden,
            ],
        )?,
        quotation_marks: weighted_vote_categorical(
            blend,
            profiles,
            |p| p.formatting.quotation_marks,
            &[
                QuotationMarks::Double,
                QuotationMarks::Single,
                QuotationMarks::Smart,
            ],
        )?,
        line_breaks: weighted_vote_categorical(
            blend,
            profiles,
            |p| p.formatting.line_breaks,
            &[LineBreaks::Single, LineBreaks::Double, LineBreaks::Rhythmic],
        )?,
    };

    let thread_opener_must_hook = weighted_vote_categorical(
        blend,
        profiles,
        |p| p.thread_opener_must_hook,
        &[false, true], // tiebreak: false wins
    )?;

    // List-of-strings: union + dedup with stable insertion order.
    let voice_traits = union_dedup_strings(blend, profiles, |p| &p.voice_traits)?;
    let ai_tells_to_avoid = union_dedup_strings(blend, profiles, |p| &p.ai_tells_to_avoid)?;
    let topical_obsessions = union_dedup_strings(blend, profiles, |p| &p.topical_obsessions)?;
    let topical_avoidances = union_dedup_strings(blend, profiles, |p| &p.topical_avoidances)?;

    // Construct merged profile.
    let mut merged = StyleProfile {
        version: 1,
        sentence_length_target,
        sentence_length_distribution,
        fragment_frequency,
        opening_patterns,
        opening_pattern_weights,
        formatting,
        emoji_policy,
        hashtag_policy,
        specificity_target,
        voice_traits,
        ai_tells_to_avoid,
        thread_rhythm,
        thread_max_length,
        thread_opener_must_hook,
        topical_obsessions,
        topical_avoidances,
    };

    // Apply user overrides last.
    apply_overrides(&mut merged, &recipe.overrides);

    // Post-merge validation.
    merged
        .validate()
        .map_err(|inner| BlendError::PostMergeValidation { inner })?;

    Ok(merged)
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

    use std::collections::HashMap;

    // ─────────────────────────────────────────────────────────────────────
    // P1.2d — test helpers
    // ─────────────────────────────────────────────────────────────────────

    /// Returns a baseline valid StyleProfile (mutate fields per test).
    /// Schema matches the §2.2 spec example.
    fn mk_profile() -> StyleProfile {
        StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![
                OpeningPattern::ClaimFirst,
                OpeningPattern::NumberFirst,
                OpeningPattern::SceneFirst,
                OpeningPattern::QuestionFirst,
            ],
            opening_pattern_weights: vec![0.4, 0.2, 0.2, 0.2],
            formatting: Formatting {
                lowercase: true,
                periods: PeriodsPolicy::Optional,
                em_dashes: EmDashPolicy::Forbidden,
                quotation_marks: QuotationMarks::Double,
                line_breaks: LineBreaks::Single,
            },
            emoji_policy: EmojiPolicy::RarePunchlineOnly,
            hashtag_policy: HashtagPolicy::Never,
            specificity_target: SpecificityTarget::High,
            voice_traits: vec!["specific".to_string(), "no_hedging".to_string()],
            ai_tells_to_avoid: vec!["delve".to_string()],
            thread_rhythm: ThreadRhythm::PunchlineCallbacks,
            thread_max_length: 10,
            thread_opener_must_hook: true,
            topical_obsessions: vec!["AI".to_string()],
            topical_avoidances: vec!["politics".to_string()],
        }
    }

    fn mk_recipe(entries: Vec<(&str, f64)>, overrides: PartialStyleProfile) -> BlendRecipe {
        BlendRecipe {
            version: 1,
            blend: entries
                .into_iter()
                .map(|(w, weight)| BlendEntry {
                    writer: w.to_string(),
                    weight,
                })
                .collect(),
            overrides,
        }
    }

    fn mk_profiles(entries: Vec<(&str, StyleProfile)>) -> HashMap<String, StyleProfile> {
        entries
            .into_iter()
            .map(|(w, p)| (w.to_string(), p))
            .collect()
    }

    // ─────────────────────────────────────────────────────────────────────
    // P1.2d — blend_profiles integration tests
    // ─────────────────────────────────────────────────────────────────────

    // ---- Single-writer identity -----------------------------------------

    #[test]
    fn single_writer_recipe_returns_input_profile_modulo_overrides() {
        let p = mk_profile();
        let recipe = mk_recipe(vec![("k", 1.0)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("k", p.clone())]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out, p);
    }

    // ---- Numeric merging ------------------------------------------------

    #[test]
    fn sentence_length_distribution_weighted_average_simple() {
        let mut a = mk_profile();
        a.sentence_length_distribution = [40, 30, 20, 10];
        let mut b = mk_profile();
        b.sentence_length_distribution = [20, 30, 30, 20];
        // 0.5 × [40,30,20,10] + 0.5 × [20,30,30,20] = [30, 30, 25, 15]
        let recipe = mk_recipe(vec![("a", 0.5), ("b", 0.5)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.sentence_length_distribution, [30, 30, 25, 15]);
    }

    #[test]
    fn sentence_length_distribution_hare_quota_residual_distribution() {
        // 0.5 × [33, 33, 33, 1] + 0.5 × [34, 33, 33, 0] = [33.5, 33.0, 33.0, 0.5]
        // Floors [33, 33, 33, 0] sum 99 → residual 1 → bucket 0 wins
        // (declaration-order tiebreak vs bucket 3, both have frac=0.5).
        let mut a = mk_profile();
        a.sentence_length_distribution = [33, 33, 33, 1];
        let mut b = mk_profile();
        b.sentence_length_distribution = [34, 33, 33, 0];
        let recipe = mk_recipe(vec![("a", 0.5), ("b", 0.5)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.sentence_length_distribution, [34, 33, 33, 0]);
    }

    #[test]
    fn thread_max_length_weighted_average_rounded() {
        let mut a = mk_profile();
        a.thread_max_length = 10;
        let mut b = mk_profile();
        b.thread_max_length = 20;
        // 0.6 × 10 + 0.4 × 20 = 14
        let recipe = mk_recipe(vec![("a", 0.6), ("b", 0.4)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.thread_max_length, 14);
    }

    #[test]
    fn thread_max_length_clamps_to_valid_range() {
        // Two profiles at max 25 each; weighted avg is 25, but f64 rounding
        // could theoretically produce 25.0000001 → 26. Clamp catches that.
        let mut a = mk_profile();
        a.thread_max_length = 25;
        let mut b = mk_profile();
        b.thread_max_length = 25;
        let recipe = mk_recipe(vec![("a", 0.5), ("b", 0.5)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.thread_max_length, 25);
    }

    // ---- Categorical voting ---------------------------------------------

    #[test]
    fn enum_vote_picks_highest_weighted_variant() {
        let mut a = mk_profile();
        a.emoji_policy = EmojiPolicy::Never;
        let mut b = mk_profile();
        b.emoji_policy = EmojiPolicy::Frequent;
        // a wins 0.7 vs b's 0.3 → Never.
        let recipe = mk_recipe(vec![("a", 0.7), ("b", 0.3)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.emoji_policy, EmojiPolicy::Never);
    }

    #[test]
    fn enum_vote_tiebreaks_by_declaration_order() {
        let mut a = mk_profile();
        a.emoji_policy = EmojiPolicy::Frequent;
        let mut b = mk_profile();
        b.emoji_policy = EmojiPolicy::Never;
        // 0.5/0.5 tie. Never declared first → wins.
        let recipe = mk_recipe(vec![("a", 0.5), ("b", 0.5)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.emoji_policy, EmojiPolicy::Never);
    }

    #[test]
    fn bool_vote_tiebreaks_to_false() {
        let mut a = mk_profile();
        a.formatting.lowercase = true;
        let mut b = mk_profile();
        b.formatting.lowercase = false;
        // 0.5/0.5 tie. false declared first in &[false, true] → wins.
        let recipe = mk_recipe(vec![("a", 0.5), ("b", 0.5)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert!(!out.formatting.lowercase);
    }

    // ---- Opening-pattern merge ------------------------------------------

    #[test]
    fn opening_patterns_merge_weighted_accumulator_normalizes_to_1() {
        let mut a = mk_profile();
        a.opening_patterns = vec![OpeningPattern::ClaimFirst, OpeningPattern::NumberFirst];
        a.opening_pattern_weights = vec![0.4, 0.6];
        let mut b = mk_profile();
        b.opening_patterns = vec![OpeningPattern::ClaimFirst, OpeningPattern::SceneFirst];
        b.opening_pattern_weights = vec![0.7, 0.3];
        // 0.6 × {claim_first: 0.4, number_first: 0.6}
        // 0.4 × {claim_first: 0.7, scene_first: 0.3}
        // = claim_first: 0.52, number_first: 0.36, scene_first: 0.12 (sum 1.0)
        let recipe = mk_recipe(vec![("a", 0.6), ("b", 0.4)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        let total: f64 = out.opening_pattern_weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "got total: {total}");
        // Sorted by weight DESC.
        assert_eq!(out.opening_patterns[0], OpeningPattern::ClaimFirst);
        assert!((out.opening_pattern_weights[0] - 0.52).abs() < 1e-9);
        assert_eq!(out.opening_patterns[1], OpeningPattern::NumberFirst);
        assert_eq!(out.opening_patterns[2], OpeningPattern::SceneFirst);
    }

    #[test]
    fn opening_patterns_output_sorted_by_weight_desc_with_declaration_tiebreak() {
        let mut a = mk_profile();
        a.opening_patterns = vec![OpeningPattern::SceneFirst, OpeningPattern::ClaimFirst];
        a.opening_pattern_weights = vec![0.5, 0.5];
        let mut b = mk_profile();
        b.opening_patterns = vec![OpeningPattern::ClaimFirst, OpeningPattern::SceneFirst];
        b.opening_pattern_weights = vec![0.5, 0.5];
        // Each pattern gets 0.5 × 0.5 + 0.5 × 0.5 = 0.5. Tied weights.
        // Declaration-order tiebreak: ClaimFirst (idx 0) before SceneFirst (idx 2).
        let recipe = mk_recipe(vec![("a", 0.5), ("b", 0.5)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(
            out.opening_patterns,
            vec![OpeningPattern::ClaimFirst, OpeningPattern::SceneFirst]
        );
    }

    // ---- List union+dedup -----------------------------------------------

    #[test]
    fn list_of_strings_union_preserves_first_occurrence_order() {
        let mut a = mk_profile();
        a.voice_traits = vec!["alpha".to_string(), "bravo".to_string()];
        let mut b = mk_profile();
        b.voice_traits = vec!["charlie".to_string(), "alpha".to_string()];
        // Recipe order: a then b.
        // Expected: ["alpha", "bravo", "charlie"] — alpha from a wins position 0,
        // bravo from a is position 1, charlie from b is position 2 (alpha is dedup'd).
        let recipe = mk_recipe(vec![("a", 0.5), ("b", 0.5)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(
            out.voice_traits,
            vec![
                "alpha".to_string(),
                "bravo".to_string(),
                "charlie".to_string()
            ]
        );
    }

    #[test]
    fn list_of_strings_dedup_is_case_sensitive() {
        let mut a = mk_profile();
        a.voice_traits = vec!["specific".to_string()];
        let mut b = mk_profile();
        b.voice_traits = vec!["Specific".to_string()];
        let recipe = mk_recipe(vec![("a", 0.5), ("b", 0.5)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", a), ("b", b)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        // Both kept: "specific" and "Specific" are distinct.
        assert_eq!(
            out.voice_traits,
            vec!["specific".to_string(), "Specific".to_string()]
        );
    }

    // ---- Override application -------------------------------------------

    #[test]
    fn overrides_replace_blended_value_unconditionally() {
        let p = mk_profile(); // thread_max_length = 10
        let recipe = mk_recipe(
            vec![("k", 1.0)],
            PartialStyleProfile {
                thread_max_length: Some(7),
                ..Default::default()
            },
        );
        let profiles = mk_profiles(vec![("k", p)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.thread_max_length, 7);
    }

    #[test]
    fn override_violation_surfaces_as_post_merge_validation() {
        let p = mk_profile();
        let recipe = mk_recipe(
            vec![("k", 1.0)],
            PartialStyleProfile {
                // 30 violates the 1..=25 invariant.
                thread_max_length: Some(30),
                ..Default::default()
            },
        );
        let profiles = mk_profiles(vec![("k", p)]);
        let err = blend_profiles(&recipe, &profiles).unwrap_err();
        match err {
            BlendError::PostMergeValidation { inner } => {
                let msg = format!("{inner}");
                assert!(msg.contains("thread_max_length"), "msg: {msg}");
                assert!(msg.contains("1..=25"), "msg: {msg}");
            }
            other => panic!("expected PostMergeValidation, got {other:?}"),
        }
    }

    #[test]
    fn partial_parallel_array_override_caught_by_validate() {
        let p = mk_profile();
        // Override patterns but not weights → length mismatch after apply.
        let recipe = mk_recipe(
            vec![("k", 1.0)],
            PartialStyleProfile {
                opening_patterns: Some(vec![
                    OpeningPattern::ClaimFirst,
                    OpeningPattern::NumberFirst,
                    OpeningPattern::SceneFirst,
                ]),
                ..Default::default()
            },
        );
        let profiles = mk_profiles(vec![("k", p)]);
        let err = blend_profiles(&recipe, &profiles).unwrap_err();
        match err {
            BlendError::PostMergeValidation { inner } => {
                let msg = format!("{inner}");
                assert!(msg.contains("opening_patterns"), "msg: {msg}");
                assert!(msg.contains("same length"), "msg: {msg}");
            }
            other => panic!("expected PostMergeValidation, got {other:?}"),
        }
    }

    // ---- Error paths ----------------------------------------------------

    #[test]
    fn missing_profile_for_writer_returns_missing_profile_error() {
        let p = mk_profile();
        // Recipe references "k" and "ghost" but profiles only has "k".
        let recipe = mk_recipe(
            vec![("k", 0.5), ("ghost", 0.5)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("k", p)]);
        let err = blend_profiles(&recipe, &profiles).unwrap_err();
        match err {
            BlendError::MissingProfile(w) => assert_eq!(w, "ghost"),
            other => panic!("expected MissingProfile, got {other:?}"),
        }
    }

    #[test]
    fn recipe_with_writer_listed_twice_uses_first_lookup() {
        // BlendRecipe::validate would reject duplicate writers, but if a
        // hand-constructed unvalidated recipe is passed, document the
        // behavior: each weighted contribution sums independently.
        let mut p = mk_profile();
        p.thread_max_length = 10;
        // Manually construct unvalidated recipe with duplicate "k".
        let recipe = BlendRecipe {
            version: 1,
            blend: vec![
                BlendEntry {
                    writer: "k".to_string(),
                    weight: 0.4,
                },
                BlendEntry {
                    writer: "k".to_string(),
                    weight: 0.6,
                },
            ],
            overrides: PartialStyleProfile::default(),
        };
        let profiles = mk_profiles(vec![("k", p)]);
        // 0.4 × 10 + 0.6 × 10 = 10 → still 10 (k contributes twice).
        let out = blend_profiles(&recipe, &profiles).unwrap();
        assert_eq!(out.thread_max_length, 10);
    }

    // ---- Determinism + happy path ---------------------------------------

    #[test]
    fn blend_is_deterministic_across_runs() {
        let p_a = {
            let mut x = mk_profile();
            x.thread_max_length = 12;
            x.emoji_policy = EmojiPolicy::Never;
            x
        };
        let p_b = {
            let mut x = mk_profile();
            x.thread_max_length = 18;
            x.emoji_policy = EmojiPolicy::Occasional;
            x
        };
        let recipe = mk_recipe(vec![("a", 0.6), ("b", 0.4)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("a", p_a), ("b", p_b)]);
        let baseline = blend_profiles(&recipe, &profiles).unwrap();
        for _ in 0..10 {
            let again = blend_profiles(&recipe, &profiles).unwrap();
            assert_eq!(again, baseline);
        }
    }

    #[test]
    fn blend_output_passes_style_profile_validate() {
        let p = mk_profile();
        let recipe = mk_recipe(vec![("k", 1.0)], PartialStyleProfile::default());
        let profiles = mk_profiles(vec![("k", p)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        out.validate().expect("merged profile is valid");
    }

    // ---- Five-writer canonical example ----------------------------------

    #[test]
    fn five_writer_canonical_blend_matches_spec_example() {
        // The umbrella spec §2.3 AI/tech blend.
        let p_karpathy = {
            let mut x = mk_profile();
            x.thread_max_length = 12;
            x.voice_traits = vec!["technical".to_string(), "humble".to_string()];
            x
        };
        let p_eladgil = {
            let mut x = mk_profile();
            x.thread_max_length = 8;
            x.voice_traits = vec!["signal_dense".to_string()];
            x
        };
        let p_swyx = {
            let mut x = mk_profile();
            x.thread_max_length = 10;
            x.voice_traits = vec!["builder".to_string()];
            x
        };
        let p_naval = {
            let mut x = mk_profile();
            x.thread_max_length = 5;
            x.voice_traits = vec!["aphoristic".to_string()];
            x
        };
        let p_sama = {
            let mut x = mk_profile();
            x.thread_max_length = 6;
            x.voice_traits = vec!["sparse".to_string()];
            x
        };
        let recipe = mk_recipe(
            vec![
                ("karpathy", 0.30),
                ("eladgil", 0.20),
                ("swyx", 0.20),
                ("naval", 0.15),
                ("sama", 0.15),
            ],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![
            ("karpathy", p_karpathy),
            ("eladgil", p_eladgil),
            ("swyx", p_swyx),
            ("naval", p_naval),
            ("sama", p_sama),
        ]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        // Smoke: the result is a valid StyleProfile with all 5 writers'
        // voice_traits unioned. Karpathy contributes 2 ("technical", "humble"),
        // each of the others 1 → 6 distinct strings total.
        out.validate().expect("merged profile is valid");
        assert_eq!(out.voice_traits.len(), 6);
        assert!(out.voice_traits.contains(&"technical".to_string()));
        assert!(out.voice_traits.contains(&"humble".to_string()));
        assert!(out.voice_traits.contains(&"sparse".to_string()));
        // 0.30×12 + 0.20×8 + 0.20×10 + 0.15×5 + 0.15×6 = 3.6 + 1.6 + 2.0 + 0.75 + 0.9 = 8.85 → 9
        assert_eq!(out.thread_max_length, 9);
    }

    // ---- Floating-point drift edge case ---------------------------------

    #[test]
    fn opening_pattern_weights_normalized_under_f64_drift() {
        // Construct profiles where the cross-product produces weights that
        // sum to a value off by >1e-9 due to f64 multiplication drift.
        let mut a = mk_profile();
        a.opening_patterns = vec![OpeningPattern::ClaimFirst];
        a.opening_pattern_weights = vec![1.0];
        let mut b = mk_profile();
        b.opening_patterns = vec![OpeningPattern::NumberFirst];
        b.opening_pattern_weights = vec![1.0];
        let mut c = mk_profile();
        c.opening_patterns = vec![OpeningPattern::SceneFirst];
        c.opening_pattern_weights = vec![1.0];
        // Weights chosen to hit f64 representation edges: 0.1 + 0.2 + 0.7
        // = 0.9999999999999999 in f64.
        let recipe = mk_recipe(
            vec![("a", 0.1), ("b", 0.2), ("c", 0.7)],
            PartialStyleProfile::default(),
        );
        let profiles = mk_profiles(vec![("a", a), ("b", b), ("c", c)]);
        let out = blend_profiles(&recipe, &profiles).unwrap();
        // After defensive normalization, sum is within 1e-6 → validate passes.
        out.validate()
            .expect("merged profile passes validate after f64 normalization");
        let total: f64 = out.opening_pattern_weights.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-6,
            "weights sum should normalize to 1.0, got {total}"
        );
    }
}
