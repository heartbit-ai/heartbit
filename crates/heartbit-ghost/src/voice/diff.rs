//! Structured per-field diff between two [`StyleProfile`] snapshots.
//!
//! See [`ProfileDiff::compute`] for the data shape and
//! [`render_profile_diff`] for the human-readable formatter used by
//! `heartbit persona profile diff`.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::voice::snapshot::SnapshotMeta;
use crate::voice::style::{Formatting, OpeningPattern, StyleProfile};

/// Render a serde-serializable value as its canonical snake_case wire
/// format. The closed-vocab enums in [`crate::voice::style`] all derive
/// `#[serde(rename_all = "snake_case")]`, so this produces the same
/// strings the user sees in TOML/JSON output (e.g., `rare_punchline_only`,
/// not the Debug-derived `RarePunchlineOnly`).
fn enum_as_snake_case<T: serde::Serialize>(val: &T) -> String {
    serde_json::to_string(val)
        .ok()
        .map(|s| s.trim_matches('"').to_string())
        .unwrap_or_default()
}

/// Structured difference between two [`StyleProfile`] values. Captures
/// only the fields that changed; identical fields are omitted.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ProfileDiff {
    /// Per-field changes, in StyleProfile declaration order.
    pub changes: Vec<FieldChange>,
}

/// One field's change (with the field's name and a typed delta).
#[derive(Debug, Clone, PartialEq)]
pub struct FieldChange {
    /// Field name as it appears in the StyleProfile struct.
    pub field: String,

    /// The kind of change (categorical / list / weighted / distribution).
    pub kind: ChangeKind,
}

/// One of four shapes a field-level change can take.
#[derive(Debug, Clone, PartialEq)]
pub enum ChangeKind {
    /// Categorical, numeric, or bool — `old → new`, both as snake_case strings.
    Scalar {
        /// Old value rendered as a string.
        old: String,
        /// New value rendered as a string.
        new: String,
    },

    /// Unordered list of strings — symmetric difference (added / removed).
    StringList {
        /// Strings in `new` but not in `old`.
        added: Vec<String>,
        /// Strings in `old` but not in `new`.
        removed: Vec<String>,
    },

    /// Parallel weighted arrays (only `opening_patterns` + weights today).
    WeightedList {
        /// Per-pattern delta. `old_weight` is `None` if the pattern was added,
        /// `new_weight` is `None` if it was removed.
        entries: Vec<WeightedEntry>,
    },

    /// 4-bucket distribution (only `sentence_length_distribution` today).
    Distribution {
        /// Old bucket values.
        old: [u8; 4],
        /// New bucket values.
        new: [u8; 4],
    },
}

/// One pattern's contribution in a weighted-list change.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedEntry {
    /// Pattern name as snake_case string.
    pub item: String,
    /// Weight in the old profile, or `None` if the pattern was added.
    pub old_weight: Option<f64>,
    /// Weight in the new profile, or `None` if the pattern was removed.
    pub new_weight: Option<f64>,
}

impl ProfileDiff {
    /// Walk both profiles, emitting one [`FieldChange`] per field that
    /// differs. Identical fields are not in `changes`. Empty `changes`
    /// means the profiles are equal.
    pub fn compute(old: &StyleProfile, new: &StyleProfile) -> Self {
        let mut changes = Vec::new();

        if old.sentence_length_target != new.sentence_length_target {
            changes.push(FieldChange {
                field: "sentence_length_target".to_string(),
                kind: ChangeKind::Scalar {
                    old: enum_as_snake_case(&old.sentence_length_target),
                    new: enum_as_snake_case(&new.sentence_length_target),
                },
            });
        }
        if old.sentence_length_distribution != new.sentence_length_distribution {
            changes.push(FieldChange {
                field: "sentence_length_distribution".to_string(),
                kind: ChangeKind::Distribution {
                    old: old.sentence_length_distribution,
                    new: new.sentence_length_distribution,
                },
            });
        }
        if old.fragment_frequency != new.fragment_frequency {
            changes.push(FieldChange {
                field: "fragment_frequency".to_string(),
                kind: ChangeKind::Scalar {
                    old: enum_as_snake_case(&old.fragment_frequency),
                    new: enum_as_snake_case(&new.fragment_frequency),
                },
            });
        }
        if old.opening_patterns != new.opening_patterns
            || old.opening_pattern_weights != new.opening_pattern_weights
        {
            changes.push(FieldChange {
                field: "opening_patterns".to_string(),
                kind: ChangeKind::WeightedList {
                    entries: weighted_list_diff(
                        &old.opening_patterns,
                        &old.opening_pattern_weights,
                        &new.opening_patterns,
                        &new.opening_pattern_weights,
                    ),
                },
            });
        }
        if old.formatting != new.formatting {
            // formatting changed — emit per-sub-field scalars
            push_formatting_changes(&mut changes, &old.formatting, &new.formatting);
        }
        if old.emoji_policy != new.emoji_policy {
            changes.push(FieldChange {
                field: "emoji_policy".to_string(),
                kind: ChangeKind::Scalar {
                    old: enum_as_snake_case(&old.emoji_policy),
                    new: enum_as_snake_case(&new.emoji_policy),
                },
            });
        }
        if old.hashtag_policy != new.hashtag_policy {
            changes.push(FieldChange {
                field: "hashtag_policy".to_string(),
                kind: ChangeKind::Scalar {
                    old: enum_as_snake_case(&old.hashtag_policy),
                    new: enum_as_snake_case(&new.hashtag_policy),
                },
            });
        }
        if old.specificity_target != new.specificity_target {
            changes.push(FieldChange {
                field: "specificity_target".to_string(),
                kind: ChangeKind::Scalar {
                    old: enum_as_snake_case(&old.specificity_target),
                    new: enum_as_snake_case(&new.specificity_target),
                },
            });
        }
        if old.voice_traits != new.voice_traits {
            changes.push(FieldChange {
                field: "voice_traits".to_string(),
                kind: string_list_diff(&old.voice_traits, &new.voice_traits),
            });
        }
        if old.ai_tells_to_avoid != new.ai_tells_to_avoid {
            changes.push(FieldChange {
                field: "ai_tells_to_avoid".to_string(),
                kind: string_list_diff(&old.ai_tells_to_avoid, &new.ai_tells_to_avoid),
            });
        }
        if old.thread_rhythm != new.thread_rhythm {
            changes.push(FieldChange {
                field: "thread_rhythm".to_string(),
                kind: ChangeKind::Scalar {
                    old: enum_as_snake_case(&old.thread_rhythm),
                    new: enum_as_snake_case(&new.thread_rhythm),
                },
            });
        }
        if old.thread_max_length != new.thread_max_length {
            changes.push(FieldChange {
                field: "thread_max_length".to_string(),
                kind: ChangeKind::Scalar {
                    old: old.thread_max_length.to_string(),
                    new: new.thread_max_length.to_string(),
                },
            });
        }
        if old.thread_opener_must_hook != new.thread_opener_must_hook {
            changes.push(FieldChange {
                field: "thread_opener_must_hook".to_string(),
                kind: ChangeKind::Scalar {
                    old: old.thread_opener_must_hook.to_string(),
                    new: new.thread_opener_must_hook.to_string(),
                },
            });
        }
        if old.topical_obsessions != new.topical_obsessions {
            changes.push(FieldChange {
                field: "topical_obsessions".to_string(),
                kind: string_list_diff(&old.topical_obsessions, &new.topical_obsessions),
            });
        }
        if old.topical_avoidances != new.topical_avoidances {
            changes.push(FieldChange {
                field: "topical_avoidances".to_string(),
                kind: string_list_diff(&old.topical_avoidances, &new.topical_avoidances),
            });
        }

        Self { changes }
    }

    /// `true` when the two profiles compared equal (no changes).
    pub fn is_empty(&self) -> bool {
        self.changes.is_empty()
    }
}

fn string_list_diff(old: &[String], new: &[String]) -> ChangeKind {
    let old_set: BTreeSet<&str> = old.iter().map(String::as_str).collect();
    let new_set: BTreeSet<&str> = new.iter().map(String::as_str).collect();
    let added: Vec<String> = new_set
        .difference(&old_set)
        .map(|s| (*s).to_string())
        .collect();
    let removed: Vec<String> = old_set
        .difference(&new_set)
        .map(|s| (*s).to_string())
        .collect();
    ChangeKind::StringList { added, removed }
}

fn weighted_list_diff(
    old_pats: &[OpeningPattern],
    old_weights: &[f64],
    new_pats: &[OpeningPattern],
    new_weights: &[f64],
) -> Vec<WeightedEntry> {
    let mut entries: Vec<WeightedEntry> = Vec::new();
    for (pat, w) in old_pats.iter().zip(old_weights.iter()) {
        entries.push(WeightedEntry {
            item: enum_as_snake_case(pat),
            old_weight: Some(*w),
            new_weight: None,
        });
    }
    for (pat, w) in new_pats.iter().zip(new_weights.iter()) {
        let item = enum_as_snake_case(pat);
        if let Some(existing) = entries.iter_mut().find(|e| e.item == item) {
            existing.new_weight = Some(*w);
        } else {
            entries.push(WeightedEntry {
                item,
                old_weight: None,
                new_weight: Some(*w),
            });
        }
    }
    entries
}

fn push_formatting_changes(changes: &mut Vec<FieldChange>, old: &Formatting, new: &Formatting) {
    if old.lowercase != new.lowercase {
        changes.push(FieldChange {
            field: "formatting.lowercase".to_string(),
            kind: ChangeKind::Scalar {
                old: old.lowercase.to_string(),
                new: new.lowercase.to_string(),
            },
        });
    }
    if old.periods != new.periods {
        changes.push(FieldChange {
            field: "formatting.periods".to_string(),
            kind: ChangeKind::Scalar {
                old: enum_as_snake_case(&old.periods),
                new: enum_as_snake_case(&new.periods),
            },
        });
    }
    if old.em_dashes != new.em_dashes {
        changes.push(FieldChange {
            field: "formatting.em_dashes".to_string(),
            kind: ChangeKind::Scalar {
                old: enum_as_snake_case(&old.em_dashes),
                new: enum_as_snake_case(&new.em_dashes),
            },
        });
    }
    if old.quotation_marks != new.quotation_marks {
        changes.push(FieldChange {
            field: "formatting.quotation_marks".to_string(),
            kind: ChangeKind::Scalar {
                old: enum_as_snake_case(&old.quotation_marks),
                new: enum_as_snake_case(&new.quotation_marks),
            },
        });
    }
    if old.line_breaks != new.line_breaks {
        changes.push(FieldChange {
            field: "formatting.line_breaks".to_string(),
            kind: ChangeKind::Scalar {
                old: enum_as_snake_case(&old.line_breaks),
                new: enum_as_snake_case(&new.line_breaks),
            },
        });
    }
}

/// Render a [`ProfileDiff`] as human-readable text for the CLI.
pub fn render_profile_diff(
    diff: &ProfileDiff,
    old_meta: &SnapshotMeta,
    new_meta: &SnapshotMeta,
) -> String {
    let recipe_note = if old_meta.recipe_hash == new_meta.recipe_hash {
        "same recipe"
    } else {
        "recipe changed"
    };
    let mut out = String::new();
    let _ = writeln!(
        out,
        "Profile diff: v{} → v{} (recipe-hash: {} → {}; {})",
        old_meta.version,
        new_meta.version,
        truncate_hash(&old_meta.recipe_hash),
        truncate_hash(&new_meta.recipe_hash),
        recipe_note
    );
    if diff.is_empty() {
        let _ = writeln!(out, "(no changes)");
        return out;
    }
    let _ = writeln!(out);
    for change in &diff.changes {
        match &change.kind {
            ChangeKind::Scalar { old, new } => {
                let _ = writeln!(out, "{}: {} → {}", change.field, old, new);
            }
            ChangeKind::Distribution { old, new } => {
                let _ = writeln!(out, "{}: {:?} → {:?}", change.field, old, new);
            }
            ChangeKind::StringList { added, removed } => {
                let _ = writeln!(out, "{}:", change.field);
                for s in added {
                    let _ = writeln!(out, "  + {s}");
                }
                for s in removed {
                    let _ = writeln!(out, "  - {s}");
                }
            }
            ChangeKind::WeightedList { entries } => {
                let _ = writeln!(out, "{}:", change.field);
                for e in entries {
                    match (e.old_weight, e.new_weight) {
                        (Some(o), Some(n)) if (o - n).abs() < f64::EPSILON => {}
                        (Some(o), Some(n)) => {
                            let _ = writeln!(out, "  {}: {:.2} → {:.2}", e.item, o, n);
                        }
                        (None, Some(n)) => {
                            let _ = writeln!(out, "  + {}: {:.2}", e.item, n);
                        }
                        (Some(o), None) => {
                            let _ = writeln!(out, "  - {}: {:.2}", e.item, o);
                        }
                        (None, None) => {}
                    }
                }
            }
        }
    }
    out
}

fn truncate_hash(hash: &str) -> &str {
    if hash.len() >= 8 { &hash[..8] } else { hash }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::style::{
        EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
        OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
        ThreadRhythm,
    };
    use chrono::Utc;

    fn mk_profile() -> StyleProfile {
        StyleProfile {
            version: 1,
            sentence_length_target: SentenceLengthTarget::Short,
            sentence_length_distribution: [40, 30, 20, 10],
            fragment_frequency: FragmentFrequency::Common,
            opening_patterns: vec![OpeningPattern::ClaimFirst, OpeningPattern::NumberFirst],
            opening_pattern_weights: vec![0.6, 0.4],
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

    fn mk_meta(version: u32, recipe_hash: &str) -> SnapshotMeta {
        SnapshotMeta {
            version,
            hash: "0".repeat(64),
            recipe_hash: recipe_hash.to_string(),
            generated_at: Utc::now(),
        }
    }

    #[test]
    fn compute_identical_profiles_produces_empty_diff() {
        let p = mk_profile();
        let diff = ProfileDiff::compute(&p, &p);
        assert!(diff.is_empty());
    }

    #[test]
    fn compute_scalar_change_is_recorded() {
        let old = mk_profile();
        let mut new = old.clone();
        new.emoji_policy = EmojiPolicy::Never;
        let diff = ProfileDiff::compute(&old, &new);
        assert_eq!(diff.changes.len(), 1);
        assert_eq!(diff.changes[0].field, "emoji_policy");
        match &diff.changes[0].kind {
            ChangeKind::Scalar { old, new } => {
                assert_eq!(old, "rare_punchline_only");
                assert_eq!(new, "never");
            }
            other => panic!("expected Scalar, got {other:?}"),
        }
    }

    #[test]
    fn compute_string_list_change_records_added_and_removed() {
        let old = mk_profile();
        let mut new = old.clone();
        new.voice_traits = vec!["specific".to_string(), "humble".to_string()];
        let diff = ProfileDiff::compute(&old, &new);
        let voice_change = diff
            .changes
            .iter()
            .find(|c| c.field == "voice_traits")
            .expect("voice_traits in diff");
        match &voice_change.kind {
            ChangeKind::StringList { added, removed } => {
                assert_eq!(added, &vec!["humble".to_string()]);
                assert_eq!(removed, &vec!["no_hedging".to_string()]);
            }
            other => panic!("expected StringList, got {other:?}"),
        }
    }

    #[test]
    fn compute_distribution_change_is_typed() {
        let old = mk_profile();
        let mut new = old.clone();
        new.sentence_length_distribution = [35, 35, 22, 8];
        let diff = ProfileDiff::compute(&old, &new);
        let change = diff
            .changes
            .iter()
            .find(|c| c.field == "sentence_length_distribution")
            .expect("present");
        match &change.kind {
            ChangeKind::Distribution { old, new } => {
                assert_eq!(old, &[40, 30, 20, 10]);
                assert_eq!(new, &[35, 35, 22, 8]);
            }
            other => panic!("expected Distribution, got {other:?}"),
        }
    }

    #[test]
    fn compute_weighted_list_change_records_per_pattern_delta() {
        let old = mk_profile(); // claim_first: 0.6, number_first: 0.4
        let mut new = old.clone();
        new.opening_patterns = vec![OpeningPattern::ClaimFirst, OpeningPattern::SceneFirst];
        new.opening_pattern_weights = vec![0.5, 0.5];
        let diff = ProfileDiff::compute(&old, &new);
        let change = diff
            .changes
            .iter()
            .find(|c| c.field == "opening_patterns")
            .expect("present");
        match &change.kind {
            ChangeKind::WeightedList { entries } => {
                let claim = entries.iter().find(|e| e.item == "claim_first").unwrap();
                assert_eq!(claim.old_weight, Some(0.6));
                assert_eq!(claim.new_weight, Some(0.5));
                let number = entries.iter().find(|e| e.item == "number_first").unwrap();
                assert_eq!(number.old_weight, Some(0.4));
                assert_eq!(number.new_weight, None);
                let scene = entries.iter().find(|e| e.item == "scene_first").unwrap();
                assert_eq!(scene.old_weight, None);
                assert_eq!(scene.new_weight, Some(0.5));
            }
            other => panic!("expected WeightedList, got {other:?}"),
        }
    }

    #[test]
    fn render_no_changes_says_so() {
        let p = mk_profile();
        let diff = ProfileDiff::compute(&p, &p);
        let m1 = mk_meta(3, &"a".repeat(64));
        let m2 = mk_meta(4, &"a".repeat(64));
        let out = render_profile_diff(&diff, &m1, &m2);
        assert!(out.contains("v3 → v4"), "got: {out}");
        assert!(out.contains("same recipe"), "got: {out}");
        assert!(out.contains("(no changes)"), "got: {out}");
    }

    #[test]
    fn render_recipe_change_label() {
        let p = mk_profile();
        let mut p2 = p.clone();
        p2.thread_max_length = 7;
        let diff = ProfileDiff::compute(&p, &p2);
        let m1 = mk_meta(3, &"a".repeat(64));
        let m2 = mk_meta(4, &"b".repeat(64));
        let out = render_profile_diff(&diff, &m1, &m2);
        assert!(out.contains("recipe changed"), "got: {out}");
        assert!(out.contains("thread_max_length: 10 → 7"), "got: {out}");
    }

    #[test]
    fn render_emits_canonical_snake_case_for_enum_variants() {
        let old = mk_profile(); // emoji_policy: RarePunchlineOnly
        let mut new = old.clone();
        new.emoji_policy = EmojiPolicy::Never;
        let diff = ProfileDiff::compute(&old, &new);
        let m1 = mk_meta(3, &"a".repeat(64));
        let m2 = mk_meta(4, &"a".repeat(64));
        let out = render_profile_diff(&diff, &m1, &m2);
        // The canonical snake_case form, NOT the Debug-derived form.
        assert!(
            out.contains("emoji_policy: rare_punchline_only → never"),
            "got: {out}"
        );
        // Negative assertion: the broken Debug-derived form must NOT appear.
        assert!(!out.contains("rarepunchlineonly"), "got: {out}");
    }
}
