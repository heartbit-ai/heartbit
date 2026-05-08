//! Render a [`StyleProfile`] as English voice guidelines for the writer's
//! user message. All 16 non-version fields are surfaced; ~200 tokens.

use crate::voice::{
    EmDashPolicy, EmojiPolicy, Formatting, FragmentFrequency, HashtagPolicy, LineBreaks,
    OpeningPattern, PeriodsPolicy, QuotationMarks, SentenceLengthTarget, SpecificityTarget,
    StyleProfile, ThreadRhythm,
};

/// Render the profile as a structured-English voice-guidelines block.
pub fn render_style_profile_as_english(profile: &StyleProfile) -> String {
    let mut out = String::new();
    out.push_str("Voice guidelines:\n");

    let dist = &profile.sentence_length_distribution;
    out.push_str(&format!(
        "- sentence length: {} ({}% short, {}% medium-short, {}% medium-long, {}% long)\n",
        sentence_length_word(profile.sentence_length_target),
        dist[0],
        dist[1],
        dist[2],
        dist[3]
    ));
    out.push_str(&format!(
        "- fragments: {}\n",
        fragment_frequency_word(profile.fragment_frequency)
    ));

    let openers = profile
        .opening_patterns
        .iter()
        .zip(profile.opening_pattern_weights.iter())
        .map(|(p, w)| {
            format!(
                "{} ({}%)",
                opening_pattern_word(*p),
                (w * 100.0).round() as u32
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    if openers.is_empty() {
        out.push_str("- opening patterns: (none)\n");
    } else {
        out.push_str(&format!("- opening patterns: {}\n", openers));
    }

    out.push_str(&format!(
        "- formatting: {}\n",
        render_formatting(&profile.formatting)
    ));
    out.push_str(&format!(
        "- emoji policy: {}\n",
        emoji_policy_word(profile.emoji_policy)
    ));
    out.push_str(&format!(
        "- hashtag policy: {}\n",
        hashtag_policy_word(profile.hashtag_policy)
    ));
    out.push_str(&format!(
        "- specificity target: {}\n",
        specificity_target_word(profile.specificity_target)
    ));
    out.push_str(&format!(
        "- voice traits: {}\n",
        render_string_list(&profile.voice_traits)
    ));
    out.push_str(&format!(
        "- ai tells to avoid: {}\n",
        render_string_list(&profile.ai_tells_to_avoid)
    ));
    out.push_str(&format!(
        "- thread rhythm: {}\n",
        thread_rhythm_word(profile.thread_rhythm)
    ));
    out.push_str(&format!(
        "- thread max length: {} ({})\n",
        profile.thread_max_length,
        if profile.thread_opener_must_hook {
            "opener must hook"
        } else {
            "opener need not hook"
        }
    ));
    out.push_str(&format!(
        "- topical obsessions: {}\n",
        render_string_list(&profile.topical_obsessions)
    ));
    out.push_str(&format!(
        "- topical avoidances: {}\n",
        render_string_list(&profile.topical_avoidances)
    ));

    out
}

fn render_string_list(v: &[String]) -> String {
    if v.is_empty() {
        "(none)".to_string()
    } else {
        v.join(", ")
    }
}

fn render_formatting(f: &Formatting) -> String {
    let mut parts = Vec::new();
    parts.push(
        if f.lowercase {
            "lowercase"
        } else {
            "sentence case"
        }
        .to_string(),
    );
    parts.push(format!("{} periods", periods_policy_word(f.periods)));
    parts.push(format!("em-dashes {}", em_dash_policy_word(f.em_dashes)));
    parts.push(format!(
        "{} quotes",
        quotation_marks_word(f.quotation_marks)
    ));
    parts.push(format!("{} line breaks", line_breaks_word(f.line_breaks)));
    parts.join(", ")
}

fn sentence_length_word(t: SentenceLengthTarget) -> &'static str {
    match t {
        SentenceLengthTarget::Short => "short",
        SentenceLengthTarget::Mixed => "mixed",
        SentenceLengthTarget::Long => "long",
    }
}

fn fragment_frequency_word(f: FragmentFrequency) -> &'static str {
    match f {
        FragmentFrequency::Rare => "rare",
        FragmentFrequency::Occasional => "occasional",
        FragmentFrequency::Common => "common",
    }
}

fn opening_pattern_word(p: OpeningPattern) -> &'static str {
    match p {
        OpeningPattern::ClaimFirst => "claim_first",
        OpeningPattern::NumberFirst => "number_first",
        OpeningPattern::SceneFirst => "scene_first",
        OpeningPattern::QuestionFirst => "question_first",
        OpeningPattern::AphoristicFirst => "aphoristic_first",
        OpeningPattern::AnecdoteFirst => "anecdote_first",
        OpeningPattern::ContrarianFirst => "contrarian_first",
    }
}

fn periods_policy_word(p: PeriodsPolicy) -> &'static str {
    match p {
        PeriodsPolicy::Always => "always",
        PeriodsPolicy::Optional => "optional",
        PeriodsPolicy::Rare => "rare",
    }
}

fn em_dash_policy_word(e: EmDashPolicy) -> &'static str {
    match e {
        EmDashPolicy::Preferred => "preferred",
        EmDashPolicy::Ok => "ok",
        EmDashPolicy::Forbidden => "forbidden",
    }
}

fn quotation_marks_word(q: QuotationMarks) -> &'static str {
    match q {
        QuotationMarks::Double => "double",
        QuotationMarks::Single => "single",
        QuotationMarks::Smart => "smart",
    }
}

fn line_breaks_word(l: LineBreaks) -> &'static str {
    match l {
        LineBreaks::Single => "single",
        LineBreaks::Double => "double",
        LineBreaks::Rhythmic => "rhythmic",
    }
}

fn emoji_policy_word(e: EmojiPolicy) -> &'static str {
    match e {
        EmojiPolicy::Never => "never",
        EmojiPolicy::RarePunchlineOnly => "rare punchline only",
        EmojiPolicy::Occasional => "occasional",
        EmojiPolicy::Frequent => "frequent",
    }
}

fn hashtag_policy_word(h: HashtagPolicy) -> &'static str {
    match h {
        HashtagPolicy::Never => "never",
        HashtagPolicy::Rare => "rare",
        HashtagPolicy::TopicRelevant => "topic-relevant",
        HashtagPolicy::Always => "always",
    }
}

fn specificity_target_word(s: SpecificityTarget) -> &'static str {
    match s {
        SpecificityTarget::Low => "low",
        SpecificityTarget::Medium => "medium",
        SpecificityTarget::High => "high",
    }
}

fn thread_rhythm_word(t: ThreadRhythm) -> &'static str {
    match t {
        ThreadRhythm::Linear => "linear",
        ThreadRhythm::ListThenPayoff => "list_then_payoff",
        ThreadRhythm::PunchlineCallbacks => "punchline_callbacks",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voice::StyleProfile;

    fn canonical_profile() -> StyleProfile {
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
            ai_tells_to_avoid: vec!["delve".to_string(), "in conclusion".to_string()],
            thread_rhythm: ThreadRhythm::PunchlineCallbacks,
            thread_max_length: 10,
            thread_opener_must_hook: true,
            topical_obsessions: vec!["AI".to_string()],
            topical_avoidances: vec!["politics".to_string()],
        }
    }

    #[test]
    fn render_canonical_profile_includes_all_16_fields() {
        let p = canonical_profile();
        let s = render_style_profile_as_english(&p);
        assert!(s.contains("Voice guidelines:"));
        assert!(s.contains("sentence length: short"));
        assert!(s.contains("40% short, 30% medium-short, 20% medium-long, 10% long"));
        assert!(s.contains("fragments: common"));
        assert!(s.contains("opening patterns: claim_first (60%), number_first (40%)"));
        assert!(s.contains("lowercase"));
        assert!(s.contains("optional periods"));
        assert!(s.contains("em-dashes forbidden"));
        assert!(s.contains("double quotes"));
        assert!(s.contains("single line breaks"));
        assert!(s.contains("emoji policy: rare punchline only"));
        assert!(s.contains("hashtag policy: never"));
        assert!(s.contains("specificity target: high"));
        assert!(s.contains("voice traits: specific, no_hedging"));
        assert!(s.contains("ai tells to avoid: delve, in conclusion"));
        assert!(s.contains("thread rhythm: punchline_callbacks"));
        assert!(s.contains("thread max length: 10 (opener must hook)"));
        assert!(s.contains("topical obsessions: AI"));
        assert!(s.contains("topical avoidances: politics"));
    }

    #[test]
    fn render_empty_string_lists_show_none_marker() {
        let mut p = canonical_profile();
        p.voice_traits.clear();
        p.ai_tells_to_avoid.clear();
        p.topical_obsessions.clear();
        p.topical_avoidances.clear();
        let s = render_style_profile_as_english(&p);
        assert!(s.contains("voice traits: (none)"));
        assert!(s.contains("ai tells to avoid: (none)"));
        assert!(s.contains("topical obsessions: (none)"));
        assert!(s.contains("topical avoidances: (none)"));
    }

    #[test]
    fn render_sentence_case_when_lowercase_false() {
        let mut p = canonical_profile();
        p.formatting.lowercase = false;
        let s = render_style_profile_as_english(&p);
        assert!(
            s.contains("sentence case"),
            "expected 'sentence case' in formatting; got: {s}"
        );
        assert!(
            !s.contains(", lowercase, "),
            "should not contain 'lowercase' when false; got: {s}"
        );
    }

    #[test]
    fn render_thread_opener_need_not_hook_when_false() {
        let mut p = canonical_profile();
        p.thread_opener_must_hook = false;
        let s = render_style_profile_as_english(&p);
        assert!(
            s.contains("opener need not hook"),
            "expected fallback wording; got: {s}"
        );
    }
}
