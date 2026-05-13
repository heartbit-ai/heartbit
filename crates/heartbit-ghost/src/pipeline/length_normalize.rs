//! Deterministic post-write length normalization.
//!
//! The writer LLM is asked to keep each tweet ≤280 chars and split with a
//! blank line into a thread. In practice ~33% of drafts overrun anyway —
//! LLMs are bad at character counting. This module is the structural
//! enforcement that the prompt can't provide.
//!
//! Strategy:
//! 1. Split the draft on `\n\n` (same rule as `publish_gate` + `parse_thread_tweets`).
//! 2. For each segment that's already ≤max, keep it.
//! 3. For each over-length segment, split on sentence boundaries
//!    (`. `, `! `, `? ` followed by an uppercase ASCII letter) and
//!    greedily pack the sentences back into ≤max-char chunks.
//! 4. If a single sentence is itself >max (rare; e.g. one long run-on),
//!    leave it intact and let `publish_gate` reject the candidate — the
//!    pre-Telegram filter will then hide it from the operator. We never
//!    truncate mid-sentence; better to drop the draft than to ship a
//!    mangled one.
//!
//! No-op when every segment already fits. Idempotent.

/// Maximum characters per tweet on X — the structural constraint the
/// `publish_gate` enforces.
pub const MAX_TWEET_CHARS: usize = 280;

/// Normalize a free-form draft so each `\n\n`-delimited segment is at most
/// `max_per_tweet` chars. Returns the (possibly rejoined) draft as a
/// String. When the input is already valid, the result is byte-equal.
pub fn normalize_tweet_length(draft: &str, max_per_tweet: usize) -> String {
    let segments: Vec<&str> = draft
        .split("\n\n")
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    let any_overruns = segments.iter().any(|s| s.chars().count() > max_per_tweet);
    if !any_overruns {
        // Fast path: nothing to do. Preserve the exact original string
        // (including any surrounding whitespace the writer emitted) so
        // downstream byte-equal comparisons in tests stay stable.
        return draft.to_string();
    }

    let mut rebuilt: Vec<String> = Vec::with_capacity(segments.len());
    for segment in segments {
        if segment.chars().count() <= max_per_tweet {
            rebuilt.push(segment.to_string());
            continue;
        }
        let chunks = split_segment_to_chunks(segment, max_per_tweet);
        for c in chunks {
            rebuilt.push(c);
        }
    }
    rebuilt.join("\n\n")
}

/// Split one over-length segment into sentence-boundary chunks.
/// If no boundary produces a ≤max chunk (e.g. one giant sentence), the
/// original segment is returned as a single element — caller's
/// `publish_gate` will reject it.
fn split_segment_to_chunks(segment: &str, max_per_tweet: usize) -> Vec<String> {
    let sentences = split_on_sentence_boundaries(segment);

    // Greedy pack: accumulate sentences into a buffer until adding the
    // next one would overflow, then commit and start a new buffer.
    let mut chunks: Vec<String> = Vec::new();
    let mut current = String::new();
    for s in sentences {
        let candidate_len = if current.is_empty() {
            s.chars().count()
        } else {
            current.chars().count() + 1 + s.chars().count() // +1 for the joining space
        };
        if candidate_len <= max_per_tweet {
            if !current.is_empty() {
                current.push(' ');
            }
            current.push_str(s);
        } else if current.is_empty() {
            // A single sentence is over the limit and we have no
            // accumulator to flush. Emit it as-is and let publish_gate
            // reject — preserving the writer's text is more honest than
            // truncating mid-thought.
            chunks.push(s.to_string());
        } else {
            chunks.push(std::mem::take(&mut current));
            current.push_str(s);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    if chunks.is_empty() {
        // Defensive: if there were no sentences at all (e.g. an
        // empty-after-trim segment, though we filtered those out
        // earlier), return the original so we never produce an empty
        // chunk list from a non-empty input.
        chunks.push(segment.to_string());
    }
    chunks
}

/// Walk `text` and split at every sentence boundary: a `.`, `!`, or `?`
/// followed by whitespace and an ASCII uppercase letter. Returns a Vec
/// of trimmed sentence slices in source order.
///
/// We use byte-level scanning rather than regex because Rust's std lacks
/// look-around and pulling in `fancy-regex` for this is overkill.
/// Conservative on edge cases: ellipsis (`...`) won't match because the
/// next char after the dot is another dot, not whitespace. URLs like
/// `https://example.com.` don't usually have an uppercase letter after
/// the trailing dot+space, so false positives are rare. We accept
/// occasional sub-optimal splits as a better failure mode than mid-
/// sentence breaks.
fn split_on_sentence_boundaries(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out: Vec<&str> = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i + 2 < bytes.len() {
        let c = bytes[i];
        if (c == b'.' || c == b'!' || c == b'?')
            && bytes[i + 1] == b' '
            && bytes[i + 2].is_ascii_uppercase()
        {
            // Boundary AFTER the punctuation. Slice up to and including it,
            // then skip the trailing space so the next sentence starts clean.
            let segment = text[start..=i].trim();
            if !segment.is_empty() {
                out.push(segment);
            }
            start = i + 2;
            i += 2;
            continue;
        }
        i += 1;
    }
    if start < text.len() {
        let tail = text[start..].trim();
        if !tail.is_empty() {
            out.push(tail);
        }
    }
    // Edge case: no boundaries found at all → return the whole thing as a
    // single "sentence" so the caller's greedy packer can decide what
    // to do (it will emit-as-is since current is empty).
    if out.is_empty() && !text.trim().is_empty() {
        out.push(text.trim());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_single_tweet_passes_through_unchanged() {
        let in_ = "a short tweet that obviously fits";
        let out = normalize_tweet_length(in_, MAX_TWEET_CHARS);
        assert_eq!(out, in_, "no-op fast path must preserve input bytes");
    }

    #[test]
    fn empty_input_returns_empty() {
        let out = normalize_tweet_length("", MAX_TWEET_CHARS);
        assert_eq!(out, "");
    }

    #[test]
    fn already_valid_thread_passes_through_unchanged() {
        let in_ = "first tweet\n\nsecond tweet\n\nthird tweet";
        let out = normalize_tweet_length(in_, MAX_TWEET_CHARS);
        assert_eq!(out, in_);
    }

    #[test]
    fn over_length_block_splits_on_sentence_boundaries() {
        // 6-sentence block of ~370 chars total. Greedy packer must split
        // into multiple ≤280-char tweets without truncating any sentence.
        let s = "Agent loops without guardrails are credit cards on a while loop. \
                 The cost is silent until production catches fire and someone pages you. \
                 Wire compaction at the orchestrator level not the leaf agent. \
                 The orchestrator owns the budget and the leaf owns the work. \
                 Token budgets without exit conditions are a bug not a feature. \
                 Bound everything that loops.";
        assert!(
            s.chars().count() > MAX_TWEET_CHARS,
            "fixture must overrun; got {} chars",
            s.chars().count()
        );
        let out = normalize_tweet_length(s, MAX_TWEET_CHARS);
        let tweets: Vec<&str> = out.split("\n\n").collect();
        assert!(
            tweets.len() >= 2,
            "expected ≥2 chunks after split; got {tweets:?}"
        );
        for (i, t) in tweets.iter().enumerate() {
            assert!(
                t.chars().count() <= MAX_TWEET_CHARS,
                "chunk {i} is {} chars (>{MAX_TWEET_CHARS}): {t:?}",
                t.chars().count()
            );
        }
    }

    #[test]
    fn unsplittable_single_long_sentence_stays_intact() {
        // One sentence, no boundaries. The function returns it as-is and
        // the caller's publish_gate will reject it. We assert no
        // truncation, no mid-word break.
        let s = "x".repeat(310); // single "sentence" of 310 chars, no `.` or `?`
        let out = normalize_tweet_length(&s, MAX_TWEET_CHARS);
        assert_eq!(out, s, "must not truncate or break mid-content");
    }

    #[test]
    fn mixed_thread_preserves_short_segments_and_splits_long_ones() {
        let s = "short opener\n\n\
                 The middle paragraph is way too long. It has several sentences. \
                 Each one is meaningful. The whole thing overruns the limit. \
                 We need it to split here. Because nobody wants a single 300-char tweet. \
                 Splitting cleanly is the goal. Mid-sentence breaks are forbidden.\n\n\
                 short closer";
        let out = normalize_tweet_length(s, MAX_TWEET_CHARS);
        let tweets: Vec<&str> = out.split("\n\n").collect();
        assert_eq!(tweets[0], "short opener");
        assert_eq!(tweets[tweets.len() - 1], "short closer");
        for t in &tweets[1..tweets.len() - 1] {
            assert!(
                t.chars().count() <= MAX_TWEET_CHARS,
                "middle chunk too long: {t:?}"
            );
        }
    }

    #[test]
    fn boundary_detection_skips_dots_not_followed_by_capital() {
        // Common false-positive trap: "e.g." or a URL like "example.com" or
        // a number "3.14". None should be treated as a sentence boundary.
        let in_ = "Use e.g. tracing and metrics. \
                   The url example.com is fine. \
                   Constants like pi (3.14) are also fine.";
        let sentences = split_on_sentence_boundaries(in_);
        // Expect 3 boundaries: between sentences ending in `.` followed by
        // a capital. NOT after "e.g." (followed by "tracing", lowercase) or
        // after "example.com" (the dot is in the middle of the token).
        assert_eq!(
            sentences.len(),
            3,
            "false-positive guard: expected 3 sentences, got {sentences:?}"
        );
        assert!(sentences[0].starts_with("Use e.g."));
        assert!(sentences[1].starts_with("The url"));
        assert!(sentences[2].starts_with("Constants"));
    }

    #[test]
    fn fast_path_preserves_input_bytes_including_extra_separators() {
        // The function's contract is length normalization, NOT formatting
        // cleanup. When every segment already fits, the input is returned
        // verbatim — including any oddities like `\n\n\n\n`. Downstream
        // `parse_thread_tweets` is responsible for filtering empties when
        // it splits for the publish_gate, so we don't duplicate that work
        // here.
        let in_ = "first\n\n\n\nsecond";
        let out = normalize_tweet_length(in_, MAX_TWEET_CHARS);
        assert_eq!(out, in_, "fast path must pass through bytes unchanged");
    }

    #[test]
    fn rebuild_path_normalizes_separators_to_single_blank_line() {
        // When the function HAS to rebuild (some segment overruns), the
        // rebuilt output uses canonical `\n\n` separators — the extra
        // blank lines from the input are not preserved through the join.
        let long = "Agent loops without guardrails are credit cards on a while loop. \
                    The cost is silent until production catches fire and someone pages you. \
                    Wire compaction at the orchestrator level not the leaf agent. \
                    The orchestrator owns the budget and the leaf owns the work. \
                    Token budgets without exit conditions are a bug not a feature. \
                    Bound everything that loops.";
        assert!(long.chars().count() > MAX_TWEET_CHARS);
        let in_ = format!("short\n\n\n\n{long}\n\n\n\nshort");
        let out = normalize_tweet_length(&in_, MAX_TWEET_CHARS);
        assert!(
            !out.contains("\n\n\n"),
            "rebuilt output should not contain triple newlines; got: {out:?}"
        );
        let tweets: Vec<&str> = out.split("\n\n").collect();
        assert_eq!(tweets[0], "short");
        assert_eq!(tweets[tweets.len() - 1], "short");
    }

    #[test]
    fn idempotent_on_already_normalized_output() {
        // Run normalize twice; second call must be a no-op.
        let in_ = "first short tweet\n\nsecond short tweet";
        let once = normalize_tweet_length(in_, MAX_TWEET_CHARS);
        let twice = normalize_tweet_length(&once, MAX_TWEET_CHARS);
        assert_eq!(once, twice, "second pass must be a no-op");
    }

    #[test]
    fn regression_actual_overrun_from_production_log_2026_05_12() {
        // The exact 1058-char block from the 2026-05-12 14:12 gate
        // rejection that motivated this whole fix. Verify normalize
        // turns it into a valid thread.
        let s = "Retrying tool calls is a reliability trap for multi-agent systems. \
                 In a shared state environment, exponential backoff creates \"ghost\" contexts. \
                 An agent reads version one of a file and fails its tool execution. \
                 While that agent backs off, a second agent updates the file to version two. \
                 The first agent eventually retries using its original cached context. \
                 It silently overwrites the new work with a stale version. \
                 Standard backoff logic ignores the race conditions inherent in concurrent agents. \
                 We are seeing token usage spike during these cascading retry storms. \
                 The solution is moving toward Write-Ahead Logging and deterministic task IDs. \
                 Operations must be idempotent to survive a retry without side effects. \
                 I prefer immediate failure propagation for semantic errors like invalid json. \
                 Save the exponential backoff for transient issues like rate limits. \
                 Saga patterns offer a better path than naive loops. \
                 You cannot hold a write lock for minutes while a model computes. \
                 Consistency in agentic systems is won or lost at the coordination layer.";
        let out = normalize_tweet_length(s, MAX_TWEET_CHARS);
        let tweets: Vec<&str> = out.split("\n\n").collect();
        assert!(
            tweets.len() >= 4,
            "1000-char fixture should split into ≥4 tweets; got {} chunks",
            tweets.len()
        );
        for (i, t) in tweets.iter().enumerate() {
            assert!(
                t.chars().count() <= MAX_TWEET_CHARS,
                "chunk {i} still over limit: {} chars",
                t.chars().count()
            );
        }
    }
}
