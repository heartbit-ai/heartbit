//! Post-action verification via state-diff (spec capability 5).
//!
//! "An Illusion of Progress" (COLM'25) found the #1 web-agent failure *class* is
//! incomplete steps — the agent fills a form but never confirms the Submit
//! landed, then reports success. Agent-E's "change observation" is the fix:
//! after every mutating action, re-observe and confirm the page actually changed
//! as intended; if nothing changed, the action was a no-op and the loop should
//! retry or replan instead of marching on.
//!
//! This module reduces a snapshot to a cheap [`SnapshotSignature`] and
//! [`diff`]s before/after signatures into an [`ActionEffect`]. Pure functions
//! over snapshot text — no browser/MCP/LLM — so fully unit-testable.

use std::collections::{BTreeMap, BTreeSet};

use super::distill::interactive_uids;

/// A cheap fingerprint of page state, captured before and after an action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotSignature {
    /// Current page URL.
    pub url: String,
    /// The page title / RootWebArea accessible name, if present.
    pub title: Option<String>,
    /// Set of interactable element handles present in the snapshot.
    pub interactable_uids: BTreeSet<String>,
    /// Per-`uid` content fingerprint covering everything after the `uid=N_M`
    /// token (role, accessible name, `value=`, and other attributes). This is
    /// what catches in-place changes such as a field whose `value="..."` was
    /// set, or an element relabeled while keeping its `uid` — changes that leave
    /// the URL, the title, and the interactable-uid set all untouched. Omitting
    /// it caused the top "Illusion of Progress / Incomplete Steps" false-`NoOp`:
    /// a successful `fill` reported as a no-op.
    pub content_digest: BTreeMap<String, String>,
    /// Count of console error lines observed (a negative signal).
    pub console_errors: usize,
}

/// The observed effect of an action, derived from a before/after [`diff`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActionEffect {
    /// The page changed; the string summarizes what (for the loop / logs).
    Changed(String),
    /// Nothing observable changed — the action was a no-op (likely a miss).
    NoOp,
}

impl ActionEffect {
    /// Whether this effect indicates real progress (a change occurred).
    pub fn is_progress(&self) -> bool {
        matches!(self, ActionEffect::Changed(_))
    }

    /// A structured hint to append to a no-op tool result so the agent loop
    /// re-grounds / replans instead of trusting an unverified "done".
    pub fn hint(&self) -> Option<&'static str> {
        match self {
            ActionEffect::NoOp => Some(
                "[browser] observed change: NO — the page did not change after this \
                 action. The element may be wrong or the action had no effect. \
                 Re-snapshot, re-check the target, and try a different approach.",
            ),
            ActionEffect::Changed(_) => None,
        }
    }
}

/// The accessible name on the `RootWebArea` line, used as the page title.
fn root_title(snapshot: &str) -> Option<String> {
    snapshot.lines().find_map(|l| {
        let t = l.trim_start();
        if t.contains("RootWebArea") {
            let start = t.find('"')?;
            let rest = &t[start + 1..];
            let end = rest.find('"')?;
            Some(rest[..end].to_string())
        } else {
            None
        }
    })
}

/// Split a snapshot line into `(uid, content)` where `content` is everything
/// after the `uid=N_M` token (role + quoted name + `value=`/`url=`/other attrs),
/// trimmed. Lines without a `uid=` token are ignored (returns `None`).
fn uid_and_content(line: &str) -> Option<(String, String)> {
    let body = line.trim_start();
    let rest = body.strip_prefix("uid=")?;
    let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
    let uid = &rest[..end];
    let content = rest[end..].trim();
    Some((uid.to_string(), content.to_string()))
}

/// Build a [`SnapshotSignature`] from a snapshot, the page URL, and console
/// lines. `console` lines containing "error" (case-insensitive) are counted.
pub fn signature(snapshot: &str, url: &str, console: &[String]) -> SnapshotSignature {
    SnapshotSignature {
        url: url.to_string(),
        title: root_title(snapshot),
        interactable_uids: interactive_uids(snapshot).into_iter().collect(),
        content_digest: snapshot.lines().filter_map(uid_and_content).collect(),
        console_errors: console
            .iter()
            .filter(|l| l.to_lowercase().contains("error"))
            .count(),
    }
}

/// Diff two signatures into an [`ActionEffect`]. Any of: URL change, title
/// change, a changed set of interactable handles, or new console errors counts
/// as a change. Identical signatures are a [`ActionEffect::NoOp`].
pub fn diff(before: &SnapshotSignature, after: &SnapshotSignature) -> ActionEffect {
    let mut reasons = Vec::new();

    if before.url != after.url {
        reasons.push(format!("url {} -> {}", before.url, after.url));
    }
    if before.title != after.title {
        reasons.push(format!("title {:?} -> {:?}", before.title, after.title));
    }
    if before.interactable_uids != after.interactable_uids {
        let added = after
            .interactable_uids
            .difference(&before.interactable_uids)
            .count();
        let removed = before
            .interactable_uids
            .difference(&after.interactable_uids)
            .count();
        reasons.push(format!("elements +{added}/-{removed}"));
    }
    // In-place content change: same `uid`, different role/name/value/attrs — a set
    // `value="..."` or a same-uid relabel (Play->Pause). URL/title/uid-set all miss
    // these; this is the fix for the #1 false-`NoOp` (Illusion-of-Progress) class.
    let changed = after
        .content_digest
        .iter()
        .filter(|(uid, content)| {
            before
                .content_digest
                .get(*uid)
                .is_some_and(|prev| prev != *content)
        })
        .count();
    if changed > 0 {
        reasons.push(format!("content changed on {changed} element(s)"));
    }
    // B2: added/removed content nodes — including non-interactive ones
    // (StaticText, alert) that `interactable_uids` does not track. A failed
    // login adding `StaticText "Invalid password"`, or a successful submit
    // adding a non-interactive success banner, is a real change that URL,
    // title, and the interactable-uid set all miss → false `NoOp` otherwise.
    let content_added = after
        .content_digest
        .keys()
        .filter(|uid| !before.content_digest.contains_key(uid.as_str()))
        .count();
    let content_removed = before
        .content_digest
        .keys()
        .filter(|uid| !after.content_digest.contains_key(uid.as_str()))
        .count();
    if content_added > 0 || content_removed > 0 {
        reasons.push(format!("content nodes +{content_added}/-{content_removed}"));
    }
    if after.console_errors > before.console_errors {
        reasons.push(format!(
            "console errors +{}",
            after.console_errors - before.console_errors
        ));
    }

    if reasons.is_empty() {
        ActionEffect::NoOp
    } else {
        ActionEffect::Changed(reasons.join("; "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SNAP_A: &str = r#"uid=1_0 RootWebArea "Login" url="https://app.test/login"
  uid=1_1 textbox "Email"
  uid=1_2 textbox "Password"
  uid=1_3 button "Sign in""#;

    // Same page, but after a successful sign-in: new URL, new title, new elements.
    const SNAP_B: &str = r#"uid=2_0 RootWebArea "Dashboard" url="https://app.test/home"
  uid=2_1 link "Profile"
  uid=2_2 button "Log out""#;

    #[test]
    fn signature_extracts_title_url_uids_and_errors() {
        let sig = signature(SNAP_A, "https://app.test/login", &["console.log ok".into()]);
        assert_eq!(sig.url, "https://app.test/login");
        assert_eq!(sig.title.as_deref(), Some("Login"));
        assert_eq!(
            sig.interactable_uids,
            ["1_1", "1_2", "1_3"]
                .iter()
                .map(|s| s.to_string())
                .collect()
        );
        assert_eq!(sig.console_errors, 0);
    }

    #[test]
    fn signature_counts_console_errors() {
        let console = vec![
            "Uncaught TypeError: x is not a function".into(),
            "info: loaded".into(),
            "ERROR: network failed".into(),
        ];
        let sig = signature(SNAP_A, "u", &console);
        assert_eq!(sig.console_errors, 2);
    }

    #[test]
    fn diff_identical_is_noop() {
        let a = signature(SNAP_A, "https://app.test/login", &[]);
        let b = signature(SNAP_A, "https://app.test/login", &[]);
        assert_eq!(diff(&a, &b), ActionEffect::NoOp);
        assert!(!diff(&a, &b).is_progress());
        assert!(diff(&a, &b).hint().is_some());
    }

    #[test]
    fn diff_navigation_is_change() {
        let a = signature(SNAP_A, "https://app.test/login", &[]);
        let b = signature(SNAP_B, "https://app.test/home", &[]);
        let effect = diff(&a, &b);
        assert!(effect.is_progress(), "navigation must register as a change");
        if let ActionEffect::Changed(summary) = effect {
            assert!(summary.contains("url"), "summary: {summary}");
            assert!(summary.contains("title"), "summary: {summary}");
            assert!(summary.contains("elements"), "summary: {summary}");
        } else {
            panic!("expected Changed");
        }
        assert!(diff(&a, &b).hint().is_none());
    }

    #[test]
    fn diff_same_url_new_element_is_change() {
        // An inline validation error appearing (same URL/title, +1 element).
        let before = signature(SNAP_A, "https://app.test/login", &[]);
        let with_error = r#"uid=1_0 RootWebArea "Login" url="https://app.test/login"
  uid=1_1 textbox "Email"
  uid=1_2 textbox "Password"
  uid=1_3 button "Sign in"
  uid=1_9 textbox "2FA code""#;
        let after = signature(with_error, "https://app.test/login", &[]);
        let effect = diff(&before, &after);
        assert!(effect.is_progress());
        if let ActionEffect::Changed(s) = effect {
            assert!(s.contains("elements +1"), "summary: {s}");
        }
    }

    #[test]
    fn diff_added_noninteractive_text_is_change() {
        // B2: a failed login adds a non-interactive StaticText error banner with
        // no URL/title/interactive-element change. URL, title, and the
        // interactable-uid set all miss it; it must still register as a change,
        // not a false NoOp ("Illusion of Progress" inverse).
        let before = signature(SNAP_A, "https://app.test/login", &[]);
        let with_banner = r#"uid=1_0 RootWebArea "Login" url="https://app.test/login"
  uid=1_1 textbox "Email"
  uid=1_2 textbox "Password"
  uid=1_3 button "Sign in"
  uid=1_9 StaticText "Invalid password""#;
        let after = signature(with_banner, "https://app.test/login", &[]);
        let effect = diff(&before, &after);
        assert!(
            effect.is_progress(),
            "an added non-interactive error banner must register as a change, got {effect:?}"
        );
        if let ActionEffect::Changed(s) = effect {
            assert!(s.contains("content nodes +1"), "summary: {s}");
        }
    }

    #[test]
    fn diff_new_console_error_is_change() {
        let a = signature(SNAP_A, "u", &[]);
        let b = signature(SNAP_A, "u", &["ERROR: submit failed".into()]);
        assert_eq!(
            diff(&a, &b),
            ActionEffect::Changed("console errors +1".to_string())
        );
    }

    // --- in-place mutation: the #1 "Illusion of Progress / Incomplete Steps"
    // failure class. Live chrome-devtools-mcp evidence (2026-05-31) confirmed the
    // textbox value IS in the snapshot as `value="..."` and that an element keeps
    // its uid across same-document mutation — so url/title/uid-set/console are all
    // unchanged when you fill a field or a label flips. The signature must still
    // register these as a Change, or the agent reports a no-op step as success.

    #[test]
    fn diff_textbox_value_set_is_change() {
        // Same page, same uids, only the typed-in value differs (fill succeeded).
        let empty = r#"uid=4_0 RootWebArea "Form" url="https://app.test/f"
  uid=4_1 textbox "Email"
  uid=4_2 button "Submit""#;
        let filled = r#"uid=4_0 RootWebArea "Form" url="https://app.test/f"
  uid=4_1 textbox "Email" value="me@x.com"
  uid=4_2 button "Submit""#;
        let before = signature(empty, "https://app.test/f", &[]);
        let after = signature(filled, "https://app.test/f", &[]);
        assert!(
            diff(&before, &after).is_progress(),
            "typing into a textbox (value set) must register as progress, not NoOp"
        );
    }

    #[test]
    fn diff_same_uid_relabel_is_change() {
        // A toggle button whose label flips Play->Pause but keeps its uid.
        let play = r#"uid=4_0 RootWebArea "Player" url="https://app.test/p"
  uid=4_1 button "Play""#;
        let pause = r#"uid=4_0 RootWebArea "Player" url="https://app.test/p"
  uid=4_1 button "Pause""#;
        let before = signature(play, "https://app.test/p", &[]);
        let after = signature(pause, "https://app.test/p", &[]);
        assert!(
            diff(&before, &after).is_progress(),
            "a same-uid label change (Play->Pause) must register as progress"
        );
    }

    #[test]
    fn diff_true_noop_still_noop_with_digest() {
        // Guard against over-sensitivity: identical snapshots (incl. values) must
        // remain NoOp even though we now hash element content.
        let snap = r#"uid=4_0 RootWebArea "Form" url="https://app.test/f"
  uid=4_1 textbox "Email" value="me@x.com"
  uid=4_2 button "Submit""#;
        let a = signature(snap, "https://app.test/f", &[]);
        let b = signature(snap, "https://app.test/f", &[]);
        assert_eq!(
            diff(&a, &b),
            ActionEffect::NoOp,
            "an unchanged page (same values) must stay NoOp"
        );
    }

    #[test]
    fn noop_hint_guides_replan() {
        assert!(
            ActionEffect::NoOp
                .hint()
                .unwrap()
                .contains("observed change: NO")
        );
        assert!(ActionEffect::Changed("x".into()).hint().is_none());
    }
}
