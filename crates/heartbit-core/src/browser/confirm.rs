//! Destructive-action confirmation (capability 21 of the browser-bot spec).
//!
//! A credentialed browser agent that can submit forms, spend money, send
//! messages, or delete data is one prompt-injection away from doing real harm —
//! the embracethered "Manus kill-chain" showed an agent with NO human-in-the-loop
//! on consequential tools could be driven to full devbox takeover by a poisoned
//! page. The deep-research synthesis ranked **human confirmation on consequential
//! actions** as the single highest-ROI safety control (above an injection
//! classifier), precisely because it bounds the blast radius of *any* failure
//! mode — injection, hallucination, or a wrong-element click.
//!
//! The hard part for an a11y/`uid` agent is that the tool call is generic: a
//! `click` on "Buy now" and a `click` on "Read more" are byte-identical except
//! for the target's accessible label. So classification is **label-aware**:
//! [`label_for_uid`] resolves the target's label from the latest snapshot, and
//! [`classify_label`] maps it to an [`ActionRisk`]. The harness routes anything
//! that [`ActionRisk::requires_confirmation`] to the existing
//! [`OnApproval`](crate::OnApproval) hook (that async routing is the thin shell,
//! layered later — this module is the deterministic, exhaustively-tested core).
//!
//! Safety bias: classification **fails toward confirming**. A false positive is
//! a harmless extra prompt; a false negative is an unconfirmed irreversible
//! action — so the keyword sets favour catching the dangerous case, and the
//! operator can extend them via [`ConfirmPolicy`] without code changes.

/// The blast radius of a browser action, inferred from its target's label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActionRisk {
    /// Safe to take autonomously (navigation, reading, non-committal UI).
    Benign,
    /// Has a real-world effect that is hard to undo — buy / pay / send /
    /// publish. Route to human confirmation.
    Consequential,
    /// Effectively permanent or destructive — delete / wipe / deactivate.
    /// Route to human confirmation; the harness may additionally refuse to
    /// auto-approve these even under a blanket allow.
    Irreversible,
}

impl ActionRisk {
    /// Whether the harness must seek human confirmation before proceeding. Only
    /// [`ActionRisk::Benign`] may proceed autonomously.
    pub fn requires_confirmation(self) -> bool {
        matches!(self, ActionRisk::Consequential | ActionRisk::Irreversible)
    }
}

/// Operator-tunable keyword policy for [`classify_label`]. Defaults catch the
/// canonical dangerous buttons; generic words like "submit"/"ok" are
/// deliberately omitted from the defaults so ordinary form/search use is not
/// crippled — add them per deployment if the threat model warrants.
#[derive(Debug, Clone)]
pub struct ConfirmPolicy {
    /// Substrings (matched case-insensitively) marking an irreversible action.
    pub irreversible_keywords: Vec<String>,
    /// Substrings (matched case-insensitively) marking a consequential action.
    pub consequential_keywords: Vec<String>,
}

impl Default for ConfirmPolicy {
    fn default() -> Self {
        let owned = |xs: &[&str]| xs.iter().map(|s| s.to_string()).collect();
        Self {
            irreversible_keywords: owned(&[
                "delete",
                "remove",
                "permanently",
                "destroy",
                "wipe",
                "erase",
                "deactivate",
                "close account",
            ]),
            consequential_keywords: owned(&[
                "buy",
                "purchase",
                "pay",
                "checkout",
                "place order",
                "order now",
                "send",
                "transfer",
                "publish",
                "subscribe",
            ]),
        }
    }
}

/// Classify a target element's accessible label into an [`ActionRisk`].
///
/// Matching is case-insensitive substring containment. Irreversible is checked
/// first so a label that trips both sets (e.g. "permanently delete and pay") is
/// classified by its most severe match. An empty/whitespace label is
/// [`ActionRisk::Benign`] (nothing to commit).
pub fn classify_label(policy: &ConfirmPolicy, label: &str) -> ActionRisk {
    let label = label.to_lowercase();
    if label.trim().is_empty() {
        return ActionRisk::Benign;
    }
    // Irreversible checked first: a label matching both sets is judged by its
    // most severe match.
    if policy
        .irreversible_keywords
        .iter()
        .any(|k| label.contains(&k.to_lowercase()))
    {
        return ActionRisk::Irreversible;
    }
    if policy
        .consequential_keywords
        .iter()
        .any(|k| label.contains(&k.to_lowercase()))
    {
        return ActionRisk::Consequential;
    }
    ActionRisk::Benign
}

/// Resolve the accessible label (first quoted span) of the element carrying
/// `uid` in a `take_snapshot` tree. Returns `None` if the `uid` is absent or the
/// line has no quoted name. Pure over the snapshot text — the harness captures
/// the snapshot, this extracts the target so [`classify_label`] can judge it.
pub fn label_for_uid(snapshot: &str, uid: &str) -> Option<String> {
    for line in snapshot.lines() {
        let body = line.trim_start();
        let Some(rest) = body.strip_prefix("uid=") else {
            continue;
        };
        let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
        if &rest[..end] != uid {
            continue;
        }
        // Matched the uid; the label is the first double-quoted span.
        let after = &rest[end..];
        let start = after.find('"')?;
        let q = &after[start + 1..];
        let close = q.find('"')?;
        return Some(q[..close].to_string());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ActionRisk::requires_confirmation ---

    #[test]
    fn only_benign_proceeds_autonomously() {
        assert!(!ActionRisk::Benign.requires_confirmation());
        assert!(ActionRisk::Consequential.requires_confirmation());
        assert!(ActionRisk::Irreversible.requires_confirmation());
    }

    // --- classify_label (the keyword brain) ---

    #[test]
    fn benign_label_is_benign() {
        let p = ConfirmPolicy::default();
        assert_eq!(classify_label(&p, "Read more"), ActionRisk::Benign);
        assert_eq!(classify_label(&p, "Next page"), ActionRisk::Benign);
        assert_eq!(classify_label(&p, "Search"), ActionRisk::Benign);
    }

    #[test]
    fn consequential_buttons_are_flagged() {
        let p = ConfirmPolicy::default();
        assert_eq!(classify_label(&p, "Buy now"), ActionRisk::Consequential);
        assert_eq!(classify_label(&p, "Place order"), ActionRisk::Consequential);
        assert_eq!(classify_label(&p, "Pay $42.00"), ActionRisk::Consequential);
        assert_eq!(
            classify_label(&p, "Send message"),
            ActionRisk::Consequential
        );
    }

    #[test]
    fn irreversible_buttons_are_flagged() {
        let p = ConfirmPolicy::default();
        assert_eq!(
            classify_label(&p, "Delete account"),
            ActionRisk::Irreversible
        );
        assert_eq!(
            classify_label(&p, "Permanently remove"),
            ActionRisk::Irreversible
        );
    }

    #[test]
    fn classification_is_case_insensitive() {
        let p = ConfirmPolicy::default();
        assert_eq!(classify_label(&p, "DELETE"), ActionRisk::Irreversible);
        assert_eq!(classify_label(&p, "bUy NoW"), ActionRisk::Consequential);
    }

    #[test]
    fn irreversible_takes_priority_over_consequential() {
        let p = ConfirmPolicy::default();
        // Label matches both sets ("delete" + "pay") → most-severe wins.
        assert_eq!(
            classify_label(&p, "Delete and pay"),
            ActionRisk::Irreversible
        );
    }

    #[test]
    fn empty_label_is_benign() {
        let p = ConfirmPolicy::default();
        assert_eq!(classify_label(&p, ""), ActionRisk::Benign);
        assert_eq!(classify_label(&p, "   "), ActionRisk::Benign);
    }

    #[test]
    fn generic_submit_is_benign_by_default_but_policy_extensible() {
        // "Submit" is intentionally NOT in the defaults (preserve autonomy)...
        let p = ConfirmPolicy::default();
        assert_eq!(classify_label(&p, "Submit"), ActionRisk::Benign);
        // ...but an operator can opt in.
        let strict = ConfirmPolicy {
            consequential_keywords: vec!["submit".to_string()],
            ..ConfirmPolicy::default()
        };
        assert_eq!(classify_label(&strict, "Submit"), ActionRisk::Consequential);
    }

    // --- label_for_uid (target resolution from snapshot) ---

    const SNAP: &str = r#"uid=1_0 RootWebArea "Checkout" url="https://shop.test/cart"
  uid=1_1 StaticText "Total: $42"
  uid=1_2 button "Pay now"
  uid=1_3 link "Keep shopping""#;

    #[test]
    fn label_for_uid_resolves_target_label() {
        assert_eq!(label_for_uid(SNAP, "1_2").as_deref(), Some("Pay now"));
        assert_eq!(label_for_uid(SNAP, "1_3").as_deref(), Some("Keep shopping"));
    }

    #[test]
    fn label_for_uid_missing_uid_is_none() {
        assert_eq!(label_for_uid(SNAP, "9_9"), None);
    }

    #[test]
    fn label_for_uid_then_classify_flags_the_pay_button() {
        // The end-to-end pure path the harness shell will run before a click.
        let p = ConfirmPolicy::default();
        let label = label_for_uid(SNAP, "1_2").expect("uid present");
        assert_eq!(classify_label(&p, &label), ActionRisk::Consequential);
        assert!(classify_label(&p, &label).requires_confirmation());
    }
}
