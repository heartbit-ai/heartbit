//! Request-intent router — picks the RESPONSE MODE a request calls for,
//! before the first LLM turn (design: `tasks/intent-mode-router-2026-06-07.md`).
//!
//! Mid-tier models read hedged requests literally ("je souhaite créer un
//! petit crm" → unilateral build; Ruis 2210.14986, Korean-ISA 2502.10995:
//! 84.7% direct vs 58.1% indirect comprehension), so the harness carries the
//! pragmatic load deterministically: Layer 0 here (marker scan, ~free,
//! multilingual by enumeration), Layer 1 (`fast`-role LLM) only for the
//! ambiguous residue, Layer 2 defaults to the SAFER mode on uncertainty.
//! The model never picks its own mode; the user always can (go-tokens,
//! pinned mode).

use std::sync::Arc;

use crate::llm::types::{CompletionRequest, ContentBlock, Message};
use crate::llm::{BoxedProvider, LlmProvider};

/// The response mode a request calls for — force × completeness 2×2
/// (DAMSL force, ClarEval completeness). See design doc §2.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestMode {
    /// Assertive / info-request, specified → answer in prose, no staging.
    Answer,
    /// Directive, specified → act directly (today's behavior).
    Execute,
    /// Investigation / under-specified-for-design → read-only; must end in a
    /// written proposal + go/no-go.
    Study,
    /// Under-specified (any force) → ask first (intake/question) before any
    /// mutation.
    Clarify,
}

impl RequestMode {
    /// Stable label for traces and prompts.
    pub fn label(self) -> &'static str {
        match self {
            RequestMode::Answer => "answer",
            RequestMode::Execute => "execute",
            RequestMode::Study => "study",
            RequestMode::Clarify => "clarify",
        }
    }

    /// Parse a label (for pinned modes / classifier replies).
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "answer" => Some(RequestMode::Answer),
            "execute" => Some(RequestMode::Execute),
            "study" => Some(RequestMode::Study),
            "clarify" => Some(RequestMode::Clarify),
            _ => None,
        }
    }
}

/// Explicit go-tokens: the user's keypress always wins — EXECUTE regardless
/// of phrasing. Checked first.
const GO_TOKENS: &[&str] = &[
    "vas-y",
    "vas y",
    "fais-le",
    "fais le",
    "do it",
    "just do it",
    "just build it",
    "go ahead",
    "lance-toi",
    "code directement",
    "code-le",
    "implémente directement",
    "implemente directement",
];

/// Wish / desiderative force markers (hedged, negotiable requests).
const WISH_MARKERS: &[&str] = &[
    "je souhaite",
    "j'aimerais",
    "je voudrais",
    "il faudrait",
    "ce serait bien",
    "j'apprécierais",
    "j'aurais besoin",
    "i'd like",
    "i would like",
    "it would be nice",
    "could you maybe",
    "i wish",
];

/// Investigation force markers → STUDY (propose, no build).
const STUDY_MARKERS: &[&str] = &[
    "regarde si",
    "étudie",
    "etudie",
    "analyse",
    "explore",
    "réfléchis",
    "reflechis",
    "compare les options",
    "est-ce qu'on peut",
    "est-ce que l'on peut",
    "peut-on",
    "serait-il possible",
    "is it possible",
    "can we",
    "could we",
    "investigate",
    "look into",
    "deep research",
    "que penses-tu",
    "what do you think",
    "évalue",
    "evalue",
];

/// Design-heavy nouns that, WITHOUT a concrete spec, signal an
/// underspecified build request.
const DESIGN_NOUNS: &[&str] = &[
    "crm",
    "app",
    "application",
    "site",
    "dashboard",
    "api",
    "service",
    "bot",
    "jeu",
    "game",
    "outil",
    "tool",
    "feature",
    "fonctionnalité",
    "fonctionnalite",
    "module",
    "système",
    "systeme",
    "plateforme",
    "platform",
    "interface",
];

/// Interrogative openers → ANSWER (when not paired with an imperative).
const QUESTION_OPENERS: &[&str] = &[
    "que ",
    "qu'est-ce",
    "quoi ",
    "comment ",
    "pourquoi ",
    "quel ",
    "quelle ",
    "quels ",
    "quelles ",
    "what ",
    "why ",
    "how ",
    "which ",
    "explique",
    "explain",
    "c'est quoi",
];

/// True when the text carries a CONCRETE spec anchor: a path-ish token, a
/// backticked symbol, or a file extension — the "specified" completeness
/// signal (a specified wish is an EXECUTE, not a CLARIFY).
fn has_concrete_anchor(lower: &str) -> bool {
    if lower.contains('`') {
        return true;
    }
    // path-ish: a token containing '/' with no spaces around the slash, or a
    // known source-file extension.
    if lower.split_whitespace().any(|w| {
        (w.contains('/') && w.len() > 3)
            || w.ends_with(".rs")
            || w.ends_with(".py")
            || w.ends_with(".ts")
            || w.ends_with(".js")
            || w.ends_with(".toml")
            || w.ends_with(".md")
            || w.ends_with(".json")
            || w.ends_with(".txt")
            || w.ends_with(".html")
            || w.ends_with(".css")
    }) {
        return true;
    }
    false
}

/// FR conditional morphology on volition/possibility lemmas — the grammatical
/// hedge marker ("voudrais", "aimerais", "faudrait", "pourrait-on"…).
fn has_fr_conditional_volition(lower: &str) -> bool {
    const STEMS: &[&str] = &[
        "voudr",
        "aimer",
        "faudr",
        "pourr",
        "souhaiter",
        "apprécier",
        "apprecier",
    ];
    lower.split_whitespace().any(|w| {
        let w = w.trim_matches(|c: char| !c.is_alphanumeric());
        (w.ends_with("rais") || w.ends_with("rait") || w.ends_with("raient"))
            && STEMS.iter().any(|s| w.starts_with(s))
    })
}

fn contains_any(lower: &str, markers: &[&str]) -> bool {
    markers.iter().any(|m| lower.contains(m))
}

/// Layer 0: deterministic classification. Returns `Some(mode)` only when
/// CONFIDENT; `None` = ambiguous residue (Layer 1 / safe default).
pub fn classify_l0(text: &str) -> Option<RequestMode> {
    let lower = text.to_lowercase();
    let trimmed = lower.trim();

    // The user's explicit go always wins.
    if contains_any(trimmed, GO_TOKENS) {
        return Some(RequestMode::Execute);
    }

    let wish = contains_any(trimmed, WISH_MARKERS) || has_fr_conditional_volition(trimmed);
    let study = contains_any(trimmed, STUDY_MARKERS);
    let design = contains_any(trimmed, DESIGN_NOUNS);
    let anchored = has_concrete_anchor(trimmed);
    let interrogative = QUESTION_OPENERS.iter().any(|q| trimmed.starts_with(q))
        || (trimmed.ends_with('?') && !wish && !design);

    // Investigation request → STUDY (checked BEFORE the interrogative
    // branch: "est-ce qu'on peut paralléliser ?" / "can we speed up X?" are
    // feasibility studies phrased as questions, not info requests).
    if study {
        return Some(RequestMode::Study);
    }
    // Pure information request → ANSWER.
    if interrogative && !design {
        return Some(RequestMode::Answer);
    }
    // Hedged force…
    if wish {
        // …with a concrete spec → the wish IS the directive (specified wish).
        if anchored && !design {
            return Some(RequestMode::Execute);
        }
        // …around a design-heavy object with no spec → the incident: CLARIFY.
        if design && !anchored {
            return Some(RequestMode::Clarify);
        }
        // hedged but neither anchored nor design-heavy → ambiguous.
        return None;
    }
    // Direct imperative…
    if design && !anchored {
        // …on a design noun with no spec → underspecified imperative: CLARIFY.
        return Some(RequestMode::Clarify);
    }
    if anchored {
        // …on a concrete artifact → EXECUTE.
        return Some(RequestMode::Execute);
    }
    None
}

/// Bare affirmation / continuation ("ok vas-y", "oui", "go") — a SHORT
/// message with no new substantive content. The follow-up policy (§6)
/// promotes the prior STUDY/CLARIFY plan to EXECUTE instead of re-routing.
pub fn is_bare_affirmation(text: &str) -> bool {
    let lower = text.trim().to_lowercase();
    if lower.chars().count() > 40 {
        return false;
    }
    const AFFIRMATIONS: &[&str] = &[
        "ok",
        "oui",
        "yes",
        "go",
        "vas-y",
        "vas y",
        "fais-le",
        "fais le",
        "do it",
        "continue",
        "d'accord",
        "daccord",
        "parfait",
        "ça marche",
        "ca marche",
        "allons-y",
        "allons y",
        "let's go",
        "lets go",
        "lance",
        "ok vas-y",
        "yep",
        "yeah",
        "sure",
        "proceed",
    ];
    // Every word-ish chunk must be affirmation-ish (allow punctuation).
    let stripped: String = lower
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace() || *c == '-' || *c == '\'')
        .collect();
    let s = stripped.trim();
    if s.is_empty() {
        return false;
    }
    AFFIRMATIONS.contains(&s)
        || s.split_whitespace()
            .all(|w| AFFIRMATIONS.contains(&w) || matches!(w, "et" | "and" | "alors" | "then"))
}

/// System prompt for the Layer-1 classifier (the `fast` role).
const CLASSIFIER_SYSTEM: &str = "\
You classify a user request sent to a coding agent into the RESPONSE MODE it \
calls for. Reply with STRICT JSON only: {\"mode\":\"answer|execute|study|clarify\",\
\"confidence\":0.0-1.0}.\n\
- answer: an information question; no work requested.\n\
- execute: a directive with a clear, specified target — act now.\n\
- study: an investigation/feasibility/comparison request — propose, don't build.\n\
- clarify: work is requested but key design choices are unspecified (what kind, \
which interface, what data) — ask before building.\n\
Judge the ILLOCUTIONARY FORCE (a hedged wish like \"je souhaite…\" over an \
unspecified object is clarify, not execute) and the COMPLETENESS of the spec.";

/// Confidence below which Layer 2 falls back to the safer mode.
const CONFIDENCE_THRESHOLD: f32 = 0.6;

/// The request-intent router: Layer 0 markers + optional Layer-1 `fast`
/// classifier + Layer-2 safe default + an optional user-pinned mode that
/// overrides everything (the user always wins; the model never does).
pub struct RequestRouter {
    fast: Option<Arc<BoxedProvider>>,
    /// 0 = auto; 1 = answer; 2 = execute; 3 = study; 4 = clarify.
    pin: Option<Arc<std::sync::atomic::AtomicU8>>,
}

impl RequestMode {
    /// Pin encoding (0 = auto/unpinned).
    pub fn as_pin_u8(self) -> u8 {
        match self {
            RequestMode::Answer => 1,
            RequestMode::Execute => 2,
            RequestMode::Study => 3,
            RequestMode::Clarify => 4,
        }
    }

    /// Decode a pin value (`None` for 0/unknown = auto).
    pub fn from_pin_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(RequestMode::Answer),
            2 => Some(RequestMode::Execute),
            3 => Some(RequestMode::Study),
            4 => Some(RequestMode::Clarify),
            _ => None,
        }
    }
}

impl RequestRouter {
    /// `fast`: the cheap classifier provider (Layer 1). `None` = degraded
    /// path — Layer 0 + safe default hold the line alone (design O1).
    pub fn new(fast: Option<Arc<BoxedProvider>>) -> Self {
        Self { fast, pin: None }
    }

    /// Share a user-pinned-mode cell (e.g. the TUI `/mode study` command):
    /// a non-zero pin short-circuits routing entirely.
    pub fn with_pin(mut self, pin: Arc<std::sync::atomic::AtomicU8>) -> Self {
        self.pin = Some(pin);
        self
    }

    /// Route a fresh request. Always returns a mode (never blocks the run):
    /// L0 when confident, else L1 when available and confident, else the
    /// SAFER default (CLARIFY for design-ish texts, STUDY otherwise).
    pub async fn route(&self, text: &str) -> RoutedMode {
        if let Some(pin) = &self.pin
            && let Some(mode) =
                RequestMode::from_pin_u8(pin.load(std::sync::atomic::Ordering::Relaxed))
        {
            return RoutedMode {
                mode,
                source: RouteSource::Pinned,
                confidence: 1.0,
            };
        }
        if let Some(mode) = classify_l0(text) {
            return RoutedMode {
                mode,
                source: RouteSource::Markers,
                confidence: 1.0,
            };
        }
        if let Some(fast) = &self.fast
            && let Some((mode, confidence)) = self.classify_l1(fast, text).await
            && confidence >= CONFIDENCE_THRESHOLD
        {
            return RoutedMode {
                mode,
                source: RouteSource::Classifier,
                confidence,
            };
        }
        // Layer 2: safe default. A wrong guess toward "ask/propose" costs one
        // round-trip; a wrong guess toward "act" costs unwanted writes.
        let lower = text.to_lowercase();
        let mode = if contains_any(&lower, DESIGN_NOUNS) {
            RequestMode::Clarify
        } else {
            RequestMode::Study
        };
        RoutedMode {
            mode,
            source: RouteSource::SafeDefault,
            confidence: 0.0,
        }
    }

    async fn classify_l1(
        &self,
        fast: &Arc<BoxedProvider>,
        text: &str,
    ) -> Option<(RequestMode, f32)> {
        let request = CompletionRequest {
            system: CLASSIFIER_SYSTEM.to_string(),
            messages: vec![Message::user(format!("REQUEST:\n{text}"))],
            tools: Vec::new(),
            max_tokens: 64,
            tool_choice: None,
            reasoning_effort: None,
        };
        let response = fast.complete(request).await.ok()?;
        let body: String = response
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        // Tolerant parse: find the outermost JSON object.
        let start = body.find('{')?;
        let end = body.rfind('}')?;
        let v: serde_json::Value = serde_json::from_str(&body[start..=end]).ok()?;
        let mode = RequestMode::parse(v.get("mode")?.as_str()?)?;
        let confidence = v.get("confidence")?.as_f64().unwrap_or(0.0) as f32;
        Some((mode, confidence))
    }
}

/// A routing decision plus its provenance (for traces and tests).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedMode {
    /// The chosen response mode.
    pub mode: RequestMode,
    /// Which layer decided.
    pub source: RouteSource,
    /// Classifier confidence (1.0 for markers, 0.0 for the safe default).
    pub confidence: f32,
}

/// Which router layer produced the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteSource {
    /// Layer 0 — deterministic markers.
    Markers,
    /// Layer 1 — `fast`-role LLM classifier.
    Classifier,
    /// Layer 2 — low confidence / no classifier → safer mode.
    SafeDefault,
    /// User-pinned mode (`/mode study` etc.) — overrides routing.
    Pinned,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P1 — the labeled NATIVE fixture set (design doc §8, ship blocker).
    /// Every later router change must keep this table green.
    const L0_FIXTURES: &[(&str, RequestMode)] = &[
        // --- the incident class: hedged wish over a design noun → CLARIFY
        (
            "je souhaite créer un petit crm dans un répertoire temporaire",
            RequestMode::Clarify,
        ),
        ("j'aimerais une petite app de notes", RequestMode::Clarify),
        (
            "je voudrais un dashboard pour suivre mes ventes",
            RequestMode::Clarify,
        ),
        ("il faudrait un outil de migration", RequestMode::Clarify),
        ("i'd like a small crm for my contacts", RequestMode::Clarify),
        // --- underspecified IMPERATIVE → CLARIFY (the conflation bug: no wish marker)
        ("construis-moi un crm", RequestMode::Clarify),
        ("crée une app de todo", RequestMode::Clarify),
        ("build me a dashboard", RequestMode::Clarify),
        // --- specified WISH → EXECUTE (the other half of the conflation bug)
        (
            "j'aimerais que tu renommes la fonction `foo` en `bar`",
            RequestMode::Execute,
        ),
        (
            "je voudrais corriger la typo dans src/main.rs",
            RequestMode::Execute,
        ),
        (
            "i'd like you to fix the import in lib.rs",
            RequestMode::Execute,
        ),
        // --- directive + concrete anchor → EXECUTE
        ("renomme foo en bar dans src/app.rs", RequestMode::Execute),
        ("corrige la typo dans README.md", RequestMode::Execute),
        (
            "ajoute un test dans crates/core/tests.rs",
            RequestMode::Execute,
        ),
        // --- go-tokens force EXECUTE regardless of phrasing
        ("vas-y code directement le crm", RequestMode::Execute),
        ("je souhaite un crm — just build it", RequestMode::Execute),
        // --- investigation → STUDY
        ("regarde si on peut accélérer le build", RequestMode::Study),
        (
            "étudie les options de persistance pour le module",
            RequestMode::Study,
        ),
        (
            "est-ce qu'on peut paralléliser les tests ?",
            RequestMode::Study,
        ),
        ("can we speed up the ci pipeline?", RequestMode::Study),
        (
            "que penses-tu d'une migration vers axum ?",
            RequestMode::Study,
        ),
        // --- information questions → ANSWER
        ("que sais-tu faire ?", RequestMode::Answer),
        ("comment fonctionne le scheduler ?", RequestMode::Answer),
        ("explique-moi le rôle du pruner", RequestMode::Answer),
        ("what does the doom loop detector do?", RequestMode::Answer),
        ("pourquoi le build est-il si lent ?", RequestMode::Answer),
    ];

    #[test]
    fn l0_fixture_set_routes_native_requests() {
        let mut failures = Vec::new();
        for (text, expected) in L0_FIXTURES {
            match classify_l0(text) {
                Some(mode) if mode == *expected => {}
                got => failures.push(format!("{text:?} → {got:?}, expected {expected:?}")),
            }
        }
        assert!(
            failures.is_empty(),
            "L0 misroutes ({}/{}):\n{}",
            failures.len(),
            L0_FIXTURES.len(),
            failures.join("\n")
        );
    }

    /// Ambiguous residue: L0 must NOT guess — these return None (→ L1/L2).
    const L0_AMBIGUOUS: &[&str] = &[
        "je souhaite améliorer les choses",
        "fais quelque chose de bien",
        "on pourrait faire mieux",
        "help me with the project",
    ];

    #[test]
    fn l0_does_not_guess_on_ambiguity() {
        for text in L0_AMBIGUOUS {
            assert_eq!(
                classify_l0(text),
                None,
                "{text:?} must fall through to L1/L2"
            );
        }
    }

    #[test]
    fn bare_affirmations_detected_long_content_not() {
        for s in ["ok", "vas-y", "ok vas-y", "oui continue", "go", "d'accord"] {
            assert!(is_bare_affirmation(s), "{s:?} is a bare affirmation");
        }
        for s in [
            "vas-y mais utilise plutôt sqlite et ajoute une api rest",
            "ok mais d'abord explique-moi le schéma",
            "",
        ] {
            assert!(!is_bare_affirmation(s), "{s:?} is NOT bare");
        }
    }

    #[test]
    fn mode_labels_roundtrip() {
        for m in [
            RequestMode::Answer,
            RequestMode::Execute,
            RequestMode::Study,
            RequestMode::Clarify,
        ] {
            assert_eq!(RequestMode::parse(m.label()), Some(m));
        }
        assert_eq!(RequestMode::parse("nonsense"), None);
    }

    // --- Layer 1 + Layer 2 (mock provider) ---

    struct StubFast {
        reply: String,
    }
    impl LlmProvider for StubFast {
        async fn complete(
            &self,
            _r: CompletionRequest,
        ) -> Result<crate::llm::types::CompletionResponse, crate::error::Error> {
            Ok(crate::llm::types::CompletionResponse {
                content: vec![ContentBlock::Text {
                    text: self.reply.clone(),
                }],
                stop_reason: crate::llm::types::StopReason::EndTurn,
                reasoning: None,
                usage: crate::llm::types::TokenUsage::default(),
                model: None,
            })
        }
    }

    fn router_with(reply: &str) -> RequestRouter {
        RequestRouter::new(Some(Arc::new(BoxedProvider::from_arc(Arc::new(
            StubFast {
                reply: reply.into(),
            },
        )))))
    }

    #[tokio::test]
    async fn pinned_mode_overrides_routing() {
        let pin = Arc::new(std::sync::atomic::AtomicU8::new(
            RequestMode::Study.as_pin_u8(),
        ));
        let r = RequestRouter::new(None).with_pin(pin.clone());
        // Even a crystal-clear EXECUTE request obeys the pin.
        let routed = r.route("corrige la typo dans src/a.rs").await;
        assert_eq!(routed.mode, RequestMode::Study);
        assert_eq!(routed.source, RouteSource::Pinned);
        // Pin back to auto → normal routing resumes.
        pin.store(0, std::sync::atomic::Ordering::Relaxed);
        let routed = r.route("corrige la typo dans src/a.rs").await;
        assert_eq!(routed.mode, RequestMode::Execute);
    }

    #[tokio::test]
    async fn l1_classifier_routes_the_ambiguous_residue() {
        let r = router_with(r#"{"mode":"study","confidence":0.9}"#);
        let routed = r.route("je souhaite améliorer les choses").await;
        assert_eq!(routed.mode, RequestMode::Study);
        assert_eq!(routed.source, RouteSource::Classifier);
    }

    #[tokio::test]
    async fn low_confidence_falls_back_to_safe_default() {
        let r = router_with(r#"{"mode":"execute","confidence":0.3}"#);
        let routed = r.route("fais quelque chose de bien").await;
        assert_eq!(routed.source, RouteSource::SafeDefault);
        assert_eq!(routed.mode, RequestMode::Study, "non-design text → STUDY");
    }

    #[tokio::test]
    async fn degraded_no_fast_path_safe_defaults() {
        let r = RequestRouter::new(None);
        // wish + design noun + concrete anchor = the genuinely ambiguous
        // shape (L0 returns None: specified-wish says EXECUTE, design says
        // CLARIFY) → degraded path must safe-default on the design signal.
        let routed = r
            .route("je souhaite un crm comme décrit dans specs/crm.md")
            .await;
        assert_eq!(routed.source, RouteSource::SafeDefault);
        assert_eq!(
            routed.mode,
            RequestMode::Clarify,
            "design-ish ambiguous text → CLARIFY"
        );
    }

    #[tokio::test]
    async fn garbage_classifier_reply_safe_defaults() {
        let r = router_with("i refuse to answer in json");
        let routed = r.route("fais quelque chose de bien").await;
        assert_eq!(routed.source, RouteSource::SafeDefault);
    }

    #[tokio::test]
    async fn l0_confident_skips_the_classifier() {
        // Even with a classifier that would say EXECUTE, a wish+design-noun
        // request routes CLARIFY at L0 (markers win; the model can't promote).
        let r = router_with(r#"{"mode":"execute","confidence":0.99}"#);
        let routed = r.route("je souhaite créer un petit crm").await;
        assert_eq!(routed.mode, RequestMode::Clarify);
        assert_eq!(routed.source, RouteSource::Markers);
    }
}
