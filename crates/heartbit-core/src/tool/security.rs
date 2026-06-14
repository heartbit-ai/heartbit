//! Lethal-trifecta exposure analysis for a tool set.
//!
//! Simon Willison's "lethal trifecta" (16 Jun 2025): an agent that
//! simultaneously can (1) **read private data**, (2) **ingest untrusted
//! content**, and (3) **communicate/exfiltrate externally** is one indirect
//! prompt-injection away from leaking that data. The security literature is
//! consistent that *detection* of injections is a failing strategy on its own
//! ("95% is a failing grade"); the cheap, high-value structural control is to
//! recognise the trifecta in an agent's *configuration* and warn / require
//! containment before it ever runs.
//!
//! This module classifies each tool's exposure (a heuristic by name, overridable
//! per tool via [`Tool::security_exposure`](crate::tool::Tool::security_exposure))
//! and reports when all three legs co-occur in one agent.
//!
//! The classification is deliberately CONSERVATIVE (bias toward flagging): a
//! false positive is a harmless warning; a false negative is an unflagged
//! exfiltration path. It is a *defense-in-depth signal*, not a guarantee — the
//! guarantee comes from structural containment (dual-LLM / capabilities).

/// Which legs of the lethal trifecta a single tool (or a whole tool set, unioned)
/// is capable of. Each flag is independent; a tool may set several.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ToolExposure {
    /// Reads data the user would consider private (local files, memory, secrets,
    /// internal resources).
    pub reads_private_data: bool,
    /// Ingests content from an untrusted source the attacker may control (web
    /// pages, fetched URLs, external documents, a rendered browser page).
    pub ingests_untrusted_content: bool,
    /// Can send data to an external destination (HTTP with a body/query,
    /// email/post/publish, an outbound shell command).
    pub can_exfiltrate: bool,
}

impl ToolExposure {
    /// No exposure on any leg.
    pub const NONE: Self = Self {
        reads_private_data: false,
        ingests_untrusted_content: false,
        can_exfiltrate: false,
    };

    /// Reads-private-data leg only.
    pub const PRIVATE: Self = Self {
        reads_private_data: true,
        ..Self::NONE
    };

    /// Ingests-untrusted-content leg only.
    pub const UNTRUSTED: Self = Self {
        ingests_untrusted_content: true,
        ..Self::NONE
    };

    /// Can-exfiltrate leg only.
    pub const EXFIL: Self = Self {
        can_exfiltrate: true,
        ..Self::NONE
    };

    /// Bitwise-OR union of two exposures (used to fold a tool set).
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            reads_private_data: self.reads_private_data || other.reads_private_data,
            ingests_untrusted_content: self.ingests_untrusted_content
                || other.ingests_untrusted_content,
            can_exfiltrate: self.can_exfiltrate || other.can_exfiltrate,
        }
    }

    /// True when this exposure covers all three legs of the lethal trifecta.
    pub fn is_lethal_trifecta(self) -> bool {
        self.reads_private_data && self.ingests_untrusted_content && self.can_exfiltrate
    }
}

/// Heuristic classification of a tool by its (lowercased) name. Covers the
/// builtins and the most common MCP/browser tool names. Unknown tools get
/// [`ToolExposure::NONE`] — callers that know better should override
/// [`Tool::security_exposure`](crate::tool::Tool::security_exposure).
///
/// `bash` is the catch-all: an unrestricted shell can read anything, fetch
/// anything, and exfiltrate via `curl`, so it trips ALL three legs by itself.
pub fn classify_tool_name(name: &str) -> ToolExposure {
    let n = name.to_lowercase();

    // An unrestricted shell is the whole trifecta on its own.
    if n == "bash" || n == "shell" || n.ends_with("_bash") {
        return ToolExposure {
            reads_private_data: true,
            ingests_untrusted_content: true,
            can_exfiltrate: true,
        };
    }

    let mut e = ToolExposure::NONE;

    // (1) Reads private data: local files, memory, knowledge, secrets.
    if matches_any(
        &n,
        &[
            "read",
            "grep",
            "glob",
            "ls",
            "list_files",
            "cat",
            "open_file",
            "fetch_full_output",
            "recall",
            "memory_recall",
            "memory_search",
            "knowledge_search",
            "search_memory",
            "get_file",
            "read_file",
        ],
    ) {
        e.reads_private_data = true;
    }

    // (2) Ingests untrusted content: web/browser/external sources.
    if matches_any(
        &n,
        &[
            "webfetch",
            "web_fetch",
            "fetch_url",
            "websearch",
            "web_search",
            "navigate",
            "take_snapshot",
            "new_page",
            "browse",
            "scrape",
            "crawl",
            "fetch_page",
            "get_url",
        ],
    ) {
        e.ingests_untrusted_content = true;
    }

    // (3) Can exfiltrate: outbound sends, posts, navigation with attacker-shaped URLs.
    if matches_any(
        &n,
        &[
            "send",
            "post",
            "publish",
            "email",
            "tweet",
            "twitter",
            "transfer",
            "upload",
            "http_post",
            "webhook",
            "navigate", // a URL with query params is an exfiltration channel
            "new_page",
            "evaluate_script", // arbitrary fetch() in-page
        ],
    ) {
        e.can_exfiltrate = true;
    }

    // A web fetch both ingests untrusted content AND can carry data out in the
    // request (query/body) — mark both legs.
    if matches_any(&n, &["webfetch", "web_fetch", "fetch_url", "http"]) {
        e.ingests_untrusted_content = true;
        e.can_exfiltrate = true;
    }

    e
}

fn matches_any(name: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| name.contains(needle))
}

/// The result of analysing a tool set for the lethal trifecta.
#[derive(Debug, Clone, Default)]
pub struct TrifectaReport {
    /// Unioned exposure across the whole tool set.
    pub exposure: ToolExposure,
    /// Tools contributing the "reads private data" leg.
    pub reads_private: Vec<String>,
    /// Tools contributing the "ingests untrusted content" leg.
    pub ingests_untrusted: Vec<String>,
    /// Tools contributing the "can exfiltrate" leg.
    pub can_exfiltrate: Vec<String>,
}

impl TrifectaReport {
    /// True when the tool set covers all three legs — the dangerous combination.
    pub fn is_lethal_trifecta(&self) -> bool {
        self.exposure.is_lethal_trifecta()
    }

    /// A human-readable warning naming each leg's contributing tools, or `None`
    /// when the trifecta is not present.
    pub fn warning(&self) -> Option<String> {
        if !self.is_lethal_trifecta() {
            return None;
        }
        Some(format!(
            "LETHAL TRIFECTA: this agent can read private data ([{}]), ingest \
             untrusted content ([{}]), AND communicate externally ([{}]) — an \
             indirect prompt injection in the untrusted content can exfiltrate \
             the private data. Break the trifecta (remove one leg), gate the \
             exfiltration tools behind human approval, or run the untrusted-content \
             path in a quarantined (no-tool) context.",
            self.reads_private.join(", "),
            self.ingests_untrusted.join(", "),
            self.can_exfiltrate.join(", "),
        ))
    }
}

/// Analyse `(name, exposure)` pairs for the lethal trifecta. Use
/// [`analyze_tools`](crate::tool::analyze_tools) for a live `&[Arc<dyn Tool>]`.
pub fn analyze_exposures<'a>(
    tools: impl IntoIterator<Item = (&'a str, ToolExposure)>,
) -> TrifectaReport {
    let mut report = TrifectaReport::default();
    for (name, exposure) in tools {
        report.exposure = report.exposure.union(exposure);
        if exposure.reads_private_data {
            report.reads_private.push(name.to_string());
        }
        if exposure.ingests_untrusted_content {
            report.ingests_untrusted.push(name.to_string());
        }
        if exposure.can_exfiltrate {
            report.can_exfiltrate.push(name.to_string());
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bash_alone_is_the_whole_trifecta() {
        let e = classify_tool_name("bash");
        assert!(e.is_lethal_trifecta());
    }

    #[test]
    fn classifies_each_leg() {
        assert!(classify_tool_name("read").reads_private_data);
        assert!(classify_tool_name("memory_recall").reads_private_data);
        assert!(classify_tool_name("navigate_page").ingests_untrusted_content);
        assert!(classify_tool_name("take_snapshot").ingests_untrusted_content);
        assert!(classify_tool_name("send_email").can_exfiltrate);
        assert!(classify_tool_name("twitter_post").can_exfiltrate);
        // webfetch is both ingest + exfil.
        let wf = classify_tool_name("webfetch");
        assert!(wf.ingests_untrusted_content && wf.can_exfiltrate);
    }

    #[test]
    fn unknown_tool_is_no_exposure() {
        assert_eq!(classify_tool_name("add_numbers"), ToolExposure::NONE);
    }

    #[test]
    fn union_folds_legs() {
        let e = ToolExposure::PRIVATE
            .union(ToolExposure::UNTRUSTED)
            .union(ToolExposure::EXFIL);
        assert!(e.is_lethal_trifecta());
    }

    #[test]
    fn analyze_flags_trifecta_across_separate_tools() {
        // read (private) + take_snapshot (untrusted) + send_email (exfil) → trifecta.
        let report = analyze_exposures([
            ("read", classify_tool_name("read")),
            ("take_snapshot", classify_tool_name("take_snapshot")),
            ("send_email", classify_tool_name("send_email")),
        ]);
        assert!(report.is_lethal_trifecta());
        let warning = report.warning().expect("trifecta present → warning");
        assert!(warning.contains("read"));
        assert!(warning.contains("take_snapshot"));
        assert!(warning.contains("send_email"));
    }

    #[test]
    fn analyze_does_not_flag_a_safe_set() {
        // Only private + untrusted, no exfiltration → not the trifecta.
        let report = analyze_exposures([
            ("read", classify_tool_name("read")),
            ("grep", classify_tool_name("grep")),
        ]);
        assert!(!report.is_lethal_trifecta());
        assert!(report.warning().is_none());
    }

    #[test]
    fn analyze_does_not_flag_two_legs() {
        // read (private) + navigate (untrusted+exfil) is the trifecta BECAUSE
        // navigate carries the exfil leg; verify a genuine two-leg set is safe.
        let report = analyze_exposures([
            ("read", classify_tool_name("read")), // private only
            ("grep", classify_tool_name("grep")), // private only
            ("ls", classify_tool_name("ls")),     // private only
        ]);
        assert!(!report.is_lethal_trifecta());
    }
}
