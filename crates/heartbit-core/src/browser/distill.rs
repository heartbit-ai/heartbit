//! Snapshot distiller (capability 2 of the browser-bot spec).
//!
//! `chrome-devtools-mcp`'s `take_snapshot` returns an indented accessibility
//! tree, one node per line:
//!
//! ```text
//! uid=1_0 RootWebArea "Hacker News" url="https://news.ycombinator.com/"
//!   uid=1_1 link "Hacker News" url="https://news.ycombinator.com/news"
//!   uid=1_9 StaticText "Hacker News"
//! ```
//!
//! The raw tree is close to optimal but carries redundant nodes — most
//! commonly a `StaticText` whose text merely echoes the accessible name of the
//! interactive element (`link`/`button`) it sits under. AgentOccam (ICLR'25)
//! showed that pruning the observation alone is the single highest-leverage
//! reliability win for a web agent. This distiller removes that redundancy
//! while **preserving every `uid`** — the agent's only handle for acting.
//!
//! It is a pure `&str -> String` transform: no browser, no MCP, no LLM, so it
//! is fully unit-testable in CI.

/// Tuning for [`distill_snapshot`].
#[derive(Debug, Clone)]
pub struct DistillConfig {
    /// Drop a `StaticText` line whose quoted text duplicates the accessible
    /// name of the interactive element immediately above it at a shallower
    /// indent (the most common redundancy in the a11y tree).
    pub drop_redundant_static_text: bool,
    /// Drop nodes with no `uid` and no quoted name (pure structural noise).
    pub drop_empty_nodes: bool,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            drop_redundant_static_text: true,
            drop_empty_nodes: true,
        }
    }
}

/// A parsed view of one snapshot line.
struct Node<'a> {
    raw: &'a str,
    indent: usize,
    /// `uid=N_M` token, if present.
    uid: Option<&'a str>,
    /// Accessibility role (`link`, `StaticText`, `heading`, `RootWebArea`, …).
    role: Option<&'a str>,
    /// The first quoted string (the accessible name), without quotes.
    name: Option<&'a str>,
}

fn parse_line(line: &str) -> Node<'_> {
    let indent = line.len() - line.trim_start().len();
    let body = line.trim_start();

    let (uid, after_uid) = match body.strip_prefix("uid=") {
        Some(rest) => {
            let end = rest.find(char::is_whitespace).unwrap_or(rest.len());
            (Some(&rest[..end]), rest[end..].trim_start())
        }
        None => (None, body),
    };

    // Role is the first token after the uid (and before the first quote).
    let role = after_uid
        .split([' ', '"'])
        .find(|t| !t.is_empty())
        .filter(|_| !after_uid.starts_with('"'));

    // Accessible name is the first double-quoted span.
    let name = after_uid.find('"').and_then(|start| {
        let rest = &after_uid[start + 1..];
        rest.find('"').map(|end| &rest[..end])
    });

    Node {
        raw: line,
        indent,
        uid,
        role,
        name,
    }
}

/// Roles an agent can actually act on (`click`/`fill`/`hover`/`select`/upload).
/// Their `uid`s are load-bearing and must never be dropped.
fn is_interactive(role: Option<&str>) -> bool {
    matches!(
        role,
        Some(
            "link"
                | "button"
                | "textbox"
                | "checkbox"
                | "radio"
                | "combobox"
                | "menuitem"
                | "tab"
                | "switch"
                | "slider"
                | "searchbox"
                | "option"
        )
    )
}

/// Distill a raw `take_snapshot` accessibility tree into a compact form.
/// Returns the pruned tree as text.
///
/// Guarantees:
/// - **Every *interactive* `uid` survives** (links/buttons/inputs/etc. — the
///   only handles `click`/`fill`/`hover` can target). Non-interactive
///   `StaticText` `uid`s, which an agent never acts on, may be pruned.
/// - Output is never larger than the input.
/// - Idempotent: distilling a distilled snapshot is a no-op.
pub fn distill_snapshot(raw: &str, cfg: &DistillConfig) -> String {
    let nodes: Vec<Node> = raw.lines().map(parse_line).collect();
    let mut keep: Vec<bool> = vec![true; nodes.len()];

    for i in 0..nodes.len() {
        let n = &nodes[i];

        // Drop a redundant StaticText whose text merely echoes the accessible
        // name of the nearest shallower interactive node (the most common a11y
        // redundancy). Safe even when the StaticText carries a uid, because
        // StaticText is NON-interactive — no tool ever targets that uid.
        if cfg.drop_redundant_static_text
            && n.role == Some("StaticText")
            && !is_interactive(n.role)
            && let Some(text) = n.name
            && let Some(parent) = (0..i)
                .rev()
                .map(|j| &nodes[j])
                .find(|p| p.indent < n.indent)
            && parent.name == Some(text)
            && is_interactive(parent.role)
        {
            keep[i] = false;
            continue;
        }

        // Drop pure structural noise: a blank/whitespace-only line.
        if cfg.drop_empty_nodes && n.uid.is_none() && n.raw.trim().is_empty() {
            keep[i] = false;
        }
    }

    let mut out = String::with_capacity(raw.len());
    for (i, n) in nodes.iter().enumerate() {
        if keep[i] {
            out.push_str(n.raw);
            out.push('\n');
        }
    }
    // Match input's trailing-newline convention.
    if !raw.ends_with('\n') {
        out.pop();
    }
    out
}

/// Extract every *interactive* `uid` (the handles an agent can act on).
pub fn interactive_uids(snapshot: &str) -> Vec<String> {
    snapshot
        .lines()
        .filter_map(|l| {
            let n = parse_line(l);
            match (n.uid, is_interactive(n.role)) {
                (Some(u), true) => Some(u.to_string()),
                _ => None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A realistic fixture in the exact format chrome-devtools-mcp emits
    /// (verified live against Hacker News this session): an indented a11y tree,
    /// `uid=N_M <Role> "name" [url=...]` per line. The HN nav is the canonical
    /// redundancy — every `link "x"` is followed by a child `StaticText "x"`
    /// echoing its name (live capture: 475 StaticText vs 226 link).
    const FIXTURE: &str = r#"uid=1_0 RootWebArea "Hacker News" url="https://news.ycombinator.com/"
  uid=1_2 link "Hacker News" url="https://news.ycombinator.com/news"
    uid=1_3 StaticText "Hacker News"
  uid=1_4 link "new" url="https://news.ycombinator.com/newest"
    uid=1_5 StaticText "new"
  uid=1_6 StaticText " | "
  uid=1_30 textbox "Search"
  uid=1_31 StaticText "Plain caption with no interactive parent""#;

    #[test]
    fn preserves_every_interactive_uid() {
        let before = interactive_uids(FIXTURE);
        let out = distill_snapshot(FIXTURE, &DistillConfig::default());
        let after = interactive_uids(&out);
        assert_eq!(
            before, after,
            "distillation must preserve every interactive uid; before={before:?} after={after:?}"
        );
        // Concretely: the links and the textbox survive.
        assert_eq!(before, vec!["1_2", "1_4", "1_30"]);
    }

    #[test]
    fn strictly_shrinks_by_dropping_echoed_static_text() {
        let out = distill_snapshot(FIXTURE, &DistillConfig::default());
        assert!(
            out.len() < FIXTURE.len(),
            "distilled output ({}) must be strictly smaller than input ({})",
            out.len(),
            FIXTURE.len()
        );
        // The redundant child StaticText echoing each link's name is gone...
        assert!(!out.contains(r#"uid=1_3 StaticText "Hacker News""#));
        assert!(!out.contains(r#"uid=1_5 StaticText "new""#));
        // ...but the interactive parents (with their names) remain.
        assert!(out.contains(r#"uid=1_2 link "Hacker News""#));
        assert!(out.contains(r#"uid=1_4 link "new""#));
    }

    #[test]
    fn keeps_non_echoing_static_text_and_inputs() {
        let out = distill_snapshot(FIXTURE, &DistillConfig::default());
        // A separator and a standalone caption are NOT echoes of a parent → kept.
        assert!(out.contains(r#"uid=1_6 StaticText " | ""#));
        assert!(out.contains("Plain caption with no interactive parent"));
        // The textbox (interactive) and its accessible name are kept.
        assert!(out.contains(r#"uid=1_30 textbox "Search""#));
        // The link urls are retained (the agent may need them).
        assert!(out.contains("https://news.ycombinator.com/news"));
    }

    #[test]
    fn is_idempotent() {
        let cfg = DistillConfig::default();
        let once = distill_snapshot(FIXTURE, &cfg);
        let twice = distill_snapshot(&once, &cfg);
        assert_eq!(once, twice, "distillation must be idempotent");
    }

    #[test]
    fn parse_line_extracts_uid_role_name() {
        let n = parse_line(r#"  uid=1_2 link "Hacker News" url="https://x""#);
        assert_eq!(n.uid, Some("1_2"));
        assert_eq!(n.role, Some("link"));
        assert_eq!(n.name, Some("Hacker News"));
        assert_eq!(n.indent, 2);
    }

    #[test]
    fn distill_disabled_is_identity() {
        let cfg = DistillConfig {
            drop_redundant_static_text: false,
            drop_empty_nodes: false,
        };
        assert_eq!(distill_snapshot(FIXTURE, &cfg), FIXTURE);
    }

    #[test]
    fn interactive_uids_excludes_static_text_and_root() {
        // Only link/textbox/etc — never StaticText, RootWebArea, or separators.
        assert_eq!(interactive_uids(FIXTURE), vec!["1_2", "1_4", "1_30"]);
    }
}
