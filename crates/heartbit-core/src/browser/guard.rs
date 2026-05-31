//! Browser safety guardrails (spec capabilities 19–21).
//!
//! A credentialed browser agent is the canonical "lethal trifecta": private
//! data + untrusted page content + an exfiltration channel (`navigate_page`,
//! `evaluate_script`, form submission). WASP found frontier agents begin
//! following injected page instructions ~17% of the time. The cheapest, highest-
//! value structural defense is a **domain allowlist** that denies navigation /
//! network / submission to any host the operator did not pre-approve — it
//! catches the dangerous *first step* even when full exploitation is rare.
//!
//! [`DomainAllowlistGuard`] implements the existing [`Guardrail`] `pre_tool`
//! hook (first-`Deny`-wins), so it composes with the rest of the guardrail
//! stack and needs no browser to test.

use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;

use crate::agent::guardrail::{GuardAction, Guardrail};
use crate::error::Error;
use crate::llm::types::ToolCall;

/// Tools whose `url` argument navigates or fetches and must be allowlist-gated.
const NAVIGATION_TOOLS: &[&str] = &["navigate_page", "new_page"];

/// Deny browser navigation to any host not on an operator allowlist.
///
/// The allowlist holds bare hosts (e.g. `example.com`). A target host matches if
/// it equals an allowed host or is a subdomain of one (`docs.example.com`
/// matches `example.com`). Matching is case-insensitive. An empty allowlist
/// denies all navigation (deny-by-default).
pub struct DomainAllowlistGuard {
    allow: HashSet<String>,
}

impl DomainAllowlistGuard {
    /// Build from an iterator of allowed hosts. Hosts are lowercased; a leading
    /// `*.`, scheme, or path is stripped to the bare host.
    pub fn new(hosts: impl IntoIterator<Item = impl Into<String>>) -> Self {
        let allow = hosts
            .into_iter()
            .map(|h| normalize_host(&h.into()))
            .collect();
        Self { allow }
    }

    /// Is `host` allowed (equal to, or a subdomain of, an allowlisted host)?
    pub fn host_allowed(&self, host: &str) -> bool {
        let host = host.to_lowercase();
        self.allow
            .iter()
            .any(|allowed| host == *allowed || host.ends_with(&format!(".{allowed}")))
    }
}

/// Reduce a host/url-ish string to a bare lowercase host: strip scheme, any
/// path/query, a leading `*.`, and a trailing port.
fn normalize_host(s: &str) -> String {
    let s = s.trim().to_lowercase();
    // `rsplit` (single-ended) yields the part after the last "://"; a multi-char
    // `&str` `Split` is not `DoubleEndedIterator`, so `next_back` won't compile.
    let s = s.rsplit("://").next().unwrap_or(&s);
    let s = s.split('/').next().unwrap_or(s);
    let s = s.strip_prefix("*.").unwrap_or(s);
    let s = s.split(':').next().unwrap_or(s);
    s.to_string()
}

/// Extract the host from a URL string in a tool-call `url` argument, without a
/// URL-parsing dependency. Returns `None` for non-`http(s)` schemes (e.g.
/// `about:blank`, `javascript:`) and for `back`/`forward`/`reload` keywords.
pub(crate) fn url_host(url: &str) -> Option<String> {
    let u = url.trim();
    // Navigation keywords are not external hosts.
    if matches!(u, "back" | "forward" | "reload") {
        return None;
    }
    let after_scheme = match u.split_once("://") {
        Some(("http" | "https", rest)) => rest,
        Some(_) => return None, // non-web scheme with authority (ftp://, ws://, …)
        None => {
            // No "://": either a bare host[:port][/path], or an opaque-scheme URL
            // such as "about:blank" / "javascript:…" (a scheme whose colon is NOT
            // followed by a numeric port). Distinguish them by the post-colon text:
            // a numeric tail is a port (host:port); anything else is an opaque
            // scheme and has no web host to gate.
            let head = u.split(['/', '?', '#']).next().unwrap_or(u);
            match head.split_once(':') {
                Some((_, tail)) if tail.is_empty() || !tail.bytes().all(|b| b.is_ascii_digit()) => {
                    return None;
                }
                _ => u, // bare host, or host:port — fall through to shared stripping
            }
        }
    };
    let host = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    let host = host.split('@').next_back().unwrap_or(host); // strip userinfo
    let host = host.split(':').next().unwrap_or(host); // strip port
    if host.is_empty() {
        None
    } else {
        Some(host.to_lowercase())
    }
}

impl Guardrail for DomainAllowlistGuard {
    fn name(&self) -> &str {
        "browser-domain-allowlist"
    }

    fn pre_tool(
        &self,
        call: &ToolCall,
    ) -> Pin<Box<dyn Future<Output = Result<GuardAction, Error>> + Send + '_>> {
        let action = if NAVIGATION_TOOLS.contains(&call.name.as_str()) {
            match call.input.get("url").and_then(|v| v.as_str()) {
                // A navigation with a real http(s) host: gate it.
                Some(url) => match url_host(url) {
                    Some(host) if self.host_allowed(&host) => GuardAction::Allow,
                    Some(host) => GuardAction::deny(format!(
                        "navigation to '{host}' denied: host not on the browser allowlist"
                    )),
                    // No web host (about:blank, javascript:, back/forward): allow.
                    None => GuardAction::Allow,
                },
                // navigate_page back/forward/reload (no url arg): allow.
                None => GuardAction::Allow,
            }
        } else {
            GuardAction::Allow
        };
        Box::pin(async move { Ok(action) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nav(url: &str) -> ToolCall {
        ToolCall {
            id: "c1".into(),
            name: "navigate_page".into(),
            input: serde_json::json!({ "url": url }),
        }
    }

    async fn decide(g: &DomainAllowlistGuard, call: &ToolCall) -> GuardAction {
        g.pre_tool(call).await.expect("guard ok")
    }

    #[test]
    fn url_host_extraction() {
        assert_eq!(
            url_host("https://example.com/a/b?x=1"),
            Some("example.com".into())
        );
        assert_eq!(
            url_host("http://Docs.Example.com:8080/"),
            Some("docs.example.com".into())
        );
        assert_eq!(
            url_host("https://user:pw@evil.com/x"),
            Some("evil.com".into())
        );
        assert_eq!(url_host("example.com/path"), Some("example.com".into()));
        assert_eq!(url_host("about:blank"), None);
        assert_eq!(url_host("javascript:alert(1)"), None);
        assert_eq!(url_host("back"), None);
    }

    #[test]
    fn subdomain_matches_allowlisted_apex() {
        let g = DomainAllowlistGuard::new(["example.com"]);
        assert!(g.host_allowed("example.com"));
        assert!(g.host_allowed("docs.example.com"));
        assert!(g.host_allowed("a.b.example.com"));
        assert!(!g.host_allowed("notexample.com"));
        assert!(!g.host_allowed("example.com.evil.com"));
    }

    #[tokio::test]
    async fn allows_navigation_to_allowlisted_host() {
        let g = DomainAllowlistGuard::new(["example.com"]);
        assert_eq!(
            decide(&g, &nav("https://example.com/login")).await,
            GuardAction::Allow
        );
        assert_eq!(
            decide(&g, &nav("https://api.example.com/v1")).await,
            GuardAction::Allow
        );
    }

    #[tokio::test]
    async fn denies_navigation_off_allowlist() {
        let g = DomainAllowlistGuard::new(["example.com"]);
        let action = decide(&g, &nav("https://evil.com/steal")).await;
        match action {
            GuardAction::Deny { reason } => {
                assert!(reason.contains("evil.com"), "reason: {reason}");
                assert!(reason.contains("allowlist"), "reason: {reason}");
            }
            other => panic!("expected Deny, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_allowlist_denies_all_navigation() {
        let g = DomainAllowlistGuard::new(Vec::<String>::new());
        assert!(matches!(
            decide(&g, &nav("https://example.com")).await,
            GuardAction::Deny { .. }
        ));
    }

    #[tokio::test]
    async fn non_navigation_tools_pass_through() {
        let g = DomainAllowlistGuard::new(["example.com"]);
        // A click is not gated by the allowlist (only navigation/network is).
        let click = ToolCall {
            id: "c2".into(),
            name: "click".into(),
            input: serde_json::json!({ "uid": "1_2" }),
        };
        assert_eq!(decide(&g, &click).await, GuardAction::Allow);
    }

    #[tokio::test]
    async fn back_forward_and_nonweb_schemes_allowed() {
        let g = DomainAllowlistGuard::new(["example.com"]);
        assert_eq!(decide(&g, &nav("back")).await, GuardAction::Allow);
        assert_eq!(decide(&g, &nav("about:blank")).await, GuardAction::Allow);
        // navigate with no url arg (e.g. reload form) → allow.
        let no_url = ToolCall {
            id: "c3".into(),
            name: "navigate_page".into(),
            input: serde_json::json!({}),
        };
        assert_eq!(decide(&g, &no_url).await, GuardAction::Allow);
    }

    #[tokio::test]
    async fn new_page_is_also_gated() {
        let g = DomainAllowlistGuard::new(["example.com"]);
        let new_evil = ToolCall {
            id: "c4".into(),
            name: "new_page".into(),
            input: serde_json::json!({ "url": "https://evil.com" }),
        };
        assert!(matches!(
            decide(&g, &new_evil).await,
            GuardAction::Deny { .. }
        ));
    }
}
