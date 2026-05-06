//! HTTP client factories and URL validation primitives.
//!
//! Two factories return preconfigured `reqwest::ClientBuilder`s:
//! - [`safe_client_builder`] — for clients that send to user-controllable URLs.
//!   Caller is expected to validate URLs via [`SafeUrl::parse`] first.
//! - [`vendor_client_builder`] — for clients that send to operator-trusted
//!   vendor APIs (Twitter, OpenAI, etc.). No URL validation is implied.
//!
//! Both builders set `redirect(Policy::none())`, `.no_proxy()`, and a
//! `connect_timeout(5s)`. They also install a custom DNS resolver
//! ([`SafeDnsResolver`]) that re-applies the IP blocklist at connect time
//! — closing the DNS-rebinding bypass that the parse-time check alone
//! left open (F-NET-2).

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;

use reqwest::dns::{Addrs, Resolve, Resolving};
use reqwest::redirect::Policy;
use reqwest::{ClientBuilder, Url};

use crate::error::Error;

/// Whether to permit requests to non-routable / private IPs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpPolicy {
    /// Reject loopback, link-local, RFC1918, CGNAT, ULA, multicast,
    /// unspecified, broadcast. Library default.
    Strict,
    /// Allow any IP. Use only for single-tenant deployments where agents
    /// legitimately need internal-network access.
    AllowPrivate,
}

impl Default for IpPolicy {
    fn default() -> Self {
        Self::from_env()
    }
}

impl IpPolicy {
    /// Read the `HEARTBIT_ALLOW_PRIVATE_IPS` environment variable.
    ///
    /// Anything other than `"1"` or `"true"` (case-insensitive, trimmed)
    /// yields `Strict`. Unset → `Strict`.
    pub fn from_env() -> Self {
        Self::from_env_value(std::env::var("HEARTBIT_ALLOW_PRIVATE_IPS").ok().as_deref())
    }

    /// Pure parser — testable without mutating real env vars.
    pub(crate) fn from_env_value(value: Option<&str>) -> Self {
        match value {
            Some(v) => match v.trim().to_ascii_lowercase().as_str() {
                "1" | "true" => Self::AllowPrivate,
                _ => Self::Strict,
            },
            None => Self::Strict,
        }
    }
}

/// A URL that has passed scheme + IP-blocklist validation.
///
/// Construction via [`SafeUrl::parse`] is the only way to satisfy this type;
/// call sites that hold a `SafeUrl` proved they validated it.
#[derive(Debug, Clone)]
pub struct SafeUrl(Url);

impl SafeUrl {
    /// Parse `s`, require an `http://` or `https://` scheme, and reject if the
    /// host is a literal blocked IP, or if any IP that the host name resolves
    /// to is in the blocked set under `policy`.
    ///
    /// DNS resolution uses [`tokio::net::lookup_host`] with the URL's port (or
    /// the scheme's default port). All resolved addresses are checked; if any
    /// resolved address is private, the URL is rejected.
    ///
    /// Under `IpPolicy::AllowPrivate`, no IP check is performed (scheme check
    /// still applies).
    pub async fn parse(s: &str, policy: IpPolicy) -> Result<Self, Error> {
        let url = Url::parse(s).map_err(|e| Error::Agent(format!("invalid URL: {e}")))?;
        let scheme = url.scheme();
        if scheme != "http" && scheme != "https" {
            return Err(Error::Agent(format!(
                "URL scheme {scheme:?} not allowed; only http and https"
            )));
        }
        if matches!(policy, IpPolicy::AllowPrivate) {
            return Ok(Self(url));
        }
        let host = url
            .host_str()
            .ok_or_else(|| Error::Agent("URL has no host".into()))?;
        let port = url.port_or_known_default().unwrap_or(80);

        // `Url::host_str` returns IPv6 hosts surrounded by brackets
        // (`[::1]`, not `::1`). `IpAddr::from_str` rejects the bracketed
        // form, so we strip a single matched pair before the literal-IP
        // check. Without this, every IPv6 URL falls through to the DNS
        // path and the v6 blocklist is effectively dead code.
        let bare_host = host
            .strip_prefix('[')
            .and_then(|h| h.strip_suffix(']'))
            .unwrap_or(host);

        // Literal IP fast-path.
        if let Ok(ip) = IpAddr::from_str(bare_host) {
            if is_blocked(&ip) {
                return Err(reject(host));
            }
            return Ok(Self(url));
        }

        // DNS path: resolve and check every returned address.
        // Use the bracket-stripped host name for the resolver — `tokio::net::lookup_host`
        // expects a bare hostname, not the URL host_str format.
        let addrs = tokio::net::lookup_host((bare_host, port))
            .await
            .map_err(|e| Error::Agent(format!("DNS lookup failed for {host}: {e}")))?;
        let mut any = false;
        for sa in addrs {
            any = true;
            if is_blocked(&sa.ip()) {
                return Err(reject(host));
            }
        }
        if !any {
            return Err(Error::Agent(format!(
                "DNS lookup for {host} returned no addresses"
            )));
        }
        Ok(Self(url))
    }

    /// Return the URL as a string slice.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }

    /// Consume this wrapper and return the inner [`Url`].
    pub fn into_url(self) -> Url {
        self.0
    }
}

fn reject(host: &str) -> Error {
    Error::Agent(format!(
        "URL host {host} resolves to a private/loopback address; \
         refused (set HEARTBIT_ALLOW_PRIVATE_IPS=1 to override)"
    ))
}

fn is_blocked(ip: &IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => is_blocked_v4(v4),
        IpAddr::V6(v6) => is_blocked_v6(v6),
    }
}

/// Synchronous best-effort URL validation: scheme + literal-IP only.
///
/// Use when the constructor is not async (`TokenExchangeAuthProvider::new`,
/// `OpenAiCompatProvider::new`, etc). Returns `Ok` for hostnames (no DNS
/// lookup performed) under `IpPolicy::Strict` — call [`SafeUrl::parse`] on
/// the actual request path for the full check including DNS resolution.
///
/// Catches the obvious misconfigurations (e.g., `http://127.0.0.1` or
/// `http://169.254.169.254`) at construction time. Defense in depth, not
/// a substitute for `SafeUrl::parse`.
pub fn validate_url_sync(s: &str, policy: IpPolicy) -> Result<(), Error> {
    let url = Url::parse(s).map_err(|e| Error::Agent(format!("invalid URL: {e}")))?;
    let scheme = url.scheme();
    if scheme != "http" && scheme != "https" {
        return Err(Error::Agent(format!(
            "URL scheme {scheme:?} not allowed; only http and https"
        )));
    }
    if matches!(policy, IpPolicy::AllowPrivate) {
        return Ok(());
    }
    let host = url
        .host_str()
        .ok_or_else(|| Error::Agent("URL has no host".into()))?;
    let bare_host = host
        .strip_prefix('[')
        .and_then(|h| h.strip_suffix(']'))
        .unwrap_or(host);
    if let Ok(ip) = IpAddr::from_str(bare_host)
        && is_blocked(&ip)
    {
        return Err(reject(host));
    }
    Ok(())
}

fn is_blocked_v4(ip: &Ipv4Addr) -> bool {
    ip.is_loopback()
        || ip.is_link_local()
        || ip.is_private()
        || ip.is_multicast()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || is_cgnat_v4(ip)
}

fn is_blocked_v6(ip: &Ipv6Addr) -> bool {
    // IPv4-mapped IPv6 addresses (`::ffff:0:0/96`) embed a v4 address in the
    // low 32 bits. A literal URL such as `http://[::ffff:127.0.0.1]/` would
    // otherwise bypass the v4 blocklist via the v6 path. Reduce to v4.
    //
    // (The deprecated IPv4-compatible form `::0:0/96`, e.g. `::127.0.0.1`,
    // is RFC 4291 §2.5.5.1-deprecated and not handled here.)
    if let Some(v4) = ip.to_ipv4_mapped() {
        return is_blocked_v4(&v4);
    }
    ip.is_loopback()
        || ip.is_multicast()
        || ip.is_unspecified()
        || is_link_local_v6(ip)
        || is_ula_v6(ip)
}

/// CGNAT range (RFC 6598): 100.64.0.0/10. `Ipv4Addr::is_private` does not
/// cover this; we check explicitly.
fn is_cgnat_v4(ip: &Ipv4Addr) -> bool {
    let [a, b, _, _] = ip.octets();
    a == 100 && (64..=127).contains(&b)
}

/// IPv6 link-local: fe80::/10. `Ipv6Addr::is_unicast_link_local` is unstable
/// as of MSRV; we check the prefix directly.
fn is_link_local_v6(ip: &Ipv6Addr) -> bool {
    let s = ip.segments()[0];
    (s & 0xffc0) == 0xfe80
}

/// IPv6 unique local (ULA): fc00::/7. `Ipv6Addr::is_unique_local` is unstable
/// as of MSRV; we check the prefix directly.
fn is_ula_v6(ip: &Ipv6Addr) -> bool {
    let s = ip.segments()[0];
    (s & 0xfe00) == 0xfc00
}

/// Default cap for vendor response bodies (5 MiB).
///
/// SECURITY (F-NET-1): vendor APIs (Twitter, OpenAI TTS, image gen, search)
/// are operator-trusted but trust ≠ unbounded. A compromised vendor (DNS
/// hijack, BGP, sub-CA) could OOM the agent by streaming gigabytes. This cap
/// is per-response.
pub const DEFAULT_VENDOR_BODY_CAP: usize = 5 * 1024 * 1024;

/// Read up to `max_bytes` from `response` and return the bytes plus a flag
/// indicating whether truncation happened.
///
/// SECURITY (F-NET-1): use this helper instead of `response.text()` /
/// `response.bytes()` / `response.json()` directly when the response is from
/// a vendor that could in principle be hostile. Streams chunk-by-chunk and
/// aborts at the cap; never holds more than `max_bytes` in memory.
pub async fn read_body_capped(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<(Vec<u8>, bool), Error> {
    use futures::TryStreamExt;
    let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024);
    let mut truncated = false;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.try_next().await.map_err(Error::Http)? {
        let remaining = max_bytes.saturating_sub(buf.len());
        if remaining == 0 {
            truncated = true;
            break;
        }
        let take = chunk.len().min(remaining);
        buf.extend_from_slice(&chunk[..take]);
        if take < chunk.len() {
            truncated = true;
            break;
        }
    }
    Ok((buf, truncated))
}

/// Read response body as text, capped at `max_bytes`. Lossy UTF-8 on
/// non-UTF-8 input.
pub async fn read_text_capped(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<String, Error> {
    let (bytes, truncated) = read_body_capped(response, max_bytes).await?;
    let mut text = String::from_utf8_lossy(&bytes).into_owned();
    if truncated {
        text.push_str("…[truncated]");
    }
    Ok(text)
}

/// Custom DNS resolver that re-validates resolved IPs against the
/// [`IpPolicy`] at connect time.
///
/// SECURITY (F-NET-2): the parse-time IP blocklist on [`SafeUrl::parse`]
/// catches `http://127.0.0.1` and `http://169.254.169.254` literally, but
/// an attacker who controls `evil.com` with TTL=0 can return `8.8.8.8`
/// (passes parse) and then `127.0.0.1` (used at TCP connect). This
/// resolver re-applies the blocklist to every resolved `SocketAddr` —
/// rebind attempts fail with a connect-time error before any byte
/// reaches the loopback / metadata service.
pub struct SafeDnsResolver {
    policy: IpPolicy,
}

impl SafeDnsResolver {
    /// Resolver under the given IP policy.
    pub fn new(policy: IpPolicy) -> Self {
        Self { policy }
    }
}

impl Resolve for SafeDnsResolver {
    fn resolve(&self, name: reqwest::dns::Name) -> Resolving {
        let host = name.as_str().to_string();
        let policy = self.policy;
        Box::pin(async move {
            let resolved: Vec<SocketAddr> =
                tokio::net::lookup_host((host.as_str(), 0)).await?.collect();
            if resolved.is_empty() {
                return Err::<Addrs, _>(
                    format!("DNS lookup for {host} returned no addresses").into(),
                );
            }
            let filtered: Vec<SocketAddr> = match policy {
                IpPolicy::AllowPrivate => resolved,
                IpPolicy::Strict => resolved
                    .into_iter()
                    .filter(|sa| !is_blocked(&sa.ip()))
                    .collect(),
            };
            if filtered.is_empty() {
                return Err::<Addrs, _>(
                    format!(
                        "host {host} resolves to private/loopback addresses; \
                         refused at connect time (set HEARTBIT_ALLOW_PRIVATE_IPS=1 to override)"
                    )
                    .into(),
                );
            }
            // SAFETY: the iterator type wants `Send`; Vec<SocketAddr>::IntoIter is Send.
            let iter: Addrs = Box::new(filtered.into_iter());
            Ok(iter)
        }) as Pin<Box<_>>
    }
}

/// `reqwest::ClientBuilder` with `redirect(Policy::none())`, `.no_proxy()`,
/// `connect_timeout(5s)`, and a [`SafeDnsResolver`] baked in.
///
/// Use for clients that send to user-controllable URLs (`webfetch`, `a2a`,
/// `rss`). The caller is responsible for validating the URL via
/// [`SafeUrl::parse`] before issuing the request.
///
/// SECURITY (F-NET-3): `.no_proxy()` refuses env-driven `HTTP_PROXY` /
/// `HTTPS_PROXY` / `ALL_PROXY` by default. A misconfigured or attacker-set
/// proxy would otherwise route every outbound call (LLM, search, fetch)
/// through an attacker MITM.
///
/// SECURITY (F-NET-4): `connect_timeout(5s)` aborts a stalled TCP handshake
/// before the longer total timeout fires — slow-loris dialing only ties up
/// an agent slot for ~5s instead of 30–120s.
///
/// SECURITY (F-NET-2): `SafeDnsResolver` filters resolved IPs at connect
/// time, defeating DNS-rebinding bypasses of the parse-time blocklist.
pub fn safe_client_builder() -> ClientBuilder {
    reqwest::Client::builder()
        .redirect(Policy::none())
        .no_proxy()
        .connect_timeout(std::time::Duration::from_secs(5))
        .dns_resolver(Arc::new(SafeDnsResolver::new(IpPolicy::default())))
}

/// `reqwest::ClientBuilder` with `redirect(Policy::none())`, `.no_proxy()`,
/// and `connect_timeout(5s)` baked in.
///
/// Use for clients that send to operator-trusted vendor APIs (Twitter, OpenAI,
/// SerpAPI, etc.). No IP validation is implied — the caller asserts the host
/// is operator-trusted. See [`safe_client_builder`] for the security
/// rationale of the redirect / proxy / connect-timeout settings.
///
/// **NOTE**: the SafeDnsResolver is also installed here under `IpPolicy::Strict`
/// — even vendor calls should not silently route to private IPs if a DNS
/// hijack swings the host. Operators that need internal vendor endpoints can
/// opt out via `HEARTBIT_ALLOW_PRIVATE_IPS=1`.
pub fn vendor_client_builder() -> ClientBuilder {
    reqwest::Client::builder()
        .redirect(Policy::none())
        .no_proxy()
        .connect_timeout(std::time::Duration::from_secs(5))
        .dns_resolver(Arc::new(SafeDnsResolver::new(IpPolicy::default())))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- IpPolicy parser ----
    //
    // We test the pure parser (`from_env_value`), not `from_env` itself.
    // Mutating real env vars races with parallel tests in the cargo harness;
    // the parser is the actual logic and is testable without that risk.

    #[test]
    fn ip_policy_unset_is_strict() {
        assert_eq!(IpPolicy::from_env_value(None), IpPolicy::Strict);
    }

    #[test]
    fn ip_policy_one_is_allow() {
        assert_eq!(IpPolicy::from_env_value(Some("1")), IpPolicy::AllowPrivate);
    }

    #[test]
    fn ip_policy_true_case_insensitive_is_allow() {
        assert_eq!(
            IpPolicy::from_env_value(Some("TRUE")),
            IpPolicy::AllowPrivate
        );
        assert_eq!(
            IpPolicy::from_env_value(Some("True")),
            IpPolicy::AllowPrivate
        );
        assert_eq!(
            IpPolicy::from_env_value(Some("  true  ")),
            IpPolicy::AllowPrivate
        );
    }

    #[test]
    fn ip_policy_zero_is_strict() {
        assert_eq!(IpPolicy::from_env_value(Some("0")), IpPolicy::Strict);
        assert_eq!(IpPolicy::from_env_value(Some("false")), IpPolicy::Strict);
    }

    #[test]
    fn ip_policy_garbage_is_strict() {
        assert_eq!(IpPolicy::from_env_value(Some("yesplz")), IpPolicy::Strict);
        assert_eq!(IpPolicy::from_env_value(Some("")), IpPolicy::Strict);
    }

    // ---- SafeUrl::parse — scheme ----

    #[tokio::test]
    async fn safe_url_rejects_non_http_scheme() {
        let err = SafeUrl::parse("file:///etc/passwd", IpPolicy::Strict)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("scheme") && msg.contains("file"), "got: {msg}");
    }

    #[tokio::test]
    async fn safe_url_rejects_invalid_url() {
        let err = SafeUrl::parse("not a url", IpPolicy::Strict)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("invalid URL"));
    }

    // ---- SafeUrl::parse — literal IP blocklist (Strict) ----

    #[tokio::test]
    async fn safe_url_rejects_loopback_v4() {
        assert!(
            SafeUrl::parse("http://127.0.0.1/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_rejects_loopback_v6() {
        assert!(
            SafeUrl::parse("http://[::1]/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_rejects_link_local_v4() {
        // AWS / GCE IMDS
        assert!(
            SafeUrl::parse("http://169.254.169.254/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_rejects_link_local_v6() {
        assert!(
            SafeUrl::parse("http://[fe80::1]/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_rejects_rfc1918() {
        for h in ["10.0.0.1", "172.16.0.1", "192.168.1.1"] {
            let r = SafeUrl::parse(&format!("http://{h}/"), IpPolicy::Strict).await;
            assert!(r.is_err(), "{h} should be rejected");
        }
    }

    #[tokio::test]
    async fn safe_url_rejects_cgnat() {
        assert!(
            SafeUrl::parse("http://100.64.0.1/", IpPolicy::Strict)
                .await
                .is_err()
        );
        assert!(
            SafeUrl::parse("http://100.127.255.1/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_rejects_ula() {
        assert!(
            SafeUrl::parse("http://[fc00::1]/", IpPolicy::Strict)
                .await
                .is_err()
        );
        assert!(
            SafeUrl::parse("http://[fd00::1]/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_rejects_multicast() {
        assert!(
            SafeUrl::parse("http://224.0.0.1/", IpPolicy::Strict)
                .await
                .is_err()
        );
        assert!(
            SafeUrl::parse("http://[ff00::1]/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_rejects_unspecified() {
        assert!(
            SafeUrl::parse("http://0.0.0.0/", IpPolicy::Strict)
                .await
                .is_err()
        );
        assert!(
            SafeUrl::parse("http://[::]/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_rejects_broadcast() {
        assert!(
            SafeUrl::parse("http://255.255.255.255/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_accepts_public_ip() {
        let safe = SafeUrl::parse("http://8.8.8.8/", IpPolicy::Strict)
            .await
            .unwrap();
        assert_eq!(safe.as_str(), "http://8.8.8.8/");
    }

    // ---- SafeUrl::parse — IPv4-mapped IPv6 (`::ffff:0:0/96`) ----

    #[tokio::test]
    async fn safe_url_rejects_ipv4_mapped_loopback() {
        // ::ffff:127.0.0.1 must be rejected via the v4 blocklist.
        assert!(
            SafeUrl::parse("http://[::ffff:127.0.0.1]/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_rejects_ipv4_mapped_imds() {
        assert!(
            SafeUrl::parse("http://[::ffff:169.254.169.254]/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_rejects_ipv4_mapped_rfc1918() {
        assert!(
            SafeUrl::parse("http://[::ffff:10.0.0.1]/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    #[tokio::test]
    async fn safe_url_accepts_ipv4_mapped_public() {
        // ::ffff:8.8.8.8 should be accepted (mapped to a public v4).
        // Note: the URL parser normalizes the v6 form to the compact
        // representation (`::ffff:808:808`); we just assert success.
        let safe = SafeUrl::parse("http://[::ffff:8.8.8.8]/", IpPolicy::Strict)
            .await
            .unwrap();
        assert!(safe.as_str().starts_with("http://[::ffff:"));
    }

    // ---- SafeUrl::parse — DNS resolution ----

    #[tokio::test]
    async fn safe_url_rejects_localhost_dns() {
        // "localhost" resolves to 127.0.0.1 / ::1 — must be rejected under Strict.
        assert!(
            SafeUrl::parse("http://localhost/", IpPolicy::Strict)
                .await
                .is_err()
        );
    }

    // ---- SafeUrl::parse — AllowPrivate bypass ----

    #[tokio::test]
    async fn safe_url_allow_private_accepts_loopback() {
        let safe = SafeUrl::parse("http://127.0.0.1/", IpPolicy::AllowPrivate)
            .await
            .unwrap();
        assert_eq!(safe.as_str(), "http://127.0.0.1/");
    }

    #[tokio::test]
    async fn safe_url_allow_private_accepts_localhost() {
        let safe = SafeUrl::parse("http://localhost/", IpPolicy::AllowPrivate)
            .await
            .unwrap();
        assert_eq!(safe.as_str(), "http://localhost/");
    }

    // ---- Rejection message guidance ----

    #[tokio::test]
    async fn safe_url_rejection_message_mentions_override() {
        let err = SafeUrl::parse("http://127.0.0.1/", IpPolicy::Strict)
            .await
            .unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("HEARTBIT_ALLOW_PRIVATE_IPS"),
            "rejection message should mention the override env var; got: {msg}"
        );
    }

    // ---- Client builders ----

    #[tokio::test]
    async fn safe_client_builder_does_not_follow_redirects() {
        // Spin up a tiny in-process listener that returns 302 → /landed.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            // Accept one request, return a 302.
            if let Ok((mut sock, _)) = listener.accept().await {
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 1024];
                let _ = sock.read(&mut buf).await;
                let resp = b"HTTP/1.1 302 Found\r\nLocation: /landed\r\nContent-Length: 0\r\n\r\n";
                let _ = sock.write_all(resp).await;
                let _ = sock.shutdown().await;
            }
        });

        let client = safe_client_builder().build().unwrap();
        let resp = client
            .get(format!("http://{addr}/start"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 302, "redirect must NOT be followed");
    }

    #[test]
    fn vendor_client_builder_compiles_and_builds() {
        let _ = vendor_client_builder().build().unwrap();
    }

    /// SECURITY (F-NET-1): `read_body_capped` MUST stop reading once the cap
    /// is reached. A hostile vendor that streams gigabytes would otherwise
    /// OOM the agent.
    #[tokio::test]
    async fn read_body_capped_truncates_at_limit() {
        use std::convert::Infallible;
        use tokio::io::AsyncWriteExt;
        // Spin a tiny TCP server that streams 10 MiB.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                // Read & discard request, then send headers + huge body.
                let mut tmp = [0u8; 1024];
                let _ = tokio::io::AsyncReadExt::read(&mut sock, &mut tmp).await;
                let _ = sock
                    .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 10485760\r\n\r\n")
                    .await;
                // Stream 10 MiB of 'A'. We don't care if the peer hangs up.
                let chunk = vec![b'A'; 64 * 1024];
                for _ in 0..160 {
                    if sock.write_all(&chunk).await.is_err() {
                        break;
                    }
                }
                Ok::<_, Infallible>(())
            } else {
                Ok(())
            }
        });

        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let resp = client.get(format!("http://{addr}/")).send().await.unwrap();
        let (bytes, truncated) = read_body_capped(resp, 1024 * 1024).await.unwrap();
        assert!(truncated, "must report truncation");
        assert!(
            bytes.len() <= 1024 * 1024 + 64 * 1024,
            "must not exceed cap by more than one chunk; got {}",
            bytes.len()
        );
    }
}
