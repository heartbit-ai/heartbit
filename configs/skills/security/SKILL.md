---
name = "security"
description = "OWASP top 10 with code patterns, auth, secrets management, and common vulnerabilities"
tags = ["security", "owasp", "auth", "secrets", "appsec"]
max_inject_tokens = 2000
---

# Application Security Expert

## Injection (OWASP A03)

Always parameterize queries. Never interpolate user input into SQL, commands, or templates.

```rust
// BAD: SQL injection
let q = format!("SELECT * FROM users WHERE id = '{}'", user_input);

// GOOD: parameterized
sqlx::query("SELECT * FROM users WHERE id = $1").bind(user_id).fetch_one(&pool).await?;
```

Command injection: avoid `shell=True` (Python) or `Command::new("sh").arg("-c")` (Rust). Pass arguments as array elements. If shell is unavoidable, use allowlists for inputs, never blocklists.

## Broken Authentication (A07)

- Hash passwords with `argon2id` (preferred) or `bcrypt`. Never SHA-256/MD5 for passwords.
- JWT: validate `iss`, `aud`, `exp` claims. Reject `alg: none`. Use asymmetric keys (RS256/ES256) for multi-service.
- Token storage: `HttpOnly; Secure; SameSite=Strict` cookies. Never localStorage for auth tokens.
- Refresh token rotation: issue new refresh token on each use, invalidate the old one (prevents replay).
- Rate limit login endpoints: 5 attempts per minute per account, exponential backoff.

## SSRF (A10)

Block internal network access from user-controllable URLs:

```rust
// Validate URL before fetching
let url: Url = input.parse()?;
match url.host() {
    Some(Host::Ipv4(ip)) if ip.is_loopback() || ip.is_private() => {
        return Err(Error::SsrfBlocked);
    }
    _ => {}
}
// Also: disable HTTP redirects (redirect to internal IP bypasses check)
let client = reqwest::Client::builder().redirect(Policy::none()).build()?;
```

DNS rebinding: resolve hostname before connecting and validate the resolved IP. Check after redirect too.

## Secrets Management

- Environment variables for runtime secrets (12-factor). Never in code, configs committed to git.
- Use `dotenv` for local dev, vault (HashiCorp/AWS Secrets Manager) for production.
- Rotate secrets on schedule. Automate rotation — manual rotation means it never happens.
- Audit secret access: log who accessed which secret, when (not the value).
- `.env` in `.gitignore`. Pre-commit hooks to scan for leaked secrets (`gitleaks`, `trufflehog`).
- In-memory: zero secrets after use. Avoid `String` (may linger in memory) — use `secrecy::Secret<String>`.

## Common Vulnerabilities

**Path Traversal**: validate file paths against a base directory. Reject `..`, canonicalize before comparison:

```rust
let canonical = base_dir.join(user_path).canonicalize()?;
if !canonical.starts_with(&base_dir) {
    return Err(Error::PathTraversal);
}
```

**Insecure Deserialization**: never deserialize untrusted data with format-specific deserializers that allow arbitrary types (Python `pickle`, Java `ObjectInputStream`). Use JSON/MessagePack with strict schemas.

**Mass Assignment**: explicitly list allowed fields. Never pass raw request body to ORM/model update.

**CORS Misconfiguration**: never reflect `Origin` header as `Access-Control-Allow-Origin`. Allowlist specific origins. Never use `*` with credentials.

## Security Headers

```
Content-Security-Policy: default-src 'self'; script-src 'self'
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: camera=(), microphone=(), geolocation=()
```

## Dependency Security

- `cargo audit` / `npm audit` / `pip-audit` in CI — fail the build on known CVEs.
- Pin dependencies with lockfiles. Review lockfile diffs in PRs.
- Dependabot/Renovate for automated updates with auto-merge for patch versions.
- Minimal dependencies: each dep is an attack surface. Audit transitive deps.
