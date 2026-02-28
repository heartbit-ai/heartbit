use std::sync::RwLock;
use std::time::{Duration, Instant};

use jsonwebtoken::{Algorithm, DecodingKey, TokenData, Validation, decode, decode_header};
use serde::{Deserialize, Serialize};

use crate::daemon::types::UserContext;
use crate::error::Error;

/// Client for fetching and caching JWKS (JSON Web Key Sets) from an identity provider.
///
/// Caches the keys with a configurable TTL to avoid hitting the JWKS endpoint on every
/// request. Thread-safe via `std::sync::RwLock` (never held across `.await`).
pub struct JwksClient {
    jwks_url: String,
    client: reqwest::Client,
    cache: RwLock<Option<CachedJwks>>,
    cache_ttl: Duration,
}

struct CachedJwks {
    keys: Vec<JwkKey>,
    fetched_at: Instant,
}

/// A single JWK key from the JWKS endpoint.
#[derive(Debug, Clone, Deserialize)]
struct JwkKey {
    kid: Option<String>,
    kty: String,
    #[serde(default)]
    #[allow(dead_code)]
    alg: Option<String>,
    /// RSA modulus (base64url-encoded).
    n: Option<String>,
    /// RSA exponent (base64url-encoded).
    e: Option<String>,
}

/// JWKS response from the identity provider.
#[derive(Debug, Deserialize)]
struct JwksResponse {
    keys: Vec<JwkKey>,
}

impl JwksClient {
    /// Create a new JWKS client with a 5-minute cache TTL.
    pub fn new(jwks_url: impl Into<String>) -> Self {
        Self::with_ttl(jwks_url, Duration::from_secs(300))
    }

    /// Create a new JWKS client with a custom cache TTL.
    pub fn with_ttl(jwks_url: impl Into<String>, cache_ttl: Duration) -> Self {
        Self {
            jwks_url: jwks_url.into(),
            client: reqwest::Client::new(),
            cache: RwLock::new(None),
            cache_ttl,
        }
    }

    /// Fetch a decoding key for the given `kid` (Key ID) from the JWKS endpoint.
    ///
    /// Uses cached keys if available and not expired. Falls back to re-fetching
    /// if the requested `kid` is not found in cache (key rotation).
    pub async fn decoding_key(&self, kid: Option<&str>) -> Result<DecodingKey, Error> {
        // Try cache first
        if let Some(key) = self.find_in_cache(kid) {
            return Self::jwk_to_decoding_key(&key);
        }

        // Fetch fresh keys
        let keys = self.fetch_keys().await?;
        let key = Self::find_key(&keys, kid)?;
        let decoding_key = Self::jwk_to_decoding_key(&key)?;

        // Update cache
        if let Ok(mut cache) = self.cache.write() {
            *cache = Some(CachedJwks {
                keys,
                fetched_at: Instant::now(),
            });
        }

        Ok(decoding_key)
    }

    fn find_in_cache(&self, kid: Option<&str>) -> Option<JwkKey> {
        let cache = self.cache.read().ok()?;
        let cached = cache.as_ref()?;

        if cached.fetched_at.elapsed() > self.cache_ttl {
            return None;
        }

        Self::find_key(&cached.keys, kid).ok()
    }

    async fn fetch_keys(&self) -> Result<Vec<JwkKey>, Error> {
        let response = self
            .client
            .get(&self.jwks_url)
            .send()
            .await
            .map_err(|e| Error::Auth(format!("JWKS fetch failed: {e}")))?;

        if !response.status().is_success() {
            return Err(Error::Auth(format!(
                "JWKS endpoint returned status {}",
                response.status()
            )));
        }

        let jwks: JwksResponse = response
            .json()
            .await
            .map_err(|e| Error::Auth(format!("JWKS parse failed: {e}")))?;

        Ok(jwks.keys)
    }

    fn find_key(keys: &[JwkKey], kid: Option<&str>) -> Result<JwkKey, Error> {
        match kid {
            Some(kid) => keys
                .iter()
                .find(|k| k.kid.as_deref() == Some(kid))
                .cloned()
                .ok_or_else(|| Error::Auth(format!("No JWK found with kid={kid}"))),
            None => {
                // No kid specified — use first RSA key
                keys.iter()
                    .find(|k| k.kty == "RSA")
                    .cloned()
                    .ok_or_else(|| Error::Auth("No RSA key in JWKS".into()))
            }
        }
    }

    fn jwk_to_decoding_key(key: &JwkKey) -> Result<DecodingKey, Error> {
        match key.kty.as_str() {
            "RSA" => {
                let n = key
                    .n
                    .as_ref()
                    .ok_or_else(|| Error::Auth("JWK missing 'n' component".into()))?;
                let e = key
                    .e
                    .as_ref()
                    .ok_or_else(|| Error::Auth("JWK missing 'e' component".into()))?;
                DecodingKey::from_rsa_components(n, e)
                    .map_err(|e| Error::Auth(format!("Invalid RSA JWK: {e}")))
            }
            other => Err(Error::Auth(format!("Unsupported key type: {other}"))),
        }
    }
}

/// JWT claims that we extract from incoming tokens.
///
/// Flexible enough to handle different IdP claim structures via configurable
/// claim names.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct JwtClaims {
    /// Standard claims
    #[serde(default)]
    sub: Option<String>,
    #[serde(default)]
    iss: Option<String>,
    #[serde(default)]
    aud: Option<ClaimAudience>,
    #[serde(default)]
    exp: Option<u64>,
    /// xavyo-idp tenant ID
    #[serde(default)]
    tid: Option<String>,
    /// Roles claim (common in many IdPs)
    #[serde(default)]
    roles: Option<Vec<String>>,
    /// Catch-all for custom claims
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

/// JWT audience can be a single string or array.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum ClaimAudience {
    Single(String),
    Multiple(Vec<String>),
}

impl ClaimAudience {
    #[allow(dead_code)]
    fn contains(&self, expected: &str) -> bool {
        match self {
            ClaimAudience::Single(s) => s == expected,
            ClaimAudience::Multiple(v) => v.iter().any(|s| s == expected),
        }
    }
}

/// Validates JWTs and extracts `UserContext` from claims.
///
/// Supports configurable claim names for user ID, tenant ID, and roles
/// to accommodate different identity providers.
pub struct JwtValidator {
    jwks: JwksClient,
    issuer: Option<String>,
    audience: Option<String>,
    user_id_claim: String,
    tenant_id_claim: String,
    roles_claim: String,
}

impl JwtValidator {
    /// Create a new JWT validator.
    ///
    /// - `jwks_url`: JWKS endpoint for key verification
    /// - `issuer`: Expected `iss` claim (optional)
    /// - `audience`: Expected `aud` claim (optional)
    pub fn new(
        jwks_url: impl Into<String>,
        issuer: Option<String>,
        audience: Option<String>,
    ) -> Self {
        Self {
            jwks: JwksClient::new(jwks_url),
            issuer,
            audience,
            user_id_claim: "sub".into(),
            tenant_id_claim: "tid".into(),
            roles_claim: "roles".into(),
        }
    }

    /// Override the claim name for user ID (default: "sub").
    pub fn with_user_id_claim(mut self, claim: impl Into<String>) -> Self {
        self.user_id_claim = claim.into();
        self
    }

    /// Override the claim name for tenant ID (default: "tid").
    pub fn with_tenant_id_claim(mut self, claim: impl Into<String>) -> Self {
        self.tenant_id_claim = claim.into();
        self
    }

    /// Override the claim name for roles (default: "roles").
    pub fn with_roles_claim(mut self, claim: impl Into<String>) -> Self {
        self.roles_claim = claim.into();
        self
    }

    /// Validate a JWT token and extract a `UserContext`.
    ///
    /// Verifies the token signature against JWKS, checks issuer/audience,
    /// and extracts user_id, tenant_id, and roles from the configured claims.
    pub async fn validate(&self, token: &str) -> Result<UserContext, Error> {
        if token.is_empty() {
            return Err(Error::Auth("JWT token is empty".into()));
        }
        // 16 KiB is generous for a JWT — reject oversized tokens to prevent DoS
        if token.len() > 16_384 {
            return Err(Error::Auth("JWT token exceeds maximum length".into()));
        }

        // Decode header to get kid
        let header =
            decode_header(token).map_err(|e| Error::Auth(format!("Invalid JWT header: {e}")))?;

        // Get decoding key from JWKS
        let decoding_key = self.jwks.decoding_key(header.kid.as_deref()).await?;

        // Build validation
        let mut validation = Validation::new(Algorithm::RS256);

        if let Some(ref iss) = self.issuer {
            validation.set_issuer(&[iss]);
        }
        // When issuer is None, don't call set_issuer — iss is not validated by default.

        if let Some(ref aud) = self.audience {
            validation.set_audience(&[aud]);
        } else {
            validation.validate_aud = false;
        }

        // Decode and validate
        let token_data: TokenData<JwtClaims> = decode(token, &decoding_key, &validation)
            .map_err(|e| Error::Auth(format!("JWT validation failed: {e}")))?;

        let claims = token_data.claims;

        // Extract user_id from configured claim
        let user_id = self.extract_string_claim(&claims, &self.user_id_claim)?;
        if user_id.is_empty() {
            return Err(Error::Auth("JWT user_id claim is empty".into()));
        }

        // Extract tenant_id from configured claim
        let tenant_id = self.extract_string_claim(&claims, &self.tenant_id_claim)?;
        if tenant_id.is_empty() {
            return Err(Error::Auth("JWT tenant_id claim is empty".into()));
        }

        // Extract roles from configured claim
        let roles = self.extract_roles(&claims);

        Ok(UserContext {
            user_id,
            tenant_id,
            roles,
            raw_token: None,
        })
    }

    fn extract_string_claim(&self, claims: &JwtClaims, claim_name: &str) -> Result<String, Error> {
        // Check standard fields first
        match claim_name {
            "sub" => {
                if let Some(ref sub) = claims.sub {
                    return Ok(sub.clone());
                }
            }
            "tid" => {
                if let Some(ref tid) = claims.tid {
                    return Ok(tid.clone());
                }
            }
            "iss" => {
                if let Some(ref iss) = claims.iss {
                    return Ok(iss.clone());
                }
            }
            _ => {}
        }

        // Fall back to extra claims
        if let Some(value) = claims.extra.get(claim_name) {
            return match value {
                serde_json::Value::String(s) => Ok(s.clone()),
                serde_json::Value::Number(n) => Ok(n.to_string()),
                _ => Err(Error::Auth(format!(
                    "Claim '{claim_name}' has unsupported type (expected string or number)"
                ))),
            };
        }

        Err(Error::Auth(format!(
            "Required claim '{claim_name}' not found in JWT"
        )))
    }

    fn extract_roles(&self, claims: &JwtClaims) -> Vec<String> {
        // Check standard roles field
        if self.roles_claim == "roles"
            && let Some(ref roles) = claims.roles
        {
            return roles.clone();
        }

        // Check extra claims
        if let Some(value) = claims.extra.get(&self.roles_claim)
            && let Some(arr) = value.as_array()
        {
            return arr
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }

        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- JwkKey tests ---

    #[test]
    fn jwk_key_deserializes() {
        let json = r#"{"kty":"RSA","kid":"key-1","n":"abc","e":"AQAB"}"#;
        let key: JwkKey = serde_json::from_str(json).unwrap();
        assert_eq!(key.kty, "RSA");
        assert_eq!(key.kid.as_deref(), Some("key-1"));
        assert_eq!(key.n.as_deref(), Some("abc"));
        assert_eq!(key.e.as_deref(), Some("AQAB"));
    }

    #[test]
    fn jwk_key_optional_kid() {
        let json = r#"{"kty":"RSA","n":"abc","e":"AQAB"}"#;
        let key: JwkKey = serde_json::from_str(json).unwrap();
        assert!(key.kid.is_none());
    }

    #[test]
    fn jwks_response_parses() {
        let json = r#"{"keys":[{"kty":"RSA","kid":"k1","n":"abc","e":"AQAB"},{"kty":"RSA","kid":"k2","n":"def","e":"AQAB"}]}"#;
        let resp: JwksResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.keys.len(), 2);
        assert_eq!(resp.keys[0].kid.as_deref(), Some("k1"));
        assert_eq!(resp.keys[1].kid.as_deref(), Some("k2"));
    }

    // --- find_key tests ---

    #[test]
    fn find_key_by_kid() {
        let keys = vec![
            JwkKey {
                kid: Some("k1".into()),
                kty: "RSA".into(),
                alg: None,
                n: Some("abc".into()),
                e: Some("AQAB".into()),
            },
            JwkKey {
                kid: Some("k2".into()),
                kty: "RSA".into(),
                alg: None,
                n: Some("def".into()),
                e: Some("AQAB".into()),
            },
        ];
        let key = JwksClient::find_key(&keys, Some("k2")).unwrap();
        assert_eq!(key.n.as_deref(), Some("def"));
    }

    #[test]
    fn find_key_by_kid_not_found() {
        let keys = vec![JwkKey {
            kid: Some("k1".into()),
            kty: "RSA".into(),
            alg: None,
            n: Some("abc".into()),
            e: Some("AQAB".into()),
        }];
        let err = JwksClient::find_key(&keys, Some("k999")).unwrap_err();
        assert!(err.to_string().contains("k999"));
    }

    #[test]
    fn find_key_no_kid_returns_first_rsa() {
        let keys = vec![
            JwkKey {
                kid: None,
                kty: "EC".into(),
                alg: None,
                n: None,
                e: None,
            },
            JwkKey {
                kid: None,
                kty: "RSA".into(),
                alg: None,
                n: Some("first-rsa".into()),
                e: Some("AQAB".into()),
            },
        ];
        let key = JwksClient::find_key(&keys, None).unwrap();
        assert_eq!(key.n.as_deref(), Some("first-rsa"));
    }

    #[test]
    fn find_key_no_rsa_errors() {
        let keys = vec![JwkKey {
            kid: None,
            kty: "EC".into(),
            alg: None,
            n: None,
            e: None,
        }];
        let err = JwksClient::find_key(&keys, None).unwrap_err();
        assert!(err.to_string().contains("No RSA key"));
    }

    // --- jwk_to_decoding_key tests ---

    #[test]
    fn jwk_to_decoding_key_unsupported_type() {
        let key = JwkKey {
            kid: None,
            kty: "EC".into(),
            alg: None,
            n: None,
            e: None,
        };
        let err = JwksClient::jwk_to_decoding_key(&key).err().unwrap();
        assert!(err.to_string().contains("Unsupported key type: EC"));
    }

    #[test]
    fn jwk_to_decoding_key_missing_n() {
        let key = JwkKey {
            kid: None,
            kty: "RSA".into(),
            alg: None,
            n: None,
            e: Some("AQAB".into()),
        };
        let err = JwksClient::jwk_to_decoding_key(&key).err().unwrap();
        assert!(err.to_string().contains("missing 'n'"));
    }

    #[test]
    fn jwk_to_decoding_key_missing_e() {
        let key = JwkKey {
            kid: None,
            kty: "RSA".into(),
            alg: None,
            n: Some("abc".into()),
            e: None,
        };
        let err = JwksClient::jwk_to_decoding_key(&key).err().unwrap();
        assert!(err.to_string().contains("missing 'e'"));
    }

    // --- ClaimAudience tests ---

    #[test]
    fn claim_audience_single() {
        let aud: ClaimAudience = serde_json::from_str(r#""my-app""#).unwrap();
        assert!(aud.contains("my-app"));
        assert!(!aud.contains("other"));
    }

    #[test]
    fn claim_audience_multiple() {
        let aud: ClaimAudience = serde_json::from_str(r#"["app1","app2"]"#).unwrap();
        assert!(aud.contains("app1"));
        assert!(aud.contains("app2"));
        assert!(!aud.contains("app3"));
    }

    // --- JwtClaims tests ---

    #[test]
    fn jwt_claims_deserialize_full() {
        let json = r#"{"sub":"user-1","iss":"https://idp.example.com","aud":"my-app","exp":9999999999,"tid":"tenant-1","roles":["admin","user"]}"#;
        let claims: JwtClaims = serde_json::from_str(json).unwrap();
        assert_eq!(claims.sub.as_deref(), Some("user-1"));
        assert_eq!(claims.iss.as_deref(), Some("https://idp.example.com"));
        assert_eq!(claims.tid.as_deref(), Some("tenant-1"));
        assert_eq!(claims.roles, Some(vec!["admin".into(), "user".into()]));
    }

    #[test]
    fn jwt_claims_deserialize_minimal() {
        let json = r#"{"sub":"u1"}"#;
        let claims: JwtClaims = serde_json::from_str(json).unwrap();
        assert_eq!(claims.sub.as_deref(), Some("u1"));
        assert!(claims.tid.is_none());
        assert!(claims.roles.is_none());
    }

    #[test]
    fn jwt_claims_extra_fields() {
        let json = r#"{"sub":"u1","custom_tenant":"acme","custom_roles":["r1"]}"#;
        let claims: JwtClaims = serde_json::from_str(json).unwrap();
        assert_eq!(
            claims.extra.get("custom_tenant").and_then(|v| v.as_str()),
            Some("acme")
        );
    }

    // --- JwtValidator extract tests (unit-testable without crypto) ---

    #[test]
    fn extract_string_claim_from_sub() {
        let claims = JwtClaims {
            sub: Some("user-42".into()),
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: None,
            extra: Default::default(),
        };
        let validator = JwtValidator::new("http://unused", None, None);
        assert_eq!(
            validator.extract_string_claim(&claims, "sub").unwrap(),
            "user-42"
        );
    }

    #[test]
    fn extract_string_claim_from_tid() {
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: Some("acme".into()),
            roles: None,
            extra: Default::default(),
        };
        let validator = JwtValidator::new("http://unused", None, None);
        assert_eq!(
            validator.extract_string_claim(&claims, "tid").unwrap(),
            "acme"
        );
    }

    #[test]
    fn extract_string_claim_from_extra() {
        let mut extra = serde_json::Map::new();
        extra.insert("org_id".into(), serde_json::Value::String("org-99".into()));
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: None,
            extra,
        };
        let validator =
            JwtValidator::new("http://unused", None, None).with_tenant_id_claim("org_id");
        assert_eq!(
            validator.extract_string_claim(&claims, "org_id").unwrap(),
            "org-99"
        );
    }

    #[test]
    fn extract_string_claim_missing() {
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: None,
            extra: Default::default(),
        };
        let validator = JwtValidator::new("http://unused", None, None);
        let err = validator.extract_string_claim(&claims, "sub").unwrap_err();
        assert!(err.to_string().contains("sub"));
    }

    #[test]
    fn extract_roles_standard() {
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: Some(vec!["admin".into(), "sales".into()]),
            extra: Default::default(),
        };
        let validator = JwtValidator::new("http://unused", None, None);
        assert_eq!(validator.extract_roles(&claims), vec!["admin", "sales"]);
    }

    #[test]
    fn extract_roles_from_custom_claim() {
        let mut extra = serde_json::Map::new();
        extra.insert("permissions".into(), serde_json::json!(["read", "write"]));
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: None,
            extra,
        };
        let validator =
            JwtValidator::new("http://unused", None, None).with_roles_claim("permissions");
        assert_eq!(validator.extract_roles(&claims), vec!["read", "write"]);
    }

    #[test]
    fn extract_roles_empty_when_absent() {
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: None,
            extra: Default::default(),
        };
        let validator = JwtValidator::new("http://unused", None, None);
        assert!(validator.extract_roles(&claims).is_empty());
    }

    // --- JwksClient cache tests ---

    #[test]
    fn jwks_client_cache_ttl() {
        let client = JwksClient::with_ttl(
            "http://example.com/.well-known/jwks.json",
            Duration::from_secs(60),
        );
        assert_eq!(client.cache_ttl, Duration::from_secs(60));
    }

    #[test]
    fn jwks_client_empty_cache_returns_none() {
        let client = JwksClient::new("http://example.com/.well-known/jwks.json");
        assert!(client.find_in_cache(Some("k1")).is_none());
    }

    // --- JwtValidator builder tests ---

    #[test]
    fn validator_custom_claim_names() {
        let validator = JwtValidator::new("http://unused", None, None)
            .with_user_id_claim("user_id")
            .with_tenant_id_claim("organization_id")
            .with_roles_claim("permissions");
        assert_eq!(validator.user_id_claim, "user_id");
        assert_eq!(validator.tenant_id_claim, "organization_id");
        assert_eq!(validator.roles_claim, "permissions");
    }

    // --- Token pre-validation tests ---

    #[tokio::test]
    async fn validate_empty_token_rejected() {
        let validator = JwtValidator::new("http://unused", None, None);
        let err = validator.validate("").await.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[tokio::test]
    async fn validate_oversized_token_rejected() {
        let validator = JwtValidator::new("http://unused", None, None);
        let huge = "x".repeat(20_000);
        let err = validator.validate(&huge).await.unwrap_err();
        assert!(err.to_string().contains("maximum length"));
    }

    // --- Non-string claim coercion tests ---

    #[test]
    fn extract_string_claim_from_numeric_extra() {
        let mut extra = serde_json::Map::new();
        extra.insert("org_id".into(), serde_json::json!(12345));
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: None,
            extra,
        };
        let validator = JwtValidator::new("http://unused", None, None);
        assert_eq!(
            validator.extract_string_claim(&claims, "org_id").unwrap(),
            "12345"
        );
    }

    #[test]
    fn extract_string_claim_rejects_boolean() {
        let mut extra = serde_json::Map::new();
        extra.insert("active".into(), serde_json::json!(true));
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: None,
            extra,
        };
        let validator = JwtValidator::new("http://unused", None, None);
        let err = validator
            .extract_string_claim(&claims, "active")
            .unwrap_err();
        assert!(err.to_string().contains("unsupported type"));
    }

    #[test]
    fn extract_string_claim_rejects_null() {
        let mut extra = serde_json::Map::new();
        extra.insert("tenant".into(), serde_json::Value::Null);
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: None,
            extra,
        };
        let validator = JwtValidator::new("http://unused", None, None);
        let err = validator
            .extract_string_claim(&claims, "tenant")
            .unwrap_err();
        assert!(err.to_string().contains("unsupported type"));
    }

    #[test]
    fn extract_string_claim_rejects_object() {
        let mut extra = serde_json::Map::new();
        extra.insert("nested".into(), serde_json::json!({"key": "value"}));
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: None,
            extra,
        };
        let validator = JwtValidator::new("http://unused", None, None);
        let err = validator
            .extract_string_claim(&claims, "nested")
            .unwrap_err();
        assert!(err.to_string().contains("unsupported type"));
    }

    // --- Error variant tests ---

    #[test]
    fn auth_errors_use_auth_variant() {
        let validator = JwtValidator::new("http://unused", None, None);
        let claims = JwtClaims {
            sub: None,
            iss: None,
            aud: None,
            exp: None,
            tid: None,
            roles: None,
            extra: Default::default(),
        };
        let err = validator.extract_string_claim(&claims, "sub").unwrap_err();
        assert!(err.to_string().contains("Authentication error"));
    }
}
