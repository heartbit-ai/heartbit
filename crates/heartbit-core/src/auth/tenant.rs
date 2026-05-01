//! Tenant + optional user identity for scoping memory, audit, and policy.

/// Tenant + optional user identity for scoping memory, audit, and policy
/// decisions. Owned (no lifetime parameter) so it composes cleanly into
/// async contexts and can be stored in `Arc`-shared state.
///
/// `tenant_id` is `String`, not `Uuid`, to match the existing
/// `UserContext.tenant_id: String` (deliberate: JWT `tid` claims from
/// Auth0 / Cognito / Okta etc. are not always UUIDs). The sentinel for
/// "single-tenant mode" is the empty string.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TenantScope {
    pub tenant_id: String,
    pub user_id: Option<String>,
}

impl TenantScope {
    /// Multi-tenant scope from an externally-supplied tenant id (typically
    /// JWT `tid` claim). Empty strings delegate to `single_tenant()` so a
    /// dropped scope can never silently widen to all tenants.
    pub fn new(tenant_id: impl Into<String>) -> Self {
        let tenant_id = tenant_id.into();
        if tenant_id.is_empty() {
            return Self::single_tenant();
        }
        Self {
            tenant_id,
            user_id: None,
        }
    }

    /// Add a user identity (typically `sub` claim from JWT). An empty string
    /// normalizes to `None` so callers can rely on `Some(_)` meaning a
    /// non-empty identity is present.
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        let user_id = user_id.into();
        self.user_id = if user_id.is_empty() {
            None
        } else {
            Some(user_id)
        };
        self
    }

    /// Single-tenant default; `tenant_id == ""`.
    pub fn single_tenant() -> Self {
        Self {
            tenant_id: String::new(),
            user_id: None,
        }
    }

    /// Returns true if this scope represents single-tenant mode (`tenant_id == ""`).
    pub fn is_single_tenant(&self) -> bool {
        self.tenant_id.is_empty()
    }
}

impl Default for TenantScope {
    fn default() -> Self {
        Self::single_tenant()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_single_tenant_with_empty_id() {
        let scope = TenantScope::default();
        assert_eq!(scope.tenant_id, "");
        assert!(scope.user_id.is_none());
        assert!(scope.is_single_tenant());
    }

    #[test]
    fn new_with_real_tenant_is_not_single_tenant() {
        let scope = TenantScope::new("tenant-123");
        assert_eq!(scope.tenant_id, "tenant-123");
        assert!(!scope.is_single_tenant());
    }

    #[test]
    fn new_with_empty_string_collapses_to_single_tenant() {
        let scope = TenantScope::new("");
        assert!(scope.is_single_tenant());
    }

    #[test]
    fn with_user_attaches_identity() {
        let scope = TenantScope::new("acme").with_user("user-42");
        assert_eq!(scope.tenant_id, "acme");
        assert_eq!(scope.user_id.as_deref(), Some("user-42"));
    }

    #[test]
    fn equal_scopes_compare_equal() {
        let a = TenantScope::new("acme").with_user("u1");
        let b = TenantScope::new("acme").with_user("u1");
        assert_eq!(a, b);
    }

    #[test]
    fn new_with_empty_string_produces_canonical_single_tenant() {
        // After normalization, new("") and single_tenant() should be equal,
        // not just equivalent under is_single_tenant().
        assert_eq!(TenantScope::new(""), TenantScope::single_tenant());
    }

    #[test]
    fn with_user_empty_string_normalizes_to_none() {
        let scope = TenantScope::new("acme").with_user("");
        assert!(scope.user_id.is_none());
    }

    #[test]
    fn with_user_overwrites_previous_user_with_none_when_given_empty() {
        let scope = TenantScope::new("acme").with_user("u1").with_user("");
        assert!(scope.user_id.is_none());
    }
}
