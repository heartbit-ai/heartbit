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
    /// JWT `tid` claim). Empty strings collapse to `single_tenant()` so a
    /// dropped scope can never silently widen to all tenants.
    pub fn new(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            user_id: None,
        }
    }

    /// Add a user identity (typically `sub` claim from JWT).
    pub fn with_user(mut self, user_id: impl Into<String>) -> Self {
        self.user_id = Some(user_id.into());
        self
    }

    /// Single-tenant default; `tenant_id == ""`.
    pub fn single_tenant() -> Self {
        Self {
            tenant_id: String::new(),
            user_id: None,
        }
    }

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
}
