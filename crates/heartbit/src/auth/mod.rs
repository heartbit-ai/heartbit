#[cfg(feature = "daemon")]
mod jwt;

#[cfg(feature = "daemon")]
pub use jwt::{JwksClient, JwtValidator};

#[cfg(feature = "vault")]
pub mod vault;
