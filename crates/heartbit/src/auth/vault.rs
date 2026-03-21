//! AES-256-GCM encrypted credential vault.
//!
//! Stores secrets in `~/.heartbit/vault.enc`, encrypted with a key derived
//! via Argon2.
//!
//! # Usage
//!
//! ```ignore
//! let mut vault = CredentialVault::new(vault_path);
//! vault.unlock_with_key("my-passphrase")?;
//! vault.set("MY_SECRET", "value")?;
//! vault.save_with_key("my-passphrase")?;
//! let val = vault.get("MY_SECRET");
//! ```

use std::collections::HashMap;
use std::path::PathBuf;

use aes_gcm::aead::{Aead, KeyInit, OsRng};
use aes_gcm::{Aes256Gcm, Nonce};
use argon2::Argon2;
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Salt length for Argon2 key derivation.
const SALT_LEN: usize = 16;
/// Nonce length for AES-256-GCM.
const NONCE_LEN: usize = 12;

/// On-disk vault format.
#[derive(Serialize, Deserialize)]
struct VaultFile {
    version: u8,
    salt: String,
    nonce: String,
    ciphertext: String,
}

/// Encrypted credential vault.
///
/// Stores key-value pairs encrypted with AES-256-GCM. The encryption key
/// is derived from a master passphrase via Argon2.
pub struct CredentialVault {
    path: PathBuf,
    entries: HashMap<String, String>,
    unlocked: bool,
}

impl CredentialVault {
    /// Create a new vault at the given path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            entries: HashMap::new(),
            unlocked: false,
        }
    }

    /// Returns the default vault path: `~/.heartbit/vault.enc`.
    pub fn default_path() -> Result<PathBuf, Error> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map_err(|_| Error::Config("cannot determine home directory".into()))?;
        Ok(PathBuf::from(home).join(".heartbit").join("vault.enc"))
    }

    /// Check if the vault file exists on disk.
    pub fn exists(&self) -> bool {
        self.path.exists()
    }

    /// Unlock the vault using `HEARTBIT_VAULT_KEY` from the environment.
    ///
    /// Convenience wrapper around [`unlock_with_key`](Self::unlock_with_key).
    pub fn unlock(&mut self) -> Result<(), Error> {
        let master_key = vault_key_from_env()?;
        self.unlock_with_key(&master_key)
    }

    /// Unlock (load and decrypt) the vault with the given master key.
    ///
    /// If the vault file doesn't exist, starts with empty entries.
    pub fn unlock_with_key(&mut self, master_key: &str) -> Result<(), Error> {
        if self.path.exists() {
            let data = std::fs::read_to_string(&self.path)
                .map_err(|e| Error::Config(format!("failed to read vault: {e}")))?;
            let vault_file: VaultFile = serde_json::from_str(&data)
                .map_err(|e| Error::Config(format!("failed to parse vault: {e}")))?;

            if vault_file.version != 1 {
                return Err(Error::Config(format!(
                    "unsupported vault version: {}",
                    vault_file.version
                )));
            }

            let salt = base64_decode(&vault_file.salt)?;
            let nonce_bytes = base64_decode(&vault_file.nonce)?;
            let ciphertext = base64_decode(&vault_file.ciphertext)?;

            let key = derive_key(master_key, &salt)?;
            let cipher = Aes256Gcm::new_from_slice(&key)
                .map_err(|e| Error::Config(format!("failed to create cipher: {e}")))?;
            let nonce = Nonce::from_slice(&nonce_bytes);

            let plaintext = cipher
                .decrypt(nonce, ciphertext.as_ref())
                .map_err(|_| Error::Config("vault decryption failed (wrong key?)".into()))?;

            let json = String::from_utf8(plaintext)
                .map_err(|e| Error::Config(format!("vault plaintext is not valid UTF-8: {e}")))?;
            self.entries = serde_json::from_str(&json)
                .map_err(|e| Error::Config(format!("failed to parse vault entries: {e}")))?;
        }

        self.unlocked = true;
        Ok(())
    }

    /// Save the vault using `HEARTBIT_VAULT_KEY` from the environment.
    ///
    /// Convenience wrapper around [`save_with_key`](Self::save_with_key).
    pub fn save(&self) -> Result<(), Error> {
        let master_key = vault_key_from_env()?;
        self.save_with_key(&master_key)
    }

    /// Save (encrypt and write) the vault to disk with the given master key.
    pub fn save_with_key(&self, master_key: &str) -> Result<(), Error> {
        if !self.unlocked {
            return Err(Error::Config("vault must be unlocked before saving".into()));
        }

        let mut salt = [0u8; SALT_LEN];
        OsRng.fill_bytes(&mut salt);
        let mut nonce_bytes = [0u8; NONCE_LEN];
        OsRng.fill_bytes(&mut nonce_bytes);

        let key = derive_key(master_key, &salt)?;
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|e| Error::Config(format!("failed to create cipher: {e}")))?;
        let nonce = Nonce::from_slice(&nonce_bytes);

        let plaintext = serde_json::to_string(&self.entries)
            .map_err(|e| Error::Config(format!("failed to serialize entries: {e}")))?;
        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_bytes())
            .map_err(|e| Error::Config(format!("encryption failed: {e}")))?;

        let vault_file = VaultFile {
            version: 1,
            salt: base64_encode(&salt),
            nonce: base64_encode(&nonce_bytes),
            ciphertext: base64_encode(&ciphertext),
        };

        // Create parent directory if needed
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| Error::Config(format!("failed to create vault directory: {e}")))?;
        }

        let json = serde_json::to_string_pretty(&vault_file)
            .map_err(|e| Error::Config(format!("failed to serialize vault: {e}")))?;
        std::fs::write(&self.path, json)
            .map_err(|e| Error::Config(format!("failed to write vault: {e}")))?;

        Ok(())
    }

    /// Set a credential. Vault must be unlocked.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) -> Result<(), Error> {
        if !self.unlocked {
            return Err(Error::Config(
                "vault must be unlocked before setting".into(),
            ));
        }
        self.entries.insert(key.into(), value.into());
        Ok(())
    }

    /// Get a credential. Vault must be unlocked.
    pub fn get(&self, key: &str) -> Result<Option<&str>, Error> {
        if !self.unlocked {
            return Err(Error::Config(
                "vault must be unlocked before reading".into(),
            ));
        }
        Ok(self.entries.get(key).map(|s| s.as_str()))
    }

    /// Remove a credential. Returns the previous value if any.
    pub fn remove(&mut self, key: &str) -> Result<Option<String>, Error> {
        if !self.unlocked {
            return Err(Error::Config(
                "vault must be unlocked before removing".into(),
            ));
        }
        Ok(self.entries.remove(key))
    }

    /// List all credential keys. Vault must be unlocked.
    pub fn keys(&self) -> Result<Vec<&str>, Error> {
        if !self.unlocked {
            return Err(Error::Config(
                "vault must be unlocked before listing".into(),
            ));
        }
        let mut keys: Vec<&str> = self.entries.keys().map(|s| s.as_str()).collect();
        keys.sort();
        Ok(keys)
    }

    /// Number of stored credentials.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the vault is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Credential resolver chain: env var → vault.
///
/// First checks the environment variable, then falls back to the vault.
pub struct CredentialResolver {
    vault: Option<CredentialVault>,
}

impl CredentialResolver {
    /// Create a resolver with an optional vault.
    pub fn new(vault: Option<CredentialVault>) -> Self {
        Self { vault }
    }

    /// Create a resolver without a vault (env-only).
    pub fn env_only() -> Self {
        Self { vault: None }
    }

    /// Resolve a credential by name.
    ///
    /// Priority: env var → vault → None.
    pub fn resolve(&self, key: &str) -> Option<String> {
        // 1. Environment variable
        if let Ok(val) = std::env::var(key) {
            return Some(val);
        }
        // 2. Vault
        if let Some(ref vault) = self.vault
            && let Ok(Some(val)) = vault.get(key)
        {
            return Some(val.to_string());
        }
        None
    }
}

// --- Internal helpers ---

fn vault_key_from_env() -> Result<String, Error> {
    std::env::var("HEARTBIT_VAULT_KEY").map_err(|_| {
        Error::Config(
            "HEARTBIT_VAULT_KEY environment variable not set. \
             Set it to a strong passphrase to encrypt/decrypt the vault."
                .into(),
        )
    })
}

fn derive_key(master_key: &str, salt: &[u8]) -> Result<Vec<u8>, Error> {
    let mut key = vec![0u8; 32]; // AES-256 = 32 bytes
    Argon2::default()
        .hash_password_into(master_key.as_bytes(), salt, &mut key)
        .map_err(|e| Error::Config(format!("key derivation failed: {e}")))?;
    Ok(key)
}

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

fn base64_decode(data: &str) -> Result<Vec<u8>, Error> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| Error::Config(format!("base64 decode failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_KEY: &str = "test-passphrase-12345";

    #[test]
    fn new_vault_is_locked() {
        let vault = CredentialVault::new("/tmp/test-vault.enc");
        assert!(!vault.unlocked);
        assert!(vault.get("key").is_err());
    }

    #[test]
    fn set_requires_unlock() {
        let mut vault = CredentialVault::new("/tmp/test-vault.enc");
        assert!(vault.set("key", "value").is_err());
    }

    #[test]
    fn keys_requires_unlock() {
        let vault = CredentialVault::new("/tmp/test-vault.enc");
        assert!(vault.keys().is_err());
    }

    #[test]
    fn remove_requires_unlock() {
        let mut vault = CredentialVault::new("/tmp/test-vault.enc");
        assert!(vault.remove("key").is_err());
    }

    #[test]
    fn save_requires_unlock() {
        let vault = CredentialVault::new("/tmp/test-vault.enc");
        assert!(vault.save_with_key(TEST_KEY).is_err());
    }

    #[test]
    fn encrypt_decrypt_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vault.enc");

        // Create, unlock, set, save
        let mut vault = CredentialVault::new(&path);
        vault.unlock_with_key(TEST_KEY).unwrap();
        vault.set("API_KEY", "sk-secret-123").unwrap();
        vault.set("DB_PASSWORD", "p@ssw0rd").unwrap();
        vault.save_with_key(TEST_KEY).unwrap();

        // Reload from disk
        let mut vault2 = CredentialVault::new(&path);
        vault2.unlock_with_key(TEST_KEY).unwrap();
        assert_eq!(vault2.get("API_KEY").unwrap(), Some("sk-secret-123"));
        assert_eq!(vault2.get("DB_PASSWORD").unwrap(), Some("p@ssw0rd"));
        assert_eq!(vault2.get("NONEXISTENT").unwrap(), None);
    }

    #[test]
    fn wrong_key_fails_decrypt() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("vault.enc");

        let mut vault = CredentialVault::new(&path);
        vault.unlock_with_key("correct-key").unwrap();
        vault.set("SECRET", "value").unwrap();
        vault.save_with_key("correct-key").unwrap();

        let mut vault2 = CredentialVault::new(&path);
        let result = vault2.unlock_with_key("wrong-key");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("decryption failed")
        );
    }

    #[test]
    fn remove_entry() {
        let mut vault = CredentialVault::new("/nonexistent-path");
        vault.unlock_with_key(TEST_KEY).unwrap();
        vault.set("KEY1", "val1").unwrap();
        vault.set("KEY2", "val2").unwrap();

        let removed = vault.remove("KEY1").unwrap();
        assert_eq!(removed, Some("val1".into()));
        assert_eq!(vault.get("KEY1").unwrap(), None);
        assert_eq!(vault.get("KEY2").unwrap(), Some("val2"));
    }

    #[test]
    fn keys_sorted() {
        let mut vault = CredentialVault::new("/nonexistent-path");
        vault.unlock_with_key(TEST_KEY).unwrap();
        vault.set("ZZZ", "z").unwrap();
        vault.set("AAA", "a").unwrap();
        vault.set("MMM", "m").unwrap();

        let keys = vault.keys().unwrap();
        assert_eq!(keys, vec!["AAA", "MMM", "ZZZ"]);
    }

    #[test]
    fn len_and_is_empty() {
        let mut vault = CredentialVault::new("/nonexistent-path");
        assert_eq!(vault.len(), 0);
        assert!(vault.is_empty());

        vault.unlock_with_key(TEST_KEY).unwrap();
        vault.set("K", "V").unwrap();
        assert_eq!(vault.len(), 1);
        assert!(!vault.is_empty());
    }

    #[test]
    fn unlock_nonexistent_starts_empty() {
        let mut vault = CredentialVault::new("/nonexistent/path/vault.enc");
        vault.unlock_with_key(TEST_KEY).unwrap();
        assert!(vault.is_empty());
    }

    #[test]
    fn credential_resolver_missing() {
        if std::env::var("NONEXISTENT_CRED_KEY_12345").is_ok() {
            return;
        }
        let resolver = CredentialResolver::env_only();
        assert!(resolver.resolve("NONEXISTENT_CRED_KEY_12345").is_none());
    }

    #[test]
    fn credential_resolver_falls_back_to_vault() {
        if std::env::var("VAULT_ONLY_KEY_TEST").is_ok() {
            return;
        }
        let mut vault = CredentialVault::new("/nonexistent-path");
        vault.unlock_with_key(TEST_KEY).unwrap();
        vault.set("VAULT_ONLY_KEY_TEST", "from-vault").unwrap();

        let resolver = CredentialResolver::new(Some(vault));
        assert_eq!(
            resolver.resolve("VAULT_ONLY_KEY_TEST"),
            Some("from-vault".into())
        );
    }

    #[test]
    fn default_path_contains_heartbit() {
        if std::env::var("HOME").is_err() && std::env::var("USERPROFILE").is_err() {
            return;
        }
        let path = CredentialVault::default_path().unwrap();
        assert!(path.to_str().unwrap().contains(".heartbit"));
        assert!(path.to_str().unwrap().contains("vault.enc"));
    }

    #[test]
    fn vault_key_from_env_missing() {
        if std::env::var("HEARTBIT_VAULT_KEY").is_ok() {
            return;
        }
        let err = vault_key_from_env().unwrap_err();
        assert!(err.to_string().contains("HEARTBIT_VAULT_KEY"));
    }

    #[test]
    fn base64_round_trip() {
        let data = b"hello world \x00\xff";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }
}
