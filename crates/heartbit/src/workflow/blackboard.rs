use restate_sdk::prelude::*;

/// Restate virtual object for a shared blackboard between agents.
///
/// The object key is the workflow/task ID. Agents can read and write
/// key-value pairs during execution. State is durable — survives crashes.
///
/// Research basis: shared blackboard pattern improves multi-agent coordination
/// by 13-57% over master-slave (arXiv:2510.01285).
#[restate_sdk::object]
pub trait BlackboardObject {
    /// Write a value to the blackboard.
    async fn write(entry: Json<BlackboardEntry>) -> Result<(), HandlerError>;

    /// Read a value from the blackboard by key.
    #[shared]
    async fn read(key: String) -> Result<Json<Option<serde_json::Value>>, HandlerError>;

    /// List all keys on the blackboard.
    #[shared]
    async fn list_keys() -> Result<Json<Vec<String>>, HandlerError>;

    /// Clear all entries from the blackboard.
    async fn clear() -> Result<(), HandlerError>;
}

/// A key-value entry for the blackboard.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BlackboardEntry {
    pub key: String,
    pub value: serde_json::Value,
}

pub struct BlackboardObjectImpl;

impl BlackboardObject for BlackboardObjectImpl {
    async fn write(
        &self,
        ctx: ObjectContext<'_>,
        Json(entry): Json<BlackboardEntry>,
    ) -> Result<(), HandlerError> {
        // Store the value under a prefixed key
        let storage_key = format!("bb:{}", entry.key);
        ctx.set(&storage_key, entry.value.to_string());

        // Track the key in the key list
        let mut keys = get_key_list(&ctx).await?;
        if !keys.contains(&entry.key) {
            keys.push(entry.key);
            let keys_json = serde_json::to_string(&keys)
                .map_err(|e| TerminalError::new(format!("failed to serialize key list: {e}")))?;
            ctx.set("bb:__keys", keys_json);
        }

        Ok(())
    }

    async fn read(
        &self,
        ctx: SharedObjectContext<'_>,
        key: String,
    ) -> Result<Json<Option<serde_json::Value>>, HandlerError> {
        let storage_key = format!("bb:{key}");
        let value = ctx.get::<String>(&storage_key).await?;
        let parsed = value.and_then(|v| match serde_json::from_str(&v) {
            Ok(val) => Some(val),
            Err(e) => {
                tracing::warn!(key = %key, error = %e, "corrupt blackboard value");
                None
            }
        });
        Ok(Json(parsed))
    }

    async fn list_keys(
        &self,
        ctx: SharedObjectContext<'_>,
    ) -> Result<Json<Vec<String>>, HandlerError> {
        let keys = get_shared_key_list(&ctx).await?;
        Ok(Json(keys))
    }

    async fn clear(&self, ctx: ObjectContext<'_>) -> Result<(), HandlerError> {
        let keys = get_key_list(&ctx).await?;
        for key in &keys {
            let storage_key = format!("bb:{key}");
            ctx.clear(&storage_key);
        }
        ctx.clear("bb:__keys");
        Ok(())
    }
}

async fn get_key_list(ctx: &ObjectContext<'_>) -> Result<Vec<String>, HandlerError> {
    let raw = ctx.get::<String>("bb:__keys").await?;
    Ok(parse_key_list(raw))
}

async fn get_shared_key_list(ctx: &SharedObjectContext<'_>) -> Result<Vec<String>, HandlerError> {
    let raw = ctx.get::<String>("bb:__keys").await?;
    Ok(parse_key_list(raw))
}

fn parse_key_list(raw: Option<String>) -> Vec<String> {
    match raw {
        None => vec![],
        Some(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "corrupt blackboard key list, resetting");
            vec![]
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blackboard_entry_roundtrips() {
        let entry = BlackboardEntry {
            key: "findings".into(),
            value: serde_json::json!({"summary": "Rust is fast"}),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: BlackboardEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.key, "findings");
        assert_eq!(parsed.value["summary"], "Rust is fast");
    }
}
