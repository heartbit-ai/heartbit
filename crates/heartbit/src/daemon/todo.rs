use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::RwLock;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use uuid::Uuid;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::builtins::{TodoPriority, TodoStatus};
use crate::tool::{Tool, ToolOutput};

/// A persistent todo entry for the daemon's task list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoEntry {
    pub id: Uuid,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub status: TodoStatus,
    pub priority: TodoPriority,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub due_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub attempt_count: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub daemon_task_id: Option<Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snoozed_until: Option<DateTime<Utc>>,
}

impl TodoEntry {
    pub fn new(title: impl Into<String>, source: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Uuid::new_v4(),
            title: title.into(),
            description: None,
            status: TodoStatus::Pending,
            priority: TodoPriority::Medium,
            created_at: now,
            updated_at: now,
            source: source.into(),
            due_at: None,
            result: None,
            tags: Vec::new(),
            attempt_count: 0,
            daemon_task_id: None,
            snoozed_until: None,
        }
    }
}

/// The persistent todo list with review tracking.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TodoList {
    pub entries: Vec<TodoEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_reviewed_at: Option<DateTime<Utc>>,
}

/// File-backed persistent todo store.
///
/// Stores entries as `{workspace}/TODO.json`. Uses `std::sync::RwLock`
/// (not tokio) because locks are never held across `.await` points.
/// Same pattern as `FileTracker`.
pub struct FileTodoStore {
    path: PathBuf,
    cache: RwLock<TodoList>,
}

impl FileTodoStore {
    /// Create a new store at the given workspace path.
    pub fn new(workspace: &Path) -> Result<Self, Error> {
        let path = workspace.join("TODO.json");
        let list = if path.exists() {
            let content = std::fs::read_to_string(&path)
                .map_err(|e| Error::Daemon(format!("failed to read {}: {e}", path.display())))?;
            if content.trim().is_empty() {
                TodoList::default()
            } else {
                serde_json::from_str(&content).map_err(|e| {
                    Error::Daemon(format!("failed to parse {}: {e}", path.display()))
                })?
            }
        } else {
            TodoList::default()
        };
        Ok(Self {
            path,
            cache: RwLock::new(list),
        })
    }

    /// Persist the current state to disk.
    fn save(&self, list: &TodoList) -> Result<(), Error> {
        let json = serde_json::to_string_pretty(list)
            .map_err(|e| Error::Daemon(format!("failed to serialize todo list: {e}")))?;
        // Write atomically: write to tmp then rename
        let tmp = self.path.with_extension("json.tmp");
        std::fs::write(&tmp, json.as_bytes())
            .map_err(|e| Error::Daemon(format!("failed to write {}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, &self.path).map_err(|e| {
            Error::Daemon(format!(
                "failed to rename {} -> {}: {e}",
                tmp.display(),
                self.path.display()
            ))
        })?;
        Ok(())
    }

    /// Get a snapshot of the current todo list.
    pub fn get_list(&self) -> TodoList {
        self.cache.read().expect("todo store lock poisoned").clone()
    }

    /// Add a new entry. Returns its ID.
    pub fn add(&self, entry: TodoEntry) -> Result<Uuid, Error> {
        let id = entry.id;
        let mut list = self.cache.write().expect("todo store lock poisoned");
        list.entries.push(entry);
        self.save(&list)?;
        Ok(id)
    }

    /// Update an entry by ID. Returns `Err` if not found.
    pub fn update(&self, id: Uuid, updater: impl FnOnce(&mut TodoEntry)) -> Result<(), Error> {
        let mut list = self.cache.write().expect("todo store lock poisoned");
        let entry = list
            .entries
            .iter_mut()
            .find(|e| e.id == id)
            .ok_or_else(|| Error::Daemon(format!("todo entry {id} not found")))?;
        updater(entry);
        entry.updated_at = Utc::now();
        self.save(&list)?;
        Ok(())
    }

    /// Remove an entry by ID. Returns `Err` if not found.
    pub fn remove(&self, id: Uuid) -> Result<(), Error> {
        let mut list = self.cache.write().expect("todo store lock poisoned");
        let len_before = list.entries.len();
        list.entries.retain(|e| e.id != id);
        if list.entries.len() == len_before {
            return Err(Error::Daemon(format!("todo entry {id} not found")));
        }
        self.save(&list)?;
        Ok(())
    }

    /// Mark the list as reviewed now.
    pub fn mark_reviewed(&self) -> Result<(), Error> {
        let mut list = self.cache.write().expect("todo store lock poisoned");
        list.last_reviewed_at = Some(Utc::now());
        self.save(&list)?;
        Ok(())
    }

    /// Format the todo list for inclusion in prompts.
    pub fn format_for_prompt(&self) -> String {
        let list = self.cache.read().expect("todo store lock poisoned");
        if list.entries.is_empty() {
            return "No items in the todo list.".into();
        }

        let mut out = String::new();
        for entry in &list.entries {
            let status_icon = match entry.status {
                TodoStatus::Pending => "[ ]",
                TodoStatus::InProgress => "[>]",
                TodoStatus::Completed => "[x]",
                TodoStatus::Cancelled => "[-]",
                TodoStatus::Failed => "[!]",
                TodoStatus::Blocked => "[B]",
            };
            let age = Utc::now().signed_duration_since(entry.created_at);
            let age_str = if age.num_days() > 0 {
                format!("{}d", age.num_days())
            } else if age.num_hours() > 0 {
                format!("{}h", age.num_hours())
            } else {
                format!("{}m", age.num_minutes())
            };
            out.push_str(&format!(
                "- {status_icon} [{priority}] {title} (id: {id}, age: {age}, src: {src}",
                priority = entry.priority,
                title = entry.title,
                id = entry.id,
                age = age_str,
                src = entry.source,
            ));
            if let Some(ref due) = entry.due_at {
                out.push_str(&format!(", due: {}", due.format("%Y-%m-%d %H:%M")));
            }
            if entry.attempt_count > 0 {
                out.push_str(&format!(", attempts: {}", entry.attempt_count));
            }
            if !entry.tags.is_empty() {
                out.push_str(&format!(", tags: {}", entry.tags.join(",")));
            }
            out.push_str(")\n");
            if let Some(ref desc) = entry.description {
                out.push_str(&format!("  {desc}\n"));
            }
        }

        if let Some(reviewed) = list.last_reviewed_at {
            let ago = Utc::now().signed_duration_since(reviewed);
            out.push_str(&format!("\nLast reviewed: {}m ago\n", ago.num_minutes()));
        }

        out
    }

    /// Format the todo list for the heartbit pulse prompt.
    ///
    /// Unlike [`format_for_prompt`], this filters out noise:
    /// - **Actionable** items (Pending/InProgress/Failed/Blocked, not snoozed): full detail
    /// - **Snoozed** items: one-line count
    /// - **Terminal** items (Completed/Cancelled): one-line count
    pub fn format_for_pulse_prompt(&self) -> String {
        let list = self.cache.read().expect("todo store lock poisoned");
        if list.entries.is_empty() {
            return "No items in the todo list.".into();
        }

        let now = Utc::now();
        let mut actionable = Vec::new();
        let mut snoozed_count: usize = 0;
        let mut completed_count: usize = 0;
        let mut cancelled_count: usize = 0;

        for entry in &list.entries {
            match entry.status {
                TodoStatus::Completed => {
                    completed_count += 1;
                    continue;
                }
                TodoStatus::Cancelled => {
                    cancelled_count += 1;
                    continue;
                }
                _ => {}
            }
            // Check if snoozed (future snooze = excluded, expired = actionable)
            if let Some(until) = entry.snoozed_until
                && until > now
            {
                snoozed_count += 1;
                continue;
            }
            actionable.push(entry);
        }

        let mut out = String::new();

        if actionable.is_empty() {
            out.push_str("No actionable items.\n");
        } else {
            for entry in &actionable {
                let status_icon = match entry.status {
                    TodoStatus::Pending => "[ ]",
                    TodoStatus::InProgress => "[>]",
                    TodoStatus::Completed => "[x]",
                    TodoStatus::Cancelled => "[-]",
                    TodoStatus::Failed => "[!]",
                    TodoStatus::Blocked => "[B]",
                };
                let age = now.signed_duration_since(entry.created_at);
                let age_str = if age.num_days() > 0 {
                    format!("{}d", age.num_days())
                } else if age.num_hours() > 0 {
                    format!("{}h", age.num_hours())
                } else {
                    format!("{}m", age.num_minutes())
                };
                out.push_str(&format!(
                    "- {status_icon} [{priority}] {title} (id: {id}, age: {age}, src: {src}",
                    priority = entry.priority,
                    title = entry.title,
                    id = entry.id,
                    age = age_str,
                    src = entry.source,
                ));
                if let Some(ref due) = entry.due_at {
                    out.push_str(&format!(", due: {}", due.format("%Y-%m-%d %H:%M")));
                }
                if entry.attempt_count > 0 {
                    out.push_str(&format!(", attempts: {}", entry.attempt_count));
                }
                if !entry.tags.is_empty() {
                    out.push_str(&format!(", tags: {}", entry.tags.join(",")));
                }
                out.push_str(")\n");
                if let Some(ref desc) = entry.description {
                    out.push_str(&format!("  {desc}\n"));
                }
            }
        }

        // Summary lines for filtered items
        let mut summary_parts = Vec::new();
        if completed_count > 0 {
            summary_parts.push(format!("{completed_count} completed"));
        }
        if cancelled_count > 0 {
            summary_parts.push(format!("{cancelled_count} cancelled"));
        }
        if !summary_parts.is_empty() {
            out.push_str(&format!(
                "\n{} — use todo_manage list for details\n",
                summary_parts.join(", ")
            ));
        }
        if snoozed_count > 0 {
            out.push_str(&format!("{snoozed_count} items snoozed\n"));
        }

        if let Some(reviewed) = list.last_reviewed_at {
            let ago = now.signed_duration_since(reviewed);
            out.push_str(&format!("\nLast reviewed: {}m ago\n", ago.num_minutes()));
        }

        out
    }

    /// Returns `true` if the given entry is actionable (non-terminal, non-snoozed).
    pub fn is_actionable(entry: &TodoEntry) -> bool {
        match entry.status {
            TodoStatus::Completed | TodoStatus::Cancelled => false,
            _ => {
                if let Some(until) = entry.snoozed_until {
                    until <= Utc::now()
                } else {
                    true
                }
            }
        }
    }

    /// Path to the backing JSON file.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Multi-action tool for managing the persistent daemon todo list.
pub struct TodoManageTool {
    store: std::sync::Arc<FileTodoStore>,
}

impl TodoManageTool {
    pub fn new(store: std::sync::Arc<FileTodoStore>) -> Self {
        Self { store }
    }
}

impl Tool for TodoManageTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "todo_manage".into(),
            description: "Manage the persistent todo list. Actions: add (create new), \
                          update (modify existing by id), remove (delete by id), \
                          snooze (suppress item for N hours), \
                          list (show all entries). The todo list persists across sessions."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["add", "update", "remove", "snooze", "list"],
                        "description": "The action to perform"
                    },
                    "id": {
                        "type": "string",
                        "description": "UUID of the entry (required for update/remove/snooze)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Title of the todo entry (required for add)"
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["critical", "high", "medium", "low"],
                        "description": "Priority level (default: medium)"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "cancelled", "failed", "blocked"],
                        "description": "Status (for update)"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Tags for categorization"
                    },
                    "due_at": {
                        "type": "string",
                        "description": "Due date in ISO 8601 format"
                    },
                    "result": {
                        "type": "string",
                        "description": "Result text (for update on completion)"
                    },
                    "source": {
                        "type": "string",
                        "description": "Source of the entry (default: self)"
                    },
                    "hours": {
                        "type": "number",
                        "description": "Hours to snooze (for snooze action, default: 24)"
                    },
                    "snoozed_until": {
                        "type": ["string", "null"],
                        "description": "ISO 8601 datetime to snooze until (for update), or null to clear snooze"
                    }
                },
                "required": ["action"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let action = input
                .get("action")
                .and_then(|v| v.as_str())
                .ok_or_else(|| Error::Agent("action is required".into()))?;

            match action {
                "add" => {
                    let title = input
                        .get("title")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::Agent("title is required for add".into()))?;

                    let mut entry = TodoEntry::new(
                        title,
                        input
                            .get("source")
                            .and_then(|v| v.as_str())
                            .unwrap_or("self"),
                    );

                    if let Some(desc) = input.get("description").and_then(|v| v.as_str()) {
                        entry.description = Some(desc.into());
                    }
                    if let Some(priority) = input.get("priority").and_then(|v| v.as_str()) {
                        entry.priority = serde_json::from_value(json!(priority))
                            .map_err(|e| Error::Agent(format!("invalid priority: {e}")))?;
                    }
                    if let Some(tags) = input.get("tags").and_then(|v| v.as_array()) {
                        entry.tags = tags
                            .iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect();
                    }
                    if let Some(due) = input.get("due_at").and_then(|v| v.as_str()) {
                        entry.due_at = Some(
                            due.parse::<DateTime<Utc>>()
                                .map_err(|e| Error::Agent(format!("invalid due_at: {e}")))?,
                        );
                    }

                    let id = self.store.add(entry)?;
                    Ok(ToolOutput::success(format!("Added todo entry {id}")))
                }
                "update" => {
                    let id_str = input
                        .get("id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::Agent("id is required for update".into()))?;
                    let id: Uuid = id_str
                        .parse()
                        .map_err(|e| Error::Agent(format!("invalid id: {e}")))?;

                    let title = input
                        .get("title")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    let description = input
                        .get("description")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    // Validate status/priority/due_at before the closure (which can't return errors)
                    let status: Option<TodoStatus> =
                        match input.get("status").and_then(|v| v.as_str()) {
                            Some(s) => Some(
                                serde_json::from_value(json!(s))
                                    .map_err(|e| Error::Agent(format!("invalid status: {e}")))?,
                            ),
                            None => None,
                        };
                    let priority: Option<TodoPriority> =
                        match input.get("priority").and_then(|v| v.as_str()) {
                            Some(p) => Some(
                                serde_json::from_value(json!(p))
                                    .map_err(|e| Error::Agent(format!("invalid priority: {e}")))?,
                            ),
                            None => None,
                        };
                    let tags: Option<Vec<String>> =
                        input.get("tags").and_then(|v| v.as_array()).map(|arr| {
                            arr.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        });
                    let due_at: Option<DateTime<Utc>> =
                        match input.get("due_at").and_then(|v| v.as_str()) {
                            Some(d) => Some(
                                d.parse::<DateTime<Utc>>()
                                    .map_err(|e| Error::Agent(format!("invalid due_at: {e}")))?,
                            ),
                            None => None,
                        };
                    let result_text = input
                        .get("result")
                        .and_then(|v| v.as_str())
                        .map(String::from);
                    // snoozed_until: string → set, null → clear, absent → no change
                    let snoozed_until_action = if let Some(val) = input.get("snoozed_until") {
                        if val.is_null() {
                            Some(None) // explicit null → clear
                        } else if let Some(s) = val.as_str() {
                            let dt = s
                                .parse::<DateTime<Utc>>()
                                .map_err(|e| Error::Agent(format!("invalid snoozed_until: {e}")))?;
                            Some(Some(dt)) // string → set
                        } else {
                            None // unexpected type → ignore
                        }
                    } else {
                        None // absent → no change
                    };

                    self.store.update(id, |entry| {
                        if let Some(ref t) = title {
                            entry.title = t.clone();
                        }
                        if let Some(ref d) = description {
                            entry.description = Some(d.clone());
                        }
                        if let Some(s) = status {
                            entry.status = s;
                        }
                        if let Some(p) = priority {
                            entry.priority = p;
                        }
                        if let Some(ref t) = tags {
                            entry.tags = t.clone();
                        }
                        if let Some(dt) = due_at {
                            entry.due_at = Some(dt);
                        }
                        if let Some(ref r) = result_text {
                            entry.result = Some(r.clone());
                        }
                        if let Some(snooze_val) = snoozed_until_action {
                            entry.snoozed_until = snooze_val;
                        }
                    })?;

                    Ok(ToolOutput::success(format!("Updated todo entry {id}")))
                }
                "snooze" => {
                    let id_str = input
                        .get("id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::Agent("id is required for snooze".into()))?;
                    let id: Uuid = id_str
                        .parse()
                        .map_err(|e| Error::Agent(format!("invalid id: {e}")))?;
                    let hours = input.get("hours").and_then(|v| v.as_f64()).unwrap_or(24.0);
                    if hours <= 0.0 {
                        return Ok(ToolOutput::error("hours must be positive".to_string()));
                    }
                    let until = Utc::now() + chrono::Duration::seconds((hours * 3600.0) as i64);

                    self.store.update(id, |entry| {
                        entry.snoozed_until = Some(until);
                    })?;

                    Ok(ToolOutput::success(format!(
                        "Snoozed todo entry {id} until {until}"
                    )))
                }
                "remove" => {
                    let id_str = input
                        .get("id")
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| Error::Agent("id is required for remove".into()))?;
                    let id: Uuid = id_str
                        .parse()
                        .map_err(|e| Error::Agent(format!("invalid id: {e}")))?;
                    self.store.remove(id)?;
                    Ok(ToolOutput::success(format!("Removed todo entry {id}")))
                }
                "list" => {
                    let formatted = self.store.format_for_prompt();
                    Ok(ToolOutput::success(formatted))
                }
                _ => Ok(ToolOutput::error(format!(
                    "Unknown action '{action}'. Use: add, update, remove, snooze, list"
                ))),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todo_entry_serde_roundtrip() {
        let entry = TodoEntry::new("Test task", "heartbit");
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: TodoEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, entry.id);
        assert_eq!(parsed.title, "Test task");
        assert_eq!(parsed.source, "heartbit");
        assert_eq!(parsed.status, TodoStatus::Pending);
        assert_eq!(parsed.priority, TodoPriority::Medium);
    }

    #[test]
    fn todo_entry_all_statuses_roundtrip() {
        let statuses: Vec<TodoStatus> = vec![
            TodoStatus::Pending,
            TodoStatus::InProgress,
            TodoStatus::Completed,
            TodoStatus::Cancelled,
            TodoStatus::Failed,
            TodoStatus::Blocked,
        ];
        for status in statuses {
            let mut entry = TodoEntry::new("test", "test");
            entry.status = status.clone();
            let json = serde_json::to_string(&entry).unwrap();
            let parsed: TodoEntry = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.status, status);
        }
    }

    #[test]
    fn todo_entry_all_priorities_roundtrip() {
        let priorities: Vec<TodoPriority> = vec![
            TodoPriority::Critical,
            TodoPriority::High,
            TodoPriority::Medium,
            TodoPriority::Low,
        ];
        for priority in priorities {
            let mut entry = TodoEntry::new("test", "test");
            entry.priority = priority.clone();
            let json = serde_json::to_string(&entry).unwrap();
            let parsed: TodoEntry = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed.priority, priority);
        }
    }

    #[test]
    fn todo_list_default_empty() {
        let list = TodoList::default();
        assert!(list.entries.is_empty());
        assert!(list.last_reviewed_at.is_none());
    }

    #[test]
    fn todo_list_serde_roundtrip() {
        let mut list = TodoList::default();
        list.entries.push(TodoEntry::new("Task 1", "user"));
        list.last_reviewed_at = Some(Utc::now());
        let json = serde_json::to_string(&list).unwrap();
        let parsed: TodoList = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.entries.len(), 1);
        assert!(parsed.last_reviewed_at.is_some());
    }

    #[test]
    fn todo_entry_optional_fields_default() {
        // Deserialize minimal JSON — optional fields should default
        let json = r#"{
            "id": "00000000-0000-0000-0000-000000000000",
            "title": "test",
            "status": "pending",
            "priority": "medium",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "source": "test"
        }"#;
        let entry: TodoEntry = serde_json::from_str(json).unwrap();
        assert!(entry.description.is_none());
        assert!(entry.due_at.is_none());
        assert!(entry.result.is_none());
        assert!(entry.tags.is_empty());
        assert_eq!(entry.attempt_count, 0);
        assert!(entry.daemon_task_id.is_none());
    }

    #[test]
    fn file_todo_store_new_creates_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();
        let list = store.get_list();
        assert!(list.entries.is_empty());
    }

    #[test]
    fn file_todo_store_add_persists() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();

        let entry = TodoEntry::new("Test task", "user");
        let id = store.add(entry).unwrap();

        // Reload from disk
        let store2 = FileTodoStore::new(dir.path()).unwrap();
        let list = store2.get_list();
        assert_eq!(list.entries.len(), 1);
        assert_eq!(list.entries[0].id, id);
        assert_eq!(list.entries[0].title, "Test task");
    }

    #[test]
    fn file_todo_store_update() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();

        let entry = TodoEntry::new("Original", "user");
        let id = store.add(entry).unwrap();

        store
            .update(id, |e| {
                e.title = "Updated".into();
                e.status = TodoStatus::InProgress;
            })
            .unwrap();

        let list = store.get_list();
        assert_eq!(list.entries[0].title, "Updated");
        assert_eq!(list.entries[0].status, TodoStatus::InProgress);
    }

    #[test]
    fn file_todo_store_update_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();
        let err = store.update(Uuid::new_v4(), |_| {}).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn file_todo_store_remove() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();

        let entry = TodoEntry::new("To remove", "user");
        let id = store.add(entry).unwrap();
        assert_eq!(store.get_list().entries.len(), 1);

        store.remove(id).unwrap();
        assert!(store.get_list().entries.is_empty());

        // Verify persisted
        let store2 = FileTodoStore::new(dir.path()).unwrap();
        assert!(store2.get_list().entries.is_empty());
    }

    #[test]
    fn file_todo_store_remove_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();
        let err = store.remove(Uuid::new_v4()).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn file_todo_store_mark_reviewed() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();

        assert!(store.get_list().last_reviewed_at.is_none());
        store.mark_reviewed().unwrap();
        assert!(store.get_list().last_reviewed_at.is_some());
    }

    #[test]
    fn file_todo_store_loads_existing_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("TODO.json");

        // Write a valid JSON file
        let list = TodoList {
            entries: vec![TodoEntry::new("Existing", "user")],
            last_reviewed_at: None,
        };
        let json = serde_json::to_string_pretty(&list).unwrap();
        std::fs::write(&path, json).unwrap();

        let store = FileTodoStore::new(dir.path()).unwrap();
        assert_eq!(store.get_list().entries.len(), 1);
        assert_eq!(store.get_list().entries[0].title, "Existing");
    }

    #[test]
    fn file_todo_store_handles_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("TODO.json");
        std::fs::write(&path, "").unwrap();

        let store = FileTodoStore::new(dir.path()).unwrap();
        assert!(store.get_list().entries.is_empty());
    }

    #[test]
    fn file_todo_store_format_for_prompt_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();
        assert_eq!(store.format_for_prompt(), "No items in the todo list.");
    }

    #[test]
    fn file_todo_store_format_for_prompt_with_entries() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();

        let mut entry = TodoEntry::new("Fix bug", "user");
        entry.priority = TodoPriority::High;
        entry.tags = vec!["urgent".into()];
        store.add(entry).unwrap();

        let prompt = store.format_for_prompt();
        assert!(prompt.contains("Fix bug"));
        assert!(prompt.contains("[high]"));
        assert!(prompt.contains("tags: urgent"));
    }

    #[tokio::test]
    async fn todo_manage_tool_add() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        let result = tool
            .execute(json!({
                "action": "add",
                "title": "New task",
                "priority": "high",
                "tags": ["test"]
            }))
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("Added todo entry"));

        let list = store.get_list();
        assert_eq!(list.entries.len(), 1);
        assert_eq!(list.entries[0].title, "New task");
        assert_eq!(list.entries[0].priority, TodoPriority::High);
        assert_eq!(list.entries[0].source, "self");
    }

    #[tokio::test]
    async fn todo_manage_tool_add_requires_title() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store);

        let result = tool.execute(json!({"action": "add"})).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn todo_manage_tool_update() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        // Add first
        let entry = TodoEntry::new("Original", "user");
        let id = store.add(entry).unwrap();

        // Update via tool
        let result = tool
            .execute(json!({
                "action": "update",
                "id": id.to_string(),
                "title": "Updated",
                "status": "in_progress"
            }))
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);

        let list = store.get_list();
        assert_eq!(list.entries[0].title, "Updated");
        assert_eq!(list.entries[0].status, TodoStatus::InProgress);
    }

    #[tokio::test]
    async fn todo_manage_tool_remove() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        let entry = TodoEntry::new("To remove", "user");
        let id = store.add(entry).unwrap();

        let result = tool
            .execute(json!({
                "action": "remove",
                "id": id.to_string()
            }))
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(store.get_list().entries.is_empty());
    }

    #[tokio::test]
    async fn todo_manage_tool_list() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        store.add(TodoEntry::new("Task 1", "user")).unwrap();
        store.add(TodoEntry::new("Task 2", "user")).unwrap();

        let result = tool.execute(json!({"action": "list"})).await.unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Task 1"));
        assert!(result.content.contains("Task 2"));
    }

    #[tokio::test]
    async fn todo_manage_tool_unknown_action() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store);

        let result = tool.execute(json!({"action": "invalid"})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Unknown action"));
    }

    #[tokio::test]
    async fn todo_manage_tool_update_rejects_invalid_status() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        let entry = TodoEntry::new("Test", "user");
        let id = store.add(entry).unwrap();

        let result = tool
            .execute(json!({
                "action": "update",
                "id": id.to_string(),
                "status": "bogus"
            }))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid status"));
    }

    #[tokio::test]
    async fn todo_manage_tool_update_rejects_invalid_priority() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        let entry = TodoEntry::new("Test", "user");
        let id = store.add(entry).unwrap();

        let result = tool
            .execute(json!({
                "action": "update",
                "id": id.to_string(),
                "priority": "bogus"
            }))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid priority"));
    }

    #[tokio::test]
    async fn todo_manage_tool_update_rejects_invalid_due_at() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        let entry = TodoEntry::new("Test", "user");
        let id = store.add(entry).unwrap();

        let result = tool
            .execute(json!({
                "action": "update",
                "id": id.to_string(),
                "due_at": "not-a-date"
            }))
            .await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("invalid due_at"));
    }

    #[test]
    fn todo_manage_tool_definition() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store);
        let def = tool.definition();
        assert_eq!(def.name, "todo_manage");
        assert!(def.description.contains("persistent"));
        assert!(def.description.contains("snooze"));
    }

    // --- snoozed_until field tests ---

    #[test]
    fn todo_entry_snoozed_until_default_none() {
        // Backwards compat: minimal JSON without snoozed_until → None
        let json = r#"{
            "id": "00000000-0000-0000-0000-000000000000",
            "title": "test",
            "status": "pending",
            "priority": "medium",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "source": "test"
        }"#;
        let entry: TodoEntry = serde_json::from_str(json).unwrap();
        assert!(entry.snoozed_until.is_none());
    }

    #[test]
    fn todo_entry_snoozed_until_roundtrip() {
        let mut entry = TodoEntry::new("test", "test");
        let snooze_time = Utc::now() + chrono::Duration::hours(24);
        entry.snoozed_until = Some(snooze_time);
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: TodoEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.snoozed_until.unwrap().timestamp(),
            snooze_time.timestamp()
        );
    }

    // --- format_for_pulse_prompt tests ---

    #[test]
    fn format_for_pulse_prompt_empty() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();
        assert_eq!(
            store.format_for_pulse_prompt(),
            "No items in the todo list."
        );
    }

    #[test]
    fn format_for_pulse_prompt_excludes_completed() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();
        let mut entry = TodoEntry::new("Done task", "user");
        entry.status = TodoStatus::Completed;
        store.add(entry).unwrap();

        let mut pending = TodoEntry::new("Open task", "user");
        pending.status = TodoStatus::Pending;
        store.add(pending).unwrap();

        let prompt = store.format_for_pulse_prompt();
        assert!(prompt.contains("Open task"));
        assert!(!prompt.contains("Done task"));
        assert!(prompt.contains("1 completed"));
    }

    #[test]
    fn format_for_pulse_prompt_excludes_cancelled() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();
        let mut entry = TodoEntry::new("Cancelled task", "user");
        entry.status = TodoStatus::Cancelled;
        store.add(entry).unwrap();

        let prompt = store.format_for_pulse_prompt();
        assert!(!prompt.contains("Cancelled task"));
        assert!(prompt.contains("1 cancelled"));
    }

    #[test]
    fn format_for_pulse_prompt_all_terminal() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();
        let mut c1 = TodoEntry::new("Done 1", "user");
        c1.status = TodoStatus::Completed;
        store.add(c1).unwrap();
        let mut c2 = TodoEntry::new("Done 2", "user");
        c2.status = TodoStatus::Completed;
        store.add(c2).unwrap();

        let prompt = store.format_for_pulse_prompt();
        assert!(prompt.contains("No actionable items."));
        assert!(prompt.contains("2 completed"));
    }

    #[test]
    fn format_for_pulse_prompt_excludes_snoozed() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();
        let mut snoozed = TodoEntry::new("Snoozed task", "user");
        snoozed.snoozed_until = Some(Utc::now() + chrono::Duration::hours(24));
        store.add(snoozed).unwrap();

        let prompt = store.format_for_pulse_prompt();
        assert!(!prompt.contains("Snoozed task"));
        assert!(prompt.contains("1 items snoozed"));
    }

    #[test]
    fn format_for_pulse_prompt_includes_expired_snooze() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();
        let mut expired = TodoEntry::new("Expired snooze", "user");
        expired.snoozed_until = Some(Utc::now() - chrono::Duration::hours(1));
        store.add(expired).unwrap();

        let prompt = store.format_for_pulse_prompt();
        assert!(prompt.contains("Expired snooze"));
        assert!(!prompt.contains("snoozed"));
    }

    #[test]
    fn format_for_pulse_prompt_mixed() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileTodoStore::new(dir.path()).unwrap();

        // Pending (actionable)
        store.add(TodoEntry::new("Pending task", "user")).unwrap();

        // InProgress (actionable)
        let mut ip = TodoEntry::new("In progress task", "user");
        ip.status = TodoStatus::InProgress;
        store.add(ip).unwrap();

        // Completed (terminal)
        let mut done = TodoEntry::new("Completed task", "user");
        done.status = TodoStatus::Completed;
        store.add(done).unwrap();

        // Snoozed (filtered)
        let mut snoozed = TodoEntry::new("Snoozed task", "user");
        snoozed.snoozed_until = Some(Utc::now() + chrono::Duration::hours(24));
        store.add(snoozed).unwrap();

        let prompt = store.format_for_pulse_prompt();
        assert!(prompt.contains("Pending task"));
        assert!(prompt.contains("In progress task"));
        assert!(!prompt.contains("Completed task"));
        assert!(!prompt.contains("Snoozed task"));
        assert!(prompt.contains("1 completed"));
        assert!(prompt.contains("1 items snoozed"));
    }

    // --- snooze action tests ---

    #[tokio::test]
    async fn todo_manage_tool_snooze_action() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        let entry = TodoEntry::new("Task to snooze", "user");
        let id = store.add(entry).unwrap();

        let result = tool
            .execute(json!({
                "action": "snooze",
                "id": id.to_string()
            }))
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("Snoozed"));

        let list = store.get_list();
        let snoozed = list.entries[0].snoozed_until.unwrap();
        // Default 24h: should be ~24h from now
        let diff = snoozed.signed_duration_since(Utc::now());
        assert!(diff.num_hours() >= 23 && diff.num_hours() <= 24);
    }

    #[tokio::test]
    async fn todo_manage_tool_snooze_custom_hours() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        let entry = TodoEntry::new("Task to snooze", "user");
        let id = store.add(entry).unwrap();

        let result = tool
            .execute(json!({
                "action": "snooze",
                "id": id.to_string(),
                "hours": 48
            }))
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);

        let list = store.get_list();
        let snoozed = list.entries[0].snoozed_until.unwrap();
        let diff = snoozed.signed_duration_since(Utc::now());
        assert!(diff.num_hours() >= 47 && diff.num_hours() <= 48);
    }

    #[tokio::test]
    async fn todo_manage_tool_snooze_requires_id() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store);

        let result = tool.execute(json!({"action": "snooze"})).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("id is required"));
    }

    #[tokio::test]
    async fn todo_manage_tool_snooze_rejects_negative_hours() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        let entry = TodoEntry::new("Task", "user");
        let id = store.add(entry).unwrap();

        let result = tool
            .execute(json!({
                "action": "snooze",
                "id": id.to_string(),
                "hours": -5
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("hours must be positive"));
    }

    #[tokio::test]
    async fn todo_manage_tool_snooze_rejects_zero_hours() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        let entry = TodoEntry::new("Task", "user");
        let id = store.add(entry).unwrap();

        let result = tool
            .execute(json!({
                "action": "snooze",
                "id": id.to_string(),
                "hours": 0
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("hours must be positive"));
    }

    #[tokio::test]
    async fn todo_manage_tool_update_snoozed_until() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        let entry = TodoEntry::new("Task", "user");
        let id = store.add(entry).unwrap();

        let snooze_time = "2026-12-31T12:00:00Z";
        let result = tool
            .execute(json!({
                "action": "update",
                "id": id.to_string(),
                "snoozed_until": snooze_time
            }))
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);

        let list = store.get_list();
        let snoozed = list.entries[0].snoozed_until.unwrap();
        assert_eq!(snoozed, snooze_time.parse::<DateTime<Utc>>().unwrap());
    }

    #[tokio::test]
    async fn todo_manage_tool_update_clear_snoozed_until() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(FileTodoStore::new(dir.path()).unwrap());
        let tool = TodoManageTool::new(store.clone());

        // Add an entry with a snooze
        let mut entry = TodoEntry::new("Task", "user");
        entry.snoozed_until = Some(Utc::now() + chrono::Duration::hours(24));
        let id = store.add(entry).unwrap();

        // Clear via null
        let result = tool
            .execute(json!({
                "action": "update",
                "id": id.to_string(),
                "snoozed_until": null
            }))
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);

        let list = store.get_list();
        assert!(list.entries[0].snoozed_until.is_none());
    }

    // --- is_actionable tests ---

    #[test]
    fn is_actionable_pending() {
        let entry = TodoEntry::new("test", "test");
        assert!(FileTodoStore::is_actionable(&entry));
    }

    #[test]
    fn is_actionable_completed_false() {
        let mut entry = TodoEntry::new("test", "test");
        entry.status = TodoStatus::Completed;
        assert!(!FileTodoStore::is_actionable(&entry));
    }

    #[test]
    fn is_actionable_cancelled_false() {
        let mut entry = TodoEntry::new("test", "test");
        entry.status = TodoStatus::Cancelled;
        assert!(!FileTodoStore::is_actionable(&entry));
    }

    #[test]
    fn is_actionable_snoozed_future_false() {
        let mut entry = TodoEntry::new("test", "test");
        entry.snoozed_until = Some(Utc::now() + chrono::Duration::hours(24));
        assert!(!FileTodoStore::is_actionable(&entry));
    }

    #[test]
    fn is_actionable_snoozed_expired_true() {
        let mut entry = TodoEntry::new("test", "test");
        entry.snoozed_until = Some(Utc::now() - chrono::Duration::hours(1));
        assert!(FileTodoStore::is_actionable(&entry));
    }
}
