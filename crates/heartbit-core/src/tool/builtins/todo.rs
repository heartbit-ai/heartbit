use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

// --- TodoStore ---

/// A single item in the agent's task list.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    /// The task description.
    pub content: String,
    /// Current status of the task.
    pub status: TodoStatus,
    /// Priority level of the task.
    pub priority: TodoPriority,
}

/// Status of a to-do item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    /// Task has not been started.
    Pending,
    /// Task is currently being worked on (at most one at a time).
    InProgress,
    /// Task has been successfully completed.
    Completed,
    /// Task was abandoned.
    Cancelled,
    /// Task could not be completed due to an error.
    Failed,
    /// Task is waiting on an external dependency.
    Blocked,
}

impl std::fmt::Display for TodoStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TodoStatus::Pending => write!(f, "pending"),
            TodoStatus::InProgress => write!(f, "in_progress"),
            TodoStatus::Completed => write!(f, "completed"),
            TodoStatus::Cancelled => write!(f, "cancelled"),
            TodoStatus::Failed => write!(f, "failed"),
            TodoStatus::Blocked => write!(f, "blocked"),
        }
    }
}

/// Priority level of a to-do item.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TodoPriority {
    /// Must be done immediately.
    Critical,
    /// Should be done next.
    High,
    /// Normal priority.
    Medium,
    /// Do when convenient.
    Low,
}

impl std::fmt::Display for TodoPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TodoPriority::Critical => write!(f, "critical"),
            TodoPriority::High => write!(f, "high"),
            TodoPriority::Medium => write!(f, "medium"),
            TodoPriority::Low => write!(f, "low"),
        }
    }
}

/// Shared in-process store for agent to-do items.
///
/// Accessed via the `todo_read` and `todo_write` builtin tools.
/// Thread-safe: backed by a `std::sync::RwLock`.
pub struct TodoStore {
    todos: RwLock<Vec<TodoItem>>,
}

impl Default for TodoStore {
    fn default() -> Self {
        Self::new()
    }
}

impl TodoStore {
    /// Create an empty `TodoStore`.
    pub fn new() -> Self {
        Self {
            todos: RwLock::new(Vec::new()),
        }
    }

    fn set(&self, todos: Vec<TodoItem>) -> Result<(), String> {
        // Validate: at most 1 item can be in_progress
        let in_progress_count = todos
            .iter()
            .filter(|t| t.status == TodoStatus::InProgress)
            .count();
        if in_progress_count > 1 {
            return Err(format!(
                "Only 1 item can be in_progress at a time (got {in_progress_count})"
            ));
        }

        let mut guard = self.todos.write().expect("todo store lock poisoned");
        *guard = todos;
        Ok(())
    }

    fn get_all(&self) -> Vec<TodoItem> {
        let guard = self.todos.read().expect("todo store lock poisoned");
        guard.clone()
    }

    /// Open (actionable) items — `Pending` or `InProgress` — in list order.
    ///
    /// Long-horizon planning (recitation): the agent loop reads this each turn
    /// and re-surfaces the live plan at the context tail so it stays in recent
    /// attention. Returns an empty Vec for trivial/chat tasks that never wrote
    /// any todos, so recitation self-gates to zero overhead.
    pub fn open_items(&self) -> Vec<TodoItem> {
        self.todos
            .read()
            .expect("todo store lock poisoned")
            .iter()
            .filter(|t| matches!(t.status, TodoStatus::Pending | TodoStatus::InProgress))
            .cloned()
            .collect()
    }
}

/// Format open todo items as a compact recitation block to append at the
/// context tail. Returns `None` when there are no open items (the caller then
/// appends nothing — trivial tasks pay no cost).
///
/// Long-horizon planning: re-surfacing the *actual* plan (not a re-derived
/// textual summary) every turn counters lost-in-the-middle and means the plan
/// survives compaction automatically (next turn re-recites from the store).
pub(crate) fn recite_open_todos(items: &[TodoItem]) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let mut s = String::from("[plan — open items, stay on these until done]\n");
    for item in items {
        let mark = if item.status == TodoStatus::InProgress {
            "[>]"
        } else {
            "[ ]"
        };
        s.push_str(mark);
        s.push(' ');
        s.push_str(&item.content);
        s.push('\n');
    }
    Some(s)
}

// --- Tools ---

/// Create the `todo_read` and `todo_write` tool pair sharing a single store.
pub fn todo_tools(store: Arc<TodoStore>) -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(TodoWriteTool {
            store: store.clone(),
        }),
        Arc::new(TodoReadTool { store }),
    ]
}

struct TodoWriteTool {
    store: Arc<TodoStore>,
}

impl Tool for TodoWriteTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "todowrite".into(),
            description:
                "Write/replace the full todo list. Only 1 item can be in_progress at a time. \
                          This replaces the entire list (not append)."
                    .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed", "cancelled", "failed", "blocked"]
                                },
                                "priority": {
                                    "type": "string",
                                    "enum": ["critical", "high", "medium", "low"]
                                }
                            },
                            "required": ["content", "status", "priority"]
                        }
                    }
                },
                "required": ["todos"]
            }),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let todos_value = input
                .get("todos")
                .ok_or_else(|| Error::Agent("todos is required".into()))?;

            let todos: Vec<TodoItem> = serde_json::from_value(todos_value.clone())
                .map_err(|e| Error::Agent(format!("Invalid todo list: {e}")))?;

            if let Err(msg) = self.store.set(todos) {
                return Ok(ToolOutput::error(msg));
            }

            let all = self.store.get_all();
            Ok(ToolOutput::success(format!(
                "Todo list updated ({} items)",
                all.len()
            )))
        })
    }
}

struct TodoReadTool {
    store: Arc<TodoStore>,
}

impl Tool for TodoReadTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "todoread".into(),
            description: "Read the current todo list.".into(),
            input_schema: json!({"type": "object"}),
        }
    }

    fn execute(
        &self,
        _ctx: &crate::ExecutionContext,
        _input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let todos = self.store.get_all();

            if todos.is_empty() {
                return Ok(ToolOutput::success("No todos."));
            }

            let mut output = String::new();
            for (i, todo) in todos.iter().enumerate() {
                let status_icon = match todo.status {
                    TodoStatus::Pending => "[ ]",
                    TodoStatus::InProgress => "[>]",
                    TodoStatus::Completed => "[x]",
                    TodoStatus::Cancelled => "[-]",
                    TodoStatus::Failed => "[!]",
                    TodoStatus::Blocked => "[B]",
                };
                let priority_tag = match todo.priority {
                    TodoPriority::Critical => " [CRITICAL]",
                    TodoPriority::High => " [HIGH]",
                    TodoPriority::Medium => "",
                    TodoPriority::Low => " [low]",
                };
                output.push_str(&format!(
                    "{}. {} {}{}\n",
                    i + 1,
                    status_icon,
                    todo.content,
                    priority_tag
                ));
            }

            Ok(ToolOutput::success(output))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(content: &str, status: TodoStatus) -> TodoItem {
        TodoItem {
            content: content.into(),
            status,
            priority: TodoPriority::Medium,
        }
    }

    #[test]
    fn open_items_filters_done_and_keeps_order() {
        let store = TodoStore::new();
        store
            .set(vec![
                item("a", TodoStatus::Pending),
                item("b", TodoStatus::Completed),
                item("c", TodoStatus::InProgress),
                item("d", TodoStatus::Cancelled),
            ])
            .unwrap();
        let open = store.open_items();
        let contents: Vec<&str> = open.iter().map(|i| i.content.as_str()).collect();
        assert_eq!(
            contents,
            vec!["a", "c"],
            "only Pending/InProgress, in order"
        );
    }

    #[test]
    fn recite_empty_is_none() {
        assert!(recite_open_todos(&[]).is_none());
    }

    #[test]
    fn recite_formats_open_items_with_marks() {
        let block = recite_open_todos(&[
            item("write tests", TodoStatus::InProgress),
            item("ship it", TodoStatus::Pending),
        ])
        .expect("non-empty");
        assert!(
            block.contains("[>] write tests"),
            "in-progress marker: {block}"
        );
        assert!(block.contains("[ ] ship it"), "pending marker: {block}");
        // in_progress comes before pending in the input order
        assert!(block.find("write tests").unwrap() < block.find("ship it").unwrap());
    }

    #[test]
    fn definition_names() {
        let store = Arc::new(TodoStore::new());
        let tools = todo_tools(store);
        let names: Vec<String> = tools.iter().map(|t| t.definition().name).collect();
        assert!(names.contains(&"todowrite".to_string()));
        assert!(names.contains(&"todoread".to_string()));
    }

    #[tokio::test]
    async fn todowrite_and_read() {
        let store = Arc::new(TodoStore::new());
        let tools = todo_tools(store);
        let write_tool = &tools[0];
        let read_tool = &tools[1];

        // Write some todos
        let result = write_tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "todos": [
                        {"content": "Fix bug", "status": "in_progress", "priority": "high"},
                        {"content": "Write tests", "status": "pending", "priority": "medium"}
                    ]
                }),
            )
            .await
            .unwrap();
        assert!(!result.is_error, "got error: {}", result.content);
        assert!(result.content.contains("2 items"));

        // Read them back
        let result = read_tool
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Fix bug"));
        assert!(result.content.contains("[HIGH]"));
        assert!(result.content.contains("Write tests"));
        assert!(result.content.contains("[>]")); // in_progress
    }

    #[tokio::test]
    async fn todowrite_rejects_multiple_in_progress() {
        let store = Arc::new(TodoStore::new());
        let tools = todo_tools(store);
        let write_tool = &tools[0];

        let result = write_tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({
                    "todos": [
                        {"content": "Task 1", "status": "in_progress", "priority": "high"},
                        {"content": "Task 2", "status": "in_progress", "priority": "high"}
                    ]
                }),
            )
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("Only 1 item"));
    }

    #[tokio::test]
    async fn todoread_empty() {
        let store = Arc::new(TodoStore::new());
        let tools = todo_tools(store);
        let read_tool = &tools[1];

        let result = read_tool
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("No todos"));
    }

    #[tokio::test]
    async fn todowrite_replaces_full_list() {
        let store = Arc::new(TodoStore::new());
        let tools = todo_tools(store);
        let write_tool = &tools[0];
        let read_tool = &tools[1];

        // First write
        write_tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"todos": [{"content": "Old", "status": "pending", "priority": "low"}]}),
            )
            .await
            .unwrap();

        // Second write replaces
        write_tool
            .execute(
                &crate::ExecutionContext::default(),
                json!({"todos": [{"content": "New", "status": "completed", "priority": "high"}]}),
            )
            .await
            .unwrap();

        let result = read_tool
            .execute(&crate::ExecutionContext::default(), json!({}))
            .await
            .unwrap();
        assert!(result.content.contains("New"));
        assert!(!result.content.contains("Old"));
    }
}
