use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::Error;
use crate::llm::types::ToolDefinition;
use crate::tool::{Tool, ToolOutput};

// --- Types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionRequest {
    pub questions: Vec<Question>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Question {
    pub question: String,
    pub header: String,
    pub options: Vec<QuestionOption>,
    pub multiple: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionOption {
    pub label: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuestionResponse {
    /// Per-question list of selected labels.
    pub answers: Vec<Vec<String>>,
}

/// Callback type for agent-to-user structured questions.
pub type OnQuestion = dyn Fn(QuestionRequest) -> Pin<Box<dyn Future<Output = Result<QuestionResponse, Error>> + Send>>
    + Send
    + Sync;

// --- Tool ---

pub struct QuestionTool {
    on_question: Arc<OnQuestion>,
}

impl QuestionTool {
    pub fn new(on_question: Arc<OnQuestion>) -> Self {
        Self { on_question }
    }
}

impl Tool for QuestionTool {
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "question".into(),
            description: "Ask the user structured questions with predefined options. \
                          Use this when you need clarification or a decision from the user."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "The question to ask"
                                },
                                "header": {
                                    "type": "string",
                                    "description": "Short label (max 12 chars)"
                                },
                                "options": {
                                    "type": "array",
                                    "minItems": 2,
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "label": {"type": "string"},
                                            "description": {"type": "string"}
                                        },
                                        "required": ["label", "description"]
                                    }
                                },
                                "multiple": {
                                    "type": "boolean",
                                    "description": "Allow multiple selections"
                                }
                            },
                            "required": ["question", "header", "options", "multiple"]
                        }
                    }
                },
                "required": ["questions"]
            }),
        }
    }

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<ToolOutput, Error>> + Send + '_>> {
        Box::pin(async move {
            let questions_value = input
                .get("questions")
                .ok_or_else(|| Error::Agent("questions is required".into()))?;

            let questions: Vec<Question> = serde_json::from_value(questions_value.clone())
                .map_err(|e| Error::Agent(format!("Invalid questions format: {e}")))?;

            if questions.is_empty() {
                return Ok(ToolOutput::error("At least one question is required."));
            }
            for q in &questions {
                if q.options.len() < 2 {
                    return Ok(ToolOutput::error(format!(
                        "Question '{}' must have at least 2 options.",
                        q.header
                    )));
                }
            }

            let request = QuestionRequest {
                questions: questions.clone(),
            };
            let response = match (self.on_question)(request).await {
                Ok(r) => r,
                Err(e) => return Ok(ToolOutput::error(format!("Question failed: {e}"))),
            };

            if response.answers.len() != questions.len() {
                return Ok(ToolOutput::error(format!(
                    "Expected {} answers but got {}",
                    questions.len(),
                    response.answers.len()
                )));
            }

            // Format answers
            let mut output = String::new();
            for (i, q) in questions.iter().enumerate() {
                let answers = &response.answers[i];
                output.push_str(&format!("{}: {}\n", q.question, answers.join(", ")));
            }

            Ok(ToolOutput::success(output))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn definition_has_correct_name() {
        let callback: Arc<OnQuestion> = Arc::new(|_| {
            Box::pin(async {
                Ok(QuestionResponse {
                    answers: vec![vec!["A".into()]],
                })
            })
        });
        let tool = QuestionTool::new(callback);
        assert_eq!(tool.definition().name, "question");
    }

    #[tokio::test]
    async fn question_tool_asks_and_returns() {
        let callback: Arc<OnQuestion> = Arc::new(|req| {
            Box::pin(async move {
                let mut answers = Vec::new();
                for q in &req.questions {
                    answers.push(vec![q.options[0].label.clone()]);
                }
                Ok(QuestionResponse { answers })
            })
        });

        let tool = QuestionTool::new(callback);
        let result = tool
            .execute(json!({
                "questions": [{
                    "question": "Which color?",
                    "header": "Color",
                    "options": [
                        {"label": "Red", "description": "A warm color"},
                        {"label": "Blue", "description": "A cool color"}
                    ],
                    "multiple": false
                }]
            }))
            .await
            .unwrap();
        assert!(!result.is_error);
        assert!(result.content.contains("Red"));
    }

    #[tokio::test]
    async fn question_tool_empty_questions() {
        let callback: Arc<OnQuestion> =
            Arc::new(|_| Box::pin(async { Ok(QuestionResponse { answers: vec![] }) }));

        let tool = QuestionTool::new(callback);
        let result = tool.execute(json!({"questions": []})).await.unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("At least one question"));
    }

    #[tokio::test]
    async fn question_with_too_few_options_rejected() {
        let callback: Arc<OnQuestion> =
            Arc::new(|_| Box::pin(async { Ok(QuestionResponse { answers: vec![] }) }));

        let tool = QuestionTool::new(callback);

        // Zero options
        let result = tool
            .execute(json!({
                "questions": [{
                    "question": "Pick one",
                    "header": "Choice",
                    "options": [],
                    "multiple": false
                }]
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("at least 2 options"));

        // One option (also rejected)
        let result = tool
            .execute(json!({
                "questions": [{
                    "question": "Pick one",
                    "header": "Choice",
                    "options": [{"label": "Only", "description": "Single option"}],
                    "multiple": false
                }]
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(result.content.contains("at least 2 options"));
    }

    #[tokio::test]
    async fn question_tool_rejects_mismatched_answer_count() {
        // Callback returns 2 answers but only 1 question asked
        let callback: Arc<OnQuestion> = Arc::new(|_| {
            Box::pin(async {
                Ok(QuestionResponse {
                    answers: vec![vec!["A".into()], vec!["B".into()]],
                })
            })
        });

        let tool = QuestionTool::new(callback);
        let result = tool
            .execute(json!({
                "questions": [{
                    "question": "Pick one",
                    "header": "Choice",
                    "options": [
                        {"label": "A", "description": "Option A"},
                        {"label": "B", "description": "Option B"}
                    ],
                    "multiple": false
                }]
            }))
            .await
            .unwrap();
        assert!(result.is_error);
        assert!(
            result.content.contains("Expected 1 answers but got 2"),
            "got: {}",
            result.content
        );
    }

    #[tokio::test]
    async fn question_tool_callback_error_returns_tool_error() {
        let callback: Arc<OnQuestion> =
            Arc::new(|_| Box::pin(async { Err(Error::Agent("User cancelled".into())) }));

        let tool = QuestionTool::new(callback);
        let result = tool
            .execute(json!({
                "questions": [{
                    "question": "Pick one",
                    "header": "Choice",
                    "options": [
                        {"label": "A", "description": "Option A"},
                        {"label": "B", "description": "Option B"}
                    ],
                    "multiple": false
                }]
            }))
            .await
            .unwrap(); // Should not propagate error
        assert!(result.is_error);
        assert!(result.content.contains("User cancelled"));
    }
}
