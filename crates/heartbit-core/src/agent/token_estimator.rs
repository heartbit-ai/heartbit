use crate::llm::types::{ContentBlock, Message};

/// Estimate token count for a text string using 4 chars/token heuristic.
///
/// This is a fast, dependency-free approximation. No external tokenizer needed.
pub(crate) fn estimate_tokens(text: &str) -> u32 {
    // 4 chars per token is a reasonable average for English + code
    (text.len() as u32).div_ceil(4)
}

/// Estimate token count for a single message, including all content blocks.
///
/// Adds a small overhead per message for role/structure tokens.
pub(crate) fn estimate_message_tokens(message: &Message) -> u32 {
    const MESSAGE_OVERHEAD: u32 = 4; // role, separators

    let content_tokens: u32 = message
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => estimate_tokens(text),
            ContentBlock::ToolUse { id, name, input } => {
                estimate_tokens(id) + estimate_tokens(name) + estimate_tokens(&input.to_string())
            }
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } => estimate_tokens(tool_use_id) + estimate_tokens(content),
            ContentBlock::Image { data, .. } => {
                // Base64 images are ~1.37x the raw size. Anthropic vision bills
                // based on image dimensions, but for context window estimation
                // we approximate: each 750 base64 chars ≈ 1 token.
                (data.len() as u32) / 750 + 85 // 85 = overhead for the image block structure
            }
            ContentBlock::Audio { data, .. } => {
                // Rough audio token estimate: base64 audio ÷ 750 + overhead.
                (data.len() as u32) / 750 + 50
            }
        })
        .sum();

    MESSAGE_OVERHEAD + content_tokens
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::types::Message;
    use serde_json::json;

    #[test]
    fn estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn estimate_tokens_short() {
        // "hello" = 5 chars → ceil(5/4) = 2 tokens
        assert_eq!(estimate_tokens("hello"), 2);
    }

    #[test]
    fn estimate_tokens_exact_multiple() {
        // 8 chars → ceil(8/4) = 2 tokens
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn estimate_tokens_longer_text() {
        // 100 chars → ceil(100/4) = 25 tokens
        let text = "a".repeat(100);
        assert_eq!(estimate_tokens(&text), 25);
    }

    #[test]
    fn estimate_message_tokens_text_block() {
        let msg = Message::user("hello world"); // 11 chars → ceil(11/4) = 3 + 4 overhead = 7
        let tokens = estimate_message_tokens(&msg);
        assert_eq!(tokens, 4 + 3); // overhead + content
    }

    #[test]
    fn estimate_message_tokens_tool_use_block() {
        let msg = Message {
            role: crate::llm::types::Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "call-1".into(),
                name: "search".into(),
                input: json!({"q": "rust"}),
            }],
        };
        let tokens = estimate_message_tokens(&msg);
        // 4 overhead + estimate("call-1") + estimate("search") + estimate(json string)
        assert!(tokens > 4);
    }

    #[test]
    fn estimate_message_tokens_tool_result_block() {
        let msg = Message {
            role: crate::llm::types::Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call-1".into(),
                content: "search results here".into(),
                is_error: false,
            }],
        };
        let tokens = estimate_message_tokens(&msg);
        // 4 overhead + estimate("call-1") + estimate("search results here")
        assert!(tokens > 4);
    }

    #[test]
    fn estimate_message_tokens_audio_block() {
        let msg = Message {
            role: crate::llm::types::Role::User,
            content: vec![ContentBlock::Audio {
                format: "ogg".into(),
                data: "a".repeat(1500), // 1500 / 750 + 50 = 52
            }],
        };
        let tokens = estimate_message_tokens(&msg);
        // 4 overhead + 52 audio = 56
        assert_eq!(tokens, 4 + 52);
    }

    #[test]
    fn estimate_message_tokens_multiple_blocks() {
        let msg = Message {
            role: crate::llm::types::Role::Assistant,
            content: vec![
                ContentBlock::Text {
                    text: "Let me search.".into(),
                },
                ContentBlock::ToolUse {
                    id: "c1".into(),
                    name: "search".into(),
                    input: json!({"q": "test"}),
                },
            ],
        };
        let tokens = estimate_message_tokens(&msg);
        // Should be more than a single text block
        assert!(tokens > estimate_message_tokens(&Message::user("Let me search.")));
    }
}
