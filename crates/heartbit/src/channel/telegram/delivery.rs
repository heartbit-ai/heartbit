use std::time::Instant;

/// Maximum message length for Telegram messages (UTF-8 characters).
const TELEGRAM_MAX_LEN: usize = 4096;

/// Split a message into chunks that fit within Telegram's 4096-character limit.
///
/// Splitting strategy:
/// 1. Try paragraph boundaries (`\n\n`)
/// 2. Fall back to sentence boundaries (`. `, `! `, `? `)
/// 3. Hard split at line boundaries (`\n`)
/// 4. Last resort: hard split at 4096 chars (UTF-8 safe)
pub fn chunk_message(text: &str) -> Vec<&str> {
    if text.is_empty() {
        return Vec::new();
    }
    if text.len() <= TELEGRAM_MAX_LEN {
        return vec![text];
    }

    let mut chunks = Vec::new();
    let mut remaining = text;

    while !remaining.is_empty() {
        if remaining.len() <= TELEGRAM_MAX_LEN {
            chunks.push(remaining);
            break;
        }

        let split_at = find_split_point(remaining, TELEGRAM_MAX_LEN);
        let (chunk, rest) = remaining.split_at(split_at);
        chunks.push(chunk);
        remaining = rest.trim_start_matches('\n');
    }

    chunks
}

/// Find the best split point within `max_len` bytes.
fn find_split_point(text: &str, max_len: usize) -> usize {
    // Ensure we don't slice mid-character
    let safe_len = floor_char_boundary(text, max_len);
    let search_region = &text[..safe_len];

    // 1. Paragraph boundary
    if let Some(pos) = search_region.rfind("\n\n")
        && pos > 0
    {
        return pos + 1; // Keep one newline at the end of the chunk
    }

    // 2. Sentence boundary
    for delim in [". ", "! ", "? "] {
        if let Some(pos) = search_region.rfind(delim)
            && pos > 0
        {
            return pos + delim.len();
        }
    }

    // 3. Line boundary
    if let Some(pos) = search_region.rfind('\n')
        && pos > 0
    {
        return pos + 1;
    }

    // 4. Hard split at char boundary
    floor_char_boundary(text, max_len)
}

/// Find the largest byte offset <= `max` that is a valid UTF-8 char boundary.
pub(crate) fn floor_char_boundary(s: &str, max: usize) -> usize {
    if max >= s.len() {
        return s.len();
    }
    let mut i = max;
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Buffer for accumulating streaming text deltas with debounced edit triggers.
pub struct StreamBuffer {
    chat_id: i64,
    message_id: Option<i32>,
    accumulated: String,
    last_edit: Instant,
    debounce_ms: u64,
}

impl StreamBuffer {
    pub fn new(chat_id: i64, debounce_ms: u64) -> Self {
        Self {
            chat_id,
            message_id: None,
            accumulated: String::new(),
            last_edit: Instant::now(),
            debounce_ms,
        }
    }

    /// Push a text delta. Returns `true` if enough time has elapsed for an edit.
    pub fn push(&mut self, delta: &str) -> bool {
        self.accumulated.push_str(delta);
        let elapsed = self.last_edit.elapsed().as_millis() as u64;
        elapsed >= self.debounce_ms
    }

    /// Mark that an edit was just sent.
    pub fn mark_edited(&mut self) {
        self.last_edit = Instant::now();
    }

    /// Get the current accumulated text.
    pub fn current_text(&self) -> &str {
        &self.accumulated
    }

    /// Set the message ID after the first `send_message`.
    pub fn set_message_id(&mut self, id: i32) {
        self.message_id = Some(id);
    }

    /// Get the message ID (if set).
    pub fn message_id(&self) -> Option<i32> {
        self.message_id
    }

    /// Get the chat ID.
    pub fn chat_id(&self) -> i64 {
        self.chat_id
    }

    /// Check if any text has accumulated.
    pub fn is_empty(&self) -> bool {
        self.accumulated.is_empty()
    }

    /// Reset the buffer for a new message.
    pub fn reset(&mut self) {
        self.accumulated.clear();
        self.message_id = None;
        self.last_edit = Instant::now();
    }
}

/// Simple token-bucket rate limiter (1 message per second per chat).
pub struct RateLimiter {
    last_send: Instant,
    min_interval_ms: u64,
}

impl RateLimiter {
    pub fn new(min_interval_ms: u64) -> Self {
        Self {
            // Allow immediate first send
            last_send: Instant::now() - std::time::Duration::from_millis(min_interval_ms),
            min_interval_ms,
        }
    }

    /// Returns the delay needed before the next send. Returns `Duration::ZERO` if OK to send now.
    pub fn check(&self) -> std::time::Duration {
        let elapsed = self.last_send.elapsed().as_millis() as u64;
        if elapsed >= self.min_interval_ms {
            std::time::Duration::ZERO
        } else {
            std::time::Duration::from_millis(self.min_interval_ms - elapsed)
        }
    }

    /// Record that a send just happened.
    pub fn record_send(&mut self) {
        self.last_send = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- chunk_message tests ---

    #[test]
    fn chunk_empty_string() {
        assert!(chunk_message("").is_empty());
    }

    #[test]
    fn chunk_short_string() {
        let chunks = chunk_message("Hello, world!");
        assert_eq!(chunks, vec!["Hello, world!"]);
    }

    #[test]
    fn chunk_exact_limit() {
        let text = "x".repeat(TELEGRAM_MAX_LEN);
        let chunks = chunk_message(&text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), TELEGRAM_MAX_LEN);
    }

    #[test]
    fn chunk_paragraph_split() {
        let para1 = "a".repeat(2000);
        let para2 = "b".repeat(2000);
        let para3 = "c".repeat(100);
        let text = format!("{para1}\n\n{para2}\n\n{para3}");
        let chunks = chunk_message(&text);
        assert!(chunks.len() >= 2);
        // First chunk should end near a paragraph boundary
        assert!(chunks[0].len() <= TELEGRAM_MAX_LEN);
    }

    #[test]
    fn chunk_sentence_split() {
        let sentence = "Hello world. ";
        let count = TELEGRAM_MAX_LEN / sentence.len() + 2;
        let text = sentence.repeat(count);
        let chunks = chunk_message(&text);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.len() <= TELEGRAM_MAX_LEN);
        }
    }

    #[test]
    fn chunk_hard_split_no_delimiters() {
        let text = "x".repeat(TELEGRAM_MAX_LEN * 2 + 100);
        let chunks = chunk_message(&text);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.len() <= TELEGRAM_MAX_LEN);
        }
    }

    #[test]
    fn chunk_utf8_safety() {
        // Multi-byte characters: 4-byte emoji
        let emoji = "🦀";
        let count = TELEGRAM_MAX_LEN / emoji.len() + 10;
        let text = emoji.repeat(count);
        let chunks = chunk_message(&text);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(chunk.len() <= TELEGRAM_MAX_LEN);
            // Verify valid UTF-8 (would panic on invalid)
            let _ = chunk.chars().count();
        }
    }

    #[test]
    fn chunk_utf8_boundary_at_limit() {
        // Construct text where byte position 4096 falls inside a multi-byte char.
        // 4-byte emoji: 🦀 = 4 bytes. 1024 emojis = 4096 bytes exactly.
        // Adding one more byte of ASCII before makes byte 4096 land mid-emoji.
        let text = format!("x{}", "🦀".repeat(1024));
        // This would panic before the floor_char_boundary fix
        let chunks = chunk_message(&text);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            // Verify valid UTF-8
            let _ = chunk.chars().count();
        }
    }

    #[test]
    fn chunk_preserves_all_content() {
        let para1 = "a".repeat(3000);
        let para2 = "b".repeat(3000);
        let text = format!("{para1}\n\n{para2}");
        let chunks = chunk_message(&text);
        let reassembled: String = chunks.join("");
        // We trim newlines at split points, so the reassembled text may differ slightly
        // but all non-whitespace content should be preserved
        let original_content: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let reassembled_content: String =
            reassembled.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(original_content, reassembled_content);
    }

    // --- StreamBuffer tests ---

    #[test]
    fn stream_buffer_empty_initially() {
        let buf = StreamBuffer::new(100, 500);
        assert!(buf.is_empty());
        assert_eq!(buf.chat_id(), 100);
        assert!(buf.message_id().is_none());
    }

    #[test]
    fn stream_buffer_push_accumulates() {
        let mut buf = StreamBuffer::new(100, 500);
        buf.push("Hello ");
        buf.push("world");
        assert_eq!(buf.current_text(), "Hello world");
        assert!(!buf.is_empty());
    }

    #[test]
    fn stream_buffer_debounce() {
        let mut buf = StreamBuffer::new(100, 10);
        // First push — debounce not elapsed yet (just created)
        let should_edit = buf.push("a");
        // May or may not trigger depending on timing
        if !should_edit {
            std::thread::sleep(std::time::Duration::from_millis(15));
            let should_edit = buf.push("b");
            assert!(should_edit);
        }
    }

    #[test]
    fn stream_buffer_message_id() {
        let mut buf = StreamBuffer::new(100, 500);
        assert!(buf.message_id().is_none());
        buf.set_message_id(42);
        assert_eq!(buf.message_id(), Some(42));
    }

    #[test]
    fn stream_buffer_reset() {
        let mut buf = StreamBuffer::new(100, 500);
        buf.push("Hello");
        buf.set_message_id(42);
        buf.reset();
        assert!(buf.is_empty());
        assert!(buf.message_id().is_none());
    }

    #[test]
    fn stream_buffer_mark_edited() {
        let mut buf = StreamBuffer::new(100, 50);
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert!(buf.push("a")); // Should trigger
        buf.mark_edited();
        assert!(!buf.push("b")); // Should not trigger (just edited)
    }

    // --- RateLimiter tests ---

    #[test]
    fn rate_limiter_first_send_immediate() {
        let limiter = RateLimiter::new(1000);
        assert_eq!(limiter.check(), std::time::Duration::ZERO);
    }

    #[test]
    fn rate_limiter_requires_delay() {
        let mut limiter = RateLimiter::new(100);
        limiter.record_send();
        let delay = limiter.check();
        assert!(delay > std::time::Duration::ZERO);
    }

    #[test]
    fn rate_limiter_allows_after_interval() {
        let mut limiter = RateLimiter::new(10);
        limiter.record_send();
        std::thread::sleep(std::time::Duration::from_millis(15));
        assert_eq!(limiter.check(), std::time::Duration::ZERO);
    }

    // --- floor_char_boundary tests ---

    #[test]
    fn floor_char_boundary_ascii() {
        assert_eq!(floor_char_boundary("hello", 3), 3);
    }

    #[test]
    fn floor_char_boundary_multibyte() {
        let s = "a🦀b"; // a=1, 🦀=4, b=1 → total 6 bytes
        assert_eq!(floor_char_boundary(s, 3), 1); // Mid-emoji, back to 'a' boundary
        assert_eq!(floor_char_boundary(s, 5), 5); // After emoji
        assert_eq!(floor_char_boundary(s, 1), 1); // After 'a'
    }

    #[test]
    fn floor_char_boundary_beyond_len() {
        assert_eq!(floor_char_boundary("abc", 10), 3);
    }
}
