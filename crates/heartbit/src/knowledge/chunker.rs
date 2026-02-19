use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::{Chunk, DocumentSource};

/// Configuration for text chunking.
#[derive(Debug, Clone)]
pub struct ChunkConfig {
    /// Maximum byte length per chunk.
    pub chunk_size: usize,
    /// Number of overlapping bytes between consecutive chunks.
    pub chunk_overlap: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
        }
    }
}

/// Generate a deterministic chunk ID from source URI and chunk index.
fn chunk_id(uri: &str, index: usize) -> String {
    let mut hasher = DefaultHasher::new();
    uri.hash(&mut hasher);
    let hash = hasher.finish();
    format!("{hash:016x}-{index}")
}

/// Split text into overlapping chunks, respecting paragraph boundaries.
///
/// Empty text produces no chunks. Paragraphs are split on double newlines.
/// If a paragraph fits within `chunk_size`, it's kept whole. Large paragraphs
/// are split at `chunk_size` boundaries with `chunk_overlap` overlap.
pub fn split_into_chunks(text: &str, source: &DocumentSource, config: &ChunkConfig) -> Vec<Chunk> {
    let text = text.trim();
    if text.is_empty() {
        return vec![];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut chunk_index = 0;

    let paragraphs: Vec<&str> = text.split("\n\n").collect();

    for para in &paragraphs {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        // If adding this paragraph would exceed chunk_size, emit current chunk first
        if !current.is_empty() && current.len() + para.len() + 2 > config.chunk_size {
            let id = chunk_id(&source.uri, chunk_index);
            chunks.push(Chunk {
                id,
                content: current.clone(),
                source: source.clone(),
                chunk_index,
            });
            chunk_index += 1;

            // Keep overlap from the end of the current chunk
            if config.chunk_overlap > 0 && current.len() > config.chunk_overlap {
                let start = current.len() - config.chunk_overlap;
                // Find a char boundary
                let start = current.ceil_char_boundary(start);
                current = current[start..].to_string();
            } else if config.chunk_overlap == 0 {
                current.clear();
            }
            // If chunk_overlap >= current.len(), keep all of current
        }

        // Handle paragraphs larger than chunk_size by splitting them
        if para.len() > config.chunk_size {
            // First flush current content if any
            if !current.is_empty() {
                let id = chunk_id(&source.uri, chunk_index);
                chunks.push(Chunk {
                    id,
                    content: current.clone(),
                    source: source.clone(),
                    chunk_index,
                });
                chunk_index += 1;
                current.clear();
            }

            // Split the large paragraph
            let mut pos = 0;
            while pos < para.len() {
                let end = (pos + config.chunk_size).min(para.len());
                let end = para.ceil_char_boundary(end);
                let end = end.min(para.len());

                let id = chunk_id(&source.uri, chunk_index);
                chunks.push(Chunk {
                    id,
                    content: para[pos..end].to_string(),
                    source: source.clone(),
                    chunk_index,
                });
                chunk_index += 1;

                if end >= para.len() {
                    break;
                }

                // Advance with overlap
                let advance = if config.chunk_overlap < config.chunk_size {
                    config.chunk_size - config.chunk_overlap
                } else {
                    1 // Avoid infinite loop
                };
                pos += advance;
                pos = para.ceil_char_boundary(pos);
            }
        } else {
            // Append paragraph to current chunk
            if !current.is_empty() {
                current.push_str("\n\n");
            }
            current.push_str(para);
        }
    }

    // Emit any remaining content
    if !current.is_empty() {
        let id = chunk_id(&source.uri, chunk_index);
        chunks.push(Chunk {
            id,
            content: current,
            source: source.clone(),
            chunk_index,
        });
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_source() -> DocumentSource {
        DocumentSource {
            uri: "test.md".into(),
            title: "Test".into(),
        }
    }

    #[test]
    fn empty_text_produces_no_chunks() {
        let chunks = split_into_chunks("", &test_source(), &ChunkConfig::default());
        assert!(chunks.is_empty());
    }

    #[test]
    fn whitespace_only_produces_no_chunks() {
        let chunks = split_into_chunks("   \n\n  ", &test_source(), &ChunkConfig::default());
        assert!(chunks.is_empty());
    }

    #[test]
    fn single_small_paragraph_is_one_chunk() {
        let text = "Hello, world!";
        let chunks = split_into_chunks(text, &test_source(), &ChunkConfig::default());
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "Hello, world!");
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[0].source.uri, "test.md");
    }

    #[test]
    fn multiple_paragraphs_within_limit_are_single_chunk() {
        let text = "First paragraph.\n\nSecond paragraph.";
        let config = ChunkConfig {
            chunk_size: 1000,
            chunk_overlap: 0,
        };
        let chunks = split_into_chunks(text, &test_source(), &config);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].content.contains("First paragraph."));
        assert!(chunks[0].content.contains("Second paragraph."));
    }

    #[test]
    fn paragraphs_exceeding_limit_split_into_multiple_chunks() {
        let para1 = "a".repeat(60);
        let para2 = "b".repeat(60);
        let text = format!("{para1}\n\n{para2}");
        let config = ChunkConfig {
            chunk_size: 80,
            chunk_overlap: 0,
        };
        let chunks = split_into_chunks(&text, &test_source(), &config);
        assert!(
            chunks.len() >= 2,
            "expected >= 2 chunks, got {}",
            chunks.len()
        );
        assert!(chunks[0].content.contains('a'));
        assert!(chunks.last().unwrap().content.contains('b'));
    }

    #[test]
    fn overlap_preserves_context() {
        let para1 = "a".repeat(60);
        let para2 = "b".repeat(60);
        let text = format!("{para1}\n\n{para2}");
        let config = ChunkConfig {
            chunk_size: 80,
            chunk_overlap: 20,
        };
        let chunks = split_into_chunks(&text, &test_source(), &config);
        assert!(
            chunks.len() >= 2,
            "expected >= 2 chunks, got {}",
            chunks.len()
        );
        // The second chunk should start with overlap from the first
        if chunks.len() >= 2 {
            // With overlap=20, the second chunk should contain some trailing 'a's
            // from the first chunk
            let c1_tail = &chunks[0].content[chunks[0].content.len().saturating_sub(20)..];
            let c2_head = &chunks[1].content[..c1_tail.len().min(chunks[1].content.len())];
            assert_eq!(c1_tail, c2_head, "overlap should match");
        }
    }

    #[test]
    fn chunk_indices_are_sequential() {
        let text = (0..10)
            .map(|i| format!("Paragraph {i}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let config = ChunkConfig {
            chunk_size: 30,
            chunk_overlap: 0,
        };
        let chunks = split_into_chunks(&text, &test_source(), &config);
        for (i, chunk) in chunks.iter().enumerate() {
            assert_eq!(chunk.chunk_index, i, "chunk {i} has wrong index");
        }
    }

    #[test]
    fn deterministic_ids() {
        let text = "Hello world.\n\nSecond paragraph.";
        let config = ChunkConfig {
            chunk_size: 20,
            chunk_overlap: 0,
        };
        let chunks1 = split_into_chunks(text, &test_source(), &config);
        let chunks2 = split_into_chunks(text, &test_source(), &config);
        assert_eq!(chunks1.len(), chunks2.len());
        for (a, b) in chunks1.iter().zip(chunks2.iter()) {
            assert_eq!(a.id, b.id, "chunk IDs should be deterministic");
        }
    }

    #[test]
    fn different_sources_produce_different_ids() {
        let text = "Hello world.";
        let config = ChunkConfig::default();
        let src1 = DocumentSource {
            uri: "file1.md".into(),
            title: "F1".into(),
        };
        let src2 = DocumentSource {
            uri: "file2.md".into(),
            title: "F2".into(),
        };
        let c1 = split_into_chunks(text, &src1, &config);
        let c2 = split_into_chunks(text, &src2, &config);
        assert_ne!(c1[0].id, c2[0].id);
    }

    #[test]
    fn utf8_safe_chunking() {
        // Multi-byte characters should not be split mid-character
        let text = "é".repeat(600); // 2 bytes each = 1200 bytes, 600 chars
        let config = ChunkConfig {
            chunk_size: 100,
            chunk_overlap: 20,
        };
        let chunks = split_into_chunks(&text, &test_source(), &config);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            // Every chunk should be valid UTF-8 (Rust strings guarantee this)
            assert!(chunk.content.is_char_boundary(0));
            assert!(chunk.content.is_char_boundary(chunk.content.len()));
        }
    }

    #[test]
    fn chunk_config_defaults() {
        let config = ChunkConfig::default();
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.chunk_overlap, 200);
    }

    #[test]
    fn large_single_paragraph_split() {
        let text = "x".repeat(3000);
        let config = ChunkConfig {
            chunk_size: 1000,
            chunk_overlap: 200,
        };
        let chunks = split_into_chunks(&text, &test_source(), &config);
        assert!(
            chunks.len() >= 3,
            "expected >= 3 chunks, got {}",
            chunks.len()
        );
        // All content should be covered
        let total_unique: usize = chunks.iter().map(|c| c.content.len()).sum();
        // With overlap, total content > original, so just check >= original
        assert!(total_unique >= 3000);
    }
}
