/// Extract text content from a document's raw bytes based on MIME type.
///
/// Returns `Ok(Some(text))` for supported text-based formats, `Ok(None)` for
/// unsupported binary formats (PDF, DOCX, etc.).
pub fn extract_text(data: &[u8], mime_type: &str) -> Option<String> {
    match mime_type {
        "text/plain" | "text/csv" | "text/markdown" | "text/html" | "application/json"
        | "application/xml" | "text/xml" | "application/x-yaml" | "text/yaml"
        | "application/toml" => Some(String::from_utf8_lossy(data).into_owned()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_text_plain() {
        let data = b"Hello, world!";
        assert_eq!(
            extract_text(data, "text/plain"),
            Some("Hello, world!".into())
        );
    }

    #[test]
    fn extract_text_json() {
        let data = br#"{"key": "value"}"#;
        let result = extract_text(data, "application/json");
        assert!(result.is_some());
        assert!(result.unwrap().contains("key"));
    }

    #[test]
    fn extract_text_csv() {
        let data = b"a,b,c\n1,2,3";
        assert_eq!(extract_text(data, "text/csv"), Some("a,b,c\n1,2,3".into()));
    }

    #[test]
    fn extract_text_markdown() {
        let data = b"# Title\nBody";
        assert!(extract_text(data, "text/markdown").is_some());
    }

    #[test]
    fn extract_text_unsupported_pdf() {
        assert!(extract_text(b"%PDF-1.4", "application/pdf").is_none());
    }

    #[test]
    fn extract_text_unsupported_binary() {
        assert!(extract_text(b"\x89PNG", "application/octet-stream").is_none());
    }

    #[test]
    fn extract_text_invalid_utf8_lossy() {
        let data = &[0xFF, 0xFE, 0x68, 0x69]; // invalid UTF-8 prefix + "hi"
        let result = extract_text(data, "text/plain").unwrap();
        assert!(result.contains("hi"));
    }
}
