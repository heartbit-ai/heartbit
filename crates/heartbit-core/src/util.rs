/// FNV-1a 64-bit hash — deterministic across all platforms and runs.
///
/// Used for chunk IDs, cache keys, and other non-cryptographic hashing.
pub fn fnv1a_hash(data: &[u8]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;
    let mut hash = FNV_OFFSET;
    for &byte in data {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Strip HTML tags from text, replacing them with spaces.
///
/// Skips content inside `<script>` and `<style>` tags. For full
/// HTML→markdown conversion a dedicated crate would be appropriate,
/// but for V1 tag stripping suffices.
pub fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut tag_name = String::new();
    let mut collecting_tag = false;
    let mut last_was_space = false;
    let mut skip_content = false; // true inside <script> or <style>

    for ch in html.chars() {
        if ch == '<' {
            in_tag = true;
            tag_name.clear();
            collecting_tag = true;
            if !skip_content && !last_was_space && !result.is_empty() {
                result.push(' ');
                last_was_space = true;
            }
        } else if ch == '>' && in_tag {
            in_tag = false;
            collecting_tag = false;
            let tag_lower = tag_name.to_lowercase();
            match tag_lower.as_str() {
                "script" | "style" => skip_content = true,
                "/script" | "/style" => skip_content = false,
                _ => {}
            }
        } else if in_tag && collecting_tag {
            if ch.is_whitespace() {
                collecting_tag = false;
            } else {
                tag_name.push(ch);
            }
        } else if !in_tag && !skip_content {
            if ch.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                    last_was_space = true;
                }
            } else {
                result.push(ch);
                last_was_space = false;
            }
        }
    }

    result.trim().to_string()
}

/// Compute the Levenshtein (edit) distance between two strings.
///
/// Uses `chars().count()` for correct unicode handling (not byte length).
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();
    let mut matrix = vec![vec![0usize; b_len + 1]; a_len + 1];

    for (i, row) in matrix.iter_mut().enumerate().take(a_len + 1) {
        row[0] = i;
    }
    for (j, val) in matrix[0].iter_mut().enumerate().take(b_len + 1) {
        *val = j;
    }

    for (i, ca) in a_chars.iter().enumerate() {
        for (j, cb) in b_chars.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            matrix[i + 1][j + 1] = (matrix[i][j + 1] + 1)
                .min(matrix[i + 1][j] + 1)
                .min(matrix[i][j] + cost);
        }
    }

    matrix[a_len][b_len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_strings() {
        assert_eq!(levenshtein("read_file", "read_file"), 0);
    }

    #[test]
    fn single_substitution() {
        assert_eq!(levenshtein("reed_file", "read_file"), 1);
    }

    #[test]
    fn empty_strings() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "xyz"), 3);
    }

    #[test]
    fn unicode_chars() {
        assert_eq!(levenshtein("café", "cafe"), 1);
    }
}
