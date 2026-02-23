/// Compute the Levenshtein (edit) distance between two strings.
///
/// Uses `chars().count()` for correct unicode handling (not byte length).
pub(crate) fn levenshtein(a: &str, b: &str) -> usize {
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
