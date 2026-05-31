//! `SKILL.md` manifest: YAML frontmatter parsing + validation.
//!
//! Implements the Claude Agent Skills `SKILL.md` format
//! (<https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview>):
//! a `---`-fenced YAML frontmatter block followed by a markdown body. The
//! frontmatter declares the skill's identity (`name`, `description`) plus
//! optional `allowed-tools`, `version`, `license`, and a freeform `metadata` map.
//!
//! Validation follows the published overview spec:
//! - `name`: ≤64 chars, `[a-z0-9-]` only, equal to the skill's directory name,
//!   not containing a reserved word (`anthropic`/`claude`).
//! - `description`: non-empty, ≤1024 chars, no angle brackets.
//!
//! The reserved-word rule comes from the Agent Skills *overview* doc ("name
//! cannot contain reserved words: 'anthropic', 'claude'"); note Anthropic's
//! `quick_validate.py` does not implement it — we follow the published doc.
//! Unknown frontmatter keys are ignored (lenient-in) rather than rejected, for
//! forward-compatibility; the rest of the rules match the spec.
//!
//! Parsing is deliberately strict so a malformed skill is rejected with an
//! actionable message rather than silently injecting garbage into a prompt.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::error::Error;

/// Maximum length of a skill `name`, per the Claude Agent Skills spec.
pub const MAX_NAME_LEN: usize = 64;

/// Maximum length of a skill `description`, per the Claude Agent Skills spec.
pub const MAX_DESCRIPTION_LEN: usize = 1024;

/// Words a skill `name` may not contain (case-insensitive), per the spec.
const RESERVED_WORDS: &[&str] = &["anthropic", "claude"];

/// A parsed, validated `SKILL.md` manifest: its frontmatter metadata plus the
/// markdown body that follows the closing `---` fence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillManifest {
    /// Unique skill identifier (equals the skill directory name).
    pub name: String,
    /// What the skill does and when to use it — the Level-1 discovery signal.
    pub description: String,
    /// Tools the skill is permitted to use while active (empty = unrestricted).
    pub allowed_tools: Vec<String>,
    /// Optional semantic version string.
    pub version: Option<String>,
    /// Optional license identifier.
    pub license: Option<String>,
    /// Freeform string metadata (nested YAML values are flattened to their
    /// scalar string form; non-scalar values are skipped).
    pub metadata: BTreeMap<String, String>,
    /// The markdown body after the frontmatter (Level-2 instructions).
    pub body: String,
}

/// Raw frontmatter as deserialized from YAML, before validation/normalization.
#[derive(Debug, Deserialize)]
struct Frontmatter {
    name: String,
    description: String,
    #[serde(rename = "allowed-tools", default)]
    allowed_tools: Option<serde_yaml::Value>,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    license: Option<String>,
    #[serde(default)]
    metadata: BTreeMap<String, serde_yaml::Value>,
}

/// Split a `SKILL.md` into `(frontmatter_yaml, body)`. The content must begin
/// with a `---` fence on its own line; the frontmatter ends at the next line
/// that is exactly `---`. Returns an error if no well-formed fence is present.
fn split_frontmatter(content: &str) -> Result<(&str, &str), Error> {
    // Normalize only for the leading-fence check; we slice the original `content`.
    let trimmed = content.strip_prefix('\u{feff}').unwrap_or(content);
    let after_open = trimmed
        .strip_prefix("---\n")
        .or_else(|| trimmed.strip_prefix("---\r\n"))
        .ok_or_else(|| {
            Error::Config("SKILL.md must begin with a '---' YAML frontmatter fence".into())
        })?;

    // Find the closing fence: a line that is exactly "---".
    let mut offset = 0usize;
    for line in after_open.split_inclusive('\n') {
        let stripped = line.trim_end_matches(['\n', '\r']);
        if stripped == "---" {
            let frontmatter = &after_open[..offset];
            let body = &after_open[offset + line.len()..];
            return Ok((frontmatter, body));
        }
        offset += line.len();
    }
    Err(Error::Config(
        "SKILL.md frontmatter is not closed by a '---' fence".into(),
    ))
}

/// Normalize the `allowed-tools` field, which the spec permits as either a
/// comma-separated string or a YAML list of strings.
fn normalize_allowed_tools(value: Option<serde_yaml::Value>) -> Result<Vec<String>, Error> {
    match value {
        None | Some(serde_yaml::Value::Null) => Ok(Vec::new()),
        Some(serde_yaml::Value::String(s)) => Ok(s
            .split(',')
            .map(str::trim)
            .filter(|t| !t.is_empty())
            .map(String::from)
            .collect()),
        Some(serde_yaml::Value::Sequence(seq)) => seq
            .into_iter()
            .map(|v| match v {
                serde_yaml::Value::String(s) => Ok(s.trim().to_string()),
                other => Err(Error::Config(format!(
                    "allowed-tools list entries must be strings, got {other:?}"
                ))),
            })
            .filter(|r| !matches!(r, Ok(s) if s.is_empty()))
            .collect(),
        Some(other) => Err(Error::Config(format!(
            "allowed-tools must be a string or a list of strings, got {other:?}"
        ))),
    }
}

/// Flatten a freeform metadata map to scalar strings; drop non-scalar values.
fn flatten_metadata(raw: BTreeMap<String, serde_yaml::Value>) -> BTreeMap<String, String> {
    raw.into_iter()
        .filter_map(|(k, v)| scalar_to_string(&v).map(|s| (k, s)))
        .collect()
}

/// Render a scalar YAML value as a string, or `None` for sequences/mappings.
fn scalar_to_string(v: &serde_yaml::Value) -> Option<String> {
    match v {
        serde_yaml::Value::String(s) => Some(s.clone()),
        serde_yaml::Value::Bool(b) => Some(b.to_string()),
        serde_yaml::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

/// Validate a skill `name` against the Claude Agent Skills spec. `dir_name` is
/// the on-disk directory the skill lives in; `name` must equal it.
fn validate_name(name: &str, dir_name: &str) -> Result<(), Error> {
    if name.is_empty() {
        return Err(Error::Config("skill name must be non-empty".into()));
    }
    if name.len() > MAX_NAME_LEN {
        return Err(Error::Config(format!(
            "skill name '{name}' exceeds {MAX_NAME_LEN} characters"
        )));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(Error::Config(format!(
            "skill name '{name}' must contain only lowercase letters, digits, and hyphens"
        )));
    }
    let lower = name.to_ascii_lowercase();
    for reserved in RESERVED_WORDS {
        if lower.contains(reserved) {
            return Err(Error::Config(format!(
                "skill name '{name}' must not contain the reserved word '{reserved}'"
            )));
        }
    }
    if name != dir_name {
        return Err(Error::Config(format!(
            "skill name '{name}' must equal its directory name '{dir_name}'"
        )));
    }
    Ok(())
}

/// Validate a skill `description` against the spec.
fn validate_description(description: &str) -> Result<(), Error> {
    if description.is_empty() {
        return Err(Error::Config("skill description must be non-empty".into()));
    }
    if description.chars().count() > MAX_DESCRIPTION_LEN {
        return Err(Error::Config(format!(
            "skill description exceeds {MAX_DESCRIPTION_LEN} characters"
        )));
    }
    if description.contains('<') || description.contains('>') {
        return Err(Error::Config(
            "skill description must not contain angle brackets ('<' or '>')".into(),
        ));
    }
    Ok(())
}

impl SkillManifest {
    /// Parse and validate a `SKILL.md`. `dir_name` is the skill's directory name,
    /// which `name` must equal (the spec's directory==name rule).
    pub fn parse(content: &str, dir_name: &str) -> Result<SkillManifest, Error> {
        let (frontmatter_yaml, body) = split_frontmatter(content)?;
        let fm: Frontmatter = serde_yaml::from_str(frontmatter_yaml)
            .map_err(|e| Error::Config(format!("invalid SKILL.md frontmatter YAML: {e}")))?;

        validate_name(&fm.name, dir_name)?;
        validate_description(&fm.description)?;

        Ok(SkillManifest {
            name: fm.name,
            description: fm.description,
            allowed_tools: normalize_allowed_tools(fm.allowed_tools)?,
            version: fm.version,
            license: fm.license,
            metadata: flatten_metadata(fm.metadata),
            body: body.to_string(),
        })
    }

    /// Render the Level-1 catalog line for this skill: `name: description`.
    pub fn catalog_line(&self) -> String {
        format!("{}: {}", self.name, self.description)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = "---\nname: pdf-processing\ndescription: Extract text and tables from PDF files. Use when working with PDFs.\n---\n# PDF Processing\n\nDo the thing.\n";

    #[test]
    fn parses_a_valid_skill() {
        let m = SkillManifest::parse(VALID, "pdf-processing").expect("valid");
        assert_eq!(m.name, "pdf-processing");
        assert!(m.description.starts_with("Extract text"));
        assert!(m.body.contains("# PDF Processing"));
        assert!(m.allowed_tools.is_empty());
        assert_eq!(m.version, None);
    }

    #[test]
    fn catalog_line_is_name_colon_description() {
        let m = SkillManifest::parse(VALID, "pdf-processing").expect("valid");
        assert_eq!(
            m.catalog_line(),
            "pdf-processing: Extract text and tables from PDF files. Use when working with PDFs."
        );
    }

    #[test]
    fn missing_frontmatter_fence_is_rejected() {
        let err = SkillManifest::parse("# Just a body, no fence\n", "x").unwrap_err();
        assert!(format!("{err}").contains("frontmatter fence"), "{err}");
    }

    #[test]
    fn unclosed_frontmatter_is_rejected() {
        let err = SkillManifest::parse("---\nname: x\ndescription: y\n", "x").unwrap_err();
        assert!(format!("{err}").contains("not closed"), "{err}");
    }

    #[test]
    fn malformed_yaml_is_rejected() {
        let bad = "---\nname: [unclosed\ndescription: y\n---\nbody\n";
        let err = SkillManifest::parse(bad, "x").unwrap_err();
        assert!(format!("{err}").contains("frontmatter YAML"), "{err}");
    }

    #[test]
    fn name_must_equal_dir_name() {
        let err = SkillManifest::parse(VALID, "other-dir").unwrap_err();
        assert!(
            format!("{err}").contains("must equal its directory name"),
            "{err}"
        );
    }

    #[test]
    fn name_with_uppercase_is_rejected() {
        let c = "---\nname: PdfProcessing\ndescription: d\n---\nb\n";
        let err = SkillManifest::parse(c, "PdfProcessing").unwrap_err();
        assert!(format!("{err}").contains("lowercase"), "{err}");
    }

    #[test]
    fn name_too_long_is_rejected() {
        let long = "a".repeat(65);
        let c = format!("---\nname: {long}\ndescription: d\n---\nb\n");
        let err = SkillManifest::parse(&c, &long).unwrap_err();
        assert!(format!("{err}").contains("64 characters"), "{err}");
    }

    #[test]
    fn reserved_word_in_name_is_rejected() {
        let c = "---\nname: claude-helper\ndescription: d\n---\nb\n";
        let err = SkillManifest::parse(c, "claude-helper").unwrap_err();
        assert!(format!("{err}").contains("reserved word"), "{err}");
    }

    #[test]
    fn empty_description_is_rejected() {
        let c = "---\nname: x\ndescription: \"\"\n---\nb\n";
        let err = SkillManifest::parse(c, "x").unwrap_err();
        assert!(
            format!("{err}").contains("description must be non-empty"),
            "{err}"
        );
    }

    #[test]
    fn description_too_long_is_rejected() {
        let long = "d".repeat(MAX_DESCRIPTION_LEN + 1);
        let c = format!("---\nname: x\ndescription: {long}\n---\nb\n");
        let err = SkillManifest::parse(&c, "x").unwrap_err();
        assert!(format!("{err}").contains("1024 characters"), "{err}");
    }

    #[test]
    fn description_with_angle_brackets_is_rejected() {
        let c = "---\nname: x\ndescription: do <script> things\n---\nb\n";
        let err = SkillManifest::parse(c, "x").unwrap_err();
        assert!(format!("{err}").contains("angle brackets"), "{err}");
    }

    #[test]
    fn allowed_tools_csv_is_split() {
        let c = "---\nname: x\ndescription: d\nallowed-tools: read, write , bash\n---\nb\n";
        let m = SkillManifest::parse(c, "x").expect("valid");
        assert_eq!(m.allowed_tools, vec!["read", "write", "bash"]);
    }

    #[test]
    fn allowed_tools_list_is_accepted() {
        let c = "---\nname: x\ndescription: d\nallowed-tools:\n  - read\n  - grep\n---\nb\n";
        let m = SkillManifest::parse(c, "x").expect("valid");
        assert_eq!(m.allowed_tools, vec!["read", "grep"]);
    }

    #[test]
    fn metadata_scalars_are_captured() {
        let c = "---\nname: x\ndescription: d\nmetadata:\n  author: pascal\n  level: 3\n---\nb\n";
        let m = SkillManifest::parse(c, "x").expect("valid");
        assert_eq!(m.metadata.get("author").map(String::as_str), Some("pascal"));
        assert_eq!(m.metadata.get("level").map(String::as_str), Some("3"));
    }

    #[test]
    fn missing_name_or_description_is_rejected() {
        let no_name = "---\ndescription: d\n---\nb\n";
        assert!(SkillManifest::parse(no_name, "x").is_err());
        let no_desc = "---\nname: x\n---\nb\n";
        assert!(SkillManifest::parse(no_desc, "x").is_err());
    }

    #[test]
    fn crlf_frontmatter_is_supported() {
        let c = "---\r\nname: x\r\ndescription: d\r\n---\r\nbody\r\n";
        let m = SkillManifest::parse(c, "x").expect("valid");
        assert_eq!(m.name, "x");
        assert!(m.body.contains("body"));
    }

    #[test]
    fn version_and_license_are_optional_and_captured() {
        let c = "---\nname: x\ndescription: d\nversion: 1.2.0\nlicense: MIT\n---\nb\n";
        let m = SkillManifest::parse(c, "x").expect("valid");
        assert_eq!(m.version.as_deref(), Some("1.2.0"));
        assert_eq!(m.license.as_deref(), Some("MIT"));
    }
}
