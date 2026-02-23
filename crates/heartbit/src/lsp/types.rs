use serde::{Deserialize, Serialize};

/// LSP diagnostic severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Error = 1,
    Warning = 2,
    Information = 3,
    Hint = 4,
}

impl DiagnosticSeverity {
    pub fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Error,
            2 => Self::Warning,
            3 => Self::Information,
            _ => Self::Hint,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warning => "warning",
            Self::Information => "info",
            Self::Hint => "hint",
        }
    }
}

/// A position in a text document (0-indexed line and character).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

/// A range in a text document.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

/// A diagnostic reported by the language server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub range: Range,
    pub severity: DiagnosticSeverity,
    pub message: String,
}

/// Parameters for `textDocument/publishDiagnostics` notification.
///
/// Not yet used — will be needed when we handle push-model diagnostics.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct PublishDiagnosticsParams {
    pub uri: String,
    pub diagnostics: Vec<RawDiagnostic>,
}

/// Raw diagnostic from JSON-RPC — severity is a number.
#[derive(Debug, Clone, Deserialize)]
pub struct RawDiagnostic {
    pub range: Range,
    #[serde(default = "default_severity")]
    pub severity: u8,
    pub message: String,
}

fn default_severity() -> u8 {
    1 // Error
}

impl RawDiagnostic {
    pub fn into_diagnostic(self) -> Diagnostic {
        Diagnostic {
            range: self.range,
            severity: DiagnosticSeverity::from_u8(self.severity),
            message: self.message,
        }
    }
}

/// Format diagnostics as an XML block for tool output.
pub fn format_diagnostics(path: &str, diagnostics: &[Diagnostic]) -> String {
    if diagnostics.is_empty() {
        return String::new();
    }
    let mut out = format!("<lsp-diagnostics file=\"{path}\">\n");
    for d in diagnostics {
        // Display 1-indexed line numbers for human readability
        let line = d.range.start.line + 1;
        out.push_str(&format!(
            "{}[line {}]: {}\n",
            d.severity.label(),
            line,
            d.message
        ));
    }
    out.push_str("</lsp-diagnostics>");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnostic_severity_from_u8() {
        assert_eq!(DiagnosticSeverity::from_u8(1), DiagnosticSeverity::Error);
        assert_eq!(DiagnosticSeverity::from_u8(2), DiagnosticSeverity::Warning);
        assert_eq!(
            DiagnosticSeverity::from_u8(3),
            DiagnosticSeverity::Information
        );
        assert_eq!(DiagnosticSeverity::from_u8(4), DiagnosticSeverity::Hint);
        assert_eq!(DiagnosticSeverity::from_u8(99), DiagnosticSeverity::Hint);
    }

    #[test]
    fn diagnostic_severity_label() {
        assert_eq!(DiagnosticSeverity::Error.label(), "error");
        assert_eq!(DiagnosticSeverity::Warning.label(), "warning");
        assert_eq!(DiagnosticSeverity::Information.label(), "info");
        assert_eq!(DiagnosticSeverity::Hint.label(), "hint");
    }

    #[test]
    fn format_diagnostics_empty() {
        assert_eq!(format_diagnostics("/tmp/test.rs", &[]), "");
    }

    #[test]
    fn format_diagnostics_single_error() {
        let diagnostics = vec![Diagnostic {
            range: Range {
                start: Position {
                    line: 41,
                    character: 0,
                },
                end: Position {
                    line: 41,
                    character: 5,
                },
            },
            severity: DiagnosticSeverity::Error,
            message: "expected `;`".into(),
        }];
        let output = format_diagnostics("/path/to/file.rs", &diagnostics);
        assert!(output.contains("<lsp-diagnostics file=\"/path/to/file.rs\">"));
        assert!(output.contains("error[line 42]: expected `;`"));
        assert!(output.contains("</lsp-diagnostics>"));
    }

    #[test]
    fn format_diagnostics_multiple() {
        let diagnostics = vec![
            Diagnostic {
                range: Range {
                    start: Position {
                        line: 0,
                        character: 0,
                    },
                    end: Position {
                        line: 0,
                        character: 1,
                    },
                },
                severity: DiagnosticSeverity::Error,
                message: "syntax error".into(),
            },
            Diagnostic {
                range: Range {
                    start: Position {
                        line: 14,
                        character: 4,
                    },
                    end: Position {
                        line: 14,
                        character: 5,
                    },
                },
                severity: DiagnosticSeverity::Warning,
                message: "unused variable `x`".into(),
            },
        ];
        let output = format_diagnostics("/tmp/test.rs", &diagnostics);
        assert!(output.contains("error[line 1]: syntax error"));
        assert!(output.contains("warning[line 15]: unused variable `x`"));
    }

    #[test]
    fn raw_diagnostic_into_diagnostic() {
        let raw = RawDiagnostic {
            range: Range {
                start: Position {
                    line: 10,
                    character: 0,
                },
                end: Position {
                    line: 10,
                    character: 5,
                },
            },
            severity: 2,
            message: "unused import".into(),
        };
        let d = raw.into_diagnostic();
        assert_eq!(d.severity, DiagnosticSeverity::Warning);
        assert_eq!(d.message, "unused import");
    }

    #[test]
    fn raw_diagnostic_default_severity() {
        let json = r#"{"range":{"start":{"line":0,"character":0},"end":{"line":0,"character":1}},"message":"oops"}"#;
        let raw: RawDiagnostic = serde_json::from_str(json).unwrap();
        assert_eq!(raw.severity, 1); // default = Error
    }

    #[test]
    fn diagnostic_serde_roundtrip() {
        let d = Diagnostic {
            range: Range {
                start: Position {
                    line: 5,
                    character: 10,
                },
                end: Position {
                    line: 5,
                    character: 15,
                },
            },
            severity: DiagnosticSeverity::Warning,
            message: "test".into(),
        };
        let json = serde_json::to_string(&d).unwrap();
        let parsed: Diagnostic = serde_json::from_str(&json).unwrap();
        assert_eq!(d, parsed);
    }
}
