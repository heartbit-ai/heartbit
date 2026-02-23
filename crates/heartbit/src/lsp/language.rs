use std::path::Path;

/// Configuration for a language server.
#[derive(Debug, Clone)]
pub struct LanguageConfig {
    /// Language identifier (e.g., "rust", "python").
    pub lang_id: &'static str,
    /// Server command (e.g., "rust-analyzer").
    pub command: &'static str,
    /// Command arguments.
    pub args: &'static [&'static str],
}

/// Built-in language server configurations.
pub const BUILTIN_SERVERS: &[LanguageConfig] = &[
    LanguageConfig {
        lang_id: "rust",
        command: "rust-analyzer",
        args: &[],
    },
    LanguageConfig {
        lang_id: "typescript",
        command: "npx",
        args: &["typescript-language-server", "--stdio"],
    },
    LanguageConfig {
        lang_id: "python",
        command: "pyright-langserver",
        args: &["--stdio"],
    },
    LanguageConfig {
        lang_id: "go",
        command: "gopls",
        args: &["serve"],
    },
    LanguageConfig {
        lang_id: "c",
        command: "clangd",
        args: &[],
    },
];

/// Detect the language ID from a file extension.
///
/// Returns `None` for unsupported extensions.
pub fn detect_language(path: &Path) -> Option<&'static str> {
    let ext = path.extension()?.to_str()?;
    match ext {
        "rs" => Some("rust"),
        "ts" | "tsx" => Some("typescript"),
        "js" | "jsx" | "mjs" | "cjs" => Some("typescript"),
        "py" | "pyi" => Some("python"),
        "go" => Some("go"),
        "c" | "h" | "cpp" | "cxx" | "cc" | "hpp" | "hxx" => Some("c"),
        _ => None,
    }
}

/// Find the built-in server config for a language ID.
pub fn find_server_config(lang_id: &str) -> Option<&'static LanguageConfig> {
    BUILTIN_SERVERS.iter().find(|c| c.lang_id == lang_id)
}

/// File-modifying tool names that should trigger LSP diagnostics.
const FILE_MODIFYING_TOOLS: &[&str] = &["write", "edit", "patch"];

/// Check if a tool name is a file-modifying tool.
pub fn is_file_modifying_tool(tool_name: &str) -> bool {
    FILE_MODIFYING_TOOLS.contains(&tool_name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn detect_rust() {
        assert_eq!(detect_language(&PathBuf::from("src/main.rs")), Some("rust"));
    }

    #[test]
    fn detect_typescript() {
        assert_eq!(
            detect_language(&PathBuf::from("app/index.ts")),
            Some("typescript")
        );
        assert_eq!(
            detect_language(&PathBuf::from("app/App.tsx")),
            Some("typescript")
        );
    }

    #[test]
    fn detect_javascript_maps_to_typescript() {
        assert_eq!(
            detect_language(&PathBuf::from("lib/utils.js")),
            Some("typescript")
        );
        assert_eq!(
            detect_language(&PathBuf::from("lib/utils.jsx")),
            Some("typescript")
        );
        assert_eq!(
            detect_language(&PathBuf::from("lib/utils.mjs")),
            Some("typescript")
        );
        assert_eq!(
            detect_language(&PathBuf::from("lib/utils.cjs")),
            Some("typescript")
        );
    }

    #[test]
    fn detect_python() {
        assert_eq!(detect_language(&PathBuf::from("script.py")), Some("python"));
        assert_eq!(detect_language(&PathBuf::from("types.pyi")), Some("python"));
    }

    #[test]
    fn detect_go() {
        assert_eq!(detect_language(&PathBuf::from("main.go")), Some("go"));
    }

    #[test]
    fn detect_c_family() {
        assert_eq!(detect_language(&PathBuf::from("main.c")), Some("c"));
        assert_eq!(detect_language(&PathBuf::from("main.cpp")), Some("c"));
        assert_eq!(detect_language(&PathBuf::from("main.h")), Some("c"));
        assert_eq!(detect_language(&PathBuf::from("main.hpp")), Some("c"));
        assert_eq!(detect_language(&PathBuf::from("main.cc")), Some("c"));
        assert_eq!(detect_language(&PathBuf::from("main.cxx")), Some("c"));
        assert_eq!(detect_language(&PathBuf::from("main.hxx")), Some("c"));
    }

    #[test]
    fn detect_unsupported_returns_none() {
        assert_eq!(detect_language(&PathBuf::from("README.md")), None);
        assert_eq!(detect_language(&PathBuf::from("Cargo.toml")), None);
        assert_eq!(detect_language(&PathBuf::from("data.json")), None);
    }

    #[test]
    fn detect_no_extension_returns_none() {
        assert_eq!(detect_language(&PathBuf::from("Makefile")), None);
    }

    #[test]
    fn find_server_config_rust() {
        let config = find_server_config("rust").unwrap();
        assert_eq!(config.command, "rust-analyzer");
    }

    #[test]
    fn find_server_config_typescript() {
        let config = find_server_config("typescript").unwrap();
        assert_eq!(config.command, "npx");
    }

    #[test]
    fn find_server_config_unknown() {
        assert!(find_server_config("brainfuck").is_none());
    }

    #[test]
    fn file_modifying_tool_detection() {
        assert!(is_file_modifying_tool("write"));
        assert!(is_file_modifying_tool("edit"));
        assert!(is_file_modifying_tool("patch"));
        assert!(!is_file_modifying_tool("read"));
        assert!(!is_file_modifying_tool("bash"));
        assert!(!is_file_modifying_tool("search"));
    }

    #[test]
    fn builtin_servers_has_five_entries() {
        assert_eq!(BUILTIN_SERVERS.len(), 5);
    }
}
