//! CI test: every canonical_file in heartbit-rs-features.toml must exist.
//!
//! This catches stale menu entries when files are renamed or deleted
//! without updating the menu.

use std::path::PathBuf;

#[derive(serde::Deserialize)]
struct FeatureMenu {
    feature: Vec<FeatureEntry>,
}

#[derive(serde::Deserialize)]
struct FeatureEntry {
    name: String,
    canonical_file: String,
}

#[test]
fn every_canonical_file_in_feature_menu_exists() {
    let menu_path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data/heartbit-rs-features.toml");
    let text = std::fs::read_to_string(&menu_path).expect("menu file readable");
    let menu: FeatureMenu = toml::from_str(&text).expect("menu parses");
    // Resolve relative to workspace root (one level up from crate dir).
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let mut missing = Vec::new();
    for f in &menu.feature {
        let p = workspace_root.join(&f.canonical_file);
        if !p.exists() {
            missing.push(format!("  {} -> {}", f.name, f.canonical_file));
        }
    }
    assert!(
        missing.is_empty(),
        "feature menu has stale canonical_file paths:\n{}",
        missing.join("\n")
    );
}
