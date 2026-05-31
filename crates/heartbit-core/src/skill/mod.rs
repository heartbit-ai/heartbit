//! Agent Skills: filesystem-based, progressively-disclosed capabilities.
//!
//! A *skill* is a directory containing a `SKILL.md` (YAML frontmatter + markdown
//! body) plus optional bundled scripts and reference files, matching the Claude
//! Agent Skills format. This module adds the two pieces the existing
//! [`SkillTool`](crate::tool) lacked for SOTA parity:
//!
//! 1. [`SkillManifest`] — a strict `SKILL.md` frontmatter parser + validator.
//! 2. [`SkillRegistry`] — discovery across skill directories with a Level-1
//!    catalog (`name: description`) for injection into the system prompt, so the
//!    model knows which skills exist and when to use them (progressive
//!    disclosure level 1). The body (level 2) and bundled files (level 3) load
//!    on demand via the skill tool and the `read` builtin.

pub mod discovery;
pub mod manifest;
pub mod registry;

pub use discovery::{catalog_for, catalog_from_dirs, search_dirs};
pub use manifest::SkillManifest;
pub use registry::SkillRegistry;
