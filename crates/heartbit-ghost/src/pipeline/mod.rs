//! Generation pipeline — wires the P1.3a sub-agent recipes into a working
//! single-candidate path.
//!
//! Public entry: [`run_pipeline`] (added in Task 3 / Task 2 wiring).

pub mod prompts;
pub mod publish_gate;
pub mod style_render;
pub mod verdicts;

pub use publish_gate::{PublishGateError, check_publish_gate};
pub use style_render::render_style_profile_as_english;
pub use verdicts::{FactVerdict, StyleVerdict, parse_critic_verdict, parse_fact_verdict};
