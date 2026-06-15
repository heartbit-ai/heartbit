//! Built-in guardrail implementations — LLM judge, secret scanner, PII detector, content fence, action budget, behavioral monitor, injection classifier, tool policy, and composition helpers.

pub mod action_budget;
pub mod behavioral;
pub mod cascade;
pub mod compose;
pub mod content_fence;
pub mod function_call;
pub mod injection;
pub mod llm_judge;
pub mod pii;
pub mod scope_guard;
pub mod secret_scanner;
pub mod sensor_security;
pub mod tool_policy;

pub use action_budget::{ActionBudgetGuardrail, ActionBudgetGuardrailBuilder, BudgetRule};
pub use behavioral::{BehaviorRule, BehavioralMonitorGuardrail, BehavioralMonitorGuardrailBuilder};
pub use cascade::{CascadingGuardrail, CheapScreen};
pub use compose::{ConditionalGuardrail, GuardrailChain, WarnToDeny};
#[allow(deprecated)]
pub use content_fence::ContentFenceGuardrail;
pub use function_call::FunctionCallGuard;
pub use injection::{GuardrailMode, InjectionClassifierGuardrail};
pub use llm_judge::{LlmJudgeGuardrail, LlmJudgeGuardrailBuilder};
pub use pii::{PiiAction, PiiDetector, PiiGuardrail};
pub use scope_guard::ScopeGuard;
pub use secret_scanner::{SecretAction, SecretScannerGuardrail, SecretScannerGuardrailBuilder};
pub use sensor_security::SensorSecurityGuardrail;
pub use tool_policy::{InputConstraint, ToolPolicyGuardrail, ToolRule};
