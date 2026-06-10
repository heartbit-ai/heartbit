//! Regression tests for `#[heartbit_tool]` with `#[tool(default = ...)]` on a
//! NON-`Option` parameter (audit 2026-06-09, heartbit-macro lib.rs:339).
//!
//! Run with: `cargo test -p heartbit --test macro_tool_default --features macro`
//! (the `heartbit_tool` re-export is behind the off-by-default `macro` feature,
//! so this file is empty without it).
#![cfg(feature = "macro")]
//!
//! Before the fix, the generated JSON schema advertised the default (telling
//! the LLM the argument is optional-with-default) while ALSO listing the param
//! as `required`, and the runtime deserializer errored with
//! "missing required field" when the argument was omitted. The macro must
//! instead APPLY the declared default: a defaulted param is not `required`,
//! and an absent argument falls back to the default value.

use heartbit::{Error, ExecutionContext, Tool, ToolOutput, heartbit_tool};
use serde_json::json;

#[heartbit_tool(description = "Echo a query with a capped result count")]
async fn capped_search(
    /// The query string
    query: String,
    /// Maximum results to return
    #[tool(default = 10)]
    max_results: u32,
) -> Result<ToolOutput, Error> {
    Ok(ToolOutput::success(format!("{query}:{max_results}")))
}

#[test]
fn defaulted_non_option_param_is_not_required_in_schema() {
    let def = CappedSearch.definition();
    let required: Vec<String> = def
        .input_schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    assert!(
        required.contains(&"query".to_string()),
        "non-defaulted param must stay required: {required:?}"
    );
    assert!(
        !required.contains(&"max_results".to_string()),
        "param with #[tool(default)] must NOT be required: {required:?}"
    );
    assert_eq!(
        def.input_schema["properties"]["max_results"]["default"],
        json!(10),
        "schema must still advertise the default"
    );
}

#[tokio::test]
async fn defaulted_non_option_param_applies_default_when_absent() {
    let ctx = ExecutionContext::default();
    let out = CappedSearch
        .execute(&ctx, json!({"query": "rust"}))
        .await
        .expect("omitting a defaulted param must not error");
    assert_eq!(out.content, "rust:10");
}

#[tokio::test]
async fn defaulted_non_option_param_applies_default_when_null() {
    let ctx = ExecutionContext::default();
    let out = CappedSearch
        .execute(&ctx, json!({"query": "rust", "max_results": null}))
        .await
        .expect("explicit null on a defaulted param must fall back to the default");
    assert_eq!(out.content, "rust:10");
}

#[tokio::test]
async fn defaulted_non_option_param_uses_provided_value() {
    let ctx = ExecutionContext::default();
    let out = CappedSearch
        .execute(&ctx, json!({"query": "rust", "max_results": 3}))
        .await
        .expect("provided value must win over the default");
    assert_eq!(out.content, "rust:3");
}

#[tokio::test]
async fn non_defaulted_param_still_errors_when_missing() {
    let ctx = ExecutionContext::default();
    let err = CappedSearch
        .execute(&ctx, json!({"max_results": 5}))
        .await
        .expect_err("a required param without a default must still error");
    assert!(
        err.to_string().contains("missing required field `query`"),
        "unexpected error: {err}"
    );
}
