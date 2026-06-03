//! # Deal Pipeline Processing
//!
//! **Pattern:** `flow::pipeline()` — no-barrier per-item streaming through stages.
//!
//! Deals advance through a multi-stage pipeline: analysis → recommendation →
//! notification. Unlike `SequentialAgent` (barrier: all items at stage N before
//! any at N+1), `pipeline()` has NO inter-item barrier — deal A can be in the
//! notification stage while deal B is still being analyzed. This gives
//! wall-clock = slowest single item, not the sum of per-stage maxima.
//!
//! Each stage receives `(prev_result, original_item, index)` and returns the
//! next flow value. An `Err` at any stage drops that item; other items are
//! unaffected.

use heartbit_core::Error;
use heartbit_core::flow::ctx::WorkflowCtx;
use heartbit_core::flow::pipeline::pipeline;
use heartbit_core::flow::{agent, log};

use serde_json::Value;
use std::sync::Arc;

use crate::models::Deal;

/// Process deals through a no-barrier pipeline of stages:
///
/// 1. **Analyze** — LLM health-check + probability-to-close
/// 2. **Recommend** — generate next-action recommendations based on analysis
/// 3. **Notify** — produce notification text for the deal owner
///
/// Each deal streams through all three stages concurrently with its siblings.
pub async fn process_deals(ctx: &WorkflowCtx, deals: &[Deal]) -> Result<(), Error> {
    log(
        ctx,
        format!("Processing {} deals through pipeline", deals.len()),
    );

    // Clone ctx so stages can capture it with 'static lifetime.
    // Each pipeline stage closure needs an owned `WorkflowCtx` (it's cheap to
    // clone — just an Arc bump) because the stage signatures require 'static.
    let stage1_ctx = ctx.clone();
    let stage2_ctx = ctx.clone();
    let stage3_ctx = ctx.clone();
    let results = pipeline(ctx, deals.to_vec())
        // Stage 1: Analyze deal health using structured output.
        .stage(move |_prev: Value, deal: Arc<Deal>, _idx: usize| {
            let ctx = stage1_ctx.clone();
            async move {
                let prompt = format!(
                    "Analyze this CRM deal for pipeline review.\n\n\
                     Deal: {name} (${value:.0})\n\
                     Stage: {stage}\n\
                     Company: {company}\n\n\
                     Assess health (green/yellow/red), recommend next action, \
                     identify risk factors, and estimate probability to close (0-100).",
                    name = deal.name,
                    value = deal.value,
                    stage = deal.stage,
                    company = deal.company_id,
                );

                // Use structured output to get a validated DealAnalysis.
                let analysis = agent(&ctx, prompt)
                    .label(format!("analyze-{}", deal.id))
                    .schema::<crate::models::DealAnalysis>()
                    .run()
                    .await?
                    .ok_or_else(|| Error::Agent("deal analysis cancelled".into()))?;

                println!(
                    "  📊 Deal {} [{}]: health={}, P(close)={}%",
                    deal.id, deal.name, analysis.health, analysis.probability_to_close
                );

                // Pass the analysis forward as JSON for the next stage.
                Ok(serde_json::to_value(&analysis)?)
            }
        })
        // Stage 2: Generate a recommendation based on the analysis.
        .stage(move |analysis: Value, deal: Arc<Deal>, _idx: usize| {
            let ctx = stage2_ctx.clone();
            async move {
                let next_action = analysis
                    .get("next_action")
                    .and_then(|v| v.as_str())
                    .unwrap_or("Review manually");
                let health = analysis
                    .get("health")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                let prompt = format!(
                    "Given deal '{}' (health: {health}, next action: {next_action}), \
                     generate a concise CRM recommendation for the account executive. \
                     Include urgency level and timeline.",
                    deal.name,
                );

                let recommendation = agent(&ctx, prompt)
                    .label(format!("recommend-{}", deal.id))
                    .run()
                    .await?
                    .unwrap_or_default();

                println!(
                    "  💡 Deal {} recommendation: {}…",
                    deal.id,
                    &recommendation[..recommendation.len().min(70)]
                );

                Ok(Value::String(recommendation))
            }
        })
        // Stage 3: Produce notification text for the deal owner.
        .stage(move |recommendation: Value, deal: Arc<Deal>, _idx: usize| {
            let ctx = stage3_ctx.clone();
            async move {
                let rec_text = recommendation.as_str().unwrap_or("(no recommendation)");

                let prompt = format!(
                    "Generate a brief CRM notification for deal '{}' based on this \
                     recommendation: {rec_text}",
                    deal.name,
                );

                let notification = agent(&ctx, prompt)
                    .label(format!("notify-{}", deal.id))
                    .run()
                    .await?
                    .unwrap_or_default();

                println!(
                    "  🔔 Deal {} notification: {}…",
                    deal.id,
                    &notification[..notification.len().min(60)]
                );

                Ok(Value::String(notification))
            }
        })
        .run()
        .await;

    let completed = results.iter().filter(|r| r.is_some()).count();
    let dropped = results.len() - completed;
    log(
        ctx,
        format!("Pipeline: {completed} deals processed, {dropped} dropped"),
    );
    println!(
        "  ── Pipeline: {completed}/{} deals completed",
        results.len()
    );

    Ok(())
}
