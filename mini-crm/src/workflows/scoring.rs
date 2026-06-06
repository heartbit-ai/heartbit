//! # Parallel Lead Scoring with Structured Output
//!
//! **Pattern:** `flow::parallel()` + `AgentCall::schema::<T>()` — fail-soft
//! fan-out where each agent must produce a validated structured output.
//!
//! Lead scoring is a classic use case for structured output: the LLM must
//! return a `{ score, tier, rationale }` object, not free-form prose. The
//! `.schema::<LeadScore>()` builder injects a `__respond__` tool, validates
//! the JSON payload against the schema, and retries on mismatch. If the
//! model fails to produce valid output, the slot collapses to `None`
//! (fail-soft) — other leads are still scored.

use heartbit_core::Error;
use heartbit_core::flow::ctx::WorkflowCtx;
use heartbit_core::flow::parallel::parallel;
use heartbit_core::flow::{agent, log};

use crate::models::{Contact, LeadScore};

/// Score all leads in parallel using structured output.
///
/// Each lead gets its own `AgentCall` with `.schema::<LeadScore>()`, forcing
/// the LLM to produce `{ score: u8, tier: "hot"|"warm"|"cold", rationale }`.
/// The `serde` deserialization is an additional safety net beyond the JSON
/// Schema validator (e.g. it catches `"tier": "maybe"` which passes the
/// string schema but isn't a valid enum variant).
pub async fn score_leads(ctx: &WorkflowCtx, contacts: &[Contact]) -> Result<(), Error> {
    log(ctx, format!("Scoring {} leads in parallel", contacts.len()));

    let thunks: Vec<_> = contacts
        .iter()
        .map(|contact| {
            let ctx = ctx.clone();
            let contact = contact.clone();
            move || async move {
                let prompt = format!(
                    "Score this CRM lead for sales prioritization.\n\n\
                     Contact: {name}\n\
                     Title: {title}\n\
                     Email: {email}\n\
                     Company ID: {company}\n\n\
                     Evaluate based on:\n\
                     - Job title and seniority (buying authority)\n\
                     - Company size and industry fit\n\
                     - Engagement signals\n\n\
                     Return score (0-100), tier (hot/warm/cold), and rationale.",
                    name = contact.name,
                    title = contact.title.as_deref().unwrap_or("Unknown"),
                    email = contact.email,
                    company = contact.company_id.as_deref().unwrap_or("None"),
                );

                // The schema() call transforms AgentCall<NoSchema> into
                // AgentCall<LeadScore>, so run() returns Option<LeadScore>.
                let scored = agent(&ctx, prompt)
                    .label(format!("score-{}", contact.id))
                    .phase("lead-scoring")
                    .schema::<LeadScore>()
                    .run()
                    .await;

                match &scored {
                    Ok(Some(lead)) => {
                        println!(
                            "  🎯 Lead {} ({}): score={}/100, tier={:?}, rationale=\"{}…\"",
                            contact.id,
                            contact.name,
                            lead.score,
                            lead.tier,
                            &lead.rationale[..lead.rationale.len().min(60)],
                        );
                    }
                    Ok(None) => {
                        println!("  ⊘ Lead {} scoring cancelled", contact.name);
                    }
                    Err(e) => {
                        println!("  ✗ Lead {} scoring failed: {}", contact.name, e);
                    }
                }

                // Fail-soft: Err -> None so other leads are unaffected.
                scored.map(|opt| opt.map(|ls| ls.score))
            }
        })
        .collect();

    let results: Vec<Option<Option<u8>>> = parallel(ctx, thunks).await;

    // Summarize scores.
    let scores: Vec<u8> = results
        .iter()
        .filter_map(|r| r.as_ref().and_then(|inner| *inner))
        .collect();

    if scores.is_empty() {
        log(ctx, "No leads were successfully scored");
        println!("  ── Scoring: 0/{} leads scored", contacts.len());
        return Ok(());
    }

    let avg = scores.iter().sum::<u8>() as f64 / scores.len() as f64;
    let hot = scores.iter().filter(|&&s| s >= 75).count();
    let warm = scores.iter().filter(|&&s| s >= 50 && s < 75).count();
    let cold = scores.iter().filter(|&&s| s < 50).count();

    log(
        ctx,
        format!(
            "Scoring complete: {} scored, avg={avg:.1}, hot={hot}, warm={warm}, cold={cold}",
            scores.len()
        ),
    );
    println!(
        "  ── Scoring: {} scored, avg={avg:.1} (🔥{hot} 🟡{warm} ❄️{cold})",
        scores.len()
    );

    Ok(())
}
