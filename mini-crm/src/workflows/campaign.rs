//! # Batch Email Campaign with Budget Control
//!
//! **Pattern:** Budget-bounded loop using `WorkflowCtx::remaining()` combined
//! with `flow::parallel()` for the send phase.
//!
//! A real campaign might have 10,000 contacts. We use the workflow budget as a
//! hard ceiling on token spend: when `remaining()` drops below the per-contact
//! cost threshold, we stop admitting new agents. The budget is shared across
//! all agents in the run (it's an atomic counter inside `WorkflowCtx`), so
//! overspend is bounded by `concurrency × per-agent-cost`.
//!
//! This demonstrates the "loop-until-budget" idiom from the Claude Code target:
//! ```text
//! while ctx.remaining() >= per_contact_cost {
//!     send_email(&ctx, contact)?;
//! }
//! ```

use heartbit_core::Error;
use heartbit_core::flow::ctx::WorkflowCtx;
use heartbit_core::flow::{agent, log};

use crate::models::Contact;

/// Estimated token cost per email draft (input + output).
/// Used as the admission threshold for the budget loop.
const COST_PER_DRAFT: u64 = 200;

/// Run a batch email campaign with budget-bounded admission control.
///
/// Contacts are processed one at a time (serial admission) but each draft
/// counts against the shared budget. When remaining budget drops below
/// `COST_PER_DRAFT`, the loop stops — the campaign is paused, not failed,
/// and can be resumed later.
///
/// This serial-with-budget pattern is the CRM analog of Claude Code's
/// "loop until budget exhausted" idiom.
pub async fn run_email_campaign(ctx: &WorkflowCtx, contacts: &[Contact]) -> Result<(), Error> {
    log(
        ctx,
        format!(
            "Starting email campaign for {} contacts (budget: {} remaining, per-draft: {})",
            contacts.len(),
            ctx.remaining(),
            COST_PER_DRAFT,
        ),
    );

    let mut drafted = 0u32;
    let mut budget_exhausted = false;

    for contact in contacts {
        // ---- Budget admission: stop if remaining < per-draft cost ----
        if ctx.remaining() < COST_PER_DRAFT {
            budget_exhausted = true;
            log(
                ctx,
                format!(
                    "Campaign paused: budget remaining ({}) < per-draft cost ({})",
                    ctx.remaining(),
                    COST_PER_DRAFT,
                ),
            );
            break;
        }

        // Check if the run was cancelled (e.g. user hit stop).
        if ctx.is_cancelled() {
            log(ctx, "Campaign cancelled by user");
            break;
        }

        let prompt = format!(
            "Draft a personalized email for the PulsarData AI Summit \n\n\
             Recipient: {name} ({title})\n\
             Email: {email}\n\
             Company engagement: active partner\n\n\
             Make it warm, professional, and reference their role.",
            name = contact.name,
            title = contact.title.as_deref().unwrap_or("Valued customer"),
            email = contact.email,
        );

        let result = agent(ctx, prompt)
            .label(format!("draft-{}", contact.id))
            .phase("email-campaign")
            .run()
            .await;

        match result {
            Ok(Some(text)) => {
                let subject_line = text.lines().next().unwrap_or("(no subject)");
                println!(
                    "  ✉️  Drafted for {}: {}…",
                    contact.name,
                    &subject_line[..subject_line.len().min(50)]
                );
                drafted += 1;
            }
            Ok(None) => {
                println!("  ⊘ Draft for {} cancelled", contact.name);
            }
            Err(e) => {
                // Agent-domain error: log but continue the campaign.
                // The budget is not yet exhausted, so we try the next contact.
                println!("  ✗ Draft for {} failed: {}", contact.name, e);
            }
        }
    }

    let status = if budget_exhausted {
        "budget exhausted"
    } else if ctx.is_cancelled() {
        "cancelled"
    } else {
        "complete"
    };

    log(
        ctx,
        format!(
            "Campaign {status}: {drafted}/{} drafted, budget spent: {}",
            contacts.len(),
            ctx.budget().spent(),
        ),
    );
    println!(
        "  ── Campaign: {drafted}/{} drafted ({status})",
        contacts.len()
    );

    Ok(())
}
