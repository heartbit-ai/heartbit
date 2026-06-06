//! # Parallel Contact Enrichment
//!
//! **Pattern:** `flow::parallel()` — fail-soft barrier fan-out.
//!
//! Each contact's enrichment (LinkedIn lookup, phone validation, engagement
//! scoring) is independent, so we fan out all contacts concurrently. If one
//! enrichment fails (e.g. data provider is down for one contact), the others
//! still complete — this is the "fail-soft" semantics of `parallel()`,
//! contrasting with the fail-fast `ParallelAgent` compile-time workflow.
//!
//! Results arrive in submission order regardless of completion order.

use heartbit_core::Error;
use heartbit_core::flow::ctx::WorkflowCtx;
use heartbit_core::flow::parallel::{BoxThunk, parallel, thunk};
use heartbit_core::flow::{agent, log};

use crate::models::{Contact, Ticket};

/// Enrich a batch of contacts in parallel.
///
/// Uses the homogeneous `map` fan-out (all thunks have the same closure type,
/// so no `BoxThunk` boxing is needed). The concurrency cap on `WorkflowCtx`
/// still bounds how many actually run simultaneously.
pub async fn enrich_contacts(ctx: &WorkflowCtx, contacts: &[Contact]) -> Result<(), Error> {
    log(
        ctx,
        format!(
            "Starting parallel enrichment of {} contacts",
            contacts.len()
        ),
    );

    // Build one thunk per contact. Each thunk calls agent() to:
    //   1. Look up external data sources (LinkedIn, Clearbit, etc.)
    //   2. Score engagement based on recent activity
    //   3. Return an enrichment summary
    //
    // These are independent — no shared mutable state, perfect for parallel().
    let thunks: Vec<_> = contacts
        .iter()
        .map(|contact| {
            let ctx = ctx.clone();
            let contact = contact.clone();
            move || async move {
                let prompt = format!(
                    "Enrich contact profile for CRM. Contact: {name} ({email}), \
                     title: {title}, company_id: {company}.\n\n\
                     Look up their LinkedIn, recent conference activity, engagement \
                     signals, and phone number. Return a summary.",
                    name = contact.name,
                    email = contact.email,
                    title = contact.title.as_deref().unwrap_or("Unknown"),
                    company = contact.company_id.as_deref().unwrap_or("None"),
                );
                let result = agent(&ctx, prompt)
                    .label(format!("enrich-{}", contact.id))
                    .phase("contact-enrichment")
                    .run()
                    .await;

                match &result {
                    Ok(Some(text)) => {
                        println!(
                            "  ✓ {} enriched: {}…",
                            contact.name,
                            &text[..text.len().min(80)]
                        );
                    }
                    Ok(None) => {
                        println!("  ⊘ {} skipped (cancelled)", contact.name);
                    }
                    Err(e) => {
                        println!("  ✗ {} failed: {}", contact.name, e);
                    }
                }

                // Fail-soft: convert Err to None so siblings are unaffected.
                result.map(|opt| opt.unwrap_or_default())
            }
        })
        .collect();

    // Fan out: all thunks run concurrently, barrier waits for all to finish.
    // Results are in submission order (matching `contacts` ordering).
    let results: Vec<Option<String>> = parallel(ctx, thunks).await;

    let enriched = results.iter().filter(|r| r.is_some()).count();
    let failed = results.len() - enriched;
    log(
        ctx,
        format!("Enrichment complete: {enriched} succeeded, {failed} failed/skipped"),
    );
    println!(
        "  ── Enrichment: {enriched}/{results_len} succeeded",
        results_len = results.len()
    );

    Ok(())
}

/// Triage support tickets using **heterogeneous** parallel fan-out.
///
/// Unlike the homogeneous contact enrichment above, each ticket might need a
/// different triage approach (bug vs. feature request vs. security). We use
/// `thunk()` to type-erase distinct closures into `BoxThunk` so they can share
/// a `Vec`. Results use structured output (`schema::<TicketTriage>()`) so
/// the LLM must produce validated fields.
///
/// **Pattern:** heterogeneous `parallel()` with `BoxThunk` + structured output.
pub async fn triage_tickets(ctx: &WorkflowCtx, tickets: &[Ticket]) -> Result<(), Error> {
    log(
        ctx,
        format!(
            "Triaging {} tickets with heterogeneous parallel",
            tickets.len()
        ),
    );

    let thunks: Vec<BoxThunk<Option<String>>> = tickets
        .iter()
        .map(|ticket| {
            let ctx = ctx.clone();
            let ticket = ticket.clone();
            thunk(move || async move {
                let prompt = format!(
                    "Triage this CRM support ticket.\n\n\
                     Ticket ID: {id}\n\
                     Subject: {subject}\n\
                     Priority: {priority}\n\
                     Reporter: {contact}\n\n\
                     Classify the ticket into category (bug|feature_request|account|\
                     performance|security), assign severity (1-5), suggest an assignee \
                     team, and determine SLA target in hours.",
                    id = ticket.id,
                    subject = ticket.subject,
                    priority = ticket.priority,
                    contact = ticket.contact_id,
                );
                let result = agent(&ctx, prompt)
                    .label(format!("triage-{}", ticket.id))
                    .schema::<crate::models::TicketTriage>()
                    .run()
                    .await;

                match &result {
                    Ok(Some(triage)) => {
                        println!(
                            "  ✓ Ticket {}: category={}, severity={}, assignee={}, SLA={}h",
                            ticket.id,
                            triage.category,
                            triage.severity,
                            triage.suggested_assignee,
                            triage.sla_hours,
                        );
                    }
                    Ok(None) => {
                        println!("  ⊘ Ticket {} skipped", ticket.id);
                    }
                    Err(e) => {
                        println!("  ✗ Ticket {} triage failed: {}", ticket.id, e);
                    }
                }
                Ok(result?.map(|t| format!("{t:?}")))
            })
        })
        .collect();

    let results: Vec<Option<Option<String>>> = parallel(ctx, thunks).await;
    let triaged = results
        .iter()
        .filter(|r| matches!(r, Some(Some(_))))
        .count();
    log(
        ctx,
        format!(
            "Ticket triage complete: {triaged}/{} triaged",
            tickets.len()
        ),
    );
    println!("  ── Triage: {triaged}/{} completed", tickets.len());

    Ok(())
}
