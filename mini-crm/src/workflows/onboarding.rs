//! # Sequential Customer Onboarding (Nested Sub-Workflow)
//!
//! **Pattern:** `flow::workflow()` — one-level-deep sub-workflow sharing the
//! parent's budget, journal, and cancellation token.
//!
//! Onboarding a new customer is a multi-step sequential process where each step
//! depends on the previous one's output:
//!
//! 1. Provision account (uses customer data)
//! 2. Import historical data (uses provisioned account details)
//! 3. Configure integrations (uses import status)
//! 4. Schedule training (uses integration status)
//! 5. Post-onboarding health check (uses all above)
//!
//! These steps MUST be sequential — you can't configure integrations before
//! the account exists. The sub-workflow shares the parent's budget pool, so
//! a budget breach during onboarding halts the run-wide budget. The parent's
//! cancellation token is also shared, so a user stop cancels onboarding.
//!
//! Inside the sub-workflow, steps 3 and 4 are independent of each other
//! (integrations don't depend on training scheduling), so they use
//! `parallel()` within the sub-workflow.

use heartbit_core::Error;
use heartbit_core::flow::ctx::WorkflowCtx;
use heartbit_core::flow::parallel::{BoxThunk, parallel, thunk};
use heartbit_core::flow::{agent, log, phase, workflow};

use crate::models::{Company, Contact};

/// Run the full onboarding workflow for a new customer.
///
/// Uses `workflow()` to create a child context that shares the parent's
/// budget, concurrency cap, and cancellation. The child gets a fresh phase
/// scope, so onboarding agents group under "customer-onboarding" without
/// affecting the parent's phase.
pub async fn onboard_customer(
    ctx: &WorkflowCtx,
    primary_contact: &Contact,
    company: &Company,
) -> Result<(), Error> {
    let contact_name = primary_contact.name.clone();
    let company_name = company.name.clone();
    let company_id = company.id.clone();

    log(
        ctx,
        format!("Starting onboarding for {contact_name} at {company_name}"),
    );

    // Step 1: Provision account (sequential — must complete before anything else).
    let _phase = phase(ctx, "provision");
    let provision_result = agent(
        ctx,
        format!(
            "Provision a new PulsarData CRM account for company '{company_name}' \
         (ID: {company_id}). Primary contact: {contact_name} ({}).\n\n\
         Set up workspace, SSO, and user seats. Return provisioning summary.",
            primary_contact.email,
        ),
    )
    .label("provision-account")
    .run()
    .await?;

    let provision_summary = provision_result.unwrap_or_default();
    println!(
        "  🏗️  Account provisioned: {}…",
        &provision_summary[..provision_summary.len().min(70)]
    );
    drop(_phase);

    // Step 2: Data import (sequential — depends on provisioned account).
    let _phase = phase(ctx, "data-import");
    let import_result = agent(ctx, format!(
        "Import historical CRM data into the newly provisioned account for '{company_name}'.\n\n\
         Provisioning summary: {provision_summary}\n\n\
         Migration plan: contacts, deals, companies CSV imports. Return import summary.",
    ))
    .label("import-data")
    .run()
    .await?;

    let import_summary = import_result.unwrap_or_default();
    println!(
        "  📦 Data imported: {}…",
        &import_summary[..import_summary.len().min(70)]
    );
    drop(_phase);

    // Step 3+4: Configure integrations AND schedule training (parallel — independent).
    // This is a nested sub-workflow: the child shares parent's budget, cancel, etc.
    let parallel_results: Vec<Option<String>> = workflow(ctx, "integration-training", |child| {
        let company_name = company_name.clone();
        let contact_name = contact_name.clone();
        let import_summary = import_summary.clone();
        async move {
            let _phase = phase(&child, "integration-training");
            log(
                &child,
                "Running integration setup and training scheduling in parallel",
            );

            // Heterogeneous parallel: two distinct tasks sharing the child ctx.
            // BoxThunk type is `Option<String>` to match the child's parallel signature.
            let thunks: Vec<BoxThunk<Option<String>>> = vec![
                thunk({
                    let child = child.clone();
                    let company_name = company_name.clone();
                    let import_summary = import_summary.clone();
                    move || async move {
                        let result = agent(
                            &child,
                            format!(
                                "Configure CRM integrations for '{company_name}'.\n\
                             Data import status: {import_summary}\n\n\
                             Set up: Salesforce sync, Slack notifications, webhook \
                             endpoints. Return integration summary.",
                            ),
                        )
                        .label("configure-integrations")
                        .run()
                        .await;

                        match &result {
                            Ok(Some(text)) => {
                                println!("  🔗 Integrations: {}…", &text[..text.len().min(60)]);
                            }
                            _ => println!("  ⊘ Integrations skipped/failed"),
                        }
                        result
                    }
                }),
                thunk({
                    let child = child.clone();
                    let contact_name = contact_name.clone();
                    move || async move {
                        let result = agent(
                            &child,
                            format!(
                                "Schedule onboarding training for {contact_name}'s team.\n\n\
                             Set up training session, prepare materials, send invites. \
                             Return training schedule.",
                            ),
                        )
                        .label("schedule-training")
                        .run()
                        .await;

                        match &result {
                            Ok(Some(text)) => {
                                println!("  📚 Training: {}…", &text[..text.len().min(60)]);
                            }
                            _ => println!("  ⊘ Training skipped/failed"),
                        }
                        result
                    }
                }),
            ];

            let results = parallel(&child, thunks).await;
            Ok(results.into_iter().flatten().collect())
        }
    })
    .await?;

    let parallel_done = parallel_results.iter().filter(|r| r.is_some()).count();
    log(
        ctx,
        format!("Integration + training: {parallel_done}/2 completed"),
    );

    // Step 5: Post-onboarding health check (sequential — depends on all above).
    let _phase = phase(ctx, "health-check");
    let health_result = agent(
        ctx,
        format!(
            "Run post-onboarding health check for '{company_name}'.\n\
         Account: provisioned\n\
         Data: imported\n\
         Integrations: {parallel_done}/2 configured\n\n\
         Verify all systems operational, confirm first sync, assign CSM, \
         schedule 30-day check-in. Return health report.",
        ),
    )
    .label("onboarding-health-check")
    .run()
    .await?;

    if let Some(health) = &health_result {
        println!("  ✅ Health check: {}…", &health[..health.len().min(70)]);
    }
    drop(_phase);

    log(
        ctx,
        format!("Onboarding complete for {contact_name} at {company_name}"),
    );
    println!("  ── Onboarding: {contact_name} fully onboarded at {company_name}");

    Ok(())
}
