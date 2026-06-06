//! # Mini-CRM — Dynamic Workflow CRM Operations
//!
//! A demonstration of how heartbit's dynamic workflow system (inspired by
//! Claude Code's dynamic workflows) powers realistic CRM operations for
//! **PulsarData CRM**.
//!
//! ## Workflow Patterns Demonstrated
//!
//! | Pattern | CRM Use Case | Key API |
//! |---------|-------------|---------|
//! | Parallel fan-out | Contact enrichment | `flow::parallel()` |
//! | Pipeline streaming | Deal stage progression | `flow::pipeline()` |
//! | Budget control | Email campaign batching | `WorkflowCtx::budget()` |
//! | Structured output | Lead scoring | `AgentCall::schema::<T>()` |
//! | Nested workflows | Customer onboarding | `flow::workflow()` |
//! | Phases + logging | Observability | `flow::phase()`, `flow::log()` |
//!
//! ## Architecture
//!
//! ```text
//! main.rs           ← orchestrator: builds WorkflowCtx, runs all demos
//! models.rs         ← CRM domain types (Contact, Deal, Company, etc.)
//! mock_provider.rs  ← mock LLM provider producing CRM-shaped responses
//! workflows/
//!   mod.rs
//!   enrichment.rs   ← parallel contact enrichment
//!   deal_pipeline.rs← no-barrier pipeline for deal stage transitions
//!   campaign.rs     ← budget-bounded batch email campaign
//!   scoring.rs      ← parallel lead scoring with structured output
//!   onboarding.rs   ← nested sub-workflow for sequential onboarding
//! ```

mod mock_provider;
mod models;
mod workflows;

use std::sync::Arc;

use heartbit_core::flow;
use heartbit_core::flow::ctx::WorkflowCtx;
use heartbit_core::flow::journal::ResumeMode;
use heartbit_core::flow::progress::ProgressTracker;
use heartbit_core::llm::BoxedProvider;

use crate::mock_provider::CrmMockProvider;
use crate::models::{Company, Contact, Deal, DealStage, Ticket};

/// Build a [`WorkflowCtx`] backed by a mock CRM provider with sensible defaults:
/// concurrency cap of 4, runaway backstop of 50, and a 10 000-token budget.
#[allow(dead_code)] // Available for integration tests
fn build_crm_ctx() -> WorkflowCtx {
    let responses = CrmMockProvider::responses_for_demo();
    let provider = Arc::new(BoxedProvider::new(CrmMockProvider::new(responses)));

    WorkflowCtx::builder(provider)
        .max_concurrency(4)
        .max_agents(50)
        .budget(10_000)
        .build()
        .expect("build CRM workflow context")
}

/// Build a ctx with journaling enabled for deterministic resume.
fn build_journaled_ctx(path: &std::path::Path) -> WorkflowCtx {
    let responses = CrmMockProvider::responses_for_demo();
    let provider = Arc::new(BoxedProvider::new(CrmMockProvider::new(responses)));

    WorkflowCtx::builder(provider)
        .max_concurrency(4)
        .max_agents(50)
        .budget(10_000)
        .journal(path, ResumeMode::Fresh)
        .expect("open journal")
        .build()
        .expect("build journaled CRM context")
}

#[tokio::main]
async fn main() -> Result<(), heartbit_core::Error> {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║  PulsarData CRM — Dynamic Workflow Engine Demo          ║");
    println!("║  Powered by heartbit-core flow combinators              ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Install a progress tracker so we can report at the end.
    let tracker = ProgressTracker::new();
    let responses = CrmMockProvider::responses_for_demo();
    let provider = Arc::new(BoxedProvider::new(CrmMockProvider::new(responses)));
    let ctx = WorkflowCtx::builder(provider)
        .max_concurrency(4)
        .max_agents(50)
        .budget(10_000)
        .on_event(tracker.callback())
        .build()
        .expect("build CRM context");

    // ---- 1. Parallel Contact Enrichment ----
    println!("━━━ Workflow 1: Parallel Contact Enrichment ━━━");
    let _phase = flow::phase(&ctx, "contact-enrichment");
    workflows::enrichment::enrich_contacts(&ctx, &sample_contacts()).await?;
    drop(_phase);
    println!();

    // ---- 2. Deal Pipeline Processing ----
    println!("━━━ Workflow 2: Deal Pipeline Processing ━━━");
    let _phase = flow::phase(&ctx, "deal-pipeline");
    workflows::deal_pipeline::process_deals(&ctx, &sample_deals()).await?;
    drop(_phase);
    println!();

    // ---- 3. Budget-Bounded Email Campaign ----
    println!("━━━ Workflow 3: Email Campaign (Budget-Bounded) ━━━");
    let _phase = flow::phase(&ctx, "email-campaign");
    workflows::campaign::run_email_campaign(&ctx, &sample_contacts()).await?;
    drop(_phase);
    println!();

    // ---- 4. Parallel Lead Scoring ----
    println!("━━━ Workflow 4: Parallel Lead Scoring ━━━");
    let _phase = flow::phase(&ctx, "lead-scoring");
    workflows::scoring::score_leads(&ctx, &sample_contacts()).await?;
    drop(_phase);
    println!();

    // ---- 5. Nested Onboarding Workflow ----
    println!("━━━ Workflow 5: Customer Onboarding (Nested Sub-Workflow) ━━━");
    let _phase = flow::phase(&ctx, "onboarding");
    workflows::onboarding::onboard_customer(&ctx, &sample_contacts()[0], &sample_companies()[0])
        .await?;
    drop(_phase);
    println!();

    // ---- 6. Heterogeneous Parallel: Ticket Triage ----
    println!("━━━ Workflow 6: Heterogeneous Parallel — Ticket Triage ━━━");
    let _phase = flow::phase(&ctx, "ticket-triage");
    workflows::enrichment::triage_tickets(&ctx, &sample_tickets()).await?;
    drop(_phase);
    println!();

    // ---- Summary ----
    let progress = tracker.snapshot();
    println!("━━━ Run Summary ━━━");
    println!("  Agents started:  {}", progress.agents_started);
    println!("  Agents finished:  {}", progress.agents_finished);
    println!("  Agents skipped:  {}", progress.agents_skipped);
    println!("  Log lines:       {}", progress.log_lines);
    println!(
        "  Tokens used:     {} in, {} out",
        progress.total_tokens.input_tokens, progress.total_tokens.output_tokens
    );

    if let Some(breach) = ctx.control_breach() {
        println!("  ⚠ Control breach: {:?}", breach);
    } else {
        println!("  ✓ No control breaches — run completed cleanly");
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Sample data generators
// ---------------------------------------------------------------------------

fn sample_contacts() -> Vec<Contact> {
    vec![
        Contact {
            id: "c-001".into(),
            name: "Alice Chen".into(),
            email: "alice@acmecorp.io".into(),
            company_id: Some("co-001".into()),
            title: Some("VP Engineering".into()),
            ..Default::default()
        },
        Contact {
            id: "c-002".into(),
            name: "Bob Martinez".into(),
            email: "bob@nexus-tech.com".into(),
            company_id: Some("co-002".into()),
            title: Some("CTO".into()),
            ..Default::default()
        },
        Contact {
            id: "c-003".into(),
            name: "Carol Dubois".into(),
            email: "carol@quantum-ai.fr".into(),
            company_id: Some("co-003".into()),
            title: Some("Head of Product".into()),
            ..Default::default()
        },
        Contact {
            id: "c-004".into(),
            name: "David Kim".into(),
            email: "david@helix-labs.io".into(),
            company_id: None,
            title: Some("Senior Engineer".into()),
            ..Default::default()
        },
    ]
}

fn sample_deals() -> Vec<Deal> {
    vec![
        Deal {
            id: "d-101".into(),
            name: "AcmeCorp Enterprise License".into(),
            company_id: "co-001".into(),
            value: 125_000.0,
            stage: DealStage::Qualified,
            contact_id: "c-001".into(),
        },
        Deal {
            id: "d-102".into(),
            name: "Nexus-Tech Platform Migration".into(),
            company_id: "co-002".into(),
            value: 85_000.0,
            stage: DealStage::Discovery,
            contact_id: "c-002".into(),
        },
        Deal {
            id: "d-103".into(),
            name: "Quantum-AI Pilot".into(),
            company_id: "co-003".into(),
            value: 35_000.0,
            stage: DealStage::Proposal,
            contact_id: "c-003".into(),
        },
    ]
}

fn sample_companies() -> Vec<Company> {
    vec![Company {
        id: "co-001".into(),
        name: "AcmeCorp".into(),
        industry: Some("Enterprise Software".into()),
        employee_count: Some(500),
    }]
}

fn sample_tickets() -> Vec<Ticket> {
    vec![
        Ticket {
            id: "t-501".into(),
            subject: "Cannot access dashboard after upgrade".into(),
            priority: "high".into(),
            contact_id: "c-001".into(),
            status: "open".into(),
        },
        Ticket {
            id: "t-502".into(),
            subject: "API rate limiting too aggressive".into(),
            priority: "medium".into(),
            contact_id: "c-002".into(),
            status: "open".into(),
        },
        Ticket {
            id: "t-503".into(),
            subject: "Feature request: bulk export contacts".into(),
            priority: "low".into(),
            contact_id: "c-003".into(),
            status: "open".into(),
        },
    ]
}
