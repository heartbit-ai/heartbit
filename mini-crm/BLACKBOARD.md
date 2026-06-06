# Blackboard

## agent:researcher — Mini-CRM Investigation Results

### 1. Does it compile?
**YES** — `cargo check -p mini-crm` succeeds with 2 warnings:
- `warning: function 'build_journaled_ctx' is never used` (mini-crm/src/main.rs:64)
- `warning: struct 'EmailDraft' is never constructed` (mini-crm/src/models.rs:93)

### 2. Is there a web interface?
**NO** — There is no web interface whatsoever.

Evidence:
- **Cargo.toml has NO web framework** — Dependencies are only: `heartbit-core`, `tokio`, `serde`, `serde_json`, `chrono`, `uuid`. No actix-web, axum, warp, rocket, or any other web framework.
- **No HTML/CSS/JS files** anywhere in `mini-crm/` — Only `.rs` and `.md` files exist.
- **No HTTP routes/handlers** — The source code has zero references to HTTP, requests, routes, servers, listen, bind, ports, etc.
- **No CLI arguments** — `cargo run -p mini-crm` runs directly with no `--help` flag support. The `main()` function is hardcoded to run all 6 demo workflows sequentially.

### 3. What would be needed to add a web interface?
To add a web UI, the following would be required:

1. **Add a web framework dependency** to `mini-crm/Cargo.toml` (e.g., `axum`, `actix-web`, or `warp`)
2. **Create HTTP route handlers** that wrap the existing workflow functions (`enrich_contacts`, `process_deals`, `run_email_campaign`, `score_leads`, `onboard_customer`, `triage_tickets`)
3. **Add a web server starter** in `main.rs` (e.g., `axum::Server::bind(...)` instead of or alongside the current batch run)
4. **Build a frontend** — HTML/CSS/JS files for a dashboard UI (contact list, deal pipeline view, campaign status, lead scores, etc.)
5. **Add state management** — Currently all sample data is hardcoded in `sample_contacts()`, `sample_deals()`, etc. A real web app would need persistent storage (DB or in-memory store)
6. **Add API endpoints** for CRUD operations on contacts, deals, companies, tickets
7. **Real-time updates** (optional) — WebSocket/SSE for live workflow progress

### 4. Main entry points and how the application works

**Entry point:** `mini-crm/src/main.rs` — `#[tokio::main] async fn main()`

**Architecture:**
```
mini-crm/
├── Cargo.toml              # Pure Rust binary, depends only on heartbit-core
├── README.md               # Detailed documentation
├── IMPLEMENTATION_SUMMARY.md
└── src/
    ├── main.rs             # Orchestrator: builds WorkflowCtx, runs 6 demo workflows sequentially
    ├── models.rs           # CRM domain types: Contact, Company, Deal, Ticket, LeadScore, DealAnalysis, TicketTriage, EmailDraft, ScoredLead
    ├── mock_provider.rs    # CrmMockProvider: mock LLM that cycles through pre-built CRM-shaped responses
    └── workflows/
        ├── mod.rs          # Module declarations
        ├── enrichment.rs   # Parallel contact enrichment + heterogeneous ticket triage
        ├── deal_pipeline.rs # 3-stage pipeline (analyze → recommend → notify) for deals
        ├── campaign.rs     # Budget-bounded serial email drafting loop
        ├── scoring.rs      # Parallel lead scoring with structured output (schema-validated)
        └── onboarding.rs   # Sequential onboarding with nested sub-workflow (parallel integration+training)
```

**How it works:**
1. Builds a `WorkflowCtx` with: max_concurrency=4, max_agents=50, budget=10,000 tokens, progress tracker
2. Runs 6 workflows sequentially, each in its own `flow::phase()` scope:
   - **Workflow 1:** Parallel enrichment of 4 contacts using `flow::parallel()` (fail-soft)
   - **Workflow 2:** 3-stage deal pipeline for 3 deals using `flow::pipeline()` (no-barrier streaming)
   - **Workflow 3:** Budget-bounded email campaign for 4 contacts (serial loop, pauses when budget < 200 tokens)
   - **Workflow 4:** Parallel lead scoring for 4 contacts with `schema::<LeadScore>()` structured output ⚠️ **FAILS** — mock provider returns text, not `__respond__` tool calls (response cycling issue)
   - **Workflow 5:** Nested sub-workflow onboarding (provision → import → [parallel: integrations + training] → health check)
   - **Workflow 6:** Heterogeneous parallel ticket triage with `BoxThunk` ⚠️ **FAILS** — same mock response cycling issue
3. Prints a run summary with agent counts, token usage, and control breach status

**Runtime issues observed:**
- Workflows 4 & 6 fail because the mock provider cycles responses sequentially — by the time scoring/triage runs, the responses at the cursor position are text responses (not `__respond__` tool-use responses), so structured output fails with "LLM returned text without calling __respond__"
- The deal pipeline (Workflow 2) shows mismatched responses because response cycling doesn't align with the expected response types per workflow stage
- Despite these failures, the application completes successfully (exit code 0) thanks to fail-soft semantics

**All output is `println!` to stdout** — no file I/O, no database, no network.
