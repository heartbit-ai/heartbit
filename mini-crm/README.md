# Mini-CRM: Dynamic Workflow Engine Demo

A demonstration of how heartbit's dynamic workflow system (inspired by Claude Code's dynamic workflows) powers realistic CRM operations for **PulsarData CRM**.

## Overview

This project showcases the `flow` module from `heartbit-core`, which provides async combinators for orchestrating AI agents with patterns like:

- **Parallel fan-out** — fail-soft barrier execution
- **Pipeline streaming** — no-barrier per-item streaming through stages
- **Budget control** — hard token budget ceiling with admission control
- **Structured output** — schema-validated LLM responses
- **Nested workflows** — one-level-deep sub-workflows sharing budget/cancellation
- **Phases & logging** — observability and progress tracking

## Architecture

```
mini-crm/
├── Cargo.toml
├── README.md
└── src/
    ├── main.rs              # Orchestrator: builds WorkflowCtx, runs all demos
    ├── models.rs            # CRM domain types (Contact, Deal, Company, etc.)
    ├── mock_provider.rs     # Mock LLM provider with CRM-shaped responses
    └── workflows/
        ├── mod.rs
        ├── enrichment.rs    # Parallel contact enrichment
        ├── deal_pipeline.rs # No-barrier pipeline for deal stages
        ├── campaign.rs      # Budget-bounded batch email campaign
        ├── scoring.rs       # Parallel lead scoring with structured output
        └── onboarding.rs    # Nested sub-workflow for sequential onboarding
```

## Workflow Patterns Demonstrated

### 1. Parallel Contact Enrichment
**Pattern:** `flow::parallel()` — fail-soft barrier fan-out

Each contact's enrichment (LinkedIn lookup, phone validation, engagement scoring) runs concurrently. If one enrichment fails, others complete — this is the "fail-soft" semantics, contrasting with fail-fast compile-time workflows.

```rust
let thunks: Vec<_> = contacts.iter().map(|contact| {
    let ctx = ctx.clone();
    move || async move {
        agent(&ctx, enrichment_prompt)
            .label(format!("enrich-{}", contact.id))
            .run()
            .await
    }
}).collect();

let results = parallel(ctx, thunks).await;
```

### 2. Deal Pipeline Processing
**Pattern:** `flow::pipeline()` — no-barrier per-item streaming

Deals advance through multi-stage pipeline (analysis → recommendation → notification). Unlike `SequentialAgent` (barrier: all items at stage N before any at N+1), `pipeline()` has NO inter-item barrier — deal A can be in notification while deal B is still being analyzed.

```rust
let results = pipeline(ctx, deals.to_vec())
    .stage(move |_prev, deal, _idx| {
        // Stage 1: Analyze with structured output
        agent(&ctx, prompt).schema::<DealAnalysis>().run()
    })
    .stage(|analysis, deal, _idx| {
        // Stage 2: Generate recommendation
    })
    .stage(|recommendation, deal, _idx| {
        // Stage 3: Produce notification
    })
    .run()
    .await;
```

### 3. Batch Email Campaign with Budget Control
**Pattern:** Budget-bounded loop using `WorkflowCtx::remaining()`

Contacts are processed one at a time. When `remaining()` drops below per-contact cost threshold, the loop stops — campaign is paused, not failed, and can be resumed later.

```rust
for contact in contacts {
    if ctx.remaining() < COST_PER_DRAFT {
        log(ctx, "Campaign paused: budget exhausted");
        break;
    }
    
    let draft = agent(ctx, prompt).run().await?;
    // ... handle result
}
```

### 4. Parallel Lead Scoring with Structured Output
**Pattern:** `flow::parallel()` + `AgentCall::schema::<T>()`

Each lead gets a validated structured output: `{ score: u8, tier: "hot"|"warm"|"cold", rationale }`. The LLM must call the `__respond__` tool with schema-valid JSON; serde deserialization adds another safety net.

```rust
let scored = agent(&ctx, prompt)
    .label(format!("score-{}", contact.id))
    .schema::<LeadScore>()
    .run()
    .await?;
```

### 5. Sequential Customer Onboarding (Nested Sub-Workflow)
**Pattern:** `flow::workflow()` — one-level-deep sub-workflow

Onboarding is multi-step sequential (provision → import → integrations → training → health-check). The sub-workflow shares parent's budget, journal, and cancellation token. Inside, steps 3+4 (integrations + training) are parallel and independent.

```rust
let results = workflow(ctx, "integration-training", |child| {
    async move {
        let _phase = phase(&child, "integration-training");
        
        // Parallel: integrations + training (independent)
        let thunks = vec![
            thunk(|| agent(&child, integration_prompt).run()),
            thunk(|| agent(&child, training_prompt).run()),
        ];
        
        let results = parallel(&child, thunks).await;
        Ok(results)
    }
}).await?;
```

### 6. Heterogeneous Parallel: Ticket Triage
**Pattern:** Heterogeneous `parallel()` with `BoxThunk` + structured output

Different tickets need different triage approaches (bug vs. feature request vs. security). Use `thunk()` to type-erase distinct closures into `BoxThunk` so they can share a `Vec`.

```rust
let thunks: Vec<BoxThunk<Option<String>>> = tickets.iter().map(|ticket| {
    thunk(move || {
        agent(&ctx, prompt)
            .schema::<TicketTriage>()
            .run()
            .await
    })
}).collect();

let results = parallel(ctx, thunks).await;
```

## Running the Demo

```bash
# Build
cargo build -p mini-crm

# Run (uses mock LLM provider)
cargo run -p mini-crm
```

The demo will:
1. Enrich 4 contacts in parallel
2. Process 3 deals through a 3-stage pipeline
3. Draft emails for 4 contacts (budget-bounded)
4. Score 4 leads with structured output
5. Onboard 1 customer with nested workflow
6. Triage 3 tickets with heterogeneous parallel

All using a mock LLM provider that returns realistic CRM-shaped responses.

## Key Features of the Workflow Engine

### WorkflowCtx Builder
```rust
let ctx = WorkflowCtx::builder(Arc::new(provider))
    .max_concurrency(4)           // Parallel execution cap
    .max_agents(50)               // Runaway agent backstop
    .budget(10_000)               // Hard token ceiling
    .journal(path, ResumeMode::Fresh)  // Deterministic resume
    .on_event(tracker.callback())      // Progress tracking
    .build()?;
```

### Agent Call Builder
```rust
let result = agent(&ctx, prompt)
    .label("enrich-alice")           // Display name
    .phase("contact-enrichment")     // Grouping scope
    .schema::<LeadScore>()           // Structured output
    .run()                           // Execute
    .await?;
```

### Control Flow
- **Fail-soft:** `parallel()` — errors collapse to `None`, siblings continue
- **Fail-fast:** `SequentialAgent` (compile-time) — first error aborts
- **Streaming:** `pipeline()` — no barrier between items
- **Barrier:** `parallel()` — wait for all thunks to complete

### Budget Control
- **Hard ceiling:** Once spent ≥ total, further admissions fail
- **Shared pool:** All agents in a run share the same budget
- **Weighted tokens:** Input/output ×1, cache-write ×1.25, cache-read ×0.1
- **Admission check:** `ctx.remaining() < threshold` before admitting

### Error Handling
- **Control errors:** Budget/backstop breach → run-wide cancellation
- **Agent errors:** Domain failures → `Ok(None)` in parallel (fail-soft)
- **Cancellation:** Check `ctx.is_cancelled()` or use `select!` pattern

## Comparison with Compile-Time Workflows

| Feature | Dynamic (`flow`) | Compile-Time |
|---------|------------------|--------------|
| Type safety | Runtime | Compile-time |
| Error handling | Fail-soft | Fail-fast |
| Streaming | Yes (pipeline) | No (barrier) |
| Budget control | Yes | No |
| Nested workflows | Yes (1 level) | No |
| Journaling | Yes | No |
| Overhead | Arc clones | Zero-cost |

## Extending the Demo

### Add a New Workflow Pattern

1. Create `src/workflows/new_pattern.rs`
2. Implement your workflow using `flow` combinators
3. Add to `src/workflows/mod.rs`
4. Call from `main.rs` with a new phase:
   ```rust
   let _phase = flow::phase(&ctx, "new-pattern");
   workflows::new_pattern::run(&ctx, &data).await?;
   ```

### Add Real LLM Calls

Replace `CrmMockProvider` with a real provider:

```rust
use heartbit_core::llm::providers::anthropic::AnthropicProvider;

let provider = Arc::new(BoxedProvider::new(
    AnthropicProvider::new(api_key, "claude-3-5-sonnet-20241022")
));

let ctx = WorkflowCtx::builder(provider)
    .max_concurrency(4)
    .budget(100_000)
    .build()?;
```

### Add Tool Integration

Use the Agent's tool system for CRM operations:

```rust
use heartbit_core::agent::ToolSet;

let tools = ToolSet::new()
    .add_tool("create_contact", create_contact_handler)
    .add_tool("update_deal", update_deal_handler);

let result = agent(&ctx, prompt)
    .tools(&tools)
    .run()
    .await?;
```

## Performance Characteristics

- **Parallel fan-out:** O(n) agents, bounded by `max_concurrency`
- **Pipeline streaming:** O(n × stages) agents, no barrier overhead
- **Budget overhead:** Atomic counter increment per agent (~ns)
- **Memory:** Arc clones for ctx sharing (~24 bytes each)
- **Cancellation:** Cooperative via `CancellationToken` (~ns check)

## License

Same as heartbit-core (MIT OR Apache-2.0).
