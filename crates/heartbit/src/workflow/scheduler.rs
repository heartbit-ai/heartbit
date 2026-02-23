use std::time::Duration;

use restate_sdk::prelude::*;
use serde::{Deserialize, Serialize};

use super::orchestrator_workflow::OrchestratorWorkflowClient;
use super::types::{AgentDef, OrchestratorTask};

/// Configuration for a recurring schedule.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleConfig {
    pub task: String,
    pub agents: Vec<AgentDef>,
    pub max_turns: usize,
    pub max_tokens: u32,
    pub interval_secs: u64,
}

/// Restate virtual object for managing recurring agent schedules.
///
/// The object key is the schedule name. Each schedule runs in a loop:
/// submit task → durable sleep → repeat. The durable sleep survives crashes.
#[restate_sdk::object]
pub trait SchedulerObject {
    /// Start a recurring schedule. Loops until stopped.
    async fn start(config: Json<ScheduleConfig>) -> Result<(), HandlerError>;

    /// Stop the schedule by setting a flag.
    async fn stop() -> Result<(), HandlerError>;

    /// Check if the schedule is running.
    #[shared]
    async fn is_running() -> Result<bool, HandlerError>;
}

pub struct SchedulerObjectImpl;

impl SchedulerObject for SchedulerObjectImpl {
    /// Start a recurring schedule. Each invocation submits one task, then
    /// schedules the next iteration via a delayed self-send. This releases
    /// the Restate exclusive lock between iterations, allowing `stop()` to
    /// interleave and set the running flag to false.
    async fn start(
        &self,
        mut ctx: ObjectContext<'_>,
        Json(config): Json<ScheduleConfig>,
    ) -> Result<(), HandlerError> {
        // Guard: distinguish "first ever call" from "scheduled self-send after stop()".
        // Both have running=false, but initialized differentiates them.
        let running = ctx.get::<bool>("running").await?.unwrap_or(false);
        let initialized = ctx.get::<bool>("initialized").await?.unwrap_or(false);

        if !running && initialized {
            // stop() was called — this is a scheduled self-send that should not restart.
            // Clear initialized so a future explicit start() call can begin fresh.
            ctx.clear("initialized");
            return Ok(());
        }

        ctx.set("running", true);
        ctx.set("initialized", true);

        let key = ctx.key().to_string();

        // Submit the task as a new orchestrator workflow
        let workflow_id = format!("{}-{}", key, ctx.rand_uuid());
        ctx.workflow_client::<OrchestratorWorkflowClient>(&workflow_id)
            .run(Json(OrchestratorTask {
                input: config.task.clone(),
                agents: config.agents.clone(),
                max_turns: config.max_turns,
                max_tokens: config.max_tokens,
                approval_required: false,
                reasoning_effort: None,
            }))
            .send();

        // Schedule next iteration after the interval. This releases the
        // exclusive lock, allowing stop() to run before the next iteration.
        let interval = Duration::from_secs(config.interval_secs);
        ctx.object_client::<SchedulerObjectClient>(&key)
            .start(Json(config))
            .send_after(interval);

        Ok(())
    }

    async fn stop(&self, ctx: ObjectContext<'_>) -> Result<(), HandlerError> {
        ctx.set("running", false);
        // Keep initialized=true so the next scheduled start() detects the stop
        // and exits. That start() clears initialized, enabling future restarts.
        Ok(())
    }

    async fn is_running(&self, ctx: SharedObjectContext<'_>) -> Result<bool, HandlerError> {
        Ok(ctx.get::<bool>("running").await?.unwrap_or(false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_config_roundtrips() {
        let config = ScheduleConfig {
            task: "Generate daily report".into(),
            agents: vec![AgentDef {
                name: "reporter".into(),
                description: "Report generator".into(),
                system_prompt: "Generate reports.".into(),
                tool_defs: vec![],
                context_window_tokens: None,
                summarize_threshold: None,
                tool_timeout_seconds: None,
                max_tool_output_bytes: None,
                max_turns: None,
                max_tokens: None,
                response_schema: None,
                reasoning_effort: None,
            }],
            max_turns: 10,
            max_tokens: 4096,
            interval_secs: 3600,
        };
        let json = serde_json::to_string(&config).unwrap();
        let parsed: ScheduleConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.task, "Generate daily report");
        assert_eq!(parsed.interval_secs, 3600);
        assert_eq!(parsed.agents.len(), 1);
    }
}
