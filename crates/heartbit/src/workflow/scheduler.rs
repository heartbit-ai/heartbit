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
    async fn start(
        &self,
        mut ctx: ObjectContext<'_>,
        Json(config): Json<ScheduleConfig>,
    ) -> Result<(), HandlerError> {
        ctx.set("running", true);
        let interval = Duration::from_secs(config.interval_secs);
        let key = ctx.key().to_string();

        loop {
            // Check if we should stop
            let running = ctx.get::<bool>("running").await?.unwrap_or(false);
            if !running {
                break;
            }

            // Submit the task as a new orchestrator workflow
            let workflow_id = format!("{}-{}", key, ctx.rand_uuid());
            ctx.workflow_client::<OrchestratorWorkflowClient>(&workflow_id)
                .run(Json(OrchestratorTask {
                    input: config.task.clone(),
                    agents: config.agents.clone(),
                    max_turns: config.max_turns,
                    max_tokens: config.max_tokens,
                    approval_required: false,
                }))
                .send();

            // Durable sleep — survives crashes
            ctx.sleep(interval).await?;
        }

        Ok(())
    }

    async fn stop(&self, ctx: ObjectContext<'_>) -> Result<(), HandlerError> {
        ctx.set("running", false);
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
