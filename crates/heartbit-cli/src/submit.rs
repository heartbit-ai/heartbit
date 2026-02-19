use std::path::Path;

use anyhow::{Context, Result};

use heartbit::HeartbitConfig;
use heartbit::config::AgentConfig;
use heartbit::llm::types::ToolDefinition;
use heartbit::store::PostgresStore;
use heartbit::workflow::types::{AgentDef, HumanDecision, OrchestratorTask};

/// Submit a task to Restate for durable execution.
pub async fn submit_task(config_path: &Path, task: &str, restate_url: &str) -> Result<()> {
    let config = HeartbitConfig::from_file(config_path)
        .with_context(|| format!("failed to load config from {}", config_path.display()))?;

    let workflow_uuid = uuid::Uuid::new_v4();
    let workflow_id = workflow_uuid.to_string();

    // Optionally record the task in PostgreSQL
    let store = if let Ok(url) = std::env::var("DATABASE_URL") {
        match PostgresStore::connect(&url).await {
            Ok(s) => {
                if let Err(e) = s.run_migration().await {
                    tracing::warn!(error = %e, "failed to run database migration, continuing without store");
                    None
                } else {
                    Some(s)
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to connect to database, continuing without store");
                None
            }
        }
    } else {
        None
    };
    if let Some(store) = &store
        && let Err(e) = store.create_task(workflow_uuid, task, None).await
    {
        tracing::warn!(error = %e, "failed to record task in database, continuing");
    }

    // Resolve per-agent tool definitions from MCP servers
    let mut agents = Vec::with_capacity(config.agents.len());
    for agent_config in &config.agents {
        let tool_defs = load_mcp_tool_defs(&agent_config.name, &agent_config.mcp_servers).await;
        agents.push(agent_config_to_def(agent_config, tool_defs));
    }

    let orchestrator_task = OrchestratorTask {
        input: task.into(),
        agents,
        max_turns: config.orchestrator.max_turns,
        max_tokens: config.orchestrator.max_tokens,
        approval_required: false,
    };

    let url = format!(
        "{}/OrchestratorWorkflow/{}/run",
        restate_url.trim_end_matches('/'),
        workflow_id
    );

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&orchestrator_task)?)
        .send()
        .await
        .context("failed to submit task to Restate")?;

    if resp.status().is_success() {
        println!("Workflow submitted: {workflow_id}");
        println!("Check status: heartbit status {workflow_id}");
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Restate returned {status}: {body}");
    }

    Ok(())
}

/// Query workflow status via Restate.
pub async fn query_status(workflow_id: &str, restate_url: &str) -> Result<()> {
    let url = format!(
        "{}/OrchestratorWorkflow/{}/status",
        restate_url.trim_end_matches('/'),
        workflow_id
    );

    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .send()
        .await
        .context("failed to query workflow status")?;

    if resp.status().is_success() {
        let body = resp.text().await?;
        // Parse to show child workflow IDs if present
        if let Ok(status) = serde_json::from_str::<heartbit::workflow::types::AgentStatus>(&body) {
            println!("State: {}", status.state);
            println!("Turn: {}/{}", status.current_turn, status.max_turns);
            if !status.child_workflows.is_empty() {
                println!("Child workflows (use with 'heartbit approve <id>'):");
                for id in &status.child_workflows {
                    println!("  - {id}");
                }
            }
        } else {
            println!("{body}");
        }
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Restate returned {status}: {body}");
    }

    Ok(())
}

/// Send human approval signal to a workflow.
pub async fn send_approval(workflow_id: &str, restate_url: &str) -> Result<()> {
    let url = format!(
        "{}/AgentWorkflow/{}/approve",
        restate_url.trim_end_matches('/'),
        workflow_id
    );

    let decision = HumanDecision {
        approved: true,
        reason: Some("Approved via CLI".into()),
        turn: None, // fall back to reading current turn from workflow state
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("content-type", "application/json")
        .body(serde_json::to_string(&decision)?)
        .send()
        .await
        .context("failed to send approval")?;

    if resp.status().is_success() {
        println!("Approval sent to workflow {workflow_id}");
    } else {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Restate returned {status}: {body}");
    }

    Ok(())
}

/// Connect to MCP servers and collect tool definitions. Failures are logged and skipped.
async fn load_mcp_tool_defs(agent_name: &str, mcp_servers: &[String]) -> Vec<ToolDefinition> {
    let mut defs = Vec::new();
    for server_url in mcp_servers {
        tracing::info!(agent = %agent_name, url = %server_url, "resolving MCP tool definitions");
        match heartbit::McpClient::connect(server_url).await {
            Ok(client) => {
                for def in client.tool_definitions() {
                    tracing::info!(agent = %agent_name, tool = %def.name, "resolved tool definition");
                    defs.push(def);
                }
            }
            Err(e) => {
                tracing::warn!(
                    agent = %agent_name,
                    url = %server_url,
                    error = %e,
                    "failed to connect to MCP server, agent will run without these tools"
                );
            }
        }
    }
    defs
}

fn agent_config_to_def(config: &AgentConfig, tool_defs: Vec<ToolDefinition>) -> AgentDef {
    let (context_window_tokens, summarize_threshold) = match &config.context_strategy {
        Some(heartbit::ContextStrategyConfig::SlidingWindow { max_tokens }) => {
            (Some(*max_tokens), None)
        }
        Some(heartbit::ContextStrategyConfig::Summarize { max_tokens }) => {
            (None, Some(*max_tokens))
        }
        _ => (None, None),
    };

    AgentDef {
        name: config.name.clone(),
        description: config.description.clone(),
        system_prompt: config.system_prompt.clone(),
        tool_defs,
        context_window_tokens,
        summarize_threshold,
    }
}
