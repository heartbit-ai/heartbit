mod bridge;
mod callbacks;
mod runtime;

use std::path::PathBuf;
use std::sync::Arc;

use clap::Parser;

use bridge::command::CockpitCommand;
use callbacks::SharedState;
use heartbit::QuestionResponse;

slint::include_modules!();

#[derive(Parser)]
#[command(name = "heartbit-cockpit", about = "Heartbit desktop cockpit UI")]
struct Args {
    /// Path to heartbit.toml config file
    #[arg(long)]
    config: Option<PathBuf>,

    /// Require human approval before each tool execution
    #[arg(long)]
    approve: bool,

    /// Task to execute immediately on startup
    #[arg(trailing_var_arg = true)]
    task: Vec<String>,
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let ui = MainWindow::new()?;

    let (cmd_tx, cmd_rx) = tokio::sync::mpsc::unbounded_channel::<CockpitCommand>();
    let shared = Arc::new(SharedState::new());

    // Wire UI callbacks → CockpitCommand
    wire_ui_callbacks(&ui, cmd_tx.clone(), Arc::clone(&shared));

    // Spawn tokio runtime on background thread
    let ui_handle = ui.as_weak();
    let config_path = args.config.clone();
    let approve = args.approve;
    let shared_rt = Arc::clone(&shared);
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("failed to create tokio runtime");
        rt.block_on(runtime::runtime_loop(
            cmd_rx,
            ui_handle,
            shared_rt,
            config_path,
            approve,
        ));
    });

    // If initial task provided, auto-submit
    let task_str = args.task.join(" ");
    if !task_str.is_empty() {
        cmd_tx
            .send(CockpitCommand::SubmitTask { task: task_str })
            .ok();
    }

    ui.run()?;

    // Signal runtime to stop
    cmd_tx.send(CockpitCommand::Stop).ok();

    Ok(())
}

/// Connect Slint UI callbacks to the command channel and shared state.
fn wire_ui_callbacks(
    ui: &MainWindow,
    cmd_tx: tokio::sync::mpsc::UnboundedSender<CockpitCommand>,
    shared: Arc<SharedState>,
) {
    // submit-task callback
    let tx = cmd_tx.clone();
    ui.on_submit_task(move |task| {
        let task_str = task.to_string();
        if !task_str.is_empty() {
            tx.send(CockpitCommand::SubmitTask { task: task_str }).ok();
        }
    });

    // approve callback — receives int: 0=Allow, 1=Deny, 2=AlwaysAllow, 3=AlwaysDeny
    let shared_approve = Arc::clone(&shared);
    ui.on_approve(move |choice| {
        let decision = match choice {
            0 => heartbit::ApprovalDecision::Allow,
            2 => heartbit::ApprovalDecision::AlwaysAllow,
            3 => heartbit::ApprovalDecision::AlwaysDeny,
            _ => heartbit::ApprovalDecision::Deny,
        };
        let mut guard = shared_approve
            .approval_tx
            .lock()
            .expect("approval_tx lock poisoned");
        if let Some(tx) = guard.take() {
            tx.send(decision).ok();
        }
    });

    // send-input callback
    let shared_input = Arc::clone(&shared);
    ui.on_send_input(move |text| {
        let msg = text.to_string();
        let mut guard = shared_input
            .input_tx
            .lock()
            .expect("input_tx lock poisoned");
        if let Some(tx) = guard.take() {
            let value = if msg.is_empty() { None } else { Some(msg) };
            tx.send(value).ok();
        }
    });

    // cancel-task callback
    let tx_cancel = cmd_tx.clone();
    ui.on_cancel_task(move || {
        tx_cancel.send(CockpitCommand::Cancel).ok();
    });

    // toggle-tool-output callback (runs on UI thread, so upgrade() is synchronous)
    let shared_toggle = Arc::clone(&shared);
    let ui_toggle = ui.as_weak();
    ui.on_toggle_tool_output(move |msg_id| {
        {
            let mut proc = shared_toggle
                .processor
                .lock()
                .expect("processor lock poisoned");
            proc.toggle_expanded(msg_id);
        }
        if let Some(w) = ui_toggle.upgrade() {
            crate::callbacks::sync_state_to_ui(&shared_toggle, &w);
        }
    });

    // select-history-run callback (runs on UI thread)
    let shared_history = Arc::clone(&shared);
    let ui_history = ui.as_weak();
    ui.on_select_history_run(move |run_id| {
        {
            let mut proc = shared_history
                .processor
                .lock()
                .expect("processor lock poisoned");
            if run_id < 0 {
                proc.view_history(None);
            } else {
                proc.view_history(Some(run_id as usize));
            }
        }
        if let Some(w) = ui_history.upgrade() {
            crate::callbacks::sync_state_to_ui(&shared_history, &w);
        }
    });

    // filter-by-agent callback (runs on UI thread)
    let shared_filter = Arc::clone(&shared);
    let ui_filter = ui.as_weak();
    ui.on_filter_by_agent(move |agent_name| {
        {
            let mut proc = shared_filter
                .processor
                .lock()
                .expect("processor lock poisoned");
            proc.toggle_agent_filter(agent_name.as_ref());
        }
        if let Some(w) = ui_filter.upgrade() {
            crate::callbacks::sync_state_to_ui(&shared_filter, &w);
        }
    });

    // select-question-option callback
    let shared_q_select = Arc::clone(&shared);
    let ui_q_select = ui.as_weak();
    ui.on_select_question_option(move |option_index| {
        let complete = {
            let mut pq_guard = shared_q_select
                .pending_question
                .lock()
                .expect("pending_question lock poisoned");
            match pq_guard.as_mut() {
                Some(pq) => pq.select_option(option_index as usize),
                None => return,
            }
        };
        if complete {
            finish_question(&shared_q_select);
        }
        if let Some(w) = ui_q_select.upgrade() {
            crate::callbacks::sync_question_to_ui(&shared_q_select, &w);
        }
    });

    // copy-to-clipboard callback
    ui.on_copy_to_clipboard(move |text| {
        let content = text.to_string();
        if content.is_empty() {
            return;
        }
        match arboard::Clipboard::new() {
            Ok(mut clipboard) => {
                if let Err(e) = clipboard.set_text(&content) {
                    tracing::warn!(error = %e, "failed to copy to clipboard");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to access clipboard");
            }
        }
    });

    // submit-question-answers callback (multi-select "Done" button)
    let shared_q_submit = Arc::clone(&shared);
    let ui_q_submit = ui.as_weak();
    ui.on_submit_question_answers(move || {
        let complete = {
            let mut pq_guard = shared_q_submit
                .pending_question
                .lock()
                .expect("pending_question lock poisoned");
            match pq_guard.as_mut() {
                Some(pq) => pq.submit_current(),
                None => return,
            }
        };
        if complete {
            finish_question(&shared_q_submit);
        }
        if let Some(w) = ui_q_submit.upgrade() {
            crate::callbacks::sync_question_to_ui(&shared_q_submit, &w);
        }
    });
}

/// Complete the pending question dialog and send the response through the channel.
fn finish_question(shared: &SharedState) {
    let pending = {
        let mut pq_guard = shared
            .pending_question
            .lock()
            .expect("pending_question lock poisoned");
        pq_guard.take()
    };
    if let Some(pq) = pending {
        // Record answers as user messages in chat timeline
        {
            let mut proc = shared.processor.lock().expect("processor lock poisoned");
            for (q, ans) in pq.questions().iter().zip(pq.answers()) {
                proc.push_question_answer(&q.header, ans);
            }
        }
        let response = QuestionResponse {
            answers: pq.into_answers(),
        };
        let mut tx_guard = shared
            .question_tx
            .lock()
            .expect("question_tx lock poisoned");
        if let Some(tx) = tx_guard.take() {
            tx.send(response).ok();
        }
    }
}
