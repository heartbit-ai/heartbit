use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use chrono::Timelike;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::Error;
use crate::config::HeartbitPulseConfig;

use super::CommandProducer;
use super::todo::FileTodoStore;
use super::types::DaemonCommand;

/// Periodic awareness loop that reviews the persistent todo list and
/// submits tasks via a [`CommandProducer`]. Separate from `CronScheduler`
/// (which handles static cron schedules); the heartbit pulse is dynamic
/// with conditions and idle backoff.
pub struct HeartbitPulseScheduler {
    interval: Duration,
    todo_store: Arc<FileTodoStore>,
    producer: Arc<dyn CommandProducer>,
    commands_topic: String,
    workspace: PathBuf,
    custom_prompt: Option<String>,
    active_hours: Option<crate::config::ActiveHoursConfig>,
    idle_backoff_threshold: u32,
    /// Names of active sensor sources (e.g., "gmail_inbox") whose data feeds
    /// are already handled by the sensor pipeline. The pulse prompt includes
    /// these so the agent avoids duplicating their work.
    active_sensors: Vec<String>,
}

impl std::fmt::Debug for HeartbitPulseScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeartbitPulseScheduler")
            .field("interval", &self.interval)
            .field("workspace", &self.workspace)
            .field("idle_backoff_threshold", &self.idle_backoff_threshold)
            .finish()
    }
}

impl HeartbitPulseScheduler {
    /// Create a new scheduler from config.
    pub fn new(
        config: &HeartbitPulseConfig,
        workspace: &Path,
        producer: Arc<dyn CommandProducer>,
        commands_topic: &str,
    ) -> Result<Self, Error> {
        if config.interval_seconds == 0 {
            return Err(Error::Config(
                "heartbit_pulse.interval_seconds must be > 0".into(),
            ));
        }

        // Validate active hours if provided
        if let Some(ref hours) = config.active_hours {
            hours.parse_start()?;
            hours.parse_end()?;
        }

        let todo_store = Arc::new(FileTodoStore::new(workspace)?);

        Ok(Self {
            interval: Duration::from_secs(config.interval_seconds),
            todo_store,
            producer,
            commands_topic: commands_topic.into(),
            workspace: workspace.to_path_buf(),
            custom_prompt: config.prompt.clone(),
            active_hours: config.active_hours.clone(),
            idle_backoff_threshold: config.idle_backoff_threshold,
            active_sensors: Vec::new(),
        })
    }

    /// Set the names of active sensor sources so the pulse prompt tells the
    /// agent not to duplicate data feeds already handled by the sensor pipeline.
    pub fn set_active_sensors(&mut self, sensors: Vec<String>) {
        self.active_sensors = sensors;
    }

    /// Get a reference to the backing todo store (for injecting into tools).
    pub fn todo_store(&self) -> &Arc<FileTodoStore> {
        &self.todo_store
    }

    /// Run the heartbit pulse loop. Submits awareness tasks via the
    /// [`CommandProducer`] at the configured interval, with active-hours
    /// gating and idle backoff.
    pub async fn run(self, cancel: CancellationToken) {
        let mut current_interval = self.interval;
        let mut consecutive_ok: u32 = 0;

        loop {
            tokio::select! {
                _ = cancel.cancelled() => {
                    tracing::info!("heartbit pulse shutting down");
                    break;
                }
                _ = tokio::time::sleep(current_interval) => {
                    // Check active hours — skips don't affect backoff state
                    if !self.is_within_active_hours() {
                        tracing::debug!("heartbit pulse: outside active hours, skipping");
                        continue;
                    }

                    // Check if there's anything to review
                    let todo_list = self.todo_store.get_list();
                    let heartbit_md = self.read_heartbit_md();
                    if todo_list.entries.is_empty() && heartbit_md.is_none() {
                        tracing::debug!("heartbit pulse: no todos and no HEARTBIT.md, skipping");
                        // Nothing to review counts as idle — increment backoff
                        consecutive_ok += 1;
                        self.maybe_backoff(&mut current_interval, consecutive_ok);
                        continue;
                    }

                    // Assemble prompt
                    let prompt = self.assemble_prompt(&heartbit_md);

                    // Submit command
                    let id = Uuid::new_v4();
                    let cmd = DaemonCommand::SubmitTask {
                        id,
                        task: prompt,
                        source: "heartbit".into(),
                        story_id: None,
                        trust_level: None,
                    };
                    let payload = match serde_json::to_vec(&cmd) {
                        Ok(p) => p,
                        Err(e) => {
                            tracing::error!(error = %e, "failed to serialize heartbit pulse command");
                            continue;
                        }
                    };
                    let send_ok = match self
                        .producer
                        .send_command(
                            &self.commands_topic,
                            &id.to_string(),
                            &payload,
                        )
                        .await
                    {
                        Ok(()) => {
                            tracing::info!(
                                task_id = %id,
                                interval_secs = current_interval.as_secs(),
                                consecutive_ok,
                                "heartbit pulse fired"
                            );
                            true
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "failed to produce heartbit pulse task");
                            false
                        }
                    };

                    // Skip backoff logic on send failure — don't penalize
                    // the scheduler for infrastructure issues. Also don't
                    // mark reviewed — the task wasn't actually submitted.
                    if !send_ok {
                        continue;
                    }

                    // Mark reviewed only after successful submission
                    if let Err(e) = self.todo_store.mark_reviewed() {
                        tracing::warn!(error = %e, "failed to mark todo list as reviewed");
                    }

                    // Backoff logic: we can't easily detect HEARTBIT_OK from
                    // the response since it flows asynchronously.
                    // Use todo-list state as a proxy: actionable items mean
                    // the agent has work to do, so reset backoff.
                    let has_actionable = todo_list.entries.iter().any(|e| {
                        matches!(
                            e.status,
                            crate::tool::builtins::TodoStatus::Pending
                                | crate::tool::builtins::TodoStatus::InProgress
                                | crate::tool::builtins::TodoStatus::Failed
                        )
                    });

                    if has_actionable {
                        consecutive_ok = 0;
                        current_interval = self.interval;
                    } else {
                        consecutive_ok += 1;
                        self.maybe_backoff(&mut current_interval, consecutive_ok);
                    }
                }
            }
        }
    }

    /// Apply idle backoff if the threshold is reached. Doubles the interval
    /// up to a cap of 4x the base interval.
    fn maybe_backoff(&self, current_interval: &mut Duration, consecutive_ok: u32) {
        if self.idle_backoff_threshold > 0 && consecutive_ok >= self.idle_backoff_threshold {
            let new_interval = *current_interval * 2;
            let max_interval = self.interval * 4;
            *current_interval = new_interval.min(max_interval);
            tracing::info!(
                consecutive_ok,
                new_interval_secs = current_interval.as_secs(),
                "heartbit pulse: idle backoff"
            );
        }
    }

    /// Check if the current time is within the configured active hours.
    fn is_within_active_hours(&self) -> bool {
        let now = chrono::Local::now();
        let current_minutes = now.hour() * 60 + now.minute();
        Self::check_active_hours(&self.active_hours, current_minutes)
    }

    /// Pure logic for active-hours check, extracted for testability.
    /// `current_minutes` is the current time as minutes since midnight.
    fn check_active_hours(
        active_hours: &Option<crate::config::ActiveHoursConfig>,
        current_minutes: u32,
    ) -> bool {
        let Some(hours) = active_hours else {
            return true; // No restriction
        };
        let (start_h, start_m) = match hours.parse_start() {
            Ok(v) => v,
            Err(_) => return true, // Malformed = no restriction
        };
        let (end_h, end_m) = match hours.parse_end() {
            Ok(v) => v,
            Err(_) => return true,
        };

        let start_minutes = start_h * 60 + start_m;
        let end_minutes = end_h * 60 + end_m;

        if start_minutes <= end_minutes {
            // Normal range: e.g., 08:00 - 22:00
            current_minutes >= start_minutes && current_minutes < end_minutes
        } else {
            // Overnight range: e.g., 22:00 - 06:00
            current_minutes >= start_minutes || current_minutes < end_minutes
        }
    }

    /// Read HEARTBIT.md from the workspace, if it exists.
    fn read_heartbit_md(&self) -> Option<String> {
        let path = self.workspace.join("HEARTBIT.md");
        std::fs::read_to_string(path).ok()
    }

    /// Assemble the heartbit pulse prompt.
    fn assemble_prompt(&self, heartbit_md: &Option<String>) -> String {
        if let Some(ref custom) = self.custom_prompt {
            return custom.clone();
        }

        let todo_section = self.todo_store.format_for_prompt();

        let mut prompt = String::from(
            "You are in HEARTBIT PULSE mode. Review your state and decide what to do.\n\n",
        );

        prompt.push_str("## Current Todo List\n");
        prompt.push_str(&todo_section);
        prompt.push('\n');

        if let Some(md) = heartbit_md {
            prompt.push_str("## Standing Orders\n");
            prompt.push_str(md);
            prompt.push('\n');
        }

        if !self.active_sensors.is_empty() {
            prompt.push_str("## Active Sensors (do NOT duplicate)\n\
                             The following data sources are already monitored by the sensor pipeline.\n\
                             Do NOT call their underlying tools directly — new items are automatically\n\
                             triaged and appear as tasks in your todo list.\n");
            for sensor in &self.active_sensors {
                prompt.push_str(&format!("- {sensor}\n"));
            }
            prompt.push('\n');
        }

        prompt.push_str(
            "## Instructions\n\
             1. Review the todo list. Pick the highest-priority actionable task.\n\
             2. If a task is overdue or urgent, execute it now via delegate_task.\n\
             3. If you identify new work, add it with todo_manage.\n\
             4. Update completed/failed tasks with todo_manage.\n\
             5. If nothing needs attention, respond with HEARTBIT_OK.\n",
        );

        prompt
    }
}

#[cfg(test)]
mod tests {
    use super::super::ChannelCommandProducer;
    use super::*;

    fn mock_producer() -> (
        Arc<dyn CommandProducer>,
        tokio::sync::mpsc::UnboundedReceiver<(String, Vec<u8>)>,
    ) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (Arc::new(ChannelCommandProducer { tx }), rx)
    }

    fn mock_producer_only() -> Arc<dyn CommandProducer> {
        mock_producer().0
    }

    #[test]
    fn assemble_prompt_default() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());

        // Add an entry so the prompt has content
        use super::super::todo::TodoEntry;
        store
            .add(TodoEntry::new("Fix the login bug", "user"))
            .unwrap();

        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: store,
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        let prompt = scheduler.assemble_prompt(&None);
        assert!(prompt.contains("HEARTBIT PULSE mode"));
        assert!(prompt.contains("Fix the login bug"));
        assert!(prompt.contains("HEARTBIT_OK"));
    }

    #[test]
    fn assemble_prompt_with_heartbit_md() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());
        store
            .add(super::super::todo::TodoEntry::new("Task 1", "user"))
            .unwrap();

        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: store,
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        let md = Some("- Always check RSS feeds first\n- Report daily".into());
        let prompt = scheduler.assemble_prompt(&md);
        assert!(prompt.contains("Standing Orders"));
        assert!(prompt.contains("Always check RSS feeds first"));
    }

    #[test]
    fn assemble_prompt_with_active_sensors() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());
        store
            .add(super::super::todo::TodoEntry::new("Task 1", "user"))
            .unwrap();

        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: store,
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec!["gmail_inbox".into(), "calendar_events".into()],
        };

        let prompt = scheduler.assemble_prompt(&None);
        assert!(prompt.contains("Active Sensors (do NOT duplicate)"));
        assert!(prompt.contains("- gmail_inbox"));
        assert!(prompt.contains("- calendar_events"));
        assert!(prompt.contains("Do NOT call their underlying tools directly"));
    }

    #[test]
    fn assemble_prompt_no_active_sensors_omits_section() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());
        store
            .add(super::super::todo::TodoEntry::new("Task 1", "user"))
            .unwrap();

        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: store,
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        let prompt = scheduler.assemble_prompt(&None);
        assert!(!prompt.contains("Active Sensors"));
    }

    #[test]
    fn assemble_prompt_custom_override() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());

        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: store,
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: Some("Custom pulse prompt".into()),
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        let prompt = scheduler.assemble_prompt(&None);
        assert_eq!(prompt, "Custom pulse prompt");
    }

    #[test]
    fn read_heartbit_md_absent() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());

        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: store,
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        assert!(scheduler.read_heartbit_md().is_none());
    }

    #[test]
    fn read_heartbit_md_present() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("HEARTBIT.md"),
            "# Standing orders\nDo stuff",
        )
        .unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());

        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: store,
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        let md = scheduler.read_heartbit_md().unwrap();
        assert!(md.contains("Standing orders"));
    }

    #[test]
    fn active_hours_no_restriction() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());

        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: store,
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        // No active_hours = always active
        assert!(scheduler.is_within_active_hours());
    }

    #[test]
    fn new_validates_zero_interval() {
        let dir = tempfile::tempdir().unwrap();
        let config = HeartbitPulseConfig {
            enabled: true,
            interval_seconds: 0,
            active_hours: None,
            prompt: None,
            idle_backoff_threshold: 6,
        };
        let err =
            HeartbitPulseScheduler::new(&config, dir.path(), mock_producer_only(), "test.commands")
                .unwrap_err();
        assert!(err.to_string().contains("interval_seconds must be > 0"));
    }

    #[test]
    fn new_validates_active_hours() {
        let dir = tempfile::tempdir().unwrap();
        let config = HeartbitPulseConfig {
            enabled: true,
            interval_seconds: 1800,
            active_hours: Some(crate::config::ActiveHoursConfig {
                start: "invalid".into(),
                end: "22:00".into(),
            }),
            prompt: None,
            idle_backoff_threshold: 6,
        };
        let err =
            HeartbitPulseScheduler::new(&config, dir.path(), mock_producer_only(), "test.commands")
                .unwrap_err();
        assert!(err.to_string().contains("invalid"));
    }

    #[test]
    fn new_succeeds_with_valid_config() {
        let dir = tempfile::tempdir().unwrap();
        let config = HeartbitPulseConfig {
            enabled: true,
            interval_seconds: 900,
            active_hours: Some(crate::config::ActiveHoursConfig {
                start: "08:00".into(),
                end: "22:00".into(),
            }),
            prompt: None,
            idle_backoff_threshold: 6,
        };
        let scheduler =
            HeartbitPulseScheduler::new(&config, dir.path(), mock_producer_only(), "test.commands")
                .unwrap();
        assert_eq!(scheduler.interval, Duration::from_secs(900));
    }

    // --- Active hours boundary tests ---

    fn normal_hours() -> Option<crate::config::ActiveHoursConfig> {
        Some(crate::config::ActiveHoursConfig {
            start: "08:00".into(),
            end: "22:00".into(),
        })
    }

    fn overnight_hours() -> Option<crate::config::ActiveHoursConfig> {
        Some(crate::config::ActiveHoursConfig {
            start: "22:00".into(),
            end: "06:00".into(),
        })
    }

    #[test]
    fn active_hours_normal_inside() {
        let hours = normal_hours();
        // 12:00 = 720 minutes — inside 08:00-22:00
        assert!(HeartbitPulseScheduler::check_active_hours(&hours, 720));
    }

    #[test]
    fn active_hours_normal_at_start() {
        let hours = normal_hours();
        // 08:00 = 480 — exactly at start (inclusive)
        assert!(HeartbitPulseScheduler::check_active_hours(&hours, 480));
    }

    #[test]
    fn active_hours_normal_before_start() {
        let hours = normal_hours();
        // 07:59 = 479 — before start
        assert!(!HeartbitPulseScheduler::check_active_hours(&hours, 479));
    }

    #[test]
    fn active_hours_normal_at_end() {
        let hours = normal_hours();
        // 22:00 = 1320 — exactly at end (exclusive)
        assert!(!HeartbitPulseScheduler::check_active_hours(&hours, 1320));
    }

    #[test]
    fn active_hours_normal_just_before_end() {
        let hours = normal_hours();
        // 21:59 = 1319 — just before end
        assert!(HeartbitPulseScheduler::check_active_hours(&hours, 1319));
    }

    #[test]
    fn active_hours_overnight_late_night() {
        let hours = overnight_hours();
        // 23:00 = 1380 — inside 22:00-06:00
        assert!(HeartbitPulseScheduler::check_active_hours(&hours, 1380));
    }

    #[test]
    fn active_hours_overnight_early_morning() {
        let hours = overnight_hours();
        // 03:00 = 180 — inside 22:00-06:00
        assert!(HeartbitPulseScheduler::check_active_hours(&hours, 180));
    }

    #[test]
    fn active_hours_overnight_outside_daytime() {
        let hours = overnight_hours();
        // 12:00 = 720 — outside 22:00-06:00
        assert!(!HeartbitPulseScheduler::check_active_hours(&hours, 720));
    }

    #[test]
    fn active_hours_overnight_at_start() {
        let hours = overnight_hours();
        // 22:00 = 1320 — exactly at start (inclusive)
        assert!(HeartbitPulseScheduler::check_active_hours(&hours, 1320));
    }

    #[test]
    fn active_hours_overnight_at_end() {
        let hours = overnight_hours();
        // 06:00 = 360 — exactly at end (exclusive)
        assert!(!HeartbitPulseScheduler::check_active_hours(&hours, 360));
    }

    // --- Backoff tests ---

    #[test]
    fn maybe_backoff_below_threshold_no_change() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: Arc::new(FileTodoStore::new(dir.path()).unwrap()),
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        let mut interval = Duration::from_secs(1800);
        scheduler.maybe_backoff(&mut interval, 5); // below threshold
        assert_eq!(interval, Duration::from_secs(1800));
    }

    #[test]
    fn maybe_backoff_at_threshold_doubles() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: Arc::new(FileTodoStore::new(dir.path()).unwrap()),
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        let mut interval = Duration::from_secs(1800);
        scheduler.maybe_backoff(&mut interval, 6); // at threshold
        assert_eq!(interval, Duration::from_secs(3600)); // doubled
    }

    #[test]
    fn maybe_backoff_caps_at_4x() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: Arc::new(FileTodoStore::new(dir.path()).unwrap()),
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        // Already at 2x, backoff should cap at 4x (7200)
        let mut interval = Duration::from_secs(3600);
        scheduler.maybe_backoff(&mut interval, 12);
        assert_eq!(interval, Duration::from_secs(7200)); // 4x cap

        // Already at 4x, should stay at 4x
        scheduler.maybe_backoff(&mut interval, 18);
        assert_eq!(interval, Duration::from_secs(7200)); // still capped
    }

    #[test]
    fn maybe_backoff_disabled_when_threshold_zero() {
        let dir = tempfile::tempdir().unwrap();
        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1800),
            todo_store: Arc::new(FileTodoStore::new(dir.path()).unwrap()),
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 0, // disabled
            active_sensors: vec![],
        };

        let mut interval = Duration::from_secs(1800);
        scheduler.maybe_backoff(&mut interval, 100);
        assert_eq!(interval, Duration::from_secs(1800)); // unchanged
    }

    // --- run() integration tests with mock producer ---

    /// Receive a command from the mock producer via spin-yield loop.
    /// Yields repeatedly to let the spawned scheduler task process.
    async fn recv_cmd(
        rx: &mut tokio::sync::mpsc::UnboundedReceiver<(String, Vec<u8>)>,
    ) -> (String, Vec<u8>) {
        for _ in 0..100 {
            tokio::task::yield_now().await;
            if let Ok(msg) = rx.try_recv() {
                return msg;
            }
        }
        panic!("timed out waiting for command on mock producer channel");
    }

    /// Assert no command is received after yielding many times.
    async fn assert_no_cmd(rx: &mut tokio::sync::mpsc::UnboundedReceiver<(String, Vec<u8>)>) {
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
        assert!(
            rx.try_recv().is_err(),
            "expected no command but received one"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn run_submits_task_when_todos_exist() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());
        store
            .add(super::super::todo::TodoEntry::new("Fix bug", "user"))
            .unwrap();

        let (producer, mut rx) = mock_producer();
        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1),
            todo_store: store,
            producer,
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        tokio::spawn(async move { scheduler.run(cancel2).await });

        // Let spawned task register its sleep
        tokio::task::yield_now().await;
        // Advance time past one interval
        tokio::time::advance(Duration::from_secs(2)).await;

        // Should receive a command
        let (key, payload) = recv_cmd(&mut rx).await;
        assert!(!key.is_empty());
        let cmd: DaemonCommand = serde_json::from_slice(&payload).unwrap();
        match cmd {
            DaemonCommand::SubmitTask {
                source, task, id, ..
            } => {
                assert_eq!(source, "heartbit");
                assert!(task.contains("Fix bug"));
                assert_eq!(key, id.to_string());
            }
            other => panic!("unexpected command: {other:?}"),
        }

        cancel.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn run_skips_when_no_todos_and_no_heartbit_md() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());
        // No todos, no HEARTBIT.md

        let (producer, mut rx) = mock_producer();
        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(1),
            todo_store: store,
            producer,
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        tokio::spawn(async move { scheduler.run(cancel2).await });

        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(5)).await;

        // Should NOT receive anything
        assert_no_cmd(&mut rx).await;

        cancel.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn run_backoff_doubles_interval_after_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());
        // Only completed entries — scheduler fires but considers them non-actionable
        let mut entry = super::super::todo::TodoEntry::new("Done task", "user");
        entry.status = crate::tool::builtins::TodoStatus::Completed;
        store.add(entry).unwrap();

        let (producer, mut rx) = mock_producer();
        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(10),
            todo_store: store,
            producer,
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 2, // backoff after 2 consecutive idle
            active_sensors: vec![],
        };

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        tokio::spawn(async move { scheduler.run(cancel2).await });

        // First tick at 10s — fires (non-actionable = consecutive_ok becomes 1)
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(11)).await;
        let _ = recv_cmd(&mut rx).await;

        // Second tick at 20s — fires (consecutive_ok becomes 2, hits threshold, interval doubles to 20s)
        tokio::time::advance(Duration::from_secs(10)).await;
        let _ = recv_cmd(&mut rx).await;

        // Third tick should now be at 20s interval (backed off)
        // At 10s after last fire: should NOT fire yet
        tokio::time::advance(Duration::from_secs(10)).await;
        assert_no_cmd(&mut rx).await;

        // At 20s after last fire: should fire
        tokio::time::advance(Duration::from_secs(10)).await;
        let _ = recv_cmd(&mut rx).await;

        cancel.cancel();
    }

    #[tokio::test(start_paused = true)]
    async fn run_stops_on_cancellation() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());

        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(60),
            todo_store: store,
            producer: mock_producer_only(),
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 6,
            active_sensors: vec![],
        };

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        let handle = tokio::spawn(async move { scheduler.run(cancel2).await });

        tokio::task::yield_now().await;
        cancel.cancel();
        tokio::time::advance(Duration::from_secs(1)).await;

        // Should complete without hanging
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("run should exit on cancel")
            .expect("task should not panic");
    }

    #[tokio::test(start_paused = true)]
    async fn run_resets_backoff_when_actionable_items_appear() {
        let dir = tempfile::tempdir().unwrap();
        let store = Arc::new(FileTodoStore::new(dir.path()).unwrap());
        // Start with only completed entries (non-actionable)
        let mut entry = super::super::todo::TodoEntry::new("Done task", "user");
        entry.status = crate::tool::builtins::TodoStatus::Completed;
        store.add(entry).unwrap();

        let (producer, mut rx) = mock_producer();
        let store_clone = store.clone();
        let scheduler = HeartbitPulseScheduler {
            interval: Duration::from_secs(10),
            todo_store: store_clone,
            producer,
            commands_topic: "test.commands".into(),
            workspace: dir.path().to_path_buf(),
            custom_prompt: None,
            active_hours: None,
            idle_backoff_threshold: 1, // backoff after 1 idle tick
            active_sensors: vec![],
        };

        let cancel = CancellationToken::new();
        let cancel2 = cancel.clone();
        tokio::spawn(async move { scheduler.run(cancel2).await });

        // First tick: fires (non-actionable → consecutive_ok=1, hits threshold, doubles to 20s)
        tokio::task::yield_now().await;
        tokio::time::advance(Duration::from_secs(11)).await;
        let _ = recv_cmd(&mut rx).await;

        // Add an actionable item
        store
            .add(super::super::todo::TodoEntry::new("New bug", "user"))
            .unwrap();

        // Next tick after 20s (backed off interval)
        tokio::time::advance(Duration::from_secs(20)).await;
        let _ = recv_cmd(&mut rx).await;

        // Interval should be reset to 10s now (actionable items detected)
        // So it should fire again after 10s, not 20s
        tokio::time::advance(Duration::from_secs(11)).await;
        let _ = recv_cmd(&mut rx).await;

        cancel.cancel();
    }
}
