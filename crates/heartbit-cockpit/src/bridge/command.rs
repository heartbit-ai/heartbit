/// Commands sent from the UI thread to the tokio runtime thread.
pub enum CockpitCommand {
    /// User submitted a task to execute.
    SubmitTask { task: String },
    /// Cancel the currently running task.
    Cancel,
    /// Shutdown the runtime.
    Stop,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::oneshot;

    fn assert_send<T: Send>() {}

    #[test]
    fn command_is_send() {
        assert_send::<CockpitCommand>();
    }

    #[test]
    fn approval_oneshot_round_trip() {
        let (tx, rx) = oneshot::channel::<bool>();
        tx.send(true).unwrap();
        assert!(rx.blocking_recv().unwrap());
    }

    #[test]
    fn approval_oneshot_denied() {
        let (tx, rx) = oneshot::channel::<bool>();
        tx.send(false).unwrap();
        assert!(!rx.blocking_recv().unwrap());
    }

    #[test]
    fn input_oneshot_round_trip() {
        let (tx, rx) = oneshot::channel::<Option<String>>();
        tx.send(Some("hello".into())).unwrap();
        assert_eq!(rx.blocking_recv().unwrap(), Some("hello".into()));
    }

    #[test]
    fn input_oneshot_none_on_drop() {
        let (tx, rx) = oneshot::channel::<Option<String>>();
        drop(tx);
        assert!(rx.blocking_recv().is_err());
    }
}
