use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::RwLock;
use std::time::SystemTime;

/// Tracks when files were last read/written.
///
/// Enforces a read-before-write guard: rejects edits to files whose on-disk
/// mtime has changed since the last recorded read. Shared across read, write,
/// edit, and patch tools via `Arc<FileTracker>`.
///
/// Uses `std::sync::RwLock` (not tokio) because locks are never held across
/// `.await` points.
pub struct FileTracker {
    records: RwLock<HashMap<PathBuf, FileRecord>>,
}

struct FileRecord {
    /// On-disk mtime captured at read time.
    modified_at: Option<SystemTime>,
}

impl Default for FileTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl FileTracker {
    pub fn new() -> Self {
        Self {
            records: RwLock::new(HashMap::new()),
        }
    }

    /// Record that `path` was just read. Captures its current mtime.
    pub fn record_read(&self, path: &Path) -> std::io::Result<()> {
        let modified_at = match std::fs::metadata(path) {
            Ok(meta) => meta.modified().ok(),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };
        let canonical = std::fs::canonicalize(path)
            .or_else(|_| std::path::absolute(path))
            .unwrap_or_else(|_| path.to_path_buf());
        let mut records = self.records.write().expect("file tracker lock poisoned");
        records.insert(canonical, FileRecord { modified_at });
        Ok(())
    }

    /// Check that `path` has not been modified since the last recorded read.
    ///
    /// Returns `Ok(())` if the file is safe to write. Returns `Err(message)`
    /// if the file was modified externally or was never read.
    pub fn check_unmodified(&self, path: &Path) -> Result<(), String> {
        let canonical = std::fs::canonicalize(path)
            .or_else(|_| std::path::absolute(path))
            .unwrap_or_else(|_| path.to_path_buf());
        let records = self.records.read().expect("file tracker lock poisoned");
        let record = records.get(&canonical).ok_or_else(|| {
            format!(
                "File {} has not been read yet. Read it first before editing.",
                path.display()
            )
        })?;

        let current_mtime = std::fs::metadata(path).ok().and_then(|m| m.modified().ok());

        match (record.modified_at, current_mtime) {
            (Some(recorded), Some(current)) if recorded == current => Ok(()),
            (None, None) => Ok(()),
            _ => Err(format!(
                "File {} has been modified since it was last read. Read it again before editing.",
                path.display()
            )),
        }
    }

    /// Check whether `path` has been previously read.
    pub fn was_read(&self, path: &Path) -> bool {
        let canonical = std::fs::canonicalize(path)
            .or_else(|_| std::path::absolute(path))
            .unwrap_or_else(|_| path.to_path_buf());
        let records = self.records.read().expect("file tracker lock poisoned");
        records.contains_key(&canonical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn record_read_and_was_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").unwrap();

        let tracker = FileTracker::new();
        assert!(!tracker.was_read(&path));

        tracker.record_read(&path).unwrap();
        assert!(tracker.was_read(&path));
    }

    #[test]
    fn check_unmodified_passes_when_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").unwrap();

        let tracker = FileTracker::new();
        tracker.record_read(&path).unwrap();
        assert!(tracker.check_unmodified(&path).is_ok());
    }

    #[test]
    fn check_unmodified_fails_when_never_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").unwrap();

        let tracker = FileTracker::new();
        let err = tracker.check_unmodified(&path).unwrap_err();
        assert!(err.contains("has not been read yet"), "got: {err}");
    }

    #[test]
    fn check_unmodified_fails_when_modified_externally() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").unwrap();

        let tracker = FileTracker::new();
        tracker.record_read(&path).unwrap();

        // Wait a bit to ensure mtime changes
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Modify the file externally
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        f.write_all(b"modified").unwrap();
        f.sync_all().unwrap();

        let err = tracker.check_unmodified(&path).unwrap_err();
        assert!(err.contains("has been modified"), "got: {err}");
    }

    #[test]
    fn record_read_updates_mtime_after_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.txt");
        std::fs::write(&path, "hello").unwrap();

        let tracker = FileTracker::new();
        tracker.record_read(&path).unwrap();

        // Modify then re-read
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(&path, "changed").unwrap();
        tracker.record_read(&path).unwrap();

        // Should pass because we re-recorded after the change
        assert!(tracker.check_unmodified(&path).is_ok());
    }

    #[test]
    fn record_read_nonexistent_file_ok() {
        let tracker = FileTracker::new();
        let path = Path::new("/tmp/nonexistent_heartbit_test_file_12345");
        // Should not panic — stores None for mtime
        tracker.record_read(path).unwrap();
    }

    #[test]
    fn check_unmodified_fails_when_file_deleted_after_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("will_delete.txt");
        std::fs::write(&path, "content").unwrap();

        let tracker = FileTracker::new();
        tracker.record_read(&path).unwrap();

        // Delete the file
        std::fs::remove_file(&path).unwrap();

        // check_unmodified should detect the deletion (Some mtime -> None)
        let err = tracker.check_unmodified(&path).unwrap_err();
        assert!(err.contains("has been modified"), "got: {err}");
    }

    #[test]
    fn check_unmodified_fails_when_file_created_after_nonexistent_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("will_appear.txt");

        let tracker = FileTracker::new();
        // Record a "read" of a nonexistent file (None mtime)
        tracker.record_read(&path).unwrap();

        // Create the file externally
        std::fs::write(&path, "surprise").unwrap();

        // check_unmodified should detect the creation (None -> Some)
        let err = tracker.check_unmodified(&path).unwrap_err();
        assert!(err.contains("has been modified"), "got: {err}");
    }
}
