use std::collections::HashSet;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::time::Duration;

use chrono::{DateTime, Utc};
use rdkafka::producer::{FutureProducer, FutureRecord};
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::sensor::{Sensor, SensorEvent, SensorModality};

/// Supported audio file extensions.
const AUDIO_EXTENSIONS: &[&str] = &["wav", "mp3", "ogg", "flac", "m4a", "aac", "wma", "opus"];

/// Audio directory-watcher sensor.
///
/// Polls a configured directory at regular intervals for new audio files.
/// Each new file produces a `SensorEvent` to the `hb.sensor.audio` Kafka topic
/// with file metadata as content. When whisper-rs is integrated, the content
/// will contain the transcription instead.
pub struct AudioSensor {
    name: String,
    watch_directory: PathBuf,
    whisper_model: String,
    poll_interval: Duration,
}

impl AudioSensor {
    pub fn new(
        name: impl Into<String>,
        watch_directory: impl Into<PathBuf>,
        whisper_model: impl Into<String>,
        poll_interval: Duration,
    ) -> Self {
        Self {
            name: name.into(),
            watch_directory: watch_directory.into(),
            whisper_model: whisper_model.into(),
            poll_interval,
        }
    }
}

/// Returns `true` if the extension string (without the dot) is a supported audio format.
fn is_audio_extension(ext: &str) -> bool {
    AUDIO_EXTENSIONS.contains(&ext.to_lowercase().as_str())
}

/// Compute an FNV-1a hash of the given bytes and return as hex string.
fn fnv1a_hex(data: &[u8]) -> String {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &byte in data {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Build a `SensorEvent` from file metadata.
///
/// The content is a descriptive string with filename, size, and modification
/// time. When whisper-rs is integrated, this will contain the transcription.
fn build_sensor_event(
    sensor_name: &str,
    path: &Path,
    size_bytes: u64,
    modified_at: DateTime<Utc>,
    whisper_model: &str,
) -> Result<SensorEvent, Error> {
    let abs_path = std::path::absolute(path)
        .map_err(|e| Error::Sensor(format!("failed to resolve absolute path: {e}")))?;
    let abs_str = abs_path.to_string_lossy().to_string();

    let filename = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_default();

    let extension = path
        .extension()
        .map(|e| e.to_string_lossy().to_string())
        .unwrap_or_default();

    let content = format!("Audio: {filename} ({size_bytes} bytes, {modified_at})",);

    let id = SensorEvent::generate_id(&content, &abs_str);

    let metadata = serde_json::json!({
        "filename": filename,
        "extension": extension,
        "size_bytes": size_bytes,
        "modified_at": modified_at.to_rfc3339(),
        "whisper_model": whisper_model,
    });

    Ok(SensorEvent {
        id,
        sensor_name: sensor_name.into(),
        modality: SensorModality::Audio,
        observed_at: Utc::now(),
        content,
        source_id: abs_str.clone(),
        metadata: Some(metadata),
        binary_ref: Some(abs_str),
        related_ids: vec![],
    })
}

impl Sensor for AudioSensor {
    fn name(&self) -> &str {
        &self.name
    }

    fn modality(&self) -> SensorModality {
        SensorModality::Audio
    }

    fn kafka_topic(&self) -> &str {
        "hb.sensor.audio"
    }

    fn run(
        &self,
        producer: FutureProducer,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            let mut seen: HashSet<PathBuf> = HashSet::new();

            loop {
                if cancel.is_cancelled() {
                    return Ok(());
                }

                match scan_directory(&self.watch_directory, &mut seen).await {
                    Ok(new_files) => {
                        for path in new_files {
                            if cancel.is_cancelled() {
                                return Ok(());
                            }

                            let meta = match tokio::fs::metadata(&path).await {
                                Ok(m) => m,
                                Err(e) => {
                                    tracing::warn!(
                                        path = %path.display(),
                                        error = %e,
                                        "failed to read audio file metadata"
                                    );
                                    continue;
                                }
                            };

                            let size_bytes = meta.len();
                            let modified_at: DateTime<Utc> = meta
                                .modified()
                                .map(DateTime::<Utc>::from)
                                .unwrap_or_else(|_| Utc::now());

                            let event = match build_sensor_event(
                                &self.name,
                                &path,
                                size_bytes,
                                modified_at,
                                &self.whisper_model,
                            ) {
                                Ok(e) => e,
                                Err(e) => {
                                    tracing::warn!(
                                        path = %path.display(),
                                        error = %e,
                                        "failed to build sensor event for audio file"
                                    );
                                    continue;
                                }
                            };

                            let payload = match serde_json::to_vec(&event) {
                                Ok(p) => p,
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        "failed to serialize audio sensor event"
                                    );
                                    continue;
                                }
                            };

                            let path_hash = fnv1a_hex(path.to_string_lossy().as_bytes());
                            let key = format!("{}:{}", self.name, path_hash);

                            if let Err((e, _)) = producer
                                .send(
                                    FutureRecord::to(self.kafka_topic())
                                        .key(&key)
                                        .payload(&payload),
                                    rdkafka::util::Timeout::After(Duration::from_secs(5)),
                                )
                                .await
                            {
                                tracing::warn!(
                                    path = %path.display(),
                                    error = %e,
                                    "failed to produce audio event to Kafka"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            directory = %self.watch_directory.display(),
                            error = %e,
                            "failed to scan audio directory"
                        );
                    }
                }

                tokio::select! {
                    _ = cancel.cancelled() => return Ok(()),
                    _ = tokio::time::sleep(self.poll_interval) => {}
                }
            }
        })
    }
}

/// Scan a directory for new audio files, updating the seen set.
/// Returns paths of newly discovered audio files.
async fn scan_directory(dir: &Path, seen: &mut HashSet<PathBuf>) -> Result<Vec<PathBuf>, Error> {
    let mut new_files = Vec::new();

    let mut entries = tokio::fs::read_dir(dir)
        .await
        .map_err(|e| Error::Sensor(format!("failed to read directory {}: {e}", dir.display())))?;

    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| Error::Sensor(format!("failed to read directory entry: {e}")))?
    {
        // Use async file_type() instead of blocking path.is_dir().
        let file_type = match entry.file_type().await {
            Ok(ft) => ft,
            Err(e) => {
                tracing::warn!(error = %e, "failed to get file type for directory entry");
                continue;
            }
        };
        if file_type.is_dir() {
            continue;
        }

        let path = entry.path();
        let has_audio_ext = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(is_audio_extension);
        if !has_audio_ext {
            continue;
        }

        if seen.insert(path.clone()) {
            new_files.push(path);
        }
    }

    Ok(new_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Sensor property tests ---

    #[test]
    fn audio_sensor_name() {
        let sensor = AudioSensor::new(
            "meeting_mic",
            "/tmp/audio",
            "base.en",
            Duration::from_secs(10),
        );
        assert_eq!(sensor.name(), "meeting_mic");
    }

    #[test]
    fn audio_sensor_modality() {
        let sensor = AudioSensor::new("mic", "/tmp/audio", "base.en", Duration::from_secs(10));
        assert_eq!(sensor.modality(), SensorModality::Audio);
    }

    #[test]
    fn audio_sensor_kafka_topic() {
        let sensor = AudioSensor::new("mic", "/tmp/audio", "base.en", Duration::from_secs(10));
        assert_eq!(sensor.kafka_topic(), "hb.sensor.audio");
    }

    #[test]
    fn audio_sensor_stores_whisper_model() {
        let sensor = AudioSensor::new("mic", "/tmp/audio", "large-v3", Duration::from_secs(10));
        assert_eq!(sensor.whisper_model, "large-v3");
    }

    #[test]
    fn audio_sensor_stores_poll_interval() {
        let sensor = AudioSensor::new("mic", "/tmp/audio", "base.en", Duration::from_secs(42));
        assert_eq!(sensor.poll_interval, Duration::from_secs(42));
    }

    #[test]
    fn audio_sensor_stores_watch_directory() {
        let sensor = AudioSensor::new(
            "mic",
            "/data/recordings",
            "base.en",
            Duration::from_secs(10),
        );
        assert_eq!(sensor.watch_directory, PathBuf::from("/data/recordings"));
    }

    // --- Extension checking tests ---

    #[test]
    fn is_audio_extension_wav() {
        assert!(is_audio_extension("wav"));
    }

    #[test]
    fn is_audio_extension_mp3() {
        assert!(is_audio_extension("mp3"));
    }

    #[test]
    fn is_audio_extension_ogg() {
        assert!(is_audio_extension("ogg"));
    }

    #[test]
    fn is_audio_extension_flac() {
        assert!(is_audio_extension("flac"));
    }

    #[test]
    fn is_audio_extension_m4a() {
        assert!(is_audio_extension("m4a"));
    }

    #[test]
    fn is_audio_extension_aac() {
        assert!(is_audio_extension("aac"));
    }

    #[test]
    fn is_audio_extension_wma() {
        assert!(is_audio_extension("wma"));
    }

    #[test]
    fn is_audio_extension_opus() {
        assert!(is_audio_extension("opus"));
    }

    #[test]
    fn is_audio_extension_case_insensitive() {
        assert!(is_audio_extension("WAV"));
        assert!(is_audio_extension("Mp3"));
        assert!(is_audio_extension("FLAC"));
    }

    #[test]
    fn is_audio_extension_rejects_text() {
        assert!(!is_audio_extension("txt"));
    }

    #[test]
    fn is_audio_extension_rejects_image() {
        assert!(!is_audio_extension("jpg"));
        assert!(!is_audio_extension("png"));
    }

    #[test]
    fn is_audio_extension_rejects_empty() {
        assert!(!is_audio_extension(""));
    }

    // --- FNV hash tests ---

    #[test]
    fn fnv1a_hex_deterministic() {
        let h1 = fnv1a_hex(b"hello");
        let h2 = fnv1a_hex(b"hello");
        assert_eq!(h1, h2);
    }

    #[test]
    fn fnv1a_hex_different_inputs() {
        let h1 = fnv1a_hex(b"hello");
        let h2 = fnv1a_hex(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn fnv1a_hex_format() {
        let h = fnv1a_hex(b"test");
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // --- build_sensor_event tests ---

    #[test]
    fn build_sensor_event_content_format() {
        let modified = "2026-02-22T10:00:00Z"
            .parse::<DateTime<Utc>>()
            .expect("parse datetime");
        let event = build_sensor_event(
            "mic",
            Path::new("/tmp/audio/meeting_notes.mp3"),
            5_432_100,
            modified,
            "base.en",
        )
        .expect("build event");
        assert!(
            event.content.starts_with("Audio: meeting_notes.mp3"),
            "content: {}",
            event.content
        );
        assert!(
            event.content.contains("5432100 bytes"),
            "content: {}",
            event.content
        );
    }

    #[test]
    fn build_sensor_event_modality_is_audio() {
        let modified = Utc::now();
        let event = build_sensor_event("mic", Path::new("/tmp/a.wav"), 100, modified, "base.en")
            .expect("build event");
        assert_eq!(event.modality, SensorModality::Audio);
    }

    #[test]
    fn build_sensor_event_binary_ref_set() {
        let modified = Utc::now();
        let event = build_sensor_event(
            "mic",
            Path::new("/tmp/audio/voice.ogg"),
            1000,
            modified,
            "base.en",
        )
        .expect("build event");
        assert!(event.binary_ref.is_some(), "binary_ref should be set");
        let binary_ref = event.binary_ref.as_deref().expect("binary_ref is Some");
        assert!(binary_ref.contains("voice.ogg"), "binary_ref: {binary_ref}");
    }

    #[test]
    fn build_sensor_event_metadata_fields() {
        let modified = Utc::now();
        let event = build_sensor_event(
            "mic",
            Path::new("/tmp/audio/call.opus"),
            2048,
            modified,
            "large-v3",
        )
        .expect("build event");
        let meta = event.metadata.as_ref().expect("metadata should be set");
        assert_eq!(meta["filename"], "call.opus");
        assert_eq!(meta["extension"], "opus");
        assert_eq!(meta["size_bytes"], 2048);
        assert_eq!(meta["whisper_model"], "large-v3");
        // modified_at should be an RFC3339 string
        assert!(meta["modified_at"].as_str().is_some());
    }

    #[test]
    fn build_sensor_event_whisper_model_in_metadata() {
        let modified = Utc::now();
        let event = build_sensor_event("mic", Path::new("/tmp/test.wav"), 100, modified, "tiny.en")
            .expect("build event");
        let meta = event.metadata.as_ref().expect("metadata");
        assert_eq!(meta["whisper_model"], "tiny.en");
    }

    #[test]
    fn build_sensor_event_deterministic_id() {
        let modified = "2026-02-22T10:00:00Z"
            .parse::<DateTime<Utc>>()
            .expect("parse");
        let e1 = build_sensor_event("mic", Path::new("/tmp/a.wav"), 100, modified, "base.en")
            .expect("build");
        let e2 = build_sensor_event("mic", Path::new("/tmp/a.wav"), 100, modified, "base.en")
            .expect("build");
        assert_eq!(e1.id, e2.id, "same inputs should produce same ID");
    }

    #[test]
    fn build_sensor_event_different_files_different_ids() {
        let modified = Utc::now();
        let e1 = build_sensor_event("mic", Path::new("/tmp/a.wav"), 100, modified, "base.en")
            .expect("build");
        let e2 = build_sensor_event("mic", Path::new("/tmp/b.wav"), 100, modified, "base.en")
            .expect("build");
        assert_ne!(e1.id, e2.id, "different files should produce different IDs");
    }

    #[test]
    fn build_sensor_event_related_ids_empty() {
        let modified = Utc::now();
        let event = build_sensor_event("mic", Path::new("/tmp/a.wav"), 100, modified, "base.en")
            .expect("build");
        assert!(event.related_ids.is_empty());
    }

    #[test]
    fn build_sensor_event_sensor_name_propagated() {
        let modified = Utc::now();
        let event = build_sensor_event(
            "office_mic",
            Path::new("/tmp/a.wav"),
            100,
            modified,
            "base.en",
        )
        .expect("build");
        assert_eq!(event.sensor_name, "office_mic");
    }

    // --- Seen file tracking tests ---

    #[tokio::test]
    async fn scan_directory_finds_audio_files() {
        let dir = tempfile::tempdir().expect("create tempdir");
        let wav_path = dir.path().join("test.wav");
        tokio::fs::write(&wav_path, b"fake wav data")
            .await
            .expect("write");

        let mut seen = HashSet::new();
        let new_files = scan_directory(dir.path(), &mut seen).await.expect("scan");
        assert_eq!(new_files.len(), 1);
        assert!(new_files[0].ends_with("test.wav"));
    }

    #[tokio::test]
    async fn scan_directory_skips_non_audio() {
        let dir = tempfile::tempdir().expect("create tempdir");
        tokio::fs::write(dir.path().join("readme.txt"), b"text")
            .await
            .expect("write");
        tokio::fs::write(dir.path().join("image.png"), b"png")
            .await
            .expect("write");

        let mut seen = HashSet::new();
        let new_files = scan_directory(dir.path(), &mut seen).await.expect("scan");
        assert!(new_files.is_empty(), "non-audio files should be skipped");
    }

    #[tokio::test]
    async fn scan_directory_does_not_reprocess_seen_files() {
        let dir = tempfile::tempdir().expect("create tempdir");
        tokio::fs::write(dir.path().join("meeting.mp3"), b"mp3 data")
            .await
            .expect("write");

        let mut seen = HashSet::new();

        // First scan picks up the file
        let first = scan_directory(dir.path(), &mut seen).await.expect("scan");
        assert_eq!(first.len(), 1);

        // Second scan should return nothing (file already seen)
        let second = scan_directory(dir.path(), &mut seen).await.expect("scan");
        assert!(second.is_empty(), "seen file should not be returned again");
    }

    #[tokio::test]
    async fn scan_directory_picks_up_new_files_on_subsequent_scans() {
        let dir = tempfile::tempdir().expect("create tempdir");
        tokio::fs::write(dir.path().join("first.wav"), b"wav")
            .await
            .expect("write");

        let mut seen = HashSet::new();
        let _ = scan_directory(dir.path(), &mut seen).await.expect("scan");

        // Add a new file
        tokio::fs::write(dir.path().join("second.flac"), b"flac")
            .await
            .expect("write");

        let new_files = scan_directory(dir.path(), &mut seen).await.expect("scan");
        assert_eq!(new_files.len(), 1);
        assert!(new_files[0].ends_with("second.flac"));
    }

    #[tokio::test]
    async fn scan_directory_nonexistent_returns_error() {
        let mut seen = HashSet::new();
        let result = scan_directory(Path::new("/nonexistent/dir/12345"), &mut seen).await;
        assert!(result.is_err());
    }

    // --- Kafka key tests ---

    #[test]
    fn kafka_key_format() {
        let sensor_name = "office_mic";
        let path_hash = fnv1a_hex(b"/tmp/audio/meeting.mp3");
        let key = format!("{sensor_name}:{path_hash}");
        assert!(key.starts_with("office_mic:"));
        assert_eq!(key.len(), "office_mic:".len() + 16);
    }

    // --- SensorEvent serde roundtrip ---

    #[test]
    fn audio_sensor_event_serde_roundtrip() {
        let modified = Utc::now();
        let event = build_sensor_event(
            "mic",
            Path::new("/tmp/audio/test.mp3"),
            9999,
            modified,
            "base.en",
        )
        .expect("build");
        let json = serde_json::to_string(&event).expect("serialize");
        let back: SensorEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, event.id);
        assert_eq!(back.sensor_name, "mic");
        assert_eq!(back.modality, SensorModality::Audio);
        assert!(back.binary_ref.is_some());
    }
}
