use std::collections::HashSet;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;

use chrono::{DateTime, Utc};
use rdkafka::producer::{FutureProducer, FutureRecord};
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::sensor::{Sensor, SensorEvent, SensorModality};

/// Supported image file extensions (lowercase).
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "bmp", "webp", "tiff", "svg"];

/// Image sensor that watches a directory for new image files.
///
/// Polls the configured `watch_directory` at `poll_interval` for files
/// matching image extensions. Tracks already-seen files to avoid re-processing.
/// Each new image produces a `SensorEvent` to the `hb.sensor.image` Kafka topic.
pub struct ImageSensor {
    name: String,
    watch_directory: PathBuf,
    poll_interval: Duration,
}

impl ImageSensor {
    pub fn new(
        name: impl Into<String>,
        watch_directory: impl Into<PathBuf>,
        poll_interval: Duration,
    ) -> Self {
        Self {
            name: name.into(),
            watch_directory: watch_directory.into(),
            poll_interval,
        }
    }
}

/// Returns `true` if the given extension (case-insensitive) is a supported image format.
fn is_image_extension(ext: &str) -> bool {
    let lower = ext.to_lowercase();
    IMAGE_EXTENSIONS.contains(&lower.as_str())
}

/// Build a hex hash of a path for use as a Kafka key component.
fn hex_hash_of_path(path: &std::path::Path) -> String {
    let path_str = path.to_string_lossy();
    let mut hash: u64 = 0xcbf29ce484222325;
    for byte in path_str.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

/// Metadata about an image file discovered in the watch directory.
struct ImageFileInfo {
    path: PathBuf,
    filename: String,
    extension: String,
    size_bytes: u64,
    modified_at: DateTime<Utc>,
}

/// Build a `SensorEvent` from image file metadata.
fn build_sensor_event(sensor_name: &str, info: &ImageFileInfo) -> SensorEvent {
    let source_id = info.path.to_string_lossy().to_string();
    let content = format!(
        "Image: {} ({} bytes, {})",
        info.filename, info.size_bytes, info.modified_at
    );
    let id = SensorEvent::generate_id(&content, &source_id);

    SensorEvent {
        id,
        sensor_name: sensor_name.to_string(),
        modality: SensorModality::Image,
        observed_at: Utc::now(),
        content,
        source_id,
        metadata: Some(serde_json::json!({
            "filename": info.filename,
            "extension": info.extension,
            "size_bytes": info.size_bytes,
            "modified_at": info.modified_at.to_rfc3339(),
        })),
        binary_ref: Some(info.path.to_string_lossy().to_string()),
        related_ids: vec![],
    }
}

impl Sensor for ImageSensor {
    fn name(&self) -> &str {
        &self.name
    }

    fn modality(&self) -> SensorModality {
        SensorModality::Image
    }

    fn kafka_topic(&self) -> &str {
        "hb.sensor.image"
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

                match scan_directory(&self.watch_directory).await {
                    Ok(files) => {
                        for info in files {
                            if seen.contains(&info.path) {
                                continue;
                            }

                            let event = build_sensor_event(&self.name, &info);

                            let payload = serde_json::to_vec(&event).map_err(|e| {
                                Error::Sensor(format!("failed to serialize sensor event: {e}"))
                            })?;

                            let key = format!("{}:{}", self.name, hex_hash_of_path(&info.path));

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
                                    file = %info.path.display(),
                                    error = %e,
                                    "failed to produce image event to Kafka"
                                );
                            }

                            seen.insert(info.path);
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            directory = %self.watch_directory.display(),
                            error = %e,
                            "failed to scan image directory"
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

/// Scan a directory for image files and return their metadata.
async fn scan_directory(dir: &std::path::Path) -> Result<Vec<ImageFileInfo>, Error> {
    let mut entries = tokio::fs::read_dir(dir)
        .await
        .map_err(|e| Error::Sensor(format!("failed to read directory {}: {e}", dir.display())))?;

    let mut files = Vec::new();

    while let Some(entry) = entries
        .next_entry()
        .await
        .map_err(|e| Error::Sensor(format!("failed to read directory entry: {e}")))?
    {
        let path = entry.path();

        // Skip directories
        let file_type = entry
            .file_type()
            .await
            .map_err(|e| Error::Sensor(format!("failed to get file type: {e}")))?;
        if file_type.is_dir() {
            continue;
        }

        // Check extension
        let ext = match path.extension().and_then(|e| e.to_str()) {
            Some(e) => e.to_string(),
            None => continue,
        };

        if !is_image_extension(&ext) {
            continue;
        }

        // Get metadata
        let metadata = tokio::fs::metadata(&path).await.map_err(|e| {
            Error::Sensor(format!(
                "failed to read metadata for {}: {e}",
                path.display()
            ))
        })?;

        let modified_at = metadata
            .modified()
            .map(DateTime::<Utc>::from)
            .unwrap_or_else(|_| Utc::now());

        let filename = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();

        files.push(ImageFileInfo {
            path,
            filename,
            extension: ext.to_lowercase(),
            size_bytes: metadata.len(),
            modified_at,
        });
    }

    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Sensor property tests ---

    #[test]
    fn image_sensor_name() {
        let sensor = ImageSensor::new("scanner", "/tmp/images", Duration::from_secs(10));
        assert_eq!(sensor.name(), "scanner");
    }

    #[test]
    fn image_sensor_modality() {
        let sensor = ImageSensor::new("scanner", "/tmp/images", Duration::from_secs(10));
        assert_eq!(sensor.modality(), SensorModality::Image);
    }

    #[test]
    fn image_sensor_kafka_topic() {
        let sensor = ImageSensor::new("scanner", "/tmp/images", Duration::from_secs(10));
        assert_eq!(sensor.kafka_topic(), "hb.sensor.image");
    }

    #[test]
    fn image_sensor_stores_watch_directory() {
        let sensor = ImageSensor::new("scanner", "/home/inbox/photos", Duration::from_secs(60));
        assert_eq!(sensor.watch_directory, PathBuf::from("/home/inbox/photos"));
    }

    #[test]
    fn image_sensor_stores_poll_interval() {
        let sensor = ImageSensor::new("scanner", "/tmp", Duration::from_secs(30));
        assert_eq!(sensor.poll_interval, Duration::from_secs(30));
    }

    // --- Extension filtering tests ---

    #[test]
    fn is_image_extension_jpg() {
        assert!(is_image_extension("jpg"));
    }

    #[test]
    fn is_image_extension_jpeg() {
        assert!(is_image_extension("jpeg"));
    }

    #[test]
    fn is_image_extension_png() {
        assert!(is_image_extension("png"));
    }

    #[test]
    fn is_image_extension_gif() {
        assert!(is_image_extension("gif"));
    }

    #[test]
    fn is_image_extension_bmp() {
        assert!(is_image_extension("bmp"));
    }

    #[test]
    fn is_image_extension_webp() {
        assert!(is_image_extension("webp"));
    }

    #[test]
    fn is_image_extension_tiff() {
        assert!(is_image_extension("tiff"));
    }

    #[test]
    fn is_image_extension_svg() {
        assert!(is_image_extension("svg"));
    }

    #[test]
    fn is_image_extension_case_insensitive() {
        assert!(is_image_extension("JPG"));
        assert!(is_image_extension("Png"));
        assert!(is_image_extension("WEBP"));
    }

    #[test]
    fn is_image_extension_rejects_non_image() {
        assert!(!is_image_extension("txt"));
        assert!(!is_image_extension("pdf"));
        assert!(!is_image_extension("mp4"));
        assert!(!is_image_extension("rs"));
        assert!(!is_image_extension("exe"));
    }

    #[test]
    fn is_image_extension_rejects_empty() {
        assert!(!is_image_extension(""));
    }

    // --- hex_hash_of_path tests ---

    #[test]
    fn hex_hash_of_path_deterministic() {
        let path = PathBuf::from("/tmp/photo.jpg");
        let h1 = hex_hash_of_path(&path);
        let h2 = hex_hash_of_path(&path);
        assert_eq!(h1, h2);
    }

    #[test]
    fn hex_hash_of_path_different_paths_differ() {
        let h1 = hex_hash_of_path(&PathBuf::from("/tmp/a.jpg"));
        let h2 = hex_hash_of_path(&PathBuf::from("/tmp/b.jpg"));
        assert_ne!(h1, h2);
    }

    #[test]
    fn hex_hash_of_path_is_16_hex_chars() {
        let h = hex_hash_of_path(&PathBuf::from("/some/path.png"));
        assert_eq!(h.len(), 16);
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    }

    // --- build_sensor_event tests ---

    fn sample_info() -> ImageFileInfo {
        ImageFileInfo {
            path: PathBuf::from("/inbox/invoice.jpg"),
            filename: "invoice.jpg".into(),
            extension: "jpg".into(),
            size_bytes: 1_234_567,
            modified_at: "2026-02-22T10:00:00Z"
                .parse::<DateTime<Utc>>()
                .expect("valid datetime"),
        }
    }

    #[test]
    fn build_sensor_event_modality_is_image() {
        let event = build_sensor_event("scanner", &sample_info());
        assert_eq!(event.modality, SensorModality::Image);
    }

    #[test]
    fn build_sensor_event_sensor_name() {
        let event = build_sensor_event("my_sensor", &sample_info());
        assert_eq!(event.sensor_name, "my_sensor");
    }

    #[test]
    fn build_sensor_event_content_format() {
        let event = build_sensor_event("scanner", &sample_info());
        assert!(
            event.content.starts_with("Image: invoice.jpg"),
            "content: {}",
            event.content
        );
        assert!(
            event.content.contains("1234567 bytes"),
            "content: {}",
            event.content
        );
        assert!(
            event.content.contains("2026-02-22"),
            "content: {}",
            event.content
        );
    }

    #[test]
    fn build_sensor_event_source_id_is_absolute_path() {
        let event = build_sensor_event("scanner", &sample_info());
        assert_eq!(event.source_id, "/inbox/invoice.jpg");
    }

    #[test]
    fn build_sensor_event_binary_ref_set() {
        let event = build_sensor_event("scanner", &sample_info());
        assert_eq!(event.binary_ref, Some("/inbox/invoice.jpg".to_string()));
    }

    #[test]
    fn build_sensor_event_metadata_fields() {
        let event = build_sensor_event("scanner", &sample_info());
        let meta = event.metadata.expect("metadata should be present");
        assert_eq!(meta["filename"], "invoice.jpg");
        assert_eq!(meta["extension"], "jpg");
        assert_eq!(meta["size_bytes"], 1_234_567);
        assert!(meta["modified_at"].as_str().is_some());
    }

    #[test]
    fn build_sensor_event_related_ids_empty() {
        let event = build_sensor_event("scanner", &sample_info());
        assert!(event.related_ids.is_empty());
    }

    #[test]
    fn build_sensor_event_deterministic_id() {
        let info = sample_info();
        let e1 = build_sensor_event("scanner", &info);
        let e2 = build_sensor_event("scanner", &info);
        assert_eq!(e1.id, e2.id, "same input should produce same ID");
    }

    #[test]
    fn build_sensor_event_id_hex_format() {
        let event = build_sensor_event("scanner", &sample_info());
        assert_eq!(event.id.len(), 16);
        assert!(event.id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn build_sensor_event_different_files_different_ids() {
        let info1 = sample_info();
        let mut info2 = sample_info();
        info2.filename = "photo.png".into();
        info2.path = PathBuf::from("/inbox/photo.png");

        let e1 = build_sensor_event("scanner", &info1);
        let e2 = build_sensor_event("scanner", &info2);
        assert_ne!(e1.id, e2.id);
    }

    // --- scan_directory tests (async, uses real filesystem) ---

    #[tokio::test]
    async fn scan_directory_empty_dir() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let files = scan_directory(dir.path())
            .await
            .expect("scan should succeed");
        assert!(files.is_empty());
    }

    #[tokio::test]
    async fn scan_directory_finds_image_files() {
        let dir = tempfile::tempdir().expect("create temp dir");
        tokio::fs::write(dir.path().join("photo.jpg"), b"fake jpeg data")
            .await
            .expect("write file");
        tokio::fs::write(dir.path().join("doc.txt"), b"not an image")
            .await
            .expect("write file");

        let files = scan_directory(dir.path())
            .await
            .expect("scan should succeed");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].filename, "photo.jpg");
        assert_eq!(files[0].extension, "jpg");
    }

    #[tokio::test]
    async fn scan_directory_skips_subdirectories() {
        let dir = tempfile::tempdir().expect("create temp dir");
        tokio::fs::create_dir(dir.path().join("subdir"))
            .await
            .expect("create subdir");
        tokio::fs::write(dir.path().join("image.png"), b"png data")
            .await
            .expect("write file");

        let files = scan_directory(dir.path())
            .await
            .expect("scan should succeed");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].filename, "image.png");
    }

    #[tokio::test]
    async fn scan_directory_multiple_image_types() {
        let dir = tempfile::tempdir().expect("create temp dir");
        for name in &["a.jpg", "b.png", "c.gif", "d.webp", "e.bmp"] {
            tokio::fs::write(dir.path().join(name), b"data")
                .await
                .expect("write file");
        }

        let files = scan_directory(dir.path())
            .await
            .expect("scan should succeed");
        assert_eq!(files.len(), 5);
    }

    #[tokio::test]
    async fn scan_directory_nonexistent_returns_error() {
        let result = scan_directory(std::path::Path::new("/nonexistent/path/abc123")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn scan_directory_file_size_correct() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let data = vec![0u8; 42];
        tokio::fs::write(dir.path().join("tiny.png"), &data)
            .await
            .expect("write file");

        let files = scan_directory(dir.path())
            .await
            .expect("scan should succeed");
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].size_bytes, 42);
    }

    // --- Seen-files tracking test ---

    #[test]
    fn seen_files_prevents_reprocessing() {
        let mut seen: HashSet<PathBuf> = HashSet::new();
        let path = PathBuf::from("/inbox/photo.jpg");

        // First time: not seen
        assert!(!seen.contains(&path));
        seen.insert(path.clone());

        // Second time: already seen
        assert!(seen.contains(&path));
    }

    // --- Serde roundtrip test for produced events ---

    #[test]
    fn sensor_event_serde_roundtrip() {
        let event = build_sensor_event("scanner", &sample_info());
        let json = serde_json::to_string(&event).expect("serialize");
        let back: SensorEvent = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.id, event.id);
        assert_eq!(back.sensor_name, "scanner");
        assert_eq!(back.modality, SensorModality::Image);
        assert_eq!(back.binary_ref, event.binary_ref);
    }
}
