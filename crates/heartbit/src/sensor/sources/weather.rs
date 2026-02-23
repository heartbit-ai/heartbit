use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use chrono::Utc;
use rdkafka::producer::{FutureProducer, FutureRecord};
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::sensor::{Sensor, SensorEvent, SensorModality};

/// Keywords that indicate severe weather in the description field.
const SEVERE_KEYWORDS: &[&str] = &["storm", "thunder", "hurricane", "tornado", "blizzard"];

/// Temperature threshold (Celsius) above which weather is considered severe.
const TEMP_HIGH_THRESHOLD: f64 = 40.0;

/// Temperature threshold (Celsius) below which weather is considered severe.
const TEMP_LOW_THRESHOLD: f64 = -20.0;

/// Wind speed threshold (m/s) above which weather is considered severe.
const WIND_SPEED_THRESHOLD: f64 = 20.0;

/// Weather sensor that polls the OpenWeatherMap API at a configurable interval
/// for each configured location and produces `SensorEvent` values to the
/// `hb.sensor.weather` Kafka topic.
///
/// When `alert_only` is `true`, only events for severe weather conditions are
/// produced; normal weather readings are silently dropped.
pub struct WeatherSensor {
    name: String,
    api_key: String,
    locations: Vec<String>,
    poll_interval: Duration,
    alert_only: bool,
}

impl WeatherSensor {
    pub fn new(
        name: impl Into<String>,
        api_key: impl Into<String>,
        locations: Vec<String>,
        poll_interval: Duration,
        alert_only: bool,
    ) -> Self {
        Self {
            name: name.into(),
            api_key: api_key.into(),
            locations,
            poll_interval,
            alert_only,
        }
    }
}

/// Returns `true` if the weather data indicates severe conditions.
///
/// Severe conditions are:
/// - Temperature above 40 C or below -20 C
/// - Wind speed above 20 m/s
/// - Description containing storm/thunder/hurricane/tornado/blizzard
fn is_severe_weather(temp_c: f64, wind_speed_ms: f64, description: &str) -> bool {
    // Temperature outside the safe range [-20, 40] is severe.
    if !(TEMP_LOW_THRESHOLD..=TEMP_HIGH_THRESHOLD).contains(&temp_c) {
        return true;
    }
    if wind_speed_ms > WIND_SPEED_THRESHOLD {
        return true;
    }
    let lower = description.to_lowercase();
    SEVERE_KEYWORDS.iter().any(|kw| lower.contains(kw))
}

/// Build a `SensorEvent` from parsed weather API data.
fn build_sensor_event(
    sensor_name: &str,
    location: &str,
    temp_c: f64,
    description: &str,
    wind_speed_ms: f64,
    humidity_pct: f64,
    is_alert: bool,
) -> SensorEvent {
    let content = serde_json::json!({
        "temperature_c": temp_c,
        "description": description,
        "wind_speed_ms": wind_speed_ms,
        "humidity_pct": humidity_pct,
    });
    let content_str =
        serde_json::to_string(&content).expect("weather content JSON is always serializable");

    let timestamp = Utc::now().timestamp();
    let source_id = format!("{location}:{timestamp}");

    let id = SensorEvent::generate_id(&content_str, &source_id);

    let metadata = serde_json::json!({
        "location": location,
        "temperature_c": temp_c,
        "description": description,
        "wind_speed_ms": wind_speed_ms,
        "humidity_pct": humidity_pct,
        "alert": is_alert,
    });

    SensorEvent {
        id,
        sensor_name: sensor_name.to_string(),
        modality: SensorModality::Structured,
        observed_at: Utc::now(),
        content: content_str,
        source_id,
        metadata: Some(metadata),
        binary_ref: None,
        related_ids: vec![],
    }
}

/// Parse the OpenWeatherMap JSON response into (temp, description, wind_speed, humidity).
fn parse_weather_response(body: &str) -> Result<(f64, String, f64, f64), Error> {
    let json: Value =
        serde_json::from_str(body).map_err(|e| Error::Sensor(format!("invalid JSON: {e}")))?;

    let temp = json["main"]["temp"]
        .as_f64()
        .ok_or_else(|| Error::Sensor("missing main.temp in API response".into()))?;

    let description = json["weather"]
        .as_array()
        .and_then(|arr| arr.first())
        .and_then(|w| w["description"].as_str())
        .unwrap_or("unknown")
        .to_string();

    let wind_speed = json["wind"]["speed"].as_f64().unwrap_or(0.0);

    let humidity = json["main"]["humidity"].as_f64().unwrap_or(0.0);

    Ok((temp, description, wind_speed, humidity))
}

impl Sensor for WeatherSensor {
    fn name(&self) -> &str {
        &self.name
    }

    fn modality(&self) -> SensorModality {
        SensorModality::Structured
    }

    fn kafka_topic(&self) -> &str {
        "hb.sensor.weather"
    }

    fn run(
        &self,
        producer: FutureProducer,
        cancel: CancellationToken,
    ) -> Pin<Box<dyn Future<Output = Result<(), Error>> + Send + '_>> {
        Box::pin(async move {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .user_agent("heartbit-sensor/0.1")
                .build()
                .map_err(|e| Error::Sensor(format!("failed to create HTTP client: {e}")))?;

            loop {
                for location in &self.locations {
                    if cancel.is_cancelled() {
                        return Ok(());
                    }

                    let url = format!(
                        "https://api.openweathermap.org/data/2.5/weather?q={}&appid={}&units=metric",
                        location, self.api_key
                    );

                    match client.get(&url).send().await {
                        Ok(response) => {
                            let body = match response.text().await {
                                Ok(b) => b,
                                Err(e) => {
                                    tracing::warn!(
                                        location = %location,
                                        error = %e,
                                        "failed to read weather API response body"
                                    );
                                    continue;
                                }
                            };

                            match parse_weather_response(&body) {
                                Ok((temp, description, wind_speed, humidity)) => {
                                    let is_alert =
                                        is_severe_weather(temp, wind_speed, &description);

                                    // When alert_only is true, skip non-severe weather
                                    if self.alert_only && !is_alert {
                                        tracing::debug!(
                                            location = %location,
                                            "skipping non-severe weather (alert_only=true)"
                                        );
                                        continue;
                                    }

                                    let event = build_sensor_event(
                                        &self.name,
                                        location,
                                        temp,
                                        &description,
                                        wind_speed,
                                        humidity,
                                        is_alert,
                                    );

                                    let payload = serde_json::to_vec(&event).map_err(|e| {
                                        Error::Sensor(format!(
                                            "failed to serialize weather event: {e}"
                                        ))
                                    })?;

                                    if let Err((e, _)) = producer
                                        .send(
                                            FutureRecord::to(self.kafka_topic())
                                                .key(location.as_str())
                                                .payload(&payload),
                                            rdkafka::util::Timeout::After(Duration::from_secs(5)),
                                        )
                                        .await
                                    {
                                        tracing::warn!(
                                            location = %location,
                                            error = %e,
                                            "failed to produce weather event to Kafka"
                                        );
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        location = %location,
                                        error = %e,
                                        "failed to parse weather API response"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                location = %location,
                                error = %e,
                                "failed to fetch weather data"
                            );
                        }
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

#[cfg(test)]
mod tests {
    use super::*;

    // --- Sensor trait property tests ---

    #[test]
    fn sensor_name() {
        let sensor = WeatherSensor::new(
            "my_weather",
            "fake-key",
            vec!["London".into()],
            Duration::from_secs(60),
            false,
        );
        assert_eq!(sensor.name(), "my_weather");
    }

    #[test]
    fn sensor_modality_is_structured() {
        let sensor = WeatherSensor::new(
            "w",
            "key",
            vec!["Paris".into()],
            Duration::from_secs(60),
            false,
        );
        assert_eq!(sensor.modality(), SensorModality::Structured);
    }

    #[test]
    fn sensor_kafka_topic() {
        let sensor = WeatherSensor::new(
            "w",
            "key",
            vec!["Berlin".into()],
            Duration::from_secs(60),
            false,
        );
        assert_eq!(sensor.kafka_topic(), "hb.sensor.weather");
    }

    // --- is_severe_weather tests ---

    #[test]
    fn severe_high_temp() {
        assert!(is_severe_weather(41.0, 5.0, "clear sky"));
    }

    #[test]
    fn severe_low_temp() {
        assert!(is_severe_weather(-21.0, 5.0, "light snow"));
    }

    #[test]
    fn severe_high_wind() {
        assert!(is_severe_weather(20.0, 21.0, "windy"));
    }

    #[test]
    fn severe_storm_keyword() {
        assert!(is_severe_weather(20.0, 5.0, "thunderstorm with rain"));
    }

    #[test]
    fn severe_hurricane_keyword() {
        assert!(is_severe_weather(25.0, 10.0, "hurricane warning"));
    }

    #[test]
    fn severe_tornado_keyword() {
        assert!(is_severe_weather(22.0, 8.0, "tornado watch"));
    }

    #[test]
    fn severe_blizzard_keyword() {
        assert!(is_severe_weather(0.0, 15.0, "blizzard conditions"));
    }

    #[test]
    fn not_severe_normal_weather() {
        assert!(!is_severe_weather(22.0, 5.0, "clear sky"));
    }

    #[test]
    fn not_severe_boundary_temp_high() {
        // Exactly 40.0 is NOT severe (must be > 40)
        assert!(!is_severe_weather(40.0, 5.0, "hot"));
    }

    #[test]
    fn not_severe_boundary_temp_low() {
        // Exactly -20.0 is NOT severe (must be < -20)
        assert!(!is_severe_weather(-20.0, 5.0, "cold"));
    }

    #[test]
    fn not_severe_boundary_wind() {
        // Exactly 20.0 is NOT severe (must be > 20)
        assert!(!is_severe_weather(20.0, 20.0, "breezy"));
    }

    #[test]
    fn severe_keyword_case_insensitive() {
        assert!(is_severe_weather(20.0, 5.0, "THUNDERSTORM"));
        assert!(is_severe_weather(20.0, 5.0, "Heavy Storm"));
    }

    // --- build_sensor_event tests ---

    #[test]
    fn event_modality_is_structured() {
        let event = build_sensor_event("weather_sensor", "London", 20.0, "clear", 5.0, 65.0, false);
        assert_eq!(event.modality, SensorModality::Structured);
    }

    #[test]
    fn event_sensor_name() {
        let event = build_sensor_event(
            "my_sensor",
            "Tokyo",
            25.0,
            "partly cloudy",
            3.0,
            70.0,
            false,
        );
        assert_eq!(event.sensor_name, "my_sensor");
    }

    #[test]
    fn event_content_is_valid_json() {
        let event = build_sensor_event("w", "Paris", 18.0, "overcast", 7.0, 80.0, false);
        let parsed: Result<Value, _> = serde_json::from_str(&event.content);
        assert!(
            parsed.is_ok(),
            "content should be valid JSON: {}",
            event.content
        );

        let json = parsed.expect("already checked");
        assert_eq!(json["temperature_c"], 18.0);
        assert_eq!(json["description"], "overcast");
        assert_eq!(json["wind_speed_ms"], 7.0);
        assert_eq!(json["humidity_pct"], 80.0);
    }

    #[test]
    fn event_source_id_format() {
        let event = build_sensor_event("w", "Berlin", 15.0, "rain", 10.0, 90.0, false);
        assert!(
            event.source_id.starts_with("Berlin:"),
            "source_id should start with location, got: {}",
            event.source_id
        );
        // After the colon should be a numeric timestamp
        let parts: Vec<&str> = event.source_id.splitn(2, ':').collect();
        assert_eq!(parts.len(), 2);
        assert!(
            parts[1].parse::<i64>().is_ok(),
            "timestamp part should be numeric, got: {}",
            parts[1]
        );
    }

    #[test]
    fn event_metadata_fields() {
        let event = build_sensor_event("w", "NYC", 30.0, "hot and humid", 8.0, 85.0, true);
        let meta = event.metadata.as_ref().expect("metadata should be present");
        assert_eq!(meta["location"], "NYC");
        assert_eq!(meta["temperature_c"], 30.0);
        assert_eq!(meta["description"], "hot and humid");
        assert_eq!(meta["wind_speed_ms"], 8.0);
        assert_eq!(meta["humidity_pct"], 85.0);
        assert_eq!(meta["alert"], true);
    }

    #[test]
    fn event_metadata_alert_false() {
        let event = build_sensor_event("w", "Oslo", 5.0, "cloudy", 3.0, 60.0, false);
        let meta = event.metadata.as_ref().expect("metadata should be present");
        assert_eq!(meta["alert"], false);
    }

    #[test]
    fn event_binary_ref_is_none() {
        let event = build_sensor_event("w", "Rome", 28.0, "sunny", 2.0, 40.0, false);
        assert!(event.binary_ref.is_none());
    }

    #[test]
    fn event_related_ids_empty() {
        let event = build_sensor_event("w", "Madrid", 35.0, "hot", 4.0, 30.0, false);
        assert!(event.related_ids.is_empty());
    }

    #[test]
    fn event_id_deterministic_for_same_content() {
        // Two events with the same content and source_id should have the same id
        let content = serde_json::json!({
            "temperature_c": 20.0,
            "description": "clear",
            "wind_speed_ms": 5.0,
            "humidity_pct": 50.0,
        });
        let content_str = serde_json::to_string(&content).expect("serializable");
        let source_id = "London:1700000000";

        let id1 = SensorEvent::generate_id(&content_str, source_id);
        let id2 = SensorEvent::generate_id(&content_str, source_id);
        assert_eq!(id1, id2);
    }

    #[test]
    fn event_id_differs_for_different_locations() {
        let content = serde_json::json!({
            "temperature_c": 20.0,
            "description": "clear",
            "wind_speed_ms": 5.0,
            "humidity_pct": 50.0,
        });
        let content_str = serde_json::to_string(&content).expect("serializable");

        let id1 = SensorEvent::generate_id(&content_str, "London:1700000000");
        let id2 = SensorEvent::generate_id(&content_str, "Paris:1700000000");
        assert_ne!(id1, id2);
    }

    // --- parse_weather_response tests ---

    #[test]
    fn parse_valid_response() {
        let body = serde_json::json!({
            "main": {"temp": 22.5, "humidity": 65.0},
            "weather": [{"description": "scattered clouds"}],
            "wind": {"speed": 4.2}
        })
        .to_string();

        let (temp, desc, wind, humidity) = parse_weather_response(&body).expect("should parse");
        assert!((temp - 22.5).abs() < f64::EPSILON);
        assert_eq!(desc, "scattered clouds");
        assert!((wind - 4.2).abs() < f64::EPSILON);
        assert!((humidity - 65.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_missing_temp_returns_error() {
        let body = serde_json::json!({
            "main": {"humidity": 50.0},
            "weather": [{"description": "clear"}],
            "wind": {"speed": 3.0}
        })
        .to_string();

        let result = parse_weather_response(&body);
        assert!(result.is_err());
    }

    #[test]
    fn parse_missing_weather_description_defaults() {
        let body = serde_json::json!({
            "main": {"temp": 20.0, "humidity": 50.0},
            "weather": [],
            "wind": {"speed": 3.0}
        })
        .to_string();

        let (_, desc, _, _) = parse_weather_response(&body).expect("should parse");
        assert_eq!(desc, "unknown");
    }

    #[test]
    fn parse_missing_wind_defaults_to_zero() {
        let body = serde_json::json!({
            "main": {"temp": 20.0, "humidity": 50.0},
            "weather": [{"description": "clear"}]
        })
        .to_string();

        let (_, _, wind, _) = parse_weather_response(&body).expect("should parse");
        assert!((wind - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn parse_invalid_json_returns_error() {
        let result = parse_weather_response("not json");
        assert!(result.is_err());
    }

    // --- alert_only filtering tests ---

    #[test]
    fn alert_only_skips_non_severe() {
        // Simulates the alert_only filter logic
        let alert_only = true;
        let is_alert = is_severe_weather(20.0, 5.0, "clear sky");
        let should_skip = alert_only && !is_alert;
        assert!(
            should_skip,
            "normal weather should be skipped when alert_only=true"
        );
    }

    #[test]
    fn alert_only_does_not_skip_severe() {
        let alert_only = true;
        let is_alert = is_severe_weather(42.0, 5.0, "extreme heat");
        let should_skip = alert_only && !is_alert;
        assert!(
            !should_skip,
            "severe weather should NOT be skipped even when alert_only=true"
        );
    }

    #[test]
    fn non_alert_mode_produces_all() {
        let alert_only = false;
        let is_alert = is_severe_weather(20.0, 5.0, "clear sky");
        let should_skip = alert_only && !is_alert;
        assert!(!should_skip, "non-alert mode should not skip anything");
    }

    // --- Constructor tests ---

    #[test]
    fn constructor_stores_fields() {
        let sensor = WeatherSensor::new(
            "city_weather",
            "api-key-123",
            vec!["London".into(), "Paris".into()],
            Duration::from_secs(1800),
            true,
        );
        assert_eq!(sensor.name, "city_weather");
        assert_eq!(sensor.api_key, "api-key-123");
        assert_eq!(sensor.locations, vec!["London", "Paris"]);
        assert_eq!(sensor.poll_interval, Duration::from_secs(1800));
        assert!(sensor.alert_only);
    }

    #[test]
    fn constructor_accepts_string_types() {
        let sensor = WeatherSensor::new(
            String::from("test"),
            String::from("key"),
            vec!["Berlin".into()],
            Duration::from_secs(60),
            false,
        );
        assert_eq!(sensor.name(), "test");
    }
}
