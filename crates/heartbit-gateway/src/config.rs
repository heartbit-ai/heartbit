use serde::Deserialize;

/// Top-level gateway configuration, loaded from a TOML file.
#[derive(Debug, Clone, Deserialize)]
pub struct GatewayConfig {
    pub server: ServerConfig,
    pub kafka: heartbit::KafkaConfig,
    #[serde(default)]
    pub schedules: Vec<heartbit::ScheduleEntry>,
    pub sensors: Option<heartbit::SensorConfig>,
}

/// HTTP server settings.
#[derive(Debug, Clone, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_listen_addr")]
    pub listen_addr: String,
}

fn default_listen_addr() -> String {
    "0.0.0.0:8080".into()
}

impl GatewayConfig {
    /// Load configuration from a TOML file.
    pub fn from_file(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserialize_minimal_config() {
        let toml = r#"
[server]
listen_addr = "127.0.0.1:9090"

[kafka]
brokers = "localhost:9092"
"#;
        let config: GatewayConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.server.listen_addr, "127.0.0.1:9090");
        assert_eq!(config.kafka.brokers, "localhost:9092");
        assert!(config.schedules.is_empty());
        assert!(config.sensors.is_none());
    }

    #[test]
    fn deserialize_with_schedules() {
        let toml = r#"
[server]

[kafka]
brokers = "kafka:9092"

[[schedules]]
name = "daily-report"
cron = "0 0 9 * * *"
task = "Generate daily report"
enabled = true
"#;
        let config: GatewayConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.server.listen_addr, "0.0.0.0:8080");
        assert_eq!(config.schedules.len(), 1);
        assert_eq!(config.schedules[0].name, "daily-report");
    }

    #[test]
    fn from_file_nonexistent_returns_error() {
        let result = GatewayConfig::from_file(std::path::Path::new("/nonexistent/path.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn server_config_default_listen_addr() {
        let toml = r#"
[server]

[kafka]
brokers = "localhost:9092"
"#;
        let config: GatewayConfig = toml::from_str(toml).unwrap();
        assert_eq!(config.server.listen_addr, "0.0.0.0:8080");
    }
}
