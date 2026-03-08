#[derive(Debug, thiserror::Error)]
#[allow(dead_code)] // Scaffolded for webhook/sensor error handling in next phase
pub enum GatewayError {
    #[error("kafka error: {0}")]
    Kafka(String),
    #[error("config error: {0}")]
    Config(String),
    #[error("sensor error: {0}")]
    Sensor(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kafka_error_display() {
        let err = GatewayError::Kafka("connection refused".into());
        assert_eq!(err.to_string(), "kafka error: connection refused");
    }

    #[test]
    fn config_error_display() {
        let err = GatewayError::Config("missing field".into());
        assert_eq!(err.to_string(), "config error: missing field");
    }

    #[test]
    fn sensor_error_display() {
        let err = GatewayError::Sensor("timeout".into());
        assert_eq!(err.to_string(), "sensor error: timeout");
    }
}
