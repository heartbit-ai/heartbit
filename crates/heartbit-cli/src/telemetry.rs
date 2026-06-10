//! OpenTelemetry tracing setup, decoupled from the Restate `serve` path.
//!
//! `setup_telemetry` is wired by `init_tracing_from_config` (always available)
//! whenever a `[telemetry]` block is present, so it must compile independently
//! of the `restate` feature. It lives here behind the `telemetry` feature.

use anyhow::{Context, Result};

/// Initialize an OpenTelemetry OTLP-exporting tracing subscriber.
pub(crate) fn setup_telemetry(otlp_endpoint: &str, service_name: &str) -> Result<()> {
    use opentelemetry::trace::TracerProvider;
    use opentelemetry_otlp::WithExportConfig;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(otlp_endpoint)
        .build()
        .context("failed to create OTLP exporter")?;

    let provider = opentelemetry_sdk::trace::SdkTracerProvider::builder()
        .with_batch_exporter(exporter)
        .with_resource(
            opentelemetry_sdk::Resource::builder()
                .with_service_name(service_name.to_string())
                .build(),
        )
        .build();

    let tracer = provider.tracer("heartbit");
    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with(tracing_subscriber::fmt::layer())
        .with(otel_layer)
        .init();

    Ok(())
}
