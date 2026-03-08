use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::get;
use axum::{Json, Router};
use std::time::Instant;

/// Shared state for the gateway HTTP server.
#[derive(Clone)]
pub struct GatewayState {
    pub start_time: Instant,
    pub cancel: tokio_util::sync::CancellationToken,
}

/// Build the Axum router with health and readiness endpoints.
pub fn build_router(state: GatewayState) -> Router {
    Router::new()
        .route("/v1/health", get(handle_health))
        .route("/v1/ready", get(handle_ready))
        .with_state(state)
}

async fn handle_health(State(state): State<GatewayState>) -> impl IntoResponse {
    let status = if state.cancel.is_cancelled() {
        "shutting_down"
    } else {
        "ok"
    };
    Json(serde_json::json!({
        "status": status,
        "uptime_seconds": state.start_time.elapsed().as_secs(),
    }))
}

async fn handle_ready(State(state): State<GatewayState>) -> impl IntoResponse {
    Json(serde_json::json!({
        "ready": !state.cancel.is_cancelled(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    fn test_state() -> GatewayState {
        GatewayState {
            start_time: Instant::now(),
            cancel: tokio_util::sync::CancellationToken::new(),
        }
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let app = build_router(test_state());
        let req = Request::get("/v1/health").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "ok");
        assert!(json["uptime_seconds"].is_number());
    }

    #[tokio::test]
    async fn ready_returns_true() {
        let app = build_router(test_state());
        let req = Request::get("/v1/ready").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ready"], true);
    }

    #[tokio::test]
    async fn ready_returns_false_when_cancelled() {
        let state = test_state();
        state.cancel.cancel();
        let app = build_router(state);
        let req = Request::get("/v1/ready").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();

        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["ready"], false);
    }

    #[tokio::test]
    async fn health_shows_shutting_down_when_cancelled() {
        let state = test_state();
        state.cancel.cancel();
        let app = build_router(state);
        let req = Request::get("/v1/health").body(Body::empty()).unwrap();
        let resp = app.oneshot(req).await.unwrap();

        let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["status"], "shutting_down");
    }
}
