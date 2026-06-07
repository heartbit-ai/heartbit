//! Mock LLM provider that returns CRM-shaped responses.
//!
//! Each call cycles through the pre-configured response pool, producing
//! realistic text or structured (`__respond__` tool) responses that the
//! workflow agents can consume.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use heartbit_core::Error;
use heartbit_core::llm::LlmProvider;
use heartbit_core::llm::types::{
    CompletionRequest, CompletionResponse, ContentBlock, RESPOND_TOOL_NAME, StopReason, TokenUsage,
};

/// Mock provider that cycles through a fixed set of responses.
///
/// Each `complete()` call returns the next response in the pool, wrapping
/// around when exhausted. Token usage is derived from the response content
/// so the workflow budget is exercised realistically.
pub struct CrmMockProvider {
    responses: Vec<CompletionResponse>,
    cursor: AtomicUsize,
    /// Captured requests for debugging (unused in demo, available for tests).
    #[allow(dead_code)]
    captured: Mutex<Vec<String>>,
}

impl CrmMockProvider {
    pub fn new(responses: Vec<CompletionResponse>) -> Self {
        Self {
            responses,
            cursor: AtomicUsize::new(0),
            captured: Mutex::new(Vec::new()),
        }
    }

    /// Build the full set of responses needed for the CRM demo:
    /// contact enrichment (4), deal analysis (3), email drafts (4),
    /// lead scores (4), onboarding (5), ticket triage (3) = ~23 responses.
    pub fn responses_for_demo() -> Vec<CompletionResponse> {
        let mut v = Vec::new();

        // --- Contact enrichment (4 contacts) ---
        for (name, linkedin) in [
            ("Alice Chen", "linkedin.com/in/alicechen"),
            ("Bob Martinez", "linkedin.com/in/bobmartinez"),
            ("Carol Dubois", "linkedin.com/in/caroldubois"),
            ("David Kim", "linkedin.com/in/davidkim"),
        ] {
            v.push(Self::text_response(
                &format!(
                    "Enriched profile for {name}: LinkedIn={linkedin}, \
                     Phone=+1-555-xxxx, Recent activity: spoke at TechConf 2025, \
                     interests in AI/ML and cloud infrastructure. Engagement score: high."
                ),
                120,
                80,
            ));
        }

        // --- Deal analysis (3 deals) ---
        for (id, health, prob) in [
            ("d-101", "green", 78),
            ("d-102", "yellow", 55),
            ("d-103", "green", 82),
        ] {
            v.push(Self::respond_response(
                serde_json::json!({
                    "deal_id": id,
                    "health": health,
                    "next_action": "Schedule executive review with VP Engineering",
                    "risk_factors": ["Competitor evaluation in progress", "Budget approval pending Q2"],
                    "probability_to_close": prob
                }),
                150,
                60,
            ));
        }

        // --- Email campaign (4 contacts) ---
        for name in ["Alice", "Bob", "Carol", "David"] {
            v.push(Self::text_response(
                &format!(
                    "Subject: Exclusive invitation to PulsarData AI Summit\n\
                     Hi {name},\n\n\
                     As a valued PulsarData partner, you're invited to our annual \
                     AI Summit on March 15th. This year features keynotes on \
                     generative AI in enterprise and a hands-on workshop.\n\n\
                     RSVP by Feb 28th.\n\nBest,\nThe PulsarData Team"
                ),
                80,
                100,
            ));
        }

        // --- Lead scoring (4 leads with __respond__) ---
        for (score, tier, rationale) in [
            (
                85,
                "hot",
                "VP Engineering at enterprise client, high engagement, recent demo request",
            ),
            (
                72,
                "warm",
                "CTO at mid-market, attended webinar, evaluating competitors",
            ),
            (
                68,
                "warm",
                "Head of Product, strong fit for pilot program, budget confirmed",
            ),
            (
                42,
                "cold",
                "Senior engineer, no buying authority, early-stage interest only",
            ),
        ] {
            v.push(Self::respond_response(
                serde_json::json!({
                    "score": score,
                    "tier": tier,
                    "rationale": rationale
                }),
                100,
                40,
            ));
        }

        // --- Onboarding workflow (sequential: 5 agents) ---
        v.push(Self::text_response(
            "Account provisioned: workspace 'acmecorp' created with 50 seats. \
             SSO configured via SAML 2.0 with IdP metadata from acmecorp.io.",
            90,
            45,
        ));
        v.push(Self::text_response(
            "Data migration plan: 3 CSV files (contacts: 1,240 rows, deals: 89, \
             companies: 56). Estimated import time: 45 seconds. Field mapping validated.",
            110,
            55,
        ));
        v.push(Self::text_response(
            "Integration configured: Salesforce bidirectional sync enabled. \
             Slack notifications channel #crm-alerts connected. Webhook endpoint \
             https://acmecorp.io/hooks/pulsardata verified.",
            100,
            50,
        ));
        v.push(Self::text_response(
            "Training session scheduled: March 10, 2:00 PM EST. \
             Attendees: Alice Chen (VP Eng), 12 team members. \
             Materials: Getting Started guide + Custom Playbook v2.1 sent.",
            85,
            40,
        ));
        v.push(Self::text_response(
            "Health check complete: all systems operational. \
             First sync completed (1,240 contacts imported). \
             CSM assigned: Sarah Thompson (sarah@pulsardata.com). \
             30-day check-in scheduled for April 10.",
            95,
            50,
        ));

        // --- Ticket triage (3 tickets with __respond__) ---
        for (id, cat, sev, assignee, sla) in [
            ("t-501", "bug", 2, "Platform Engineering", 4),
            ("t-502", "performance", 3, "API Team", 8),
            ("t-503", "feature_request", 4, "Product Management", 72),
        ] {
            v.push(Self::respond_response(
                serde_json::json!({
                    "ticket_id": id,
                    "category": cat,
                    "severity": sev,
                    "suggested_assignee": assignee,
                    "sla_hours": sla
                }),
                80,
                35,
            ));
        }

        v
    }

    /// Build a text completion response.
    fn text_response(text: &str, input_tokens: u32, output_tokens: u32) -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens,
                output_tokens,
                ..Default::default()
            },
            model: Some("crm-mock-v1".into()),
            reasoning: None,
        }
    }

    /// Build a structured-output response using the `__respond__` tool pattern.
    fn respond_response(
        payload: serde_json::Value,
        input_tokens: u32,
        output_tokens: u32,
    ) -> CompletionResponse {
        CompletionResponse {
            content: vec![ContentBlock::ToolUse {
                id: "resp-1".into(),
                name: RESPOND_TOOL_NAME.into(),
                input: payload,
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage {
                input_tokens,
                output_tokens,
                ..Default::default()
            },
            model: Some("crm-mock-v1".into()),
            reasoning: None,
        }
    }
}

impl LlmProvider for CrmMockProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse, Error> {
        // Record the request's system prompt for debugging.
        if let Ok(mut captured) = self.captured.lock() {
            captured.push(request.system.clone());
        }

        // Cycle through the response pool.
        if self.responses.is_empty() {
            return Err(Error::Agent(
                "CrmMockProvider: no responses configured".into(),
            ));
        }
        let idx = self.cursor.fetch_add(1, Ordering::Relaxed) % self.responses.len();
        Ok(self.responses[idx].clone())
    }

    fn model_name(&self) -> Option<&str> {
        Some("crm-mock-v1")
    }
}
