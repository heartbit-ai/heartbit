//! CRM domain types for PulsarData CRM.
//!
//! These are plain Rust structs — not tied to the LLM layer. They get
//! serialized into agent prompts so the LLM can reason over CRM data.

use serde::{Deserialize, Serialize};

/// A CRM contact (person).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Contact {
    pub id: String,
    pub name: String,
    pub email: String,
    pub company_id: Option<String>,
    pub title: Option<String>,
    /// Populated by the enrichment workflow.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub linkedin_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub phone: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enrichment_notes: Option<String>,
}

/// A CRM company (account).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Company {
    pub id: String,
    pub name: String,
    pub industry: Option<String>,
    pub employee_count: Option<u32>,
}

/// Pipeline stage for a deal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DealStage {
    Discovery,
    Qualified,
    Proposal,
    Negotiation,
    ClosedWon,
    ClosedLost,
}

impl std::fmt::Display for DealStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Discovery => write!(f, "discovery"),
            Self::Qualified => write!(f, "qualified"),
            Self::Proposal => write!(f, "proposal"),
            Self::Negotiation => write!(f, "negotiation"),
            Self::ClosedWon => write!(f, "closed_won"),
            Self::ClosedLost => write!(f, "closed_lost"),
        }
    }
}

/// A CRM deal (opportunity).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deal {
    pub id: String,
    pub name: String,
    pub company_id: String,
    pub value: f64,
    pub stage: DealStage,
    pub contact_id: String,
}

/// A support ticket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticket {
    pub id: String,
    pub subject: String,
    pub priority: String,
    pub contact_id: String,
    pub status: String,
}

/// A scored lead with structured data produced by the LLM via
/// [`AgentCall::schema`](heartbit_core::flow::agent::AgentCall::schema).
#[allow(dead_code)] // Used for demonstration purposes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredLead {
    pub contact_id: String,
    pub score: u8,
    pub tier: String,
    pub rationale: String,
}

/// Email campaign result, one per contact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailDraft {
    pub contact_id: String,
    pub subject: String,
    pub body_preview: String,
    pub status: String,
}

/// Lead scoring schema — used with `AgentCall::schema::<LeadScore>()`
/// to force the LLM to produce validated structured output.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct LeadScore {
    pub score: u8,
    pub tier: LeadTier,
    pub rationale: String,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum LeadTier {
    Hot,
    Warm,
    Cold,
}

impl heartbit_core::StructuredSchema for LeadScore {
    fn json_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["score", "tier", "rationale"],
            "properties": {
                "score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Lead score from 0-100"
                },
                "tier": {
                    "type": "string",
                    "enum": ["hot", "warm", "cold"],
                    "description": "Lead classification tier"
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief explanation for the score"
                }
            }
        })
    }
}

/// Deal analysis schema — used for pipeline processing with structured output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DealAnalysis {
    pub deal_id: String,
    pub health: String,
    pub next_action: String,
    pub risk_factors: Vec<String>,
    pub probability_to_close: u8,
}

impl heartbit_core::StructuredSchema for DealAnalysis {
    fn json_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["deal_id", "health", "next_action", "risk_factors", "probability_to_close"],
            "properties": {
                "deal_id": { "type": "string" },
                "health": {
                    "type": "string",
                    "enum": ["green", "yellow", "red"],
                    "description": "Deal health status"
                },
                "next_action": {
                    "type": "string",
                    "description": "Recommended next step"
                },
                "risk_factors": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Identified risk factors"
                },
                "probability_to_close": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Estimated probability to close (percentage)"
                }
            }
        })
    }
}

/// Ticket triage schema — used for heterogeneous parallel ticket classification.
#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct TicketTriage {
    pub ticket_id: String,
    pub category: String,
    pub severity: u8,
    pub suggested_assignee: String,
    pub sla_hours: u32,
}

impl heartbit_core::StructuredSchema for TicketTriage {
    fn json_schema() -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["ticket_id", "category", "severity", "suggested_assignee", "sla_hours"],
            "properties": {
                "ticket_id": { "type": "string" },
                "category": {
                    "type": "string",
                    "enum": ["bug", "feature_request", "account", "performance", "security"],
                    "description": "Ticket category"
                },
                "severity": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Severity level (1=critical, 5=cosmetic)"
                },
                "suggested_assignee": {
                    "type": "string",
                    "description": "Recommended team or person"
                },
                "sla_hours": {
                    "type": "integer",
                    "description": "SLA target in hours"
                }
            }
        })
    }
}
