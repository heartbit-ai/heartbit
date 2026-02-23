use std::collections::HashSet;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use chrono::{DateTime, Utc};
use rdkafka::producer::{FutureProducer, FutureRecord};
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::sensor::{Sensor, SensorEvent, SensorModality};

// ---------------------------------------------------------------------------
// JMAP protocol types
// ---------------------------------------------------------------------------

/// Parsed JMAP session document from `/.well-known/jmap`.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JmapSession {
    api_url: String,
    primary_accounts: serde_json::Value,
}

impl JmapSession {
    /// Extract the primary mail account ID from the session.
    fn mail_account_id(&self) -> Result<String, Error> {
        self.primary_accounts
            .get("urn:ietf:params:jmap:mail")
            .and_then(|v| v.as_str())
            .map(String::from)
            .ok_or_else(|| Error::Sensor("no primary mail account in JMAP session".into()))
    }
}

/// A single JMAP method invocation: `["methodName", {args}, "callId"]`.
#[derive(Debug, Serialize)]
struct JmapMethodCall {
    using: Vec<&'static str>,
    #[serde(rename = "methodCalls")]
    method_calls: Vec<serde_json::Value>,
}

/// Parsed JMAP method response.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct JmapResponse {
    method_responses: Vec<serde_json::Value>,
}

/// Subset of JMAP `Email` properties we care about.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JmapEmail {
    pub id: String,
    #[serde(default)]
    pub message_id: Option<Vec<String>>,
    #[serde(default)]
    pub subject: Option<String>,
    #[serde(default)]
    pub from: Option<Vec<JmapEmailAddress>>,
    #[serde(default)]
    pub to: Option<Vec<JmapEmailAddress>>,
    #[serde(default)]
    pub received_at: Option<String>,
    #[serde(default)]
    pub text_body: Option<Vec<JmapBodyPart>>,
    #[serde(default)]
    pub body_values: Option<serde_json::Map<String, serde_json::Value>>,
    #[serde(default)]
    pub in_reply_to: Option<Vec<String>>,
    #[serde(default)]
    pub references: Option<Vec<String>>,
}

/// A JMAP email address object (`{name, email}`).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct JmapEmailAddress {
    #[serde(default)]
    pub name: Option<String>,
    pub email: String,
}

/// A JMAP body part reference.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct JmapBodyPart {
    pub part_id: String,
    /// MIME type (e.g. `text/plain`). Deserialized but not directly read;
    /// kept for diagnostics and future content-type filtering.
    #[serde(default)]
    #[allow(dead_code)]
    pub r#type: Option<String>,
}

impl JmapEmail {
    /// Extract the first sender email address, or `"unknown"` if missing.
    fn sender(&self) -> &str {
        self.from
            .as_ref()
            .and_then(|addrs| addrs.first())
            .map(|a| a.email.as_str())
            .unwrap_or("unknown")
    }

    /// Extract the first RFC 5322 Message-ID, or fall back to the JMAP id.
    fn rfc_message_id(&self) -> &str {
        self.message_id
            .as_ref()
            .and_then(|ids| ids.first())
            .map(String::as_str)
            .unwrap_or(&self.id)
    }

    /// Build the text body from `bodyValues` keyed by `textBody` part IDs.
    fn text_content(&self) -> String {
        let Some(parts) = &self.text_body else {
            return String::new();
        };
        let Some(values) = &self.body_values else {
            return String::new();
        };
        let mut buf = String::new();
        for part in parts {
            if let Some(val) = values.get(&part.part_id)
                && let Some(text) = val.get("value").and_then(|v| v.as_str())
            {
                if !buf.is_empty() {
                    buf.push('\n');
                }
                buf.push_str(text);
            }
        }
        buf
    }

    /// Collect related message IDs from In-Reply-To and References headers.
    fn related_message_ids(&self) -> Vec<String> {
        let mut seen = HashSet::new();
        let mut ids = Vec::new();
        if let Some(reply_to) = &self.in_reply_to {
            for id in reply_to {
                if seen.insert(id) {
                    ids.push(id.clone());
                }
            }
        }
        if let Some(refs) = &self.references {
            for r in refs {
                if seen.insert(r) {
                    ids.push(r.clone());
                }
            }
        }
        ids
    }
}

// ---------------------------------------------------------------------------
// Sensor implementation
// ---------------------------------------------------------------------------

/// JMAP email sensor. Polls a JMAP server for new emails and produces
/// `SensorEvent` values to the `hb.sensor.email` Kafka topic.
pub struct JmapEmailSensor {
    name: String,
    server_url: String,
    username: String,
    password: String,
    priority_senders: Vec<String>,
    poll_interval: Duration,
}

impl JmapEmailSensor {
    pub fn new(
        name: impl Into<String>,
        server_url: impl Into<String>,
        username: impl Into<String>,
        password: impl Into<String>,
        priority_senders: Vec<String>,
        poll_interval: Duration,
    ) -> Self {
        Self {
            name: name.into(),
            server_url: server_url.into(),
            username: username.into(),
            password: password.into(),
            priority_senders,
            poll_interval,
        }
    }
}

impl Sensor for JmapEmailSensor {
    fn name(&self) -> &str {
        &self.name
    }

    fn modality(&self) -> SensorModality {
        SensorModality::Text
    }

    fn kafka_topic(&self) -> &str {
        "hb.sensor.email"
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

            // Discover the JMAP session.
            let session =
                discover_session(&client, &self.server_url, &self.username, &self.password).await?;
            let account_id = session.mail_account_id()?;
            let api_url = &session.api_url;

            tracing::info!(
                sensor = %self.name,
                account_id = %account_id,
                api_url = %api_url,
                "JMAP session established"
            );

            // Track the last seen email date to avoid re-processing.
            let mut last_seen: Option<DateTime<Utc>> = None;

            loop {
                if cancel.is_cancelled() {
                    return Ok(());
                }

                match fetch_new_emails(
                    &client,
                    api_url,
                    &self.username,
                    &self.password,
                    &account_id,
                    last_seen.as_ref(),
                )
                .await
                {
                    Ok(emails) => {
                        for email in &emails {
                            // Update high-water mark.
                            if let Some(ref ts) = email.received_at
                                && let Ok(parsed) = ts.parse::<DateTime<Utc>>()
                                && last_seen.is_none_or(|ls| parsed > ls)
                            {
                                last_seen = Some(parsed);
                            }

                            let event =
                                build_sensor_event(&self.name, email, &self.priority_senders);

                            let payload = serde_json::to_vec(&event).map_err(|e| {
                                Error::Sensor(format!("failed to serialize sensor event: {e}"))
                            })?;

                            let key = format!("{}:{}", email.sender(), email.rfc_message_id());

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
                                    sensor = %self.name,
                                    message_id = %email.rfc_message_id(),
                                    error = %e,
                                    "failed to produce email event to Kafka"
                                );
                            }
                        }

                        if !emails.is_empty() {
                            tracing::debug!(
                                sensor = %self.name,
                                count = emails.len(),
                                "processed new emails"
                            );
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            sensor = %self.name,
                            error = %e,
                            "failed to fetch JMAP emails"
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

// ---------------------------------------------------------------------------
// JMAP API helpers
// ---------------------------------------------------------------------------

/// Fetch the JMAP session document from `{server_url}/.well-known/jmap`.
async fn discover_session(
    client: &reqwest::Client,
    server_url: &str,
    username: &str,
    password: &str,
) -> Result<JmapSession, Error> {
    let url = format!("{}/.well-known/jmap", server_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .basic_auth(username, Some(password))
        .send()
        .await
        .map_err(|e| Error::Sensor(format!("JMAP session discovery failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(Error::Sensor(format!(
            "JMAP session discovery returned HTTP {}",
            resp.status()
        )));
    }

    resp.json::<JmapSession>()
        .await
        .map_err(|e| Error::Sensor(format!("failed to parse JMAP session: {e}")))
}

/// Query for new emails and fetch their details in a single JMAP request.
///
/// Uses `Email/query` to find IDs, then `Email/get` (back-referenced) to
/// fetch the email properties we need.
async fn fetch_new_emails(
    client: &reqwest::Client,
    api_url: &str,
    username: &str,
    password: &str,
    account_id: &str,
    since: Option<&DateTime<Utc>>,
) -> Result<Vec<JmapEmail>, Error> {
    // Build the filter: optionally restrict to emails received after `since`.
    let filter = if let Some(ts) = since {
        serde_json::json!({
            "after": ts.to_rfc3339()
        })
    } else {
        serde_json::json!({})
    };

    let request = JmapMethodCall {
        using: vec!["urn:ietf:params:jmap:core", "urn:ietf:params:jmap:mail"],
        method_calls: vec![
            // Call 0: Email/query
            serde_json::json!([
                "Email/query",
                {
                    "accountId": account_id,
                    "filter": filter,
                    "sort": [{ "property": "receivedAt", "isAscending": true }],
                    "limit": 50
                },
                "q"
            ]),
            // Call 1: Email/get with back-reference to query results
            serde_json::json!([
                "Email/get",
                {
                    "accountId": account_id,
                    "#ids": {
                        "resultOf": "q",
                        "name": "Email/query",
                        "path": "/ids"
                    },
                    "properties": [
                        "id", "messageId", "subject", "from", "to",
                        "receivedAt", "textBody", "bodyValues",
                        "inReplyTo", "references"
                    ],
                    "fetchTextBodyValues": true
                },
                "g"
            ]),
        ],
    };

    let resp = client
        .post(api_url)
        .basic_auth(username, Some(password))
        .json(&request)
        .send()
        .await
        .map_err(|e| Error::Sensor(format!("JMAP API request failed: {e}")))?;

    if !resp.status().is_success() {
        return Err(Error::Sensor(format!(
            "JMAP API returned HTTP {}",
            resp.status()
        )));
    }

    let jmap_resp: JmapResponse = resp
        .json()
        .await
        .map_err(|e| Error::Sensor(format!("failed to parse JMAP response: {e}")))?;

    parse_email_get_response(&jmap_resp)
}

/// Extract `JmapEmail` values from the `Email/get` response.
///
/// The `Email/get` response is the second method response (index 1).
fn parse_email_get_response(resp: &JmapResponse) -> Result<Vec<JmapEmail>, Error> {
    // Find the Email/get response (should be at index 1, but search by name).
    let get_resp = resp
        .method_responses
        .iter()
        .find(|r| r.get(0).and_then(|v| v.as_str()) == Some("Email/get"))
        .ok_or_else(|| Error::Sensor("no Email/get response in JMAP reply".into()))?;

    let data = get_resp
        .get(1)
        .ok_or_else(|| Error::Sensor("malformed Email/get response".into()))?;

    let list = data
        .get("list")
        .ok_or_else(|| Error::Sensor("no 'list' in Email/get response".into()))?;

    let emails: Vec<JmapEmail> = serde_json::from_value(list.clone())
        .map_err(|e| Error::Sensor(format!("failed to parse email list: {e}")))?;

    Ok(emails)
}

/// Build a `SensorEvent` from a parsed JMAP email.
fn build_sensor_event(
    sensor_name: &str,
    email: &JmapEmail,
    priority_senders: &[String],
) -> SensorEvent {
    let content = email.text_content();
    let source_id = email.rfc_message_id().to_string();
    let sender = email.sender().to_string();

    // Build related IDs from threading headers.
    let related_ids: Vec<String> = email
        .related_message_ids()
        .into_iter()
        .map(|mid| SensorEvent::generate_id("", &mid))
        .collect();

    // Serialize to/from addresses for metadata.
    let to_addrs: Vec<String> = email
        .to
        .as_ref()
        .map(|addrs| addrs.iter().map(|a| a.email.clone()).collect())
        .unwrap_or_default();

    let is_priority = priority_senders
        .iter()
        .any(|ps| sender.eq_ignore_ascii_case(ps));

    let mut metadata = serde_json::json!({
        "subject": email.subject.as_deref().unwrap_or(""),
        "from": sender,
        "to": to_addrs,
        "message_id": email.rfc_message_id(),
    });

    if let Some(reply_to) = &email.in_reply_to {
        metadata["in_reply_to"] = serde_json::json!(reply_to);
    }
    if let Some(refs) = &email.references {
        metadata["references"] = serde_json::json!(refs);
    }
    if is_priority {
        metadata["priority_sender"] = serde_json::json!(true);
    }

    SensorEvent {
        id: SensorEvent::generate_id(&content, &source_id),
        sensor_name: sensor_name.to_string(),
        modality: SensorModality::Text,
        observed_at: email
            .received_at
            .as_ref()
            .and_then(|ts| ts.parse::<DateTime<Utc>>().ok())
            .unwrap_or_else(Utc::now),
        content,
        source_id,
        metadata: Some(metadata),
        binary_ref: None,
        related_ids,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_session_json() -> &'static str {
        r#"{
            "apiUrl": "https://mail.example.com/api/",
            "downloadUrl": "https://mail.example.com/download/{accountId}/{blobId}/{name}",
            "uploadUrl": "https://mail.example.com/upload/{accountId}/",
            "eventSourceUrl": "https://mail.example.com/events/",
            "primaryAccounts": {
                "urn:ietf:params:jmap:mail": "acc-001",
                "urn:ietf:params:jmap:contacts": "acc-001"
            },
            "capabilities": {}
        }"#
    }

    fn sample_email_get_response_json() -> &'static str {
        r#"{
            "methodResponses": [
                ["Email/query", {"ids": ["e1", "e2"]}, "q"],
                ["Email/get", {
                    "accountId": "acc-001",
                    "list": [
                        {
                            "id": "e1",
                            "messageId": ["msg-001@example.com"],
                            "subject": "Project Update",
                            "from": [{"name": "Alice", "email": "alice@example.com"}],
                            "to": [{"name": "Bob", "email": "bob@example.com"}],
                            "receivedAt": "2026-02-20T10:00:00Z",
                            "textBody": [{"partId": "1", "type": "text/plain"}],
                            "bodyValues": {
                                "1": {"value": "Here is the project update."}
                            },
                            "inReplyTo": null,
                            "references": null
                        },
                        {
                            "id": "e2",
                            "messageId": ["msg-002@example.com"],
                            "subject": "Re: Project Update",
                            "from": [{"name": "Bob", "email": "bob@example.com"}],
                            "to": [{"name": "Alice", "email": "alice@example.com"}],
                            "receivedAt": "2026-02-20T11:00:00Z",
                            "textBody": [{"partId": "1", "type": "text/plain"}],
                            "bodyValues": {
                                "1": {"value": "Thanks for the update!"}
                            },
                            "inReplyTo": ["msg-001@example.com"],
                            "references": ["msg-001@example.com"]
                        }
                    ]
                }, "g"]
            ]
        }"#
    }

    // -- Session parsing tests -----------------------------------------------

    #[test]
    fn parse_jmap_session() {
        let session: JmapSession =
            serde_json::from_str(sample_session_json()).expect("session parse");
        assert_eq!(session.api_url, "https://mail.example.com/api/");
    }

    #[test]
    fn session_mail_account_id() {
        let session: JmapSession =
            serde_json::from_str(sample_session_json()).expect("session parse");
        let account_id = session.mail_account_id().expect("mail account");
        assert_eq!(account_id, "acc-001");
    }

    #[test]
    fn session_mail_account_id_missing() {
        let json = r#"{
            "apiUrl": "https://example.com/api/",
            "primaryAccounts": {}
        }"#;
        let session: JmapSession = serde_json::from_str(json).expect("parse");
        let result = session.mail_account_id();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("no primary mail account")
        );
    }

    // -- Email response parsing tests ----------------------------------------

    #[test]
    fn parse_email_get_response_success() {
        let resp: JmapResponse =
            serde_json::from_str(sample_email_get_response_json()).expect("response parse");
        let emails = parse_email_get_response(&resp).expect("email parse");
        assert_eq!(emails.len(), 2);
    }

    #[test]
    fn parsed_email_fields() {
        let resp: JmapResponse =
            serde_json::from_str(sample_email_get_response_json()).expect("response parse");
        let emails = parse_email_get_response(&resp).expect("email parse");

        let e1 = &emails[0];
        assert_eq!(e1.id, "e1");
        assert_eq!(e1.subject.as_deref(), Some("Project Update"));
        assert_eq!(e1.sender(), "alice@example.com");
        assert_eq!(e1.rfc_message_id(), "msg-001@example.com");
        assert_eq!(e1.received_at.as_deref(), Some("2026-02-20T10:00:00Z"));
    }

    #[test]
    fn parsed_email_threading() {
        let resp: JmapResponse =
            serde_json::from_str(sample_email_get_response_json()).expect("response parse");
        let emails = parse_email_get_response(&resp).expect("email parse");

        let e2 = &emails[1];
        assert_eq!(
            e2.in_reply_to.as_deref(),
            Some(&["msg-001@example.com".to_string()][..])
        );
        assert_eq!(
            e2.references.as_deref(),
            Some(&["msg-001@example.com".to_string()][..])
        );

        let related = e2.related_message_ids();
        assert_eq!(related.len(), 1);
        assert_eq!(related[0], "msg-001@example.com");
    }

    #[test]
    fn parse_email_get_response_no_email_get() {
        let json = r#"{"methodResponses": [["Email/query", {"ids": []}, "q"]]}"#;
        let resp: JmapResponse = serde_json::from_str(json).expect("parse");
        let result = parse_email_get_response(&resp);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("no Email/get response")
        );
    }

    #[test]
    fn parse_email_get_response_empty_list() {
        let json = r#"{
            "methodResponses": [
                ["Email/get", {"accountId": "a", "list": []}, "g"]
            ]
        }"#;
        let resp: JmapResponse = serde_json::from_str(json).expect("parse");
        let emails = parse_email_get_response(&resp).expect("should succeed");
        assert!(emails.is_empty());
    }

    // -- JmapEmail helper tests ----------------------------------------------

    #[test]
    fn email_sender_present() {
        let email = make_test_email();
        assert_eq!(email.sender(), "alice@example.com");
    }

    #[test]
    fn email_sender_missing() {
        let mut email = make_test_email();
        email.from = None;
        assert_eq!(email.sender(), "unknown");
    }

    #[test]
    fn email_rfc_message_id_present() {
        let email = make_test_email();
        assert_eq!(email.rfc_message_id(), "msg-123@example.com");
    }

    #[test]
    fn email_rfc_message_id_fallback() {
        let mut email = make_test_email();
        email.message_id = None;
        assert_eq!(email.rfc_message_id(), "jmap-id-1");
    }

    #[test]
    fn email_text_content() {
        let email = make_test_email();
        assert_eq!(email.text_content(), "Hello, world!");
    }

    #[test]
    fn email_text_content_no_body() {
        let mut email = make_test_email();
        email.text_body = None;
        assert!(email.text_content().is_empty());
    }

    #[test]
    fn email_text_content_no_values() {
        let mut email = make_test_email();
        email.body_values = None;
        assert!(email.text_content().is_empty());
    }

    #[test]
    fn email_text_content_multiple_parts() {
        let mut email = make_test_email();
        email.text_body = Some(vec![
            JmapBodyPart {
                part_id: "1".into(),
                r#type: Some("text/plain".into()),
            },
            JmapBodyPart {
                part_id: "2".into(),
                r#type: Some("text/plain".into()),
            },
        ]);
        let mut values = serde_json::Map::new();
        values.insert("1".into(), serde_json::json!({"value": "Part one."}));
        values.insert("2".into(), serde_json::json!({"value": "Part two."}));
        email.body_values = Some(values);

        let content = email.text_content();
        assert!(content.contains("Part one."));
        assert!(content.contains("Part two."));
        assert!(content.contains('\n'));
    }

    #[test]
    fn email_related_ids_empty() {
        let email = make_test_email();
        assert!(email.related_message_ids().is_empty());
    }

    #[test]
    fn email_related_ids_from_in_reply_to() {
        let mut email = make_test_email();
        email.in_reply_to = Some(vec!["parent@example.com".into()]);
        let ids = email.related_message_ids();
        assert_eq!(ids, vec!["parent@example.com"]);
    }

    #[test]
    fn email_related_ids_dedup_references() {
        let mut email = make_test_email();
        email.in_reply_to = Some(vec!["parent@example.com".into()]);
        email.references = Some(vec![
            "root@example.com".into(),
            "parent@example.com".into(), // duplicate of in_reply_to
        ]);
        let ids = email.related_message_ids();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"parent@example.com".to_string()));
        assert!(ids.contains(&"root@example.com".to_string()));
    }

    // -- Sensor property tests -----------------------------------------------

    #[test]
    fn sensor_properties() {
        let sensor = JmapEmailSensor::new(
            "work_email",
            "https://mail.example.com",
            "user@example.com",
            "secret",
            vec!["ceo@example.com".into()],
            Duration::from_secs(60),
        );
        assert_eq!(sensor.name(), "work_email");
        assert_eq!(sensor.modality(), SensorModality::Text);
        assert_eq!(sensor.kafka_topic(), "hb.sensor.email");
    }

    #[test]
    fn sensor_stores_priority_senders() {
        let sensor = JmapEmailSensor::new(
            "inbox",
            "https://mail.example.com",
            "user@example.com",
            "secret",
            vec!["vip@example.com".into(), "boss@example.com".into()],
            Duration::from_secs(120),
        );
        assert_eq!(sensor.priority_senders.len(), 2);
        assert_eq!(sensor.priority_senders[0], "vip@example.com");
    }

    // -- SensorEvent building tests ------------------------------------------

    #[test]
    fn build_event_basic() {
        let email = make_test_email();
        let event = build_sensor_event("work_email", &email, &[]);

        assert_eq!(event.sensor_name, "work_email");
        assert_eq!(event.modality, SensorModality::Text);
        assert_eq!(event.content, "Hello, world!");
        assert_eq!(event.source_id, "msg-123@example.com");
        assert!(event.binary_ref.is_none());
        assert!(event.related_ids.is_empty());
    }

    #[test]
    fn build_event_metadata_fields() {
        let email = make_test_email();
        let event = build_sensor_event("test", &email, &[]);

        let meta = event.metadata.expect("metadata should be present");
        assert_eq!(meta["subject"], "Test Subject");
        assert_eq!(meta["from"], "alice@example.com");
        assert_eq!(meta["message_id"], "msg-123@example.com");
        let to = meta["to"].as_array().expect("to should be array");
        assert_eq!(to[0], "bob@example.com");
    }

    #[test]
    fn build_event_with_threading() {
        let mut email = make_test_email();
        email.in_reply_to = Some(vec!["parent-msg@example.com".into()]);
        email.references = Some(vec![
            "root-msg@example.com".into(),
            "parent-msg@example.com".into(),
        ]);

        let event = build_sensor_event("test", &email, &[]);

        // related_ids should contain hashed IDs for the two unique references.
        assert_eq!(event.related_ids.len(), 2);

        let meta = event.metadata.expect("metadata");
        assert!(meta.get("in_reply_to").is_some());
        assert!(meta.get("references").is_some());
    }

    #[test]
    fn build_event_priority_sender() {
        let email = make_test_email();
        let event = build_sensor_event("test", &email, &["alice@example.com".into()]);

        let meta = event.metadata.expect("metadata");
        assert_eq!(meta["priority_sender"], true);
    }

    #[test]
    fn build_event_priority_sender_case_insensitive() {
        let email = make_test_email();
        let event = build_sensor_event("test", &email, &["ALICE@EXAMPLE.COM".into()]);

        let meta = event.metadata.expect("metadata");
        assert_eq!(meta["priority_sender"], true);
    }

    #[test]
    fn build_event_non_priority_sender() {
        let email = make_test_email();
        let event = build_sensor_event("test", &email, &["vip@other.com".into()]);

        let meta = event.metadata.expect("metadata");
        assert!(meta.get("priority_sender").is_none());
    }

    #[test]
    fn build_event_deterministic_id() {
        let email = make_test_email();
        let event1 = build_sensor_event("s", &email, &[]);
        let event2 = build_sensor_event("s", &email, &[]);
        assert_eq!(event1.id, event2.id);
    }

    #[test]
    fn build_event_observed_at_from_received() {
        let email = make_test_email();
        let event = build_sensor_event("s", &email, &[]);
        // The test email has receivedAt = 2026-02-20T10:00:00Z
        assert_eq!(
            event.observed_at,
            "2026-02-20T10:00:00Z"
                .parse::<DateTime<Utc>>()
                .expect("parse ts")
        );
    }

    #[test]
    fn build_event_observed_at_fallback_to_now() {
        let mut email = make_test_email();
        email.received_at = None;
        let before = Utc::now();
        let event = build_sensor_event("s", &email, &[]);
        let after = Utc::now();
        assert!(event.observed_at >= before);
        assert!(event.observed_at <= after);
    }

    // -- Kafka key format test -----------------------------------------------

    #[test]
    fn kafka_key_format() {
        let email = make_test_email();
        let key = format!("{}:{}", email.sender(), email.rfc_message_id());
        assert_eq!(key, "alice@example.com:msg-123@example.com");
    }

    // -- Test helper ---------------------------------------------------------

    fn make_test_email() -> JmapEmail {
        let mut body_values = serde_json::Map::new();
        body_values.insert("1".into(), serde_json::json!({"value": "Hello, world!"}));

        JmapEmail {
            id: "jmap-id-1".into(),
            message_id: Some(vec!["msg-123@example.com".into()]),
            subject: Some("Test Subject".into()),
            from: Some(vec![JmapEmailAddress {
                name: Some("Alice".into()),
                email: "alice@example.com".into(),
            }]),
            to: Some(vec![JmapEmailAddress {
                name: Some("Bob".into()),
                email: "bob@example.com".into(),
            }]),
            received_at: Some("2026-02-20T10:00:00Z".into()),
            text_body: Some(vec![JmapBodyPart {
                part_id: "1".into(),
                r#type: Some("text/plain".into()),
            }]),
            body_values: Some(body_values),
            in_reply_to: None,
            references: None,
        }
    }
}
