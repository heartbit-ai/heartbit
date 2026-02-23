use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use chrono::Utc;
use rdkafka::producer::{FutureProducer, FutureRecord};
use tokio_util::sync::CancellationToken;

use crate::Error;
use crate::sensor::{Sensor, SensorEvent, SensorModality};

/// RSS/Atom feed sensor. Polls configured feeds at a regular interval and
/// produces `SensorEvent` values to the `hb.sensor.rss` Kafka topic.
pub struct RssSensor {
    name: String,
    feeds: Vec<String>,
    poll_interval: Duration,
}

impl RssSensor {
    pub fn new(name: impl Into<String>, feeds: Vec<String>, poll_interval: Duration) -> Self {
        Self {
            name: name.into(),
            feeds,
            poll_interval,
        }
    }
}

impl Sensor for RssSensor {
    fn name(&self) -> &str {
        &self.name
    }

    fn modality(&self) -> SensorModality {
        SensorModality::Text
    }

    fn kafka_topic(&self) -> &str {
        "hb.sensor.rss"
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
                for feed_url in &self.feeds {
                    if cancel.is_cancelled() {
                        return Ok(());
                    }

                    match fetch_and_parse_feed(&client, feed_url).await {
                        Ok(items) => {
                            for item in items {
                                let event = SensorEvent {
                                    id: SensorEvent::generate_id(&item.content, &item.link),
                                    sensor_name: self.name.clone(),
                                    modality: SensorModality::Text,
                                    observed_at: Utc::now(),
                                    content: item.content,
                                    source_id: item.link.clone(),
                                    metadata: Some(serde_json::json!({
                                        "title": item.title,
                                        "feed_url": feed_url,
                                    })),
                                    binary_ref: None,
                                    related_ids: vec![],
                                };

                                let payload = serde_json::to_vec(&event).map_err(|e| {
                                    Error::Sensor(format!("failed to serialize sensor event: {e}"))
                                })?;

                                let key = format!(
                                    "{}:{}",
                                    SensorEvent::generate_id(feed_url, ""),
                                    event.id
                                );

                                // Fire-and-forget produce (log errors, don't halt)
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
                                        feed = %feed_url,
                                        error = %e,
                                        "failed to produce RSS event to Kafka"
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                feed = %feed_url,
                                error = %e,
                                "failed to fetch RSS feed"
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

/// A parsed RSS/Atom feed item.
struct FeedItem {
    title: String,
    link: String,
    content: String,
}

/// Fetch and parse an RSS or Atom feed, returning extracted items.
async fn fetch_and_parse_feed(client: &reqwest::Client, url: &str) -> Result<Vec<FeedItem>, Error> {
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| Error::Sensor(format!("HTTP request failed for {url}: {e}")))?;

    let body = response
        .text()
        .await
        .map_err(|e| Error::Sensor(format!("failed to read response body from {url}: {e}")))?;

    parse_feed_xml(&body)
}

fn apply_text_to_field(
    current_tag: &str,
    text: &str,
    title: &mut String,
    link: &mut String,
    description: &mut String,
) {
    match current_tag {
        "title" => *title = text.to_string(),
        "link" => {
            if link.is_empty() {
                *link = text.to_string();
            }
        }
        "description" | "summary" | "content" | "content:encoded" => {
            *description = text.to_string();
        }
        _ => {}
    }
}

/// Parse RSS 2.0 or Atom XML into `FeedItem` values.
///
/// Supports both `<item>` (RSS 2.0) and `<entry>` (Atom) elements.
fn parse_feed_xml(xml: &str) -> Result<Vec<FeedItem>, Error> {
    use quick_xml::Reader;
    use quick_xml::events::Event;

    let mut reader = Reader::from_str(xml);
    let mut items = Vec::new();

    let mut in_item = false;
    let mut current_tag = String::new();
    let mut title = String::new();
    let mut link = String::new();
    let mut description = String::new();

    loop {
        match reader.read_event() {
            Ok(Event::Empty(ref e)) => {
                let tag_name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                // Atom feeds: <link href="..." /> (self-closing)
                if tag_name == "link" && in_item {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"href" {
                            link = String::from_utf8_lossy(&attr.value).to_string();
                        }
                    }
                }
            }
            Ok(Event::Start(ref e)) => {
                let tag_name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                match tag_name.as_str() {
                    "item" | "entry" => {
                        in_item = true;
                        title.clear();
                        link.clear();
                        description.clear();
                    }
                    _ if in_item => {
                        current_tag = tag_name;
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(ref e)) if in_item => {
                let text = e.unescape().unwrap_or_default().to_string();
                apply_text_to_field(&current_tag, &text, &mut title, &mut link, &mut description);
            }
            Ok(Event::CData(ref e)) if in_item => {
                let text = String::from_utf8_lossy(e.as_ref()).to_string();
                apply_text_to_field(&current_tag, &text, &mut title, &mut link, &mut description);
            }
            Ok(Event::End(ref e)) => {
                let tag_name = String::from_utf8_lossy(e.name().as_ref()).to_string();
                if tag_name == "item" || tag_name == "entry" {
                    in_item = false;
                    // Build content: prefer description, fall back to title
                    let content = if description.is_empty() {
                        title.clone()
                    } else {
                        format!("{title}\n\n{description}")
                    };
                    if !link.is_empty() {
                        items.push(FeedItem {
                            title: title.clone(),
                            link: link.clone(),
                            content,
                        });
                    }
                }
                current_tag.clear();
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(Error::Sensor(format!("XML parse error: {e}")));
            }
            _ => {}
        }
    }

    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rss2_feed() {
        let xml = r#"<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <title>Test Feed</title>
    <item>
      <title>Article One</title>
      <link>https://example.com/1</link>
      <description>First article content.</description>
    </item>
    <item>
      <title>Article Two</title>
      <link>https://example.com/2</link>
      <description>Second article content.</description>
    </item>
  </channel>
</rss>"#;

        let items = parse_feed_xml(xml).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].title, "Article One");
        assert_eq!(items[0].link, "https://example.com/1");
        assert!(items[0].content.contains("First article content"));
        assert_eq!(items[1].title, "Article Two");
    }

    #[test]
    fn parse_atom_feed() {
        let xml = r#"<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Atom Feed</title>
  <entry>
    <title>Atom Entry</title>
    <link href="https://example.com/atom/1"/>
    <summary>Atom summary here.</summary>
  </entry>
</feed>"#;

        let items = parse_feed_xml(xml).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0].title, "Atom Entry");
        assert!(items[0].content.contains("Atom summary here"));
    }

    #[test]
    fn parse_empty_feed() {
        let xml = r#"<?xml version="1.0"?>
<rss version="2.0">
  <channel><title>Empty</title></channel>
</rss>"#;
        let items = parse_feed_xml(xml).unwrap();
        assert!(items.is_empty());
    }

    #[test]
    fn parse_item_without_link_skipped() {
        let xml = r#"<?xml version="1.0"?>
<rss version="2.0">
  <channel>
    <item>
      <title>No Link</title>
      <description>Content without link.</description>
    </item>
  </channel>
</rss>"#;
        let items = parse_feed_xml(xml).unwrap();
        assert!(items.is_empty(), "items without link should be skipped");
    }

    #[test]
    fn parse_invalid_xml() {
        let xml = "this is not valid xml <<<<";
        let result = parse_feed_xml(xml);
        assert!(result.is_err());
    }

    #[test]
    fn rss_sensor_properties() {
        let sensor = RssSensor::new(
            "tech_news",
            vec!["https://example.com/feed".into()],
            Duration::from_secs(900),
        );
        assert_eq!(sensor.name(), "tech_news");
        assert_eq!(sensor.modality(), SensorModality::Text);
        assert_eq!(sensor.kafka_topic(), "hb.sensor.rss");
    }

    #[test]
    fn parse_feed_with_content_encoded() {
        let xml = r#"<?xml version="1.0"?>
<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
  <channel>
    <item>
      <title>Rich Content</title>
      <link>https://example.com/rich</link>
      <content:encoded><![CDATA[<p>Rich HTML content here.</p>]]></content:encoded>
    </item>
  </channel>
</rss>"#;

        let items = parse_feed_xml(xml).unwrap();
        assert_eq!(items.len(), 1);
        assert!(items[0].content.contains("Rich HTML content"));
    }
}
