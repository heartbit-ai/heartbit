use crate::error::Error;

const WHISPER_URL: &str = "https://api.openai.com/v1/audio/transcriptions";

/// Transcribe audio bytes using OpenAI's Whisper API.
///
/// The audio is sent as a multipart form upload. Supports `.ogg`, `.mp3`, `.wav`,
/// `.m4a`, and other formats accepted by the Whisper API.
pub async fn transcribe_audio(
    client: &reqwest::Client,
    audio_bytes: &[u8],
    api_key: &str,
) -> Result<String, Error> {
    let file_part = reqwest::multipart::Part::bytes(audio_bytes.to_vec())
        .file_name("voice.ogg")
        .mime_str("audio/ogg")
        .map_err(|e| Error::Telegram(format!("multipart error: {e}")))?;

    let form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("model", "whisper-1");

    let response = client
        .post(WHISPER_URL)
        .header("Authorization", format!("Bearer {api_key}"))
        .multipart(form)
        .send()
        .await
        .map_err(|e| Error::Telegram(format!("whisper request failed: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "<body read error>".into());
        return Err(Error::Telegram(format!(
            "whisper API error (HTTP {status}): {body}"
        )));
    }

    #[derive(serde::Deserialize)]
    struct WhisperResponse {
        text: String,
    }

    let result: WhisperResponse = response
        .json()
        .await
        .map_err(|e| Error::Telegram(format!("whisper response parse error: {e}")))?;

    Ok(result.text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn transcribe_audio_invalid_endpoint_returns_error() {
        // With an invalid API key, the request should return an auth error
        let client = reqwest::Client::new();
        let result = transcribe_audio(&client, b"fake audio data", "invalid-key").await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("whisper") || err.contains("401") || err.contains("error"),
            "expected whisper/auth error, got: {err}"
        );
    }
}
