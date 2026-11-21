//! Serves Cursor `RunPoll` clients by replaying transport output as framed
//! `BidiPollResponse` messages with a strictly increasing `seqno`.
//!
//! Cursor's `http1_poll` transport sends `BidiAppend` upstream and pulls
//! downstream frames with repeated `RunPoll` calls. A single poll stream may
//! only ever receive `seqno = 0, 1, 2, ...` without gaps or duplicates, so the
//! delivery cursor lives on the transport instead of on the HTTP request.
use axum::{
    body::Body,
    http::{header, HeaderValue, Response, StatusCode},
};
use bytes::Bytes;
use std::time::Duration;

use crate::{
    cursor::{
        protocol::{
            connect::{self, END_STREAM_FLAG},
            proto::agent::v1 as agent,
        },
        transport::{PollLease, TransportHandle, TransportRegistry},
    },
    Result,
};

/// Default how long one poll request waits for the next frame before ending the
/// stream, letting the client re-poll. Cursor's connect-es parser requires a
/// proper Connect end-stream envelope, so the timeout ends the stream with
/// `connect::encode_end_stream()` rather than a bare EOF.
const DEFAULT_POLL_IDLE_TIMEOUT: Duration = Duration::from_secs(25);

/// Overridable through `BYOK_POLL_IDLE_MS`; integration tests shorten it to
/// exercise a second poll without waiting 25 seconds.
fn poll_idle_timeout() -> Duration {
    std::env::var("BYOK_POLL_IDLE_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_POLL_IDLE_TIMEOUT)
}

pub async fn poll(registry: TransportRegistry, request_id: &str) -> Result<Response<Body>> {
    // A finished or unknown run must not silently create a fresh empty
    // transport: the poll would wait forever for frames that can never come.
    let handle = registry
        .local(request_id)
        .await
        .ok_or_else(|| crate::Error::RunNotFound(request_id.into()))?;
    let Some(lease) = handle.try_begin_poll() else {
        return Err(crate::Error::Protocol(
            "a RunPoll stream is already active for this request".into(),
        ));
    };
    if let Some(trace) = handle.trace() {
        trace.response_started(StatusCode::OK.as_u16());
        trace.request(
            "run_poll_request",
            Bytes::new(),
            serde_json::json!({"request_id": request_id}),
        );
    }
    let stream = poll_stream(handle, lease);
    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/connect+proto"),
    );
    response
        .headers_mut()
        .insert("connect-protocol-version", HeaderValue::from_static("1"));
    Ok(response)
}

fn poll_stream(
    handle: TransportHandle,
    lease: PollLease,
) -> impl tokio_stream::Stream<Item = std::result::Result<Bytes, std::convert::Infallible>> {
    async_stream::stream! {
        // Held until the response body is dropped, so a second poll is rejected
        // while this one owns the shared delivery cursor.
        let _lease = lease;
        let trace = handle.trace().cloned();
        let idle_timeout = poll_idle_timeout();
        loop {
            let (start, frames, closed) = handle.poll_frames();
            if frames.is_empty() {
                if closed {
                    if let Some(trace) = &trace {
                        trace.finish(None);
                    }
                    yield Ok::<Bytes, std::convert::Infallible>(end_frame(start as u64));
                    return;
                }
                let waited = tokio::time::timeout(idle_timeout, handle.wait_for_output()).await;
                if waited.is_err() {
                    // Hand the stream back with a valid Connect end-stream
                    // envelope; Cursor re-polls and the cursor is kept.
                    yield Ok::<Bytes, std::convert::Infallible>(connect::encode_end_stream());
                    return;
                }
                continue;
            }
            // Advance before yielding: a response dropped mid-stream then loses
            // frames (the client restarts the run with a new request id) instead
            // of replaying them, which would duplicate a seqno.
            handle.advance_poll_cursor(frames.len());
            for (offset, frame) in frames.into_iter().enumerate() {
                let seqno = (start + offset) as u64;
                if is_end_stream(&frame) {
                    match end_stream_error(&frame) {
                        Some(error) => {
                            // Preserve the provider/run error: rewriting it into
                            // a clean `eof` makes Cursor treat the turn as a
                            // dropped connection and retry it.
                            if let Some(trace) = &trace {
                                trace.finish(Some(&error));
                            }
                            yield Ok::<Bytes, std::convert::Infallible>(frame);
                        }
                        None => {
                            if let Some(trace) = &trace {
                                trace.finish(None);
                            }
                            yield Ok::<Bytes, std::convert::Infallible>(end_frame(seqno));
                        }
                    }
                    return;
                }
                let Some(payload) = payload_of(&frame) else {
                    tracing::error!(
                        request_id = handle.request_id(),
                        seqno,
                        "RunPoll received a frame without a Connect payload"
                    );
                    if let Some(trace) = &trace {
                        trace.finish(Some("invalid poll frame"));
                    }
                    yield Ok::<Bytes, std::convert::Infallible>(connect::encode_end_stream());
                    return;
                };
                let message = agent::BidiPollResponse {
                    seqno: seqno as i64,
                    data: hex::encode(&payload),
                    eof: None,
                };
                let encoded = match connect::encode_message(&message) {
                    Ok(encoded) => encoded,
                    Err(error) => {
                        tracing::error!(
                            request_id = handle.request_id(),
                            seqno,
                            %error,
                            "failed to encode BidiPollResponse"
                        );
                        if let Some(trace) = &trace {
                            trace.finish(Some("failed to encode poll response"));
                        }
                        yield Ok::<Bytes, std::convert::Infallible>(connect::encode_end_stream());
                        return;
                    }
                };
                // Record the same framing RunSSE records so both downlinks are
                // comparable in run traces.
                if let Some(trace) = &trace {
                    trace.response_chunk("byok_server", encoded.clone());
                }
                yield Ok::<Bytes, std::convert::Infallible>(encoded);
            }
        }
    }
}

fn end_frame(seqno: u64) -> Bytes {
    let message = agent::BidiPollResponse {
        seqno: seqno as i64,
        data: String::new(),
        eof: Some(true),
    };
    connect::encode_message(&message).unwrap_or_else(|error| {
        tracing::error!(%error, "failed to encode poll eof frame");
        connect::encode_end_stream()
    })
}

fn is_end_stream(frame: &Bytes) -> bool {
    frame
        .first()
        .is_some_and(|flags| flags & END_STREAM_FLAG != 0)
}

fn payload_of(frame: &Bytes) -> Option<Bytes> {
    let mut frames = connect::decode_frames(frame).ok()?;
    if frames.len() != 1 {
        return None;
    }
    frames.pop().map(|(_, payload)| payload)
}

/// Extracts the error message from a Connect end-stream frame, mirroring the
/// RunSSE path so poll clients see provider/run failures instead of a clean eof.
fn end_stream_error(frame: &Bytes) -> Option<String> {
    connect::decode_frames(frame)
        .ok()?
        .into_iter()
        .find_map(|(flags, payload)| {
            if flags & END_STREAM_FLAG == 0 {
                return None;
            }
            let value = serde_json::from_slice::<serde_json::Value>(&payload).ok()?;
            let error = value.get("error")?;
            let code = error.get("code").and_then(serde_json::Value::as_str);
            let message = error
                .get("message")
                .and_then(serde_json::Value::as_str)
                .filter(|message| !message.is_empty());
            Some(match (code, message) {
                (Some(code), Some(message)) => format!("{code}: {message}"),
                (Some(code), None) => code.to_string(),
                (None, Some(message)) => message.to_string(),
                (None, None) => error.to_string(),
            })
        })
}
