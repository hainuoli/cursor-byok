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
        transport::{TransportHandle, TransportRegistry},
    },
    Result,
};

/// How long a single poll request may wait for the next frame before ending the
/// stream. Cursor re-polls immediately, and the delivery cursor is persistent,
/// so ending early never loses or duplicates frames.
const POLL_IDLE_TIMEOUT: Duration = Duration::from_secs(25);

pub async fn poll(registry: TransportRegistry, request_id: &str) -> Result<Response<Body>> {
    let handle = registry.get_or_create(request_id).await?;
    if let Some(trace) = handle.trace() {
        trace.request(
            "run_poll_request",
            Bytes::new(),
            serde_json::json!({"request_id": request_id}),
        );
    }
    let stream = poll_stream(handle);
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
) -> impl tokio_stream::Stream<Item = std::result::Result<Bytes, std::convert::Infallible>> {
    async_stream::stream! {
        let trace = handle.trace().cloned();
        loop {
            let (start, frames, closed) = handle.poll_frames();
            if frames.is_empty() {
                if closed {
                    yield Ok::<Bytes, std::convert::Infallible>(end_frame(start as u64));
                    return;
                }
                let waited =
                    tokio::time::timeout(POLL_IDLE_TIMEOUT, handle.wait_for_output()).await;
                if waited.is_err() {
                    // Hand the stream back; Cursor re-polls and the cursor is kept.
                    return;
                }
                continue;
            }
            let mut consumed = 0usize;
            for (offset, frame) in frames.into_iter().enumerate() {
                let seqno = (start + offset) as u64;
                consumed = offset + 1;
                if is_end_stream(&frame) {
                    handle.advance_poll_cursor(consumed);
                    yield Ok::<Bytes, std::convert::Infallible>(end_frame(seqno));
                    return;
                }
                let Some(payload) = payload_of(&frame) else {
                    continue;
                };
                // Poll delivery is this transport's downlink, so record it as the
                // run's response bytes like `run_sse` does for its chunks.
                if let Some(trace) = &trace {
                    trace.response_chunk("byok_server", payload.clone());
                }
                let message = agent::BidiPollResponse {
                    seqno: seqno as i64,
                    data: hex::encode(&payload),
                    eof: None,
                };
                match connect::encode_message(&message) {
                    Ok(encoded) => yield Ok::<Bytes, std::convert::Infallible>(encoded),
                    Err(_) => return,
                }
            }
            handle.advance_poll_cursor(consumed);
        }
    }
}

fn end_frame(seqno: u64) -> Bytes {
    let message = agent::BidiPollResponse {
        seqno: seqno as i64,
        data: String::new(),
        eof: Some(true),
    };
    connect::encode_message(&message).unwrap_or_default()
}

fn is_end_stream(frame: &Bytes) -> bool {
    frame
        .first()
        .is_some_and(|flags| flags & END_STREAM_FLAG != 0)
}

fn payload_of(frame: &Bytes) -> Option<Bytes> {
    connect::decode_frames(frame)
        .ok()?
        .into_iter()
        .next()
        .map(|(_, payload)| payload)
}
