//! Verifies the `http1_poll` downlink that Cursor 3.24 selects when the client
//! cannot use HTTP/2: `RunPoll` must replay transport output with a strictly
//! increasing `seqno`, work whether the client opens the poll before or after
//! the append, and pass provider errors through instead of masking them as a
//! clean eof.
#[path = "support/fake_provider.rs"]
mod fake_provider;
#[path = "support/fixtures.rs"]
mod fixtures;

use std::sync::Arc;

use axum::{
    body::{Body, Bytes},
    http::{header, Request, StatusCode},
};
use cursor_server::{
    api::cursor,
    cursor::prompting::{PromptAssets, PromptCompiler},
    cursor::protocol::{
        connect,
        proto::{agent::v1 as pb, aiserver::v1 as ai},
    },
    cursor::transport::TransportRegistry,
    network::NetworkClients,
};
use futures_util::StreamExt;
use prost::Message;
use tower::ServiceExt;

const REQUEST_ID: &str = "http1-poll-request";

fn router(
    store: cursor_server::store::Store,
    provider: fake_provider::FakeProvider,
) -> axum::Router {
    let assets =
        PromptAssets::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("prompt/cursor"))
            .unwrap();
    let clients = NetworkClients::new(store.clone());
    let registry = TransportRegistry::new(store, Arc::new(provider), PromptCompiler::new(assets));
    cursor::router(registry, clients).unwrap()
}

/// What one poll stream delivered.
#[derive(Default)]
struct PollOutcome {
    text: String,
    /// Terminal error carried by a Connect end-stream frame, if any.
    error: Option<String>,
}

/// Reads one poll stream, asserting `seqno` continuity from `expected`, and
/// answering the KV requests the run emits.
async fn drain_poll(
    router: &axum::Router,
    body: Bytes,
    append_seqno: &mut i64,
    expected: &mut i64,
) -> PollOutcome {
    let response = router
        .clone()
        .oneshot(
            Request::post("/agent.v1.AgentService/RunPoll")
                .header(header::CONTENT_TYPE, "application/connect+proto")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);

    let mut outcome = PollOutcome::default();
    let mut stream = response.into_body().into_data_stream();
    loop {
        let chunk = tokio::time::timeout(std::time::Duration::from_secs(20), stream.next())
            .await
            .expect("RunPoll response timed out")
            .expect("RunPoll ended without a terminal frame")
            .unwrap();
        for (flags, payload) in connect::decode_frames(&chunk).unwrap() {
            if flags & connect::END_STREAM_FLAG != 0 {
                // Clean end-of-stream is the idle hand-back; an error end-stream
                // carries the run failure.
                outcome.error = serde_json::from_slice::<serde_json::Value>(&payload)
                    .ok()
                    .and_then(|value| value.get("error").cloned())
                    .and_then(|error| error.get("message").cloned())
                    .and_then(|message| message.as_str().map(str::to_owned));
                return outcome;
            }
            let message = pb::BidiPollResponse::decode(payload).unwrap();
            if message.eof.unwrap_or(false) {
                assert_eq!(
                    message.seqno, *expected,
                    "eof frame must carry the next seqno"
                );
                return outcome;
            }
            assert_eq!(
                message.seqno, *expected,
                "poll frames must be delivered with contiguous seqnos"
            );
            *expected += 1;
            let payload = hex::decode(&message.data).unwrap();
            let server = pb::AgentServerMessage::decode(payload.as_slice()).unwrap();
            match server.message {
                Some(pb::agent_server_message::Message::KvServerMessage(kv)) => {
                    let reply = match kv.message {
                        Some(pb::kv_server_message::Message::GetBlobArgs(_)) => {
                            pb::kv_client_message::Message::GetBlobResult(pb::GetBlobResult {
                                blob_data: None,
                                error: Some(pb::Error {
                                    message: "absent".into(),
                                }),
                            })
                        }
                        Some(pb::kv_server_message::Message::SetBlobArgs(_)) => {
                            pb::kv_client_message::Message::SetBlobResult(pb::SetBlobResult {
                                error: None,
                            })
                        }
                        None => continue,
                    };
                    append(
                        router,
                        *append_seqno,
                        pb::AgentClientMessage {
                            message: Some(pb::agent_client_message::Message::KvClientMessage(
                                pb::KvClientMessage {
                                    id: kv.id,
                                    message: Some(reply),
                                },
                            )),
                        },
                    )
                    .await;
                    *append_seqno += 1;
                }
                Some(pb::agent_server_message::Message::InteractionUpdate(update)) => {
                    if let Some(pb::interaction_update::Message::TextDelta(delta)) = update.message
                    {
                        outcome.text.push_str(&delta.text);
                    }
                }
                _ => {}
            }
        }
    }
}

async fn append(router: &axum::Router, seqno: i64, message: pb::AgentClientMessage) {
    let response = router
        .clone()
        .oneshot(
            Request::post("/aiserver.v1.BidiService/BidiAppend")
                .header(header::CONTENT_TYPE, "application/proto")
                .body(Body::from(
                    ai::BidiAppendRequest {
                        request_id: Some(ai::BidiRequestId {
                            request_id: REQUEST_ID.into(),
                        }),
                        append_seqno: seqno,
                        data_binary: message.encode_to_vec(),
                        ..Default::default()
                    }
                    .encode_to_vec(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}

fn run_request() -> pb::AgentClientMessage {
    pb::AgentClientMessage {
        message: Some(pb::agent_client_message::Message::RunRequest(
            pb::AgentRunRequest {
                conversation_id: Some("poll-conversation".into()),
                run_id: Some(REQUEST_ID.into()),
                requested_model: Some(pb::RequestedModel {
                    model_id: "plugin:test/provider/model".into(),
                    ..Default::default()
                }),
                action: Some(pb::ConversationAction {
                    action: Some(pb::conversation_action::Action::UserMessageAction(
                        pb::UserMessageAction {
                            user_message: Some(pb::UserMessage {
                                text: "Hello".into(),
                                message_id: "poll-user".into(),
                                mode: pb::AgentMode::Agent as i32,
                                ..Default::default()
                            }),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )),
    }
}

/// A `RunPoll` request is a Connect client stream carrying one
/// `BidiPollRequest`; Cursor terminates that stream immediately, so the body is
/// a single framed message.
fn poll_body(start_request: bool) -> Bytes {
    let message = pb::BidiPollRequest {
        request_id: Some(pb::BidiRequestId {
            request_id: REQUEST_ID.into(),
        }),
        start_request: start_request.then_some(true),
    };
    connect::encode_message(&message).unwrap()
}

fn provider() -> fake_provider::FakeProvider {
    use cursor_server::provider::{FinishReason, ModelEvent};

    let provider = fake_provider::FakeProvider::default();
    provider.push(vec![
        ModelEvent::Start {
            model_call_id: "poll-call".into(),
        },
        ModelEvent::TextStart,
        ModelEvent::TextDelta("poll chat works".into()),
        ModelEvent::TextEnd,
        ModelEvent::Done(FinishReason::Stop),
    ]);
    provider
}

#[tokio::test]
async fn run_poll_streams_the_run_that_bidi_append_started() {
    let provider = provider();
    let (_directory, store) = fixtures::temp_store().await;
    let router = router(store, provider.clone());

    let mut append_seqno = 0_i64;
    append(&router, append_seqno, run_request()).await;
    append_seqno += 1;

    let mut expected = 0_i64;
    let outcome = drain_poll(&router, poll_body(true), &mut append_seqno, &mut expected).await;

    assert_eq!(outcome.text, "poll chat works");
    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert!(expected > 0, "expected frames before the terminal frame");
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn run_poll_opened_before_the_append_still_serves_the_run() {
    // Cursor opens the poll stream before the append reaches the server; the
    // handler must wait for the route instead of forwarding upstream, otherwise
    // the client receives upstream seqno 0 and the next poll replays it.
    let provider = provider();
    let (_directory, store) = fixtures::temp_store().await;
    let router = router(store, provider.clone());

    let poll_router = router.clone();
    let poll = tokio::spawn(async move {
        let mut append_seqno = 1_i64;
        let mut expected = 0_i64;
        drain_poll(
            &poll_router,
            poll_body(true),
            &mut append_seqno,
            &mut expected,
        )
        .await
    });

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    append(&router, 0, run_request()).await;

    let outcome = poll.await.expect("poll task panicked");
    assert_eq!(outcome.text, "poll chat works");
    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    assert_eq!(provider.requests().len(), 1);
}

#[tokio::test]
async fn run_poll_preserves_provider_errors() {
    // Masking a provider failure as a clean eof makes Cursor report "stream
    // ended without turnEnded" and retry the turn; the error must pass through.
    let provider = fake_provider::FakeProvider::default();
    // 400 is classified as a rejected request (not retried), so the run fails
    // once and the poll stream carries that failure to the client.
    provider.push_error(cursor_server::Error::Provider(
        "Test 400 Bad Request: bad input".into(),
    ));
    let (_directory, store) = fixtures::temp_store().await;
    let router = router(store, provider.clone());

    append(&router, 0, run_request()).await;

    let mut append_seqno = 1_i64;
    let mut expected = 0_i64;
    let outcome = drain_poll(&router, poll_body(true), &mut append_seqno, &mut expected).await;

    let error = outcome.error.expect("provider error must reach the client");
    assert!(error.contains("400 Bad Request"), "{error}");
}
