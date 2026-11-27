//! Implements Cursor HTTP endpoints outside the Agent Run stream.
use std::sync::Arc;

use axum::{
    body::{to_bytes, Body, Bytes},
    extract::{DefaultBodyLimit, Extension, State},
    http::{header, HeaderMap, HeaderValue, Request, Response, StatusCode},
    routing::{get, post},
    Router,
};
use futures_util::StreamExt;
use tower_http::decompression::RequestDecompressionLayer;

use crate::{
    api::cursor::{
        bidi,
        proxy::{self, CursorProxy},
        run_sse,
    },
    cursor::{
        inference::{
            decode_append,
            upstream_sse::{prepare_upstream_headers, UpstreamRewrite},
            upstream_wire, InferenceRoute, InferenceService,
        },
        protocol::{
            connect,
            proto::{agent::v1 as agent, aiserver::v1 as ai, inference},
        },
        services::{
            account, analytics, commit_message, compatibility, entitlement::FreeEntitlementCache,
            knowledge, model_catalog, official_catalog::OfficialCatalog, server_config, tab,
        },
        transport::{TransportParent, TransportRegistry},
    },
    Result,
};

pub fn router(
    registry: TransportRegistry,
    clients: crate::network::NetworkClients,
) -> Result<Router> {
    let proxy = CursorProxy::cursor(clients);
    let knowledge = knowledge::KnowledgeService::managed()?;
    Ok(router_with_proxy(registry, proxy, knowledge).0)
}

fn router_with_proxy(
    registry: TransportRegistry,
    proxy: CursorProxy,
    knowledge_service: knowledge::KnowledgeService,
) -> (Router, Arc<InferenceService>) {
    let web_cache = registry.web_cache().router();
    let free_entitlements = FreeEntitlementCache::default();
    let official_catalog = OfficialCatalog::new();
    let inference_service =
        InferenceService::new(registry.clone(), official_catalog.clone(), proxy.clone());
    let router = Router::new()
        .route("/__byok-api__/healthz", get(health))
        .route("/agent.v1.AgentService/RunSSE", post(run_sse_handler))
        .route(
            "/aiserver.v1.InferenceService/RunInferenceSSE",
            post(run_inference_sse_handler),
        )
        .route("/aiserver.v1.BidiService/BidiAppend", post(bidi_handler))
        .route(
            "/aiserver.v1.AiService/AvailableDocs",
            post(compatibility::available_docs),
        )
        .route(
            "/aiserver.v1.DashboardService/GetEffectiveUserPlugins",
            post(compatibility::effective_user_plugins),
        )
        .route(
            "/aiserver.v1.DashboardService/GetUserPrivacyMode",
            post(compatibility::user_privacy_mode),
        )
        .route(
            "/agent.v1.AgentService/UpdateConversationMetadata",
            post(compatibility::update_conversation_metadata),
        )
        .route(
            "/aiserver.v1.AiService/GetServerConfig",
            post(server_config::get),
        )
        .route(
            "/aiserver.v1.ServerConfigService/GetServerConfig",
            post(server_config::get),
        )
        .route(
            "/aiserver.v1.AiService/AvailableModels",
            post(model_catalog::available_models),
        )
        .route(
            "/agent.v1.AgentService/GetUsableModels",
            post(model_catalog::usable_models),
        )
        .route(
            "/aiserver.v1.AiService/GetUsableModels",
            post(model_catalog::usable_models),
        )
        .route(
            "/aiserver.v1.AiService/WriteGitCommitMessage",
            post(commit_message::write_git_commit_message),
        )
        .route(
            "/aiserver.v1.NetworkService/IsConnected",
            post(is_connected),
        )
        .route(
            "/agent.v1.AgentService/GetDefaultModelForCli",
            post(model_catalog::default_model_for_cli),
        )
        .route(
            "/aiserver.v1.AiService/GetDefaultModelForCli",
            post(model_catalog::default_model_for_cli),
        )
        .route(
            "/aiserver.v1.AiService/GetDefaultModel",
            post(model_catalog::default_model),
        )
        .route(
            "/aiserver.v1.AiService/GetDefaultModelNudgeData",
            post(model_catalog::default_model_nudge),
        )
        .route(
            "/aiserver.v1.AuthService/GetEmail",
            post(account::get_email),
        )
        .route(
            "/aiserver.v1.AuthService/GetUserMeta",
            post(account::get_user_meta),
        )
        .route("/aiserver.v1.DashboardService/GetMe", post(account::get_me))
        .route(
            "/aiserver.v1.DashboardService/GetTeams",
            post(account::get_teams),
        )
        .route(
            "/aiserver.v1.DashboardService/GetUserProfile",
            post(account::get_user_profile),
        )
        .route(
            "/aiserver.v1.DashboardService/GetCurrentPeriodUsage",
            post(account::current_period_usage),
        )
        .route(
            "/aiserver.v1.DashboardService/GetUsageLimitStatusAndActiveGrants",
            post(account::usage_limit_status),
        )
        .route(
            "/aiserver.v1.AiService/KnowledgeBaseAdd",
            post(knowledge::add),
        )
        .route(
            "/aiserver.v1.AiService/KnowledgeBaseList",
            post(knowledge::list),
        )
        .route(
            "/aiserver.v1.AiService/KnowledgeBaseUpdate",
            post(knowledge::update),
        )
        .route(
            "/aiserver.v1.AiService/KnowledgeBaseRemove",
            post(knowledge::remove),
        )
        .route(
            analytics::BOOTSTRAP_STATSIG_PATH,
            post(analytics::bootstrap_statsig),
        )
        .route("/auth/full_stripe_profile", get(account::stripe_profile))
        .route("/auth/stripe_profile", get(account::stripe_profile))
        .merge(tab::router())
        .route_layer(DefaultBodyLimit::disable())
        .route_layer(RequestDecompressionLayer::new())
        .fallback(proxy::forward)
        .method_not_allowed_fallback(proxy::forward)
        .layer(Extension(proxy))
        .layer(Extension(knowledge_service))
        .layer(Extension(free_entitlements))
        .layer(Extension(inference_service.clone()))
        .layer(Extension(official_catalog))
        .with_state(registry)
        .merge(web_cache);
    (router, Arc::new(inference_service))
}

async fn health() -> StatusCode {
    StatusCode::NO_CONTENT
}

/// `NetworkService/IsConnected` probe. Cursor's always-local extension checks
/// connectivity roughly 10s after any slow request starts; a 404/error here is
/// treated as "network disconnected" and aborts in-flight work (e.g. commit
/// message generation) even while the model is still streaming. Always answer
/// connected with an empty `IsConnectedResponse` so local BYOK generation is
/// never cancelled by this probe.
async fn is_connected() -> Result<Response<Body>> {
    let payload = connect::encode_message(&ai::IsConnectedResponse {})?;
    let mut response = Response::new(Body::from(payload));
    *response.status_mut() = StatusCode::OK;
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/proto"),
    );
    Ok(response)
}

fn extract_sse_request_id(body: &[u8]) -> Result<String> {
    match connect::decode_unary::<agent::BidiRequestId>(body) {
        Ok(request) if !request.request_id.is_empty() => Ok(request.request_id),
        Ok(_) => upstream_wire::sniff_bidi_request_id(body)
            .ok_or_else(|| crate::Error::Protocol("SSE request_id is required".into())),
        Err(error) => upstream_wire::sniff_bidi_request_id(body).ok_or(error),
    }
}

async fn fail_open_upstream(
    inference_service: &InferenceService,
    registry: &TransportRegistry,
    request_id: &str,
) {
    inference_service.mark_upstream(request_id);
    registry.mark_upstream(request_id).await;
}

async fn run_sse_handler(
    State(registry): State<TransportRegistry>,
    Extension(proxy): Extension<CursorProxy>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    let (parts, body) = buffered(request).await?;
    let request_id = match extract_sse_request_id(&body) {
        Ok(request_id) => request_id,
        Err(error)
            if !body.is_empty()
                && !crate::local_app::request_uses_local_cursor_token(&parts.headers) =>
        {
            tracing::warn!(%error, "RunSSE request_id decode failed; failing open to Cursor upstream");
            return proxy::forward(
                Extension(proxy),
                Request::from_parts(parts, Body::from(body)),
            )
            .await;
        }
        Err(error) => return Err(error),
    };
    let route = registry.wait_route(&request_id).await;
    let trace = registry.trace(&request_id);
    trace.resume();
    trace.request(
        "run_sse_request",
        body.clone(),
        serde_json::json!({"request_id": request_id}),
    );
    match route {
        crate::cursor::transport::TransportRoute::Local => {
            run_sse::stream(&registry, &request_id).await
        }
        crate::cursor::transport::TransportRoute::Upstream(generation) => {
            let response = proxy::forward(
                Extension(proxy),
                Request::from_parts(parts, Body::from(body)),
            )
            .await?;
            Ok(run_sse::upstream(registry, request_id, generation, response, Some(trace)).await)
        }
    }
}

async fn run_inference_sse_handler(
    State(registry): State<TransportRegistry>,
    Extension(inference_service): Extension<InferenceService>,
    Extension(proxy): Extension<CursorProxy>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    let (parts, body) = buffered(request).await?;
    let request_id = match extract_sse_request_id(&body) {
        Ok(request_id) => request_id,
        Err(error)
            if !body.is_empty()
                && !crate::local_app::request_uses_local_cursor_token(&parts.headers) =>
        {
            tracing::warn!(%error, "RunInferenceSSE request_id decode failed; failing open to Cursor upstream");
            return proxy::forward(
                Extension(proxy),
                Request::from_parts(parts, Body::from(body)),
            )
            .await;
        }
        Err(error) => return Err(error),
    };
    match inference_service.wait_route(&request_id).await {
        InferenceRoute::Upstream => {
            let models = registry.store().models().await.unwrap_or_default();
            let reserved = inference_service
                .catalog()
                .slug_set(OfficialCatalog::token_key(&parts.headers));
            let byok = crate::cursor::tools::subagent_models::byok_policies(&models, &reserved);
            let token = OfficialCatalog::token_key(&parts.headers);
            let mut request = Request::from_parts(parts, Body::from(body));
            prepare_upstream_headers(request.headers_mut());
            let upstream = proxy::forward(Extension(proxy.clone()), request).await?;
            let status = upstream.status();
            let headers = upstream.headers().clone();
            let mut rewrite =
                UpstreamRewrite::new(byok, inference_service.catalog().clone(), token);
            let stream = async_stream::stream! {
                let mut upstream = upstream.into_body().into_data_stream();
                while let Some(chunk) = upstream.next().await {
                    match chunk {
                        Ok(bytes) => {
                            match rewrite.ingest(bytes.clone()) {
                                Ok(frames) => {
                                    for frame in frames {
                                        yield Ok::<Bytes, std::convert::Infallible>(frame);
                                    }
                                }
                                Err(error) => {
                                    tracing::warn!(%error, "upstream RunInference rewrite failed");
                                    yield Ok(bytes);
                                }
                            }
                        }
                        Err(error) => {
                            tracing::warn!(?error, "upstream RunInference stream ended with error");
                            return;
                        }
                    }
                }
            };
            let mut response = Response::new(Body::from_stream(stream));
            *response.status_mut() = status;
            *response.headers_mut() = headers;
            Ok(response)
        }
        InferenceRoute::Local(run) => {
            let (mut receiver, subscription) = inference_service.subscribe(request_id, &run);
            let stream = async_stream::stream! {
                let _subscription = subscription;
                while let Some(frame) = receiver.recv().await {
                    let complete = frame.first().is_some_and(|flag| flag & connect::END_STREAM_FLAG != 0);
                    yield Ok::<Bytes, std::convert::Infallible>(frame);
                    if complete {
                        return;
                    }
                }
            };
            let mut response = Response::new(Body::from_stream(stream));
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            );
            response
                .headers_mut()
                .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
            response
                .headers_mut()
                .insert("connect-protocol-version", HeaderValue::from_static("1"));
            Ok(response)
        }
    }
}

struct ParsedBidiEnvelope {
    request_id: String,
    append_seqno: i64,
    payload: Vec<u8>,
}

enum EnvelopeParse {
    Parsed(ParsedBidiEnvelope),
    FailOpenUpstream(crate::Error),
}

fn parse_bidi_envelope(body: &[u8], can_fail_open: bool) -> Result<EnvelopeParse> {
    match connect::decode_unary::<ai::BidiAppendRequest>(body) {
        Ok(request) => {
            let request_id = request
                .request_id
                .as_ref()
                .map(|id| id.request_id.as_str())
                .filter(|id| !id.is_empty());
            let Some(request_id) = request_id else {
                if can_fail_open
                    && !body.is_empty()
                    && request.data.is_empty()
                    && request.data_binary.is_empty()
                    && request.append_seqno == 0
                {
                    return Ok(EnvelopeParse::FailOpenUpstream(crate::Error::Protocol(
                        "BidiAppend request_id missing on unrecognized envelope".into(),
                    )));
                }
                return Err(crate::Error::Protocol(
                    "BidiAppend request_id is required".into(),
                ));
            };
            let payload = match (request.data.is_empty(), request.data_binary.is_empty()) {
                (true, false) => request.data_binary,
                (false, true) => hex::decode(&request.data).map_err(|error| {
                    crate::Error::Protocol(format!("invalid BidiAppend hex: {error}"))
                })?,
                (true, true) => {
                    return Err(crate::Error::Protocol(
                        "BidiAppend contains no AgentClientMessage".into(),
                    ));
                }
                (false, false) => {
                    return Err(crate::Error::Protocol(
                        "BidiAppend contains both data and data_binary".into(),
                    ));
                }
            };
            Ok(EnvelopeParse::Parsed(ParsedBidiEnvelope {
                request_id: request_id.to_owned(),
                append_seqno: request.append_seqno,
                payload,
            }))
        }
        Err(error) => {
            if let Some(sniffed) = upstream_wire::sniff_bidi_append_envelope(body) {
                return Ok(EnvelopeParse::Parsed(ParsedBidiEnvelope {
                    request_id: sniffed.request_id,
                    append_seqno: 0,
                    payload: sniffed.payload,
                }));
            }
            if can_fail_open && !body.is_empty() {
                Ok(EnvelopeParse::FailOpenUpstream(error))
            } else {
                Err(error)
            }
        }
    }
}

fn decode_agent_append(
    request_id: &str,
    append_seqno: i64,
    payload: &[u8],
) -> Result<bidi::DecodedAppend> {
    use prost::Message;
    let message = agent::AgentClientMessage::decode(payload)?;
    Ok(bidi::DecodedAppend {
        request_id: request_id.to_owned(),
        seqno: append_seqno,
        message,
    })
}

async fn bidi_handler(
    State(registry): State<TransportRegistry>,
    Extension(proxy): Extension<CursorProxy>,
    Extension(inference_service): Extension<InferenceService>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    let (parts, body) = buffered(request).await?;
    let can_fail_open = !crate::local_app::request_uses_local_cursor_token(&parts.headers);
    let ParsedBidiEnvelope {
        request_id,
        append_seqno,
        payload,
    } = match parse_bidi_envelope(&body, can_fail_open)? {
        EnvelopeParse::Parsed(parsed) => parsed,
        EnvelopeParse::FailOpenUpstream(error) => {
            tracing::warn!(%error, "BidiAppend outer decode failed; failing open to Cursor upstream");
            return proxy::forward(
                Extension(proxy),
                Request::from_parts(parts, Body::from(body)),
            )
            .await;
        }
    };

    // Fast-path 1: already routed on InferenceService.
    match inference_service.route(&request_id) {
        Some(InferenceRoute::Local(run)) => {
            let message = decode_append(&payload)?;
            run.append(append_seqno, message).await?;
            let mut response = Response::new(Body::empty());
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/proto"),
            );
            return Ok(response);
        }
        Some(InferenceRoute::Upstream) => {
            inference_service.mark_upstream(&request_id);
            let finished = upstream_wire::sniff_client_finish_run(&payload);
            let response = proxy::forward(
                Extension(proxy),
                Request::from_parts(parts, Body::from(body)),
            )
            .await;
            if finished {
                inference_service.finish(&request_id);
            }
            return response;
        }
        None => {}
    }

    // Fast-path 2: already routed on legacy TransportRegistry.
    if registry.local(&request_id).await.is_some() {
        let decoded = decode_agent_append(&request_id, append_seqno, &payload)?;
        let trace_metadata = decoded.trace_metadata();
        let trace = registry.trace(&decoded.request_id);
        trace.resume();
        let parent = match parent_headers(&parts.headers) {
            Ok(parent) => parent,
            Err(error) => {
                trace.request(
                    "bidi_request",
                    body,
                    trace_outcome(
                        trace_metadata,
                        false,
                        "invalid_parent",
                        Some(error.to_string()),
                    ),
                );
                return Err(error);
            }
        };
        match bidi::append(&registry, decoded, parent).await {
            Ok(_) => trace.request(
                "bidi_request",
                body,
                trace_outcome(trace_metadata, true, "local", None),
            ),
            Err(error) => {
                trace.request(
                    "bidi_request",
                    body,
                    trace_outcome(
                        trace_metadata,
                        false,
                        "command_rejected",
                        Some(error.to_string()),
                    ),
                );
                return Err(error);
            }
        }
        let mut response = Response::new(axum::body::Body::empty());
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/proto"),
        );
        return Ok(response);
    }

    if registry.upstream(&request_id).await {
        registry.mark_upstream(&request_id).await;
        let trace = registry.trace(&request_id);
        trace.resume();
        let trace_metadata = decode_agent_append(&request_id, append_seqno, &payload)
            .map(|decoded| decoded.trace_metadata())
            .unwrap_or_else(|_| {
                serde_json::json!({
                    "append_seqno": append_seqno,
                    "message_type": "raw_upstream_passthrough",
                })
            });
        trace.request(
            "bidi_request",
            body.clone(),
            trace_outcome(trace_metadata, true, "upstream", None),
        );
        return proxy::forward(
            Extension(proxy),
            Request::from_parts(parts, Body::from(body)),
        )
        .await;
    }

    // Route discovery for a new request_id.
    let inference_decode = decode_append(&payload);
    if let Ok(message) = &inference_decode {
        let starts_inference = matches!(
            message.message.as_ref(),
            Some(inference::run_inference_client_message::Message::RunRequest(run))
                if !run.conversation_id.is_empty()
                    && run.requested_model.as_ref().is_some_and(|model| !model.model_id.is_empty())
        );
        if starts_inference {
            let route = inference_service
                .begin(&request_id, message, &parts.headers)
                .await?;
            match route {
                InferenceRoute::Local(run) => {
                    run.append(append_seqno, message.clone()).await?;
                    let mut response = Response::new(Body::empty());
                    response.headers_mut().insert(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static("application/proto"),
                    );
                    return Ok(response);
                }
                InferenceRoute::Upstream => {
                    return proxy::forward(
                        Extension(proxy),
                        Request::from_parts(parts, Body::from(body)),
                    )
                    .await;
                }
            }
        }
    }

    if let Ok(decoded) = decode_agent_append(&request_id, append_seqno, &payload) {
        if let Some(model_id) = decoded.model_id().map(str::to_owned) {
            let conversation_id = decoded.conversation_id().map(str::to_owned);
            let trace_metadata = decoded.trace_metadata();
            let trace = registry.trace(&decoded.request_id);
            // 插件模型 ID 只在本地有意义,永远不转发到 Cursor 官方上游。
            let local = if model_id.starts_with(crate::plugin::ADAPTER_ID_PREFIX)
                || registry.store().model(&model_id).await?.is_some()
            {
                tracing::info!(
                    request_id = decoded.request_id,
                    model_id,
                    "routing Cursor Run to BYOK provider"
                );
                true
            } else {
                tracing::info!(
                    request_id = decoded.request_id,
                    model_id,
                    "routing Cursor Run to Cursor upstream"
                );
                false
            };
            trace.begin(
                conversation_id.as_deref(),
                if local {
                    "local_byok"
                } else {
                    "cursor_official"
                },
                Some(model_id.as_str()),
            );
            if !local {
                registry.mark_upstream(&decoded.request_id).await;
                trace.request(
                    "bidi_request",
                    body.clone(),
                    trace_outcome(trace_metadata, true, "upstream", None),
                );
                return proxy::forward(
                    Extension(proxy),
                    Request::from_parts(parts, Body::from(body)),
                )
                .await;
            }
            let parent = match parent_headers(&parts.headers) {
                Ok(parent) => parent,
                Err(error) => {
                    trace.request(
                        "bidi_request",
                        body,
                        trace_outcome(
                            trace_metadata,
                            false,
                            "invalid_parent",
                            Some(error.to_string()),
                        ),
                    );
                    return Err(error);
                }
            };
            match bidi::append(&registry, decoded, parent).await {
                Ok(_) => trace.request(
                    "bidi_request",
                    body,
                    trace_outcome(trace_metadata, true, "local", None),
                ),
                Err(error) => {
                    trace.request(
                        "bidi_request",
                        body,
                        trace_outcome(
                            trace_metadata,
                            false,
                            "command_rejected",
                            Some(error.to_string()),
                        ),
                    );
                    return Err(error);
                }
            }
            let mut response = Response::new(axum::body::Body::empty());
            *response.status_mut() = StatusCode::OK;
            response.headers_mut().insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/proto"),
            );
            return Ok(response);
        }
    }

    // Tier 2: prost failed to decode a modified RunRequest, inspect model_id via raw TLV scan.
    if let Some(sniffed) = upstream_wire::sniff_client_run_start(&payload) {
        let is_local = inference_service
            .is_local_model(&sniffed.model_id, &parts.headers)
            .await
            .unwrap_or(false);
        if is_local {
            let reason = inference_decode
                .err()
                .map(|err| err.to_string())
                .unwrap_or_else(|| "incomplete RunRequest".into());
            return Err(crate::Error::Protocol(format!(
                "RunRequest for local BYOK model '{}' failed protobuf decode: {reason}",
                sniffed.model_id
            )));
        }
        if can_fail_open {
            tracing::warn!(
                request_id,
                model_id = %sniffed.model_id,
                "client RunRequest prost decode failed; TLV sniffer failing open to Cursor upstream"
            );
            fail_open_upstream(&inference_service, &registry, &request_id).await;
            return proxy::forward(
                Extension(proxy),
                Request::from_parts(parts, Body::from(body)),
            )
            .await;
        }
    }

    // Tier 3: Ultimate fail-open for unrecognized or pre-start upstream BidiAppend payloads.
    if can_fail_open {
        tracing::warn!(
            request_id,
            payload_len = payload.len(),
            "unrecognized or non-start BidiAppend on new request_id; failing open to Cursor upstream"
        );
        fail_open_upstream(&inference_service, &registry, &request_id).await;
        return proxy::forward(
            Extension(proxy),
            Request::from_parts(parts, Body::from(body)),
        )
        .await;
    }

    let trace = registry.trace(&request_id);
    trace.resume();
    trace.request(
        "bidi_request",
        body,
        trace_outcome(
            serde_json::json!({
                "append_seqno": append_seqno,
                "message_type": "unrecognized",
            }),
            false,
            "missing_transport",
            None,
        ),
    );
    Err(crate::Error::Protocol(
        "first BidiAppend message must select a model".into(),
    ))
}

fn trace_outcome(
    mut metadata: serde_json::Value,
    accepted: bool,
    route_outcome: &str,
    error: Option<String>,
) -> serde_json::Value {
    if let Some(metadata) = metadata.as_object_mut() {
        metadata.insert("accepted".into(), accepted.into());
        metadata.insert("route_outcome".into(), route_outcome.into());
        if let Some(error) = error {
            metadata.insert("error".into(), error.into());
        }
    }
    metadata
}

#[cfg(test)]
#[path = "handlers/run_inference_tests.rs"]
mod run_inference_tests;

async fn buffered(request: Request<Body>) -> Result<(axum::http::request::Parts, Bytes)> {
    let (parts, body) = request.into_parts();
    let body = to_bytes(body, usize::MAX)
        .await
        .map_err(|error| crate::Error::Protocol(format!("cannot read request body: {error}")))?;
    Ok((parts, body))
}

fn parent_headers(headers: &HeaderMap) -> Result<Option<TransportParent>> {
    Ok(header_text(headers, "x-parent-agent-tool-call-id")?
        .filter(|value| !value.is_empty())
        .map(|tool_call_id| TransportParent {
            tool_call_id: tool_call_id.into(),
        }))
}

fn header_text<'a>(headers: &'a HeaderMap, name: &str) -> Result<Option<&'a str>> {
    headers
        .get(name)
        .map(|value| value.to_str())
        .transpose()
        .map_err(|error| crate::Error::Protocol(format!("invalid {name} header: {error}")))
}

#[cfg(test)]
mod parent_header_tests {
    use super::*;

    fn headers(request_id: Option<&str>, tool_call_id: Option<&str>) -> HeaderMap {
        let mut headers = HeaderMap::new();
        if let Some(value) = request_id {
            headers.insert("x-parent-request-id", value.parse().unwrap());
        }
        if let Some(value) = tool_call_id {
            headers.insert("x-parent-agent-tool-call-id", value.parse().unwrap());
        }
        headers
    }

    fn parent(tool_call_id: &str) -> TransportParent {
        TransportParent {
            tool_call_id: tool_call_id.into(),
        }
    }

    #[test]
    fn both_headers_preserve_the_parent_tool_call() {
        assert_eq!(
            parent_headers(&headers(Some("req"), Some("tool"))).unwrap(),
            Some(parent("tool"))
        );
    }

    #[test]
    fn missing_request_id_keeps_the_parent_tool_call() {
        assert_eq!(
            parent_headers(&headers(None, Some("tool"))).unwrap(),
            Some(parent("tool"))
        );
    }

    #[test]
    fn missing_tool_call_id_means_no_parent() {
        assert_eq!(parent_headers(&headers(Some("req"), None)).unwrap(), None);
    }

    #[test]
    fn no_headers_or_empty_headers_mean_no_parent() {
        assert_eq!(parent_headers(&headers(None, None)).unwrap(), None);
        assert_eq!(parent_headers(&headers(Some(""), Some(""))).unwrap(), None);
        assert_eq!(
            parent_headers(&headers(Some(""), Some("tool"))).unwrap(),
            Some(parent("tool"))
        );
    }
}
