//! Owns one client-loop inference stream per Cursor request ID.
use std::{collections::HashMap, sync::Arc};

use axum::http::{header, HeaderMap};
use bytes::Bytes;
use futures_util::StreamExt;
use parking_lot::Mutex;
use prost::Message;
use tokio::{
    sync::{mpsc, Mutex as AsyncMutex, Notify},
    task::JoinSet,
};
use tokio_util::sync::CancellationToken;

use crate::{
    api::cursor::proxy::CursorProxy,
    cursor::{
        protocol::{connect, proto::inference as pb},
        services::official_catalog::{OfficialCatalog, OfficialSlug, TokenKey},
        tools::subagent_models::{self},
        transport::{OutputHub, TransportRegistry},
    },
    model::{ModelConfig, ModelInvocation, ModelType, Usage},
    provider::{ModelEvent, Provider},
    store::Store,
    Error, Result,
};

use super::project;

/// Cursor's subagent model policies; `none` leaves model selection to the client.
const SUBAGENT_FORCE_POLICY_NONE: &str = "none";

#[derive(Clone)]
pub(crate) struct InferenceService {
    state: Arc<Mutex<HashMap<String, InferenceRoute>>>,
    changed: Arc<Notify>,
    registry: TransportRegistry,
    catalog: OfficialCatalog,
    proxy: CursorProxy,
}

#[derive(Clone)]
pub(crate) enum InferenceRoute {
    Local(Arc<Run>),
    Upstream,
}

pub(crate) struct Run {
    input: mpsc::Sender<pb::RunInferenceClientMessage>,
    output: Arc<OutputHub>,
    next_seqno: AsyncMutex<i64>,
    subscriptions: Mutex<Subscriptions>,
}

#[derive(Default)]
struct Subscriptions {
    active: usize,
    generation: u64,
}

pub(crate) struct Subscription {
    service: InferenceService,
    request_id: String,
    run: Arc<Run>,
}

impl Drop for Subscription {
    fn drop(&mut self) {
        let mut subscriptions = self.run.subscriptions.lock();
        subscriptions.active -= 1;
        if subscriptions.active != 0 {
            return;
        }
        let generation = subscriptions.generation;
        let state = Arc::downgrade(&self.service.state);
        let run = Arc::downgrade(&self.run);
        let request_id = self.request_id.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(300)).await;
            if let (Some(state), Some(run)) = (state.upgrade(), run.upgrade()) {
                let mut routes = state.lock();
                if matches!(routes.get(&request_id), Some(InferenceRoute::Local(current)) if Arc::ptr_eq(current, &run))
                    && run.subscriptions.lock().generation == generation
                {
                    routes.remove(&request_id);
                }
            }
        });
    }
}

struct LocalModel {
    display_name: String,
    model_type: ModelType,
    context_window_tokens: Option<u64>,
}

struct ServeContext {
    output: Arc<OutputHub>,
    provider: Arc<dyn Provider>,
    store: Store,
    run_id: String,
    conversation_id: String,
    model_id: String,
    model: Option<LocalModel>,
    token: TokenKey,
    headers: HeaderMap,
    catalog: OfficialCatalog,
    proxy: CursorProxy,
}

impl InferenceService {
    pub fn new(registry: TransportRegistry, catalog: OfficialCatalog, proxy: CursorProxy) -> Self {
        Self {
            state: Arc::new(Mutex::new(HashMap::new())),
            changed: Arc::new(Notify::new()),
            registry,
            catalog,
            proxy,
        }
    }

    pub fn catalog(&self) -> &OfficialCatalog {
        &self.catalog
    }

    pub fn route(&self, request_id: &str) -> Option<InferenceRoute> {
        self.state.lock().get(request_id).cloned()
    }

    pub async fn wait_route(&self, request_id: &str) -> InferenceRoute {
        loop {
            let changed = self.changed.notified();
            tokio::pin!(changed);
            changed.as_mut().enable();
            if let Some(route) = self.route(request_id) {
                return route;
            }
            changed.await;
        }
    }

    pub async fn begin(
        &self,
        request_id: &str,
        message: &pb::RunInferenceClientMessage,
        headers: &HeaderMap,
    ) -> Result<InferenceRoute> {
        let Some(pb::run_inference_client_message::Message::RunRequest(request)) =
            message.message.as_ref()
        else {
            return Err(Error::Protocol(
                "first RunInference message must start a run".into(),
            ));
        };
        let model_id = request
            .requested_model
            .as_ref()
            .map(|model| model.model_id.as_str())
            .filter(|model| !model.is_empty())
            .ok_or_else(|| Error::Protocol("RunInference model is required".into()))?;
        if request.conversation_id.is_empty() {
            return Err(Error::Protocol(
                "RunInference conversation ID is required".into(),
            ));
        }
        let token = OfficialCatalog::token_key(headers);
        let store_models = self.registry.store().models().await?;
        let byok_hashes = store_models
            .iter()
            .map(|model| model.model_hash.clone())
            .collect::<std::collections::HashSet<_>>();
        let configured = self.registry.store().model(model_id).await?;
        let (model_id, configured, upstream) = if let Some(config) = configured {
            (model_id.to_string(), Some(config), false)
        } else if model_id.starts_with(crate::plugin::ADAPTER_ID_PREFIX) {
            (model_id.to_string(), None, false)
        } else {
            let official = self
                .catalog
                .get_or_fetch(token, headers, &self.proxy, &byok_hashes)
                .await;
            let reserved = official_reserved(&official);
            match self.resolve_by_display_name(model_id, &reserved).await? {
                Some(config) => {
                    tracing::info!(
                        request_id,
                        requested = model_id,
                        resolved = %config.model_hash,
                        "resolved BYOK subagent model by display name"
                    );
                    (config.model_hash.clone(), Some(config), false)
                }
                None => (model_id.to_string(), None, true),
            }
        };
        let model_id = model_id.as_str();
        let local = !upstream
            && (configured.is_some() || model_id.starts_with(crate::plugin::ADAPTER_ID_PREFIX));
        let request_headers = capture_request_headers(headers);
        let route = if local {
            tracing::info!(
                request_id,
                model_id,
                "routing Cursor RunInference to BYOK provider"
            );
            let (sender, receiver) = mpsc::channel(64);
            let output = Arc::new(OutputHub::default());
            let run = Arc::new(Run {
                input: sender,
                output: output.clone(),
                next_seqno: AsyncMutex::new(0),
                subscriptions: Mutex::default(),
            });
            tokio::spawn(serve(
                ServeContext {
                    output,
                    provider: self.registry.provider(),
                    store: self.registry.store().clone(),
                    run_id: request_id.into(),
                    conversation_id: request.conversation_id.clone(),
                    model_id: model_id.into(),
                    model: configured.map(|model| LocalModel {
                        display_name: model.display_name,
                        model_type: model.model_type,
                        context_window_tokens: model.context_window_tokens,
                    }),
                    token,
                    headers: request_headers,
                    catalog: self.catalog.clone(),
                    proxy: self.proxy.clone(),
                },
                receiver,
            ));
            InferenceRoute::Local(run)
        } else {
            tracing::info!(
                request_id,
                model_id,
                "routing Cursor RunInference to Cursor upstream"
            );
            InferenceRoute::Upstream
        };
        let mut state = self.state.lock();
        let route = state
            .entry(request_id.into())
            .or_insert_with(|| route.clone())
            .clone();
        drop(state);
        self.changed.notify_waiters();
        Ok(route)
    }

    pub fn subscribe(
        &self,
        request_id: String,
        run: &Arc<Run>,
    ) -> (mpsc::UnboundedReceiver<Bytes>, Subscription) {
        let mut subscriptions = run.subscriptions.lock();
        subscriptions.active += 1;
        subscriptions.generation += 1;
        drop(subscriptions);
        (
            run.output.subscribe(),
            Subscription {
                service: self.clone(),
                request_id,
                run: run.clone(),
            },
        )
    }

    pub fn finish(&self, request_id: &str) {
        self.state.lock().remove(request_id);
    }

    /// Resolves a Task `model` argument that names a model by display name.
    async fn resolve_by_display_name(
        &self,
        requested: &str,
        reserved: &std::collections::HashSet<String>,
    ) -> Result<Option<ModelConfig>> {
        let models = self.registry.store().models().await?;
        let mut byok = subagent_models::models_from_configs(&models);
        subagent_models::assign_slugs(&mut byok, reserved);
        let Ok(hash) = subagent_models::resolve(&byok, requested) else {
            return Ok(None);
        };
        Ok(models.into_iter().find(|model| model.model_hash == hash))
    }
}

impl Run {
    pub async fn append(&self, seqno: i64, message: pb::RunInferenceClientMessage) -> Result<()> {
        let mut next = self.next_seqno.lock().await;
        if seqno < *next {
            return Ok(());
        }
        if seqno != *next {
            return Err(Error::Protocol(format!(
                "RunInference append sequence {seqno} differs from expected {}",
                *next
            )));
        }
        self.input
            .send(message)
            .await
            .map_err(|_| Error::Protocol("RunInference stream has ended".into()))?;
        *next += 1;
        Ok(())
    }
}

async fn serve(context: ServeContext, mut input: mpsc::Receiver<pb::RunInferenceClientMessage>) {
    let ServeContext {
        output,
        provider,
        store,
        run_id,
        conversation_id,
        model_id,
        model,
        token,
        headers,
        catalog,
        proxy,
    } = context;
    let invocations = Arc::new(Mutex::new(HashMap::<String, CancellationToken>::new()));
    let mut tasks = JoinSet::new();
    let mut finished = false;
    let mut provider_call_index = 0;
    let configured_context_window = model.as_ref().and_then(|model| model.context_window_tokens);
    let mut run_context_window = configured_context_window;
    let models = store.models().await.unwrap_or_default();
    while let Some(message) = input.recv().await {
        use pb::run_inference_client_message::Message;
        match message.message {
            Some(Message::RunRequest(request)) => {
                let requested_window =
                    project::requested_context_window(request.requested_model.as_ref())
                        .ok()
                        .flatten();
                run_context_window = requested_window.or(configured_context_window);
                if run_context_window.is_none() {
                    tracing::info!(
                        request_id = %run_id,
                        model_id,
                        "inference_context_window_unknown"
                    );
                }
                let byok_hashes = models
                    .iter()
                    .map(|model| model.model_hash.clone())
                    .collect::<std::collections::HashSet<_>>();
                let official = catalog
                    .get_or_fetch(token, &headers, &proxy, &byok_hashes)
                    .await;
                let ready = run_ready(&request, model.as_ref(), &models, &model_id, &official);
                emit(
                    &output,
                    pb::run_inference_server_message::Message::RunReady(ready),
                );
            }
            Some(Message::InvokeModel(invoke)) => {
                let id = invoke.invocation_id.clone();
                let Some(request) = invoke.request else {
                    invocation_error(&output, id, "RunInference model request is missing");
                    continue;
                };
                if invoke.delta_base.is_some() || invoke.compressed_request.is_some() {
                    invocation_error(
                        &output,
                        id,
                        "RunInference delta or compressed requests are not supported",
                    );
                    continue;
                }
                let request =
                    match project::project(&request, &model_id, run_context_window, &models) {
                        Ok(request) => request,
                        Err(error) => {
                            invocation_error(&output, id, &error.to_string());
                            continue;
                        }
                    };
                let context_window_tokens = request.model.context_window_tokens;
                let cancellation = CancellationToken::new();
                invocations.lock().insert(id.clone(), cancellation.clone());
                tasks.spawn(serve_invocation(
                    provider.clone(),
                    output.clone(),
                    invocations.clone(),
                    cancellation,
                    ModelInvocation {
                        call_id: id,
                        run_id: run_id.clone(),
                        conversation_id: conversation_id.clone(),
                        provider_call_index,
                        request,
                    },
                    context_window_tokens,
                ));
                provider_call_index += 1;
            }
            Some(Message::CancelInvocation(cancel)) => {
                if let Some(token) = invocations.lock().remove(&cancel.invocation_id) {
                    token.cancel();
                }
            }
            Some(Message::FinishRun(finish)) => {
                finished = true;
                if finish.aborted {
                    for (_, token) in invocations.lock().drain() {
                        token.cancel();
                    }
                }
                break;
            }
            Some(Message::ResolveSubagentModels(resolve)) => {
                let official = catalog.get(token);
                let resolved = resolve_subagent_models(resolve, &models, &official);
                emit(
                    &output,
                    pb::run_inference_server_message::Message::SubagentModelsResolved(resolved),
                );
            }
            Some(
                Message::ClientHeartbeat(_)
                | Message::AgentEvent(_)
                | Message::RunRequestLog(_)
                | Message::ClientFirstCallTimings(_),
            )
            | None => {}
        }
    }
    if !finished {
        for (_, token) in invocations.lock().drain() {
            token.cancel();
        }
    }
    while tasks.join_next().await.is_some() {}
    output.emit(connect::encode_end_stream());
    output.close();
}

fn official_reserved(official: &[OfficialSlug]) -> std::collections::HashSet<String> {
    official
        .iter()
        .flat_map(|slug| std::iter::once(slug.slug.clone()).chain(slug.legacy_aliases.clone()))
        .collect()
}

fn is_official_slug(official: &[OfficialSlug], slug: &str) -> bool {
    if slug.is_empty() {
        return false;
    }
    official
        .iter()
        .any(|model| model.slug == slug || model.legacy_aliases.iter().any(|alias| alias == slug))
}

fn is_byok_owned_slug(models: &[subagent_models::ByokModel], slug: &str) -> bool {
    models.iter().any(|model| {
        model.hash == slug
            || (!model.advertised_slug.is_empty() && model.advertised_slug == slug)
            || slug == subagent_models::hash6_slug(&subagent_models::slug_base(model), &model.hash)
    })
}

fn capture_request_headers(headers: &HeaderMap) -> HeaderMap {
    static CONNECT_PROTOCOL_VERSION: axum::http::HeaderName =
        axum::http::HeaderName::from_static("connect-protocol-version");
    let mut captured = HeaderMap::new();
    if let Some(value) = headers.get(header::AUTHORIZATION) {
        captured.insert(header::AUTHORIZATION, value.clone());
    }
    if let Some(value) = headers.get(header::COOKIE) {
        captured.insert(header::COOKIE, value.clone());
    }
    if let Some(value) = headers.get(&CONNECT_PROTOCOL_VERSION) {
        captured.insert(CONNECT_PROTOCOL_VERSION.clone(), value.clone());
    }
    captured
}

fn local_feature_flags() -> HashMap<String, pb::RunInferenceFlagValue> {
    use pb::run_inference_flag_value::Value;
    let mut flags = HashMap::new();
    let entries: &[(&str, Value)] = &[
        ("closeStdin", Value::BoolValue(true)),
        ("disableBackgroundTaskFollowUp", Value::BoolValue(false)),
        ("enableAwaitForSubagents", Value::BoolValue(true)),
        ("enableBackgroundTaskProgress", Value::BoolValue(true)),
        ("enableBoundedSubagentAwait", Value::BoolValue(true)),
        ("enableDebugSubagent", Value::BoolValue(true)),
        ("enableEmptyResponseRetry", Value::BoolValue(true)),
        ("enableExploreSubagent", Value::BoolValue(true)),
        ("enableReadToolNegativeOffset", Value::BoolValue(true)),
        ("enableSandboxSharedBuildCache", Value::BoolValue(true)),
        ("longRunningJobs", Value::BoolValue(true)),
        ("nalLoopDetection", Value::BoolValue(true)),
        ("outputNotificationLimit", Value::NumberValue(1000.0)),
        ("subagentSupportInterrupt", Value::BoolValue(true)),
        ("useClientSideSubagent", Value::BoolValue(true)),
        ("dynamicToolProfile", Value::StringValue("final".into())),
    ];
    for (name, value) in entries {
        flags.insert(
            (*name).into(),
            pb::RunInferenceFlagValue {
                value: Some(value.clone()),
            },
        );
    }
    flags
}

fn run_ready(
    request: &pb::RunInferenceRunRequest,
    model: Option<&LocalModel>,
    models: &[ModelConfig],
    model_id: &str,
    official: &[OfficialSlug],
) -> pb::RunInferenceRunReady {
    let (display_name, model_type) = model
        .map(|model| (model.display_name.clone(), model.model_type))
        .unwrap_or_else(|| (String::new(), ModelType::OpenAi));
    let anthropic = model_type == ModelType::Anthropic;
    let name = display_name.to_ascii_lowercase();
    let claude = anthropic && name.contains("claude");
    let gpt5 = !anthropic && name.starts_with("gpt-5");
    pb::RunInferenceRunReady {
        resolved_model: request.requested_model.clone(),
        routed_model_display_name: None,
        prompt_model_metadata: Some(pb::RunInferencePromptModelMetadata {
            vendor: if anthropic { "anthropic" } else { "openai" }.into(),
            prompt_version: "latest".into(),
            is_claude_4x: claude,
            is_sonnet4: claude && name.contains("sonnet"),
            is_gpt5_family: gpt5,
            is_gpt5: gpt5,
            persona: "auto".into(),
            self_identity_name: "Cursor".into(),
            enable_line_numbers: true,
            agent_mode_configs: agent_mode_configs(),
            ask_question_config: Some(pb::RunInferenceAskQuestionConfig {
                tool_name: "AskQuestion".into(),
                ..Default::default()
            }),
            feature_flags: local_feature_flags(),
            subagent_model_config: Some(subagent_model_config(model_id, models, request, official)),
            ..Default::default()
        }),
        accepts_aborted_finish: true,
        accepts_subagent_model_resolution: true,
        ..Default::default()
    }
}

/// Advertises every configured BYOK model as a legal subagent model.
///
/// Cursor falls back to pinning subagents to the parent model (`force_policy =
/// "parent_pin"`) unless the run ships a subagent model catalog, which is why
/// Task calls collapsed to the parent model. `force_policy = "none"` keeps the
/// client's own resolution and the catalog is what the Task tool description
/// and the model argument are resolved against.
///
/// `parent_model` must be the model **slug** (`model_hash`), not the display
/// name: the client uses it as a key into `modelsBySlug`, so a display name
/// makes the parent model unresolvable and breaks Task calls that do not name
/// a model explicitly.
fn subagent_model_config(
    parent_model: &str,
    models: &[ModelConfig],
    request: &pb::RunInferenceRunRequest,
    official: &[OfficialSlug],
) -> pb::RunInferenceSubagentModelConfig {
    use pb::run_inference_subagent_model_override::Selection;
    let reserved = official_reserved(official);
    let mut policies = subagent_models::byok_policies(models, &reserved);
    let existing = policies
        .iter()
        .map(|policy| policy.slug.clone())
        .collect::<std::collections::HashSet<_>>();
    let byok_hashes = models
        .iter()
        .map(|model| model.model_hash.as_str())
        .collect::<std::collections::HashSet<_>>();
    for selected in &request.selected_subagent_models {
        let slug = selected.model_id.as_str();
        if slug.is_empty() || slug == "default" {
            continue;
        }
        if byok_hashes.contains(slug) {
            if let Some(policy) = policies.iter_mut().find(|policy| policy.slug == slug) {
                policy.advertised = false;
            }
            continue;
        }
        if existing.contains(slug) {
            continue;
        }
        policies.push(pb::RunInferenceSubagentModelPolicy {
            slug: slug.into(),
            advertised: true,
            valid: true,
            blocked: false,
            requires_max_mode: false,
            representative_cost_dollars: 0.0,
            legacy_aliases: Vec::new(),
            turned_off_by_user: false,
        });
    }
    for slug in official {
        if existing.contains(&slug.slug) || policies.iter().any(|policy| policy.slug == slug.slug) {
            continue;
        }
        policies.push(pb::RunInferenceSubagentModelPolicy {
            slug: slug.slug.clone(),
            advertised: slug.advertised,
            valid: slug.valid,
            blocked: slug.blocked,
            requires_max_mode: false,
            representative_cost_dollars: 0.0,
            legacy_aliases: slug.legacy_aliases.clone(),
            turned_off_by_user: false,
        });
    }
    let overrides = request
        .subagent_model_overrides
        .iter()
        .filter_map(|entry| {
            let (kind, model_id) = match entry.selection.as_ref()? {
                Selection::Model(m) if m.model_id == "default" => ("inherit", None),
                Selection::Model(m) if !m.model_id.is_empty() => {
                    ("model", Some(m.model_id.clone()))
                }
                Selection::Inherit(true) => ("inherit", None),
                Selection::Disabled(true) => ("disabled", None),
                _ => return None,
            };
            Some(pb::RunInferenceResolvedSubagentModelOverride {
                subagent_type: entry.subagent_type.clone(),
                kind: kind.into(),
                model_id,
            })
        })
        .collect();
    pb::RunInferenceSubagentModelConfig {
        force_policy: SUBAGENT_FORCE_POLICY_NONE.into(),
        force_model_id: None,
        parent_max_mode: false,
        models: policies,
        overrides,
        task_tool_omitted: false,
        parent_requested_model_name: parent_model.into(),
    }
}

/// Resolves the model slugs a client asks about before spawning subagents.
///
/// Only configured BYOK models can be resolved here; anything else is echoed
/// back untouched so the client keeps its own upstream resolution.
fn resolve_subagent_models(
    resolve: pb::RunInferenceResolveSubagentModels,
    models: &[ModelConfig],
    official: &[OfficialSlug],
) -> pb::RunInferenceSubagentModelsResolved {
    let mut byok = subagent_models::models_from_configs(models);
    subagent_models::assign_slugs(&mut byok, &official_reserved(official));
    let resolved = resolve
        .requested_models
        .iter()
        .map(|requested| {
            let hash = if byok.iter().any(|model| &model.hash == requested) {
                Some(requested.clone())
            } else if let Some(resolved) = subagent_models::resolve(&byok, requested)
                .ok()
                .map(str::to_owned)
                .filter(|hash| byok.iter().any(|model| &model.hash == hash))
            {
                Some(resolved)
            } else if is_official_slug(official, requested) && !is_byok_owned_slug(&byok, requested)
            {
                Some(requested.clone())
            } else {
                None
            };
            pb::RunInferenceResolvedSubagentModel {
                requested: requested.clone(),
                resolved_slug: hash,
                policy: None,
            }
        })
        .collect();
    pb::RunInferenceSubagentModelsResolved {
        resolution_id: resolve.resolution_id,
        models: resolved,
    }
}

async fn serve_invocation(
    provider: Arc<dyn Provider>,
    output: Arc<OutputHub>,
    invocations: Arc<Mutex<HashMap<String, CancellationToken>>>,
    cancellation: CancellationToken,
    invocation: ModelInvocation,
    context_window_tokens: Option<u64>,
) {
    let id = invocation.call_id.clone();
    let mut stream = provider.stream(invocation, cancellation);
    let mut tool_calls = HashMap::<usize, (String, String, String)>::new();
    let mut error = None;
    while let Some(next) = stream.next().await {
        match next {
            Ok(event) => {
                for response in model_responses(event, &mut tool_calls, context_window_tokens) {
                    emit(
                        &output,
                        pb::run_inference_server_message::Message::InvocationResponse(
                            pb::RunInferenceInvocationResponse {
                                invocation_id: id.clone(),
                                response: Some(response),
                            },
                        ),
                    );
                }
            }
            Err(value) => {
                error = Some(value.to_string());
                break;
            }
        }
    }
    invocations.lock().remove(&id);
    let message = pb::RunInferenceInvocationEnd {
        invocation_id: id,
        error: error.map(|message| pb::RunInferenceInvocationError {
            code: 13,
            message,
            ..Default::default()
        }),
        ..Default::default()
    };
    emit(
        &output,
        pb::run_inference_server_message::Message::InvocationEnd(message),
    );
}

fn model_responses(
    event: ModelEvent,
    tool_calls: &mut HashMap<usize, (String, String, String)>,
    context_window_tokens: Option<u64>,
) -> Vec<pb::InferenceStreamResponse> {
    use pb::inference_stream_response::Response;
    let response = match event {
        ModelEvent::Usage(usage) => {
            return usage_responses(usage, context_window_tokens);
        }
        ModelEvent::TextDelta(text) => Response::TextPart(pb::InferenceTextStreamPart {
            text,
            is_final: false,
        }),
        ModelEvent::TextEnd => Response::TextPart(pb::InferenceTextStreamPart {
            text: String::new(),
            is_final: true,
        }),
        ModelEvent::ThinkingDelta(text) => {
            Response::ThinkingPart(pb::InferenceThinkingStreamPart {
                text,
                is_final: false,
                ..Default::default()
            })
        }
        ModelEvent::ThinkingEnd => Response::ThinkingPart(pb::InferenceThinkingStreamPart {
            text: String::new(),
            is_final: true,
            ..Default::default()
        }),
        ModelEvent::ThinkingSignature(signature) => {
            Response::ThinkingPart(pb::InferenceThinkingStreamPart {
                text: String::new(),
                signature: Some(signature),
                is_final: true,
                ..Default::default()
            })
        }
        ModelEvent::RedactedThinking(data) => {
            Response::ThinkingPart(pb::InferenceThinkingStreamPart {
                text: String::new(),
                redacted_data: Some(data),
                is_final: true,
                ..Default::default()
            })
        }
        ModelEvent::ToolCallStart {
            index,
            call_id,
            name,
        } => {
            tool_calls.insert(index, (call_id.clone(), name.clone(), String::new()));
            Response::ToolCallPart(pb::InferenceToolCallStreamPart {
                tool_call_id: call_id,
                tool_name: name,
                args: String::new(),
                is_complete: false,
                tool_index: i32::try_from(index).ok(),
                ..Default::default()
            })
        }
        ModelEvent::ToolCallArgumentsDelta { index, delta } => {
            let Some((call_id, _, arguments)) = tool_calls.get_mut(&index) else {
                return Vec::new();
            };
            arguments.push_str(&delta);
            Response::ToolCallPart(pb::InferenceToolCallStreamPart {
                tool_call_id: call_id.clone(),
                tool_name: String::new(),
                args: delta,
                is_complete: false,
                tool_index: i32::try_from(index).ok(),
                ..Default::default()
            })
        }
        ModelEvent::ToolCallEnd { index } => {
            let Some((call_id, name, arguments)) = tool_calls.remove(&index) else {
                return Vec::new();
            };
            Response::ToolCallPart(pb::InferenceToolCallStreamPart {
                tool_call_id: call_id,
                tool_name: name,
                args: arguments,
                is_complete: true,
                tool_index: i32::try_from(index).ok(),
                ..Default::default()
            })
        }
        ModelEvent::Start { model_call_id } => {
            Response::InvocationId(pb::InferenceInvocationIdInfo {
                invocation_id: model_call_id,
            })
        }
        ModelEvent::TextStart
        | ModelEvent::ThinkingStart
        | ModelEvent::ProviderReplayState(_)
        | ModelEvent::Done(_) => return Vec::new(),
    };
    vec![pb::InferenceStreamResponse {
        response: Some(response),
        ..Default::default()
    }]
}

fn usage_responses(
    usage: Usage,
    context_window_tokens: Option<u64>,
) -> Vec<pb::InferenceStreamResponse> {
    use pb::inference_stream_response::Response;
    let prompt_tokens = usage.context_input_tokens.or(usage.input_tokens);
    let completion_tokens = usage.output_tokens;
    let total_tokens = usage
        .total_tokens
        .or_else(|| match (prompt_tokens, completion_tokens) {
            (Some(prompt), Some(completion)) => Some(prompt.saturating_add(completion)),
            (Some(prompt), None) => Some(prompt),
            (None, Some(completion)) => Some(completion),
            (None, None) => None,
        });
    vec![
        stream_response(Response::Usage(pb::InferenceUsageInfo {
            prompt_tokens: count(prompt_tokens),
            completion_tokens: count(completion_tokens),
            total_tokens: total_tokens.map(count_value),
        })),
        stream_response(Response::ExtendedUsage(pb::InferenceExtendedUsageInfo {
            input_tokens: count(usage.input_tokens),
            output_tokens: count(usage.output_tokens),
            cache_read_tokens: count(usage.cache_read_tokens),
            cache_write_tokens: count(usage.cache_write_tokens),
            max_tokens: count(context_window_tokens),
            reasoning_tokens: usage.reasoning_tokens.map(count_value),
        })),
    ]
}

fn stream_response(
    response: pb::inference_stream_response::Response,
) -> pb::InferenceStreamResponse {
    pb::InferenceStreamResponse {
        response: Some(response),
        ..Default::default()
    }
}

fn agent_mode_configs() -> Vec<pb::RunInferenceAgentModeConfig> {
    const MODES: [&str; 6] = ["agent", "chat", "plan", "debug", "triage", "multitask"];
    const SWITCH: [&str; 5] = ["agent", "chat", "plan", "debug", "multitask"];
    MODES
        .into_iter()
        .map(|mode| pb::RunInferenceAgentModeConfig {
            mode_id: mode.into(),
            enable_ask_question_tool: true,
            switch_mode_tool_config: (mode == "agent").then(|| {
                pb::RunInferenceSwitchModeToolConfig {
                    target_modes: SWITCH.into_iter().map(str::to_owned).collect(),
                }
            }),
        })
        .collect()
}

fn count(value: Option<u64>) -> i32 {
    count_value(value.unwrap_or_default())
}

fn count_value(value: u64) -> i32 {
    value.min(i32::MAX as u64) as i32
}

fn invocation_error(output: &OutputHub, id: String, message: &str) {
    emit(
        output,
        pb::run_inference_server_message::Message::InvocationEnd(pb::RunInferenceInvocationEnd {
            invocation_id: id,
            error: Some(pb::RunInferenceInvocationError {
                code: 3,
                message: message.into(),
                ..Default::default()
            }),
            ..Default::default()
        }),
    );
}

fn emit(output: &OutputHub, message: pb::run_inference_server_message::Message) {
    let message = pb::RunInferenceServerMessage {
        message: Some(message),
    };
    if let Ok(frame) = connect::encode_message(&message) {
        output.emit(frame);
    }
}

pub(crate) fn decode_append(data: &[u8]) -> Result<pb::RunInferenceClientMessage> {
    pb::RunInferenceClientMessage::decode(data).map_err(Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        api::cursor::proxy::CursorProxy,
        cursor::{
            prompting::{PromptAssets, PromptCompiler},
            transport::TransportRegistry,
        },
        model::{ModelConfigInput, OPENAI_CHAT_ENDPOINT},
        network::NetworkClients,
        provider::{Provider, ProviderStream},
        store::Store,
    };
    use axum::http::HeaderMap;
    use futures_util::stream;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    fn byok(hash: &str, model_id: &str, display_name: &str) -> ModelConfig {
        ModelConfig {
            model_hash: hash.into(),
            sort_order: 0,
            display_name: display_name.into(),
            group_name: None,
            model_type: ModelType::OpenAi,
            base_url: "http://127.0.0.1:15721".into(),
            use_full_url: false,
            api_key: "PROXY_MANAGED".into(),
            tooltip_data: "0.11x".into(),
            model_id: model_id.into(),
            reasoning_effort: None,
            openai_endpoint: "/v1/responses".into(),
            openai_extra_params_enabled: false,
            openai_extra_params: serde_json::json!({}),
            custom_headers_enabled: false,
            custom_headers: serde_json::json!({}),
            anthropic_extra_params_enabled: false,
            anthropic_extra_params: serde_json::json!({}),
            context_window_tokens: None,
            max_completion_tokens: None,
            anthropic_max_tokens: None,
            anthropic_thinking_effort: None,
            thinking_budget_tokens: None,
            created_at_ms: 0,
            updated_at_ms: 0,
        }
    }

    fn request(resolution_id: &str, requested: &[&str]) -> pb::RunInferenceResolveSubagentModels {
        pb::RunInferenceResolveSubagentModels {
            resolution_id: resolution_id.into(),
            requested_models: requested.iter().map(|value| (*value).to_owned()).collect(),
        }
    }

    #[test]
    fn resolves_hashes_ids_and_display_names() {
        let models = [
            byok(
                "8c1030decdf858d7",
                "ttsw-codex-deepseek-v4.1-flash",
                "Deepseek V4.1 Flash (TT Switch)",
            ),
            byok(
                "0b9f3dcf3d1bcef5",
                "ttsw-codex-claude-sonnet-5.5",
                "Claude 5.5 Sonnet (TT Switch)",
            ),
        ];
        let resolved = resolve_subagent_models(
            request(
                "resolution-1",
                &[
                    "8c1030decdf858d7",
                    "ttsw-codex-claude-sonnet-5.5",
                    "Claude 5.5 Sonnet (TT Switch)",
                    "Claude 5.5 Sonnet",
                    "gpt-5.6-sol-medium",
                ],
            ),
            &models,
            &[],
        );
        assert_eq!(resolved.resolution_id, "resolution-1");
        let slugs = resolved
            .models
            .iter()
            .map(|model| (model.requested.as_str(), model.resolved_slug.as_deref()))
            .collect::<Vec<_>>();
        assert_eq!(
            slugs,
            vec![
                ("8c1030decdf858d7", Some("8c1030decdf858d7")),
                ("ttsw-codex-claude-sonnet-5.5", Some("0b9f3dcf3d1bcef5")),
                ("Claude 5.5 Sonnet (TT Switch)", Some("0b9f3dcf3d1bcef5")),
                ("Claude 5.5 Sonnet", Some("0b9f3dcf3d1bcef5")),
                ("gpt-5.6-sol-medium", None),
            ]
        );
    }

    #[test]
    fn keeps_one_entry_per_requested_model_without_models() {
        let resolved = resolve_subagent_models(
            request("resolution-2", &["inherit", "8c1030decdf858d7"]),
            &[],
            &[],
        );
        assert_eq!(resolved.resolution_id, "resolution-2");
        assert_eq!(resolved.models.len(), 2);
        assert!(resolved
            .models
            .iter()
            .all(|model| model.resolved_slug.is_none()));
    }

    #[test]
    fn echoes_unknown_frames_with_their_own_resolution_id() {
        let models = [byok(
            "0b9f3dcf3d1bcef5",
            "ttsw-codex-claude-sonnet-5.5",
            "Claude 5.5 Sonnet (TT Switch)",
        )];
        let resolved = resolve_subagent_models(request("", &[]), &models, &[]);
        assert!(resolved.resolution_id.is_empty());
        assert!(resolved.models.is_empty());
    }

    #[test]
    fn resolves_official_slug_when_catalog_contains_it() {
        let official = [OfficialSlug {
            slug: "gpt-5.6-sol-medium".into(),
            advertised: true,
            valid: true,
            blocked: false,
            legacy_aliases: vec![],
        }];
        let resolved = resolve_subagent_models(
            request("resolution-official", &["gpt-5.6-sol-medium"]),
            &[],
            &official,
        );
        assert_eq!(
            resolved.models[0].resolved_slug.as_deref(),
            Some("gpt-5.6-sol-medium")
        );
    }

    #[test]
    fn run_ready_advertises_every_byok_model_without_pinning() {
        let models = [
            byok(
                "8c1030decdf858d7",
                "ttsw-codex-deepseek-v4.1-flash",
                "Deepseek V4.1 Flash (TT Switch)",
            ),
            byok(
                "0b9f3dcf3d1bcef5",
                "ttsw-codex-claude-sonnet-5.5",
                "Claude 5.5 Sonnet (TT Switch)",
            ),
        ];
        let local = LocalModel {
            display_name: "Deepseek V4.1 Flash (TT Switch)".into(),
            model_type: ModelType::OpenAi,
            context_window_tokens: None,
        };
        let ready = run_ready(
            &Default::default(),
            Some(&local),
            &models,
            "8c1030decdf858d7",
            &[],
        );
        assert!(ready.routed_model_display_name.is_none());
        let metadata = ready
            .prompt_model_metadata
            .expect("prompt metadata is required");
        assert!(metadata.enable_line_numbers);
        assert_eq!(
            metadata
                .feature_flags
                .get("dynamicToolProfile")
                .and_then(|flag| flag.value.as_ref()),
            Some(&pb::run_inference_flag_value::Value::StringValue(
                "final".into()
            ))
        );
        let config = metadata
            .subagent_model_config
            .expect("subagent model config must be advertised");
        assert_eq!(config.force_policy, SUBAGENT_FORCE_POLICY_NONE);
        assert!(config.force_model_id.is_none());
        assert_eq!(config.parent_requested_model_name, "8c1030decdf858d7");
        assert!(!config.task_tool_omitted);
        assert!(
            config
                .models
                .iter()
                .any(|model| model.slug == config.parent_requested_model_name),
            "the parent slug must be part of the catalog so unnamed Task calls resolve"
        );
        let parent_hash = config
            .models
            .iter()
            .find(|model| model.slug == config.parent_requested_model_name)
            .expect("parent hash entry");
        assert!(!parent_hash.advertised);
        assert!(parent_hash.valid);
        let advertised = config
            .models
            .iter()
            .filter(|model| model.advertised)
            .map(|model| model.slug.as_str())
            .collect::<std::collections::HashSet<_>>();
        assert!(advertised.contains("deepseek-v4.1-flash-tt-switch"));
        assert!(advertised.contains("claude-5.5-sonnet-tt-switch"));
        assert!(config
            .models
            .iter()
            .all(|model| !model.blocked && !model.turned_off_by_user && !model.requires_max_mode));
    }

    #[test]
    fn run_ready_still_advertises_an_empty_catalog_without_models() {
        let local = LocalModel {
            display_name: "GPT 6.1 Sol (TT Switch)".into(),
            model_type: ModelType::OpenAi,
            context_window_tokens: None,
        };
        let ready = run_ready(
            &Default::default(),
            Some(&local),
            &[],
            "4c3368f26587305a",
            &[],
        );
        assert!(ready.routed_model_display_name.is_none());
        let config = ready
            .prompt_model_metadata
            .expect("prompt metadata is required")
            .subagent_model_config
            .expect("config is advertised even with no BYOK models");
        assert_eq!(config.force_policy, SUBAGENT_FORCE_POLICY_NONE);
        assert!(config.models.is_empty());
        assert_eq!(config.parent_requested_model_name, "4c3368f26587305a");
    }

    #[test]
    fn subagent_catalog_excludes_display_names_from_parent_lookup() {
        let models = [byok(
            "0b9f3dcf3d1bcef5",
            "ttsw-codex-claude-sonnet-5.5",
            "Claude 5.5 Sonnet (TT Switch)",
        )];
        let config = subagent_model_config("8c1030decdf858d7", &models, &Default::default(), &[]);
        assert_eq!(config.parent_requested_model_name, "8c1030decdf858d7");
        assert!(
            !config
                .models
                .iter()
                .any(|model| model.slug == "Deepseek V4.1 Flash (TT Switch)"),
            "display names must never be used as slugs"
        );
        assert!(config
            .models
            .iter()
            .any(|model| { model.slug == "claude-5.5-sonnet-tt-switch" && model.advertised }));
    }

    #[test]
    fn resolves_hash6_collision_slug_even_when_catalog_lists_it() {
        let models = [byok(
            "8c1030decdf858d7",
            "ttsw-codex-deepseek-v4.1-flash",
            "Deepseek V4.1 Flash (TT Switch)",
        )];
        let official = [OfficialSlug {
            slug: "deepseek-v4.1-flash-tt-switch-8c1030".into(),
            advertised: true,
            valid: true,
            blocked: false,
            legacy_aliases: vec![],
        }];
        let resolved = resolve_subagent_models(
            request(
                "resolution-hash6",
                &[
                    "deepseek-v4.1-flash-tt-switch",
                    "deepseek-v4.1-flash-tt-switch-8c1030",
                ],
            ),
            &models,
            &official,
        );
        assert_eq!(
            resolved.models[0].resolved_slug.as_deref(),
            Some("8c1030decdf858d7")
        );
        assert_eq!(
            resolved.models[1].resolved_slug.as_deref(),
            Some("8c1030decdf858d7")
        );
    }

    #[test]
    fn true_official_collision_keeps_clean_slug_official() {
        let models = [byok(
            "8c1030decdf858d7",
            "ttsw-codex-deepseek-v4.1-flash",
            "Deepseek V4.1 Flash (TT Switch)",
        )];
        let official = [OfficialSlug {
            slug: "deepseek-v4.1-flash-tt-switch".into(),
            advertised: true,
            valid: true,
            blocked: false,
            legacy_aliases: vec![],
        }];
        let resolved = resolve_subagent_models(
            request(
                "resolution-true-collision",
                &[
                    "deepseek-v4.1-flash-tt-switch",
                    "deepseek-v4.1-flash-tt-switch-8c1030",
                ],
            ),
            &models,
            &official,
        );
        assert_eq!(
            resolved.models[0].resolved_slug.as_deref(),
            Some("deepseek-v4.1-flash-tt-switch")
        );
        assert_eq!(
            resolved.models[1].resolved_slug.as_deref(),
            Some("8c1030decdf858d7")
        );
    }

    #[derive(Default)]
    struct SilentProvider;

    impl Provider for SilentProvider {
        fn stream(
            &self,
            _invocation: ModelInvocation,
            _cancellation: CancellationToken,
        ) -> ProviderStream {
            let events: Vec<crate::Result<ModelEvent>> = Vec::new();
            Box::pin(stream::iter(events))
        }
    }

    fn deepseek_input() -> ModelConfigInput {
        ModelConfigInput {
            sort_order: 1,
            display_name: "Deepseek V4.1 Flash (TT Switch)".into(),
            group_name: None,
            model_type: ModelType::OpenAi,
            base_url: "http://127.0.0.1:15721".into(),
            use_full_url: false,
            api_key: "PROXY_MANAGED".into(),
            tooltip_data: "0.11x".into(),
            model_id: "ttsw-codex-deepseek-v4.1-flash".into(),
            reasoning_effort: None,
            openai_endpoint: OPENAI_CHAT_ENDPOINT.into(),
            openai_extra_params_enabled: false,
            openai_extra_params: serde_json::json!({}),
            custom_headers_enabled: false,
            custom_headers: serde_json::json!({}),
            anthropic_extra_params_enabled: false,
            anthropic_extra_params: serde_json::json!({}),
            context_window_tokens: Some(128_000),
            max_completion_tokens: None,
            anthropic_max_tokens: None,
            anthropic_thinking_effort: None,
            thinking_budget_tokens: None,
        }
    }

    async fn inference_with_deepseek() -> (tempfile::TempDir, TransportRegistry, ModelConfig) {
        let directory = tempfile::tempdir().unwrap();
        let url = format!("sqlite://{}", directory.path().join("test.db").display());
        let store = Store::connect(&url).await.unwrap();
        let model = store.create_model(&deepseek_input()).await.unwrap();
        let assets = PromptAssets::load(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("prompt/cursor")
                .as_path(),
        )
        .unwrap();
        let registry =
            TransportRegistry::new(store, Arc::new(SilentProvider), PromptCompiler::new(assets));
        (directory, registry, model)
    }

    fn run_request(model_id: &str) -> pb::RunInferenceClientMessage {
        pb::RunInferenceClientMessage {
            message: Some(pb::run_inference_client_message::Message::RunRequest(
                pb::RunInferenceRunRequest {
                    conversation_id: "conversation-hash6".into(),
                    requested_model: Some(pb::InferenceRequestedModel {
                        model_id: model_id.into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )),
        }
    }

    fn assert_local(route: &InferenceRoute) {
        assert!(
            matches!(route, InferenceRoute::Local(_)),
            "expected BYOK Local route, got Upstream"
        );
    }

    #[tokio::test]
    async fn begin_routes_clean_and_hash6_byok_slugs_to_local() {
        let (_directory, registry, model) = inference_with_deepseek().await;
        let token = [0u8; 32];
        let catalog = OfficialCatalog::new();
        catalog.remember_run_ready(
            token,
            vec![
                OfficialSlug {
                    slug: "composer-2".into(),
                    advertised: true,
                    valid: true,
                    blocked: false,
                    legacy_aliases: vec![],
                },
                OfficialSlug {
                    slug: "grok-4.6".into(),
                    advertised: true,
                    valid: true,
                    blocked: false,
                    legacy_aliases: vec![],
                },
            ],
        );
        let proxy = CursorProxy::cursor(NetworkClients::new(registry.store().clone()));
        let service = InferenceService::new(registry.clone(), catalog, proxy);

        let clean = "deepseek-v4.1-flash-tt-switch";
        let hash6 = format!("{clean}-{}", &model.model_hash[..6]);
        let headers = HeaderMap::new();

        let clean_route = service
            .begin("req-clean", &run_request(clean), &headers)
            .await
            .unwrap();
        assert_local(&clean_route);

        let hash6_route = service
            .begin("req-hash6", &run_request(&hash6), &headers)
            .await
            .unwrap();
        assert_local(&hash6_route);

        let polluted = OfficialCatalog::new();
        polluted.remember_run_ready(
            token,
            vec![OfficialSlug {
                slug: hash6.clone(),
                advertised: true,
                valid: true,
                blocked: false,
                legacy_aliases: vec![],
            }],
        );
        assert!(polluted.is_official_slug(token, &hash6));
        let proxy = CursorProxy::cursor(NetworkClients::new(registry.store().clone()));
        let polluted_service = InferenceService::new(registry, polluted, proxy);
        let polluted_route = polluted_service
            .begin("req-polluted-hash6", &run_request(&hash6), &headers)
            .await
            .unwrap();
        assert_local(&polluted_route);
    }
}
