//! Owns one client-loop inference stream per Cursor request ID.
use std::{collections::HashMap, sync::Arc};

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
    cursor::{
        protocol::{connect, proto::inference as pb},
        tools::subagent_models::{self, ByokModel},
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
}

impl InferenceService {
    pub fn new(registry: TransportRegistry) -> Self {
        Self {
            state: Arc::new(Mutex::new(HashMap::new())),
            changed: Arc::new(Notify::new()),
            registry,
        }
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
        let configured = self.registry.store().model(model_id).await?;
        // A Task call may name a BYOK model by display name; resolve it to the
        // hash so the run is billed and routed to the model the caller meant.
        let (model_id, configured) = match configured {
            Some(config) => (model_id.to_string(), Some(config)),
            None => match self.resolve_by_display_name(model_id).await? {
                Some(config) => {
                    tracing::info!(
                        request_id,
                        requested = model_id,
                        resolved = %config.model_hash,
                        "resolved BYOK subagent model by display name"
                    );
                    (config.model_hash.clone(), Some(config))
                }
                None => (model_id.to_string(), None),
            },
        };
        let model_id = model_id.as_str();
        let local = configured.is_some() || model_id.starts_with(crate::plugin::ADAPTER_ID_PREFIX);
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

    pub fn finish(&self, request_id: &str) {
        self.state.lock().remove(request_id);
    }

    /// Resolves a Task `model` argument that names a model by display name.
    async fn resolve_by_display_name(&self, requested: &str) -> Result<Option<ModelConfig>> {
        let models = self.registry.store().models().await?;
        let byok = models
            .iter()
            .map(|model| ByokModel {
                hash: model.model_hash.clone(),
                model_id: model.model_id.clone(),
                display_name: model.display_name.clone(),
            })
            .collect::<Vec<_>>();
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

    pub fn subscribe(&self) -> mpsc::UnboundedReceiver<Bytes> {
        self.output.subscribe()
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
                let ready = run_ready(request.requested_model, model.as_ref(), &models, &model_id);
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
                let resolved = resolve_subagent_models(resolve, &models);
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

fn run_ready(
    requested_model: Option<pb::InferenceRequestedModel>,
    model: Option<&LocalModel>,
    models: &[ModelConfig],
    model_id: &str,
) -> pb::RunInferenceRunReady {
    let (display_name, model_type) = model
        .map(|model| (model.display_name.clone(), model.model_type))
        .unwrap_or_else(|| (String::new(), ModelType::OpenAi));
    let anthropic = model_type == ModelType::Anthropic;
    let name = display_name.to_ascii_lowercase();
    let claude = anthropic && name.contains("claude");
    let gpt5 = !anthropic && name.starts_with("gpt-5");
    pb::RunInferenceRunReady {
        resolved_model: requested_model,
        routed_model_display_name: (!display_name.is_empty()).then_some(display_name.clone()),
        prompt_model_metadata: Some(pb::RunInferencePromptModelMetadata {
            vendor: if anthropic { "anthropic" } else { "openai" }.into(),
            prompt_version: "latest".into(),
            is_claude_4x: claude,
            is_sonnet4: claude && name.contains("sonnet"),
            is_gpt5_family: gpt5,
            is_gpt5: gpt5,
            persona: "auto".into(),
            self_identity_name: "Cursor".into(),
            agent_mode_configs: agent_mode_configs(),
            ask_question_config: Some(pb::RunInferenceAskQuestionConfig {
                tool_name: "AskQuestion".into(),
                ..Default::default()
            }),
            subagent_model_config: Some(subagent_model_config(model_id, models)),
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
) -> pb::RunInferenceSubagentModelConfig {
    pb::RunInferenceSubagentModelConfig {
        force_policy: SUBAGENT_FORCE_POLICY_NONE.into(),
        force_model_id: None,
        parent_max_mode: false,
        models: models
            .iter()
            .map(|model| pb::RunInferenceSubagentModelPolicy {
                slug: model.model_hash.clone(),
                advertised: true,
                valid: true,
                blocked: false,
                requires_max_mode: false,
                representative_cost_dollars: 0.0,
                legacy_aliases: Vec::new(),
                turned_off_by_user: false,
            })
            .collect(),
        overrides: Vec::new(),
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
) -> pb::RunInferenceSubagentModelsResolved {
    let byok = models
        .iter()
        .map(|model| ByokModel {
            hash: model.model_hash.clone(),
            model_id: model.model_id.clone(),
            display_name: model.display_name.clone(),
        })
        .collect::<Vec<_>>();
    let resolved = resolve
        .requested_models
        .iter()
        .map(|requested| {
            let hash = if byok.iter().any(|model| &model.hash == requested) {
                Some(requested.clone())
            } else {
                subagent_models::resolve(&byok, requested)
                    .ok()
                    .map(str::to_owned)
                    .filter(|hash| byok.iter().any(|model| &model.hash == hash))
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
        let resolved = resolve_subagent_models(request("", &[]), &models);
        assert!(resolved.resolution_id.is_empty());
        assert!(resolved.models.is_empty());
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
        let ready = run_ready(None, Some(&local), &models, "8c1030decdf858d7");
        let config = ready
            .prompt_model_metadata
            .expect("prompt metadata is required")
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
        let slugs = config
            .models
            .iter()
            .map(|model| (model.slug.as_str(), model.advertised, model.valid))
            .collect::<Vec<_>>();
        assert_eq!(
            slugs,
            vec![
                ("8c1030decdf858d7", true, true),
                ("0b9f3dcf3d1bcef5", true, true),
            ]
        );
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
        let ready = run_ready(None, Some(&local), &[], "4c3368f26587305a");
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
        let config = subagent_model_config("8c1030decdf858d7", &models);
        assert_eq!(config.parent_requested_model_name, "8c1030decdf858d7");
        assert!(
            !config
                .models
                .iter()
                .any(|model| model.slug == "Deepseek V4.1 Flash (TT Switch)"),
            "display names must never be used as slugs"
        );
    }
}
