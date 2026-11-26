//! Rewrites upstream RunInferenceSSE Connect frames for BYOK subagent injection.

use bytes::{BufMut, Bytes, BytesMut};
use flate2::{write::GzEncoder, Compression};
use std::io::Write;
use tracing::{info, warn};

use super::upstream_wire::{
    contains_run_ready_field, patch_run_inference_server_message, ServerPatch,
};

use crate::{
    cursor::{
        protocol::{connect, proto::inference as pb},
        services::official_catalog::{OfficialCatalog, TokenKey},
    },
    Result,
};

pub struct UpstreamRewrite {
    decoder: connect::FrameDecoder,
    byok: Vec<pb::RunInferenceSubagentModelPolicy>,
    catalog: OfficialCatalog,
    token: TokenKey,
    ready_rewritten: bool,
}

impl UpstreamRewrite {
    pub fn new(
        byok: Vec<pb::RunInferenceSubagentModelPolicy>,
        catalog: OfficialCatalog,
        token: TokenKey,
    ) -> Self {
        Self {
            decoder: connect::FrameDecoder::new(),
            byok,
            catalog,
            token,
            ready_rewritten: false,
        }
    }

    pub fn ingest(&mut self, chunk: Bytes) -> Result<Vec<Bytes>> {
        let mut out = Vec::new();
        for (flags, payload, envelope) in self.decoder.push(chunk)? {
            if flags & connect::END_STREAM_FLAG != 0 {
                out.push(envelope);
                continue;
            }
            if self.ready_rewritten {
                out.push(envelope);
                continue;
            }
            let raw = match connect::decompress_payload(flags, payload) {
                Ok(value) => value,
                Err(error) => {
                    warn!(%error, "upstream_sse: decompress failed, forwarding frame unchanged");
                    out.push(envelope);
                    continue;
                }
            };
            let had_run_ready = contains_run_ready_field(raw.as_ref());
            match patch_run_inference_server_message(raw.as_ref(), &self.byok) {
                Ok(ServerPatch::Unchanged) => {
                    out.push(envelope);
                }
                Ok(ServerPatch::Patched(patched)) => {
                    self.harvest_from_config(&patched.raw_config);
                    info!(
                        appended_byok = patched.appended_byok,
                        harvested_official = patched.raw_config.models.len(),
                        config_fingerprint = %patched.fingerprint,
                        "upstream_sse: patched RunReady with BYOK subagent models"
                    );
                    let body = compress_if_needed(flags, &patched.body)?;
                    out.push(encode_connect_frame(flags, body));
                    self.ready_rewritten = true;
                }
                Err(error) => {
                    if had_run_ready {
                        warn!(
                            %error,
                            "upstream_sse: RunReady wire patch failed, withholding frame"
                        );
                    } else {
                        warn!(
                            %error,
                            "upstream_sse: protobuf wire scan failed on non-RunReady frame, forwarding unchanged"
                        );
                        out.push(envelope);
                    }
                }
            }
        }
        Ok(out)
    }

    fn harvest_from_config(&self, config: &pb::RunInferenceSubagentModelConfig) {
        let mut harvested = config.clone();
        harvested
            .models
            .retain(|model| !is_known_byok_identity(&self.byok, &model.slug));
        let ready = pb::RunInferenceRunReady {
            prompt_model_metadata: Some(pb::RunInferencePromptModelMetadata {
                subagent_model_config: Some(harvested),
                ..Default::default()
            }),
            ..Default::default()
        };
        self.catalog.harvest_run_ready(self.token, &ready);
    }
}

fn is_known_byok_identity(byok: &[pb::RunInferenceSubagentModelPolicy], slug: &str) -> bool {
    byok.iter().any(|policy| {
        policy.slug == slug
            || policy.legacy_aliases.iter().any(|alias| alias == slug)
            || is_hash6_suffix_of(slug, &policy.slug)
    })
}

fn looks_like_model_hash(value: &str) -> bool {
    value.len() == 16 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_hash6_suffix_of(slug: &str, hash: &str) -> bool {
    looks_like_model_hash(hash)
        && slug
            .rsplit_once('-')
            .is_some_and(|(_, suffix)| suffix == &hash[..hash.len().min(6)])
}

pub fn prepare_upstream_headers(headers: &mut axum::http::HeaderMap) {
    headers.insert(
        "connect-accept-encoding",
        axum::http::HeaderValue::from_static("identity"),
    );
    headers.insert(
        axum::http::header::ACCEPT_ENCODING,
        axum::http::HeaderValue::from_static("identity"),
    );
}

fn encode_connect_frame(flags: u8, payload: Bytes) -> Bytes {
    let mut output = BytesMut::with_capacity(5 + payload.len());
    output.put_u8(flags);
    output.put_u32(payload.len() as u32);
    output.extend_from_slice(&payload);
    output.freeze()
}

fn compress_if_needed(flags: u8, body: &[u8]) -> Result<Bytes> {
    if flags & connect::COMPRESSION_FLAG == 0 {
        return Ok(Bytes::copy_from_slice(body));
    }
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(body)
        .map_err(|error| crate::Error::Protocol(format!("gzip compress failed: {error}")))?;
    Ok(Bytes::from(encoder.finish().map_err(|error| {
        crate::Error::Protocol(format!("gzip compress finish failed: {error}"))
    })?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::inference::upstream_wire::{
        append_length_delimited_field, byok_digest, decode_varint, encode_length_delimited_field,
        encode_varint, salt_config_fingerprint, scan_fields, PROMPT_METADATA_TAG_SUBAGENT_CONFIG,
        RUN_READY_TAG_CONFIG_FINGERPRINT, RUN_READY_TAG_PROMPT_METADATA, SERVER_TAG_RUN_READY,
        WIRE_LEN,
    };
    use crate::cursor::tools::subagent_models::byok_policies;
    use crate::model::{ModelConfig, ModelType};
    use bytes::BufMut;
    use flate2::{write::GzEncoder, Compression};
    use prost::Message;
    use std::collections::HashSet;
    use std::io::Write;

    fn ready_message() -> pb::RunInferenceServerMessage {
        pb::RunInferenceServerMessage {
            message: Some(pb::run_inference_server_message::Message::RunReady(
                pb::RunInferenceRunReady {
                    prompt_model_metadata: Some(pb::RunInferencePromptModelMetadata {
                        subagent_model_config: Some(pb::RunInferenceSubagentModelConfig {
                            force_policy: "none".into(),
                            models: vec![pb::RunInferenceSubagentModelPolicy {
                                slug: "official".into(),
                                advertised: true,
                                valid: true,
                                ..Default::default()
                            }],
                            ..Default::default()
                        }),
                        ..Default::default()
                    }),
                    config_fingerprint: b"upstream-fp".to_vec(),
                    ..Default::default()
                },
            )),
        }
    }

    fn gzip_frame(message: &pb::RunInferenceServerMessage) -> Bytes {
        let raw = connect::encode_message(message).unwrap();
        let payload = raw.slice(5..);
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&payload).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut out = bytes::BytesMut::with_capacity(5 + compressed.len());
        out.put_u8(connect::COMPRESSION_FLAG);
        out.put_u32(compressed.len() as u32);
        out.extend_from_slice(&compressed);
        out.freeze()
    }

    fn grok_like_undecodable_server_message() -> Bytes {
        let baseline = ready_message();
        let outer = connect::encode_message(&baseline).unwrap();
        let payload = outer.slice(5..);
        let fields = scan_fields(&payload).unwrap();
        let run_ready = fields
            .iter()
            .find(|field| field.number == SERVER_TAG_RUN_READY)
            .unwrap()
            .value;
        let run_fields = scan_fields(run_ready).unwrap();
        let metadata = run_fields
            .iter()
            .find(|field| field.number == RUN_READY_TAG_PROMPT_METADATA)
            .unwrap()
            .value;
        let mut poisoned_metadata = metadata.to_vec();
        append_length_delimited_field(&mut poisoned_metadata, 999, b"unknown-upstream-chunk");
        // Field 3 is proto `bool is_sonnet45` (varint); length-delimited bytes conflict.
        append_length_delimited_field(&mut poisoned_metadata, 3, b"wire-type-conflict");

        let mut poisoned_run_ready = Vec::new();
        for field in run_fields {
            if field.number == RUN_READY_TAG_PROMPT_METADATA {
                poisoned_run_ready.extend(encode_length_delimited_field(
                    RUN_READY_TAG_PROMPT_METADATA,
                    &poisoned_metadata,
                ));
            } else {
                poisoned_run_ready.extend_from_slice(field.raw);
            }
        }
        append_length_delimited_field(&mut poisoned_run_ready, 999, b"run-ready-unknown");
        // Field 2 is proto `bool supports_self_summary` (varint); length-delimited bytes conflict.
        append_length_delimited_field(&mut poisoned_run_ready, 2, b"bool-wire-conflict");

        let mut poisoned_outer = Vec::new();
        for field in fields {
            if field.number == SERVER_TAG_RUN_READY {
                poisoned_outer.extend(encode_length_delimited_field(
                    SERVER_TAG_RUN_READY,
                    &poisoned_run_ready,
                ));
            } else {
                poisoned_outer.extend_from_slice(field.raw);
            }
        }
        append_length_delimited_field(&mut poisoned_outer, 999, b"server-unknown");
        assert!(pb::RunInferenceServerMessage::decode(poisoned_outer.as_slice()).is_err());
        let mut frame = bytes::BytesMut::with_capacity(5 + poisoned_outer.len());
        frame.put_u8(0);
        frame.put_u32(poisoned_outer.len() as u32);
        frame.extend_from_slice(&poisoned_outer);
        frame.freeze()
    }

    fn slugs_from_frame(frame: &Bytes) -> Vec<String> {
        let flags = frame[0];
        let payload = frame.slice(5..);
        let decoded = connect::decompress_payload(flags, payload).unwrap();
        let fields = scan_fields(&decoded).unwrap();
        let run_ready = fields
            .iter()
            .find(|field| field.number == SERVER_TAG_RUN_READY)
            .unwrap()
            .value;
        let run_fields = scan_fields(run_ready).unwrap();
        let metadata = run_fields
            .iter()
            .find(|field| field.number == RUN_READY_TAG_PROMPT_METADATA)
            .unwrap()
            .value;
        let meta_fields = scan_fields(metadata).unwrap();
        let config = meta_fields
            .iter()
            .find(|field| field.number == PROMPT_METADATA_TAG_SUBAGENT_CONFIG)
            .unwrap()
            .value;
        pb::RunInferenceSubagentModelConfig::decode(config)
            .unwrap()
            .models
            .into_iter()
            .map(|model| model.slug)
            .collect()
    }

    fn fingerprint_from_frame(frame: &Bytes) -> Vec<u8> {
        let payload = connect::decompress_payload(frame[0], frame.slice(5..)).unwrap();
        let run_ready = scan_fields(&payload)
            .unwrap()
            .into_iter()
            .find(|field| field.number == SERVER_TAG_RUN_READY)
            .unwrap();
        scan_fields(run_ready.value)
            .unwrap()
            .into_iter()
            .find(|field| field.number == RUN_READY_TAG_CONFIG_FINGERPRINT)
            .map(|field| field.value.to_vec())
            .unwrap_or_default()
    }

    fn assert_non_rewritten_siblings_preserved(input_payload: &[u8], output_payload: &[u8]) {
        let input_fields = scan_fields(input_payload).unwrap();
        let output_fields = scan_fields(output_payload).unwrap();
        for field in &input_fields {
            if field.number == SERVER_TAG_RUN_READY && field.wire_type == WIRE_LEN {
                continue;
            }
            assert!(
                output_fields.iter().any(|out| out.raw == field.raw),
                "server-message sibling tag {} must be preserved byte-for-byte",
                field.number
            );
        }

        let input_run = scan_fields(
            input_fields
                .iter()
                .find(|field| field.number == SERVER_TAG_RUN_READY)
                .unwrap()
                .value,
        )
        .unwrap();
        let output_run = scan_fields(
            output_fields
                .iter()
                .find(|field| field.number == SERVER_TAG_RUN_READY)
                .unwrap()
                .value,
        )
        .unwrap();
        for field in &input_run {
            if field.number == RUN_READY_TAG_PROMPT_METADATA && field.wire_type == WIRE_LEN {
                continue;
            }
            if field.number == RUN_READY_TAG_CONFIG_FINGERPRINT && field.wire_type == WIRE_LEN {
                continue;
            }
            assert!(
                output_run.iter().any(|out| out.raw == field.raw),
                "run-ready sibling tag {} must be preserved byte-for-byte",
                field.number
            );
        }

        let input_metadata = scan_fields(
            input_run
                .iter()
                .find(|field| field.number == RUN_READY_TAG_PROMPT_METADATA)
                .unwrap()
                .value,
        )
        .unwrap();
        let output_metadata = scan_fields(
            output_run
                .iter()
                .find(|field| field.number == RUN_READY_TAG_PROMPT_METADATA)
                .unwrap()
                .value,
        )
        .unwrap();
        for field in &input_metadata {
            if field.number == PROMPT_METADATA_TAG_SUBAGENT_CONFIG && field.wire_type == WIRE_LEN {
                continue;
            }
            assert!(
                output_metadata.iter().any(|out| out.raw == field.raw),
                "prompt-metadata sibling tag {} must be preserved byte-for-byte",
                field.number
            );
        }
    }

    #[test]
    fn rewrites_plain_run_ready_and_appends_byok() {
        let frame = connect::encode_message(&ready_message()).unwrap();
        let byok = vec![pb::RunInferenceSubagentModelPolicy {
            slug: "byok-slug".into(),
            advertised: true,
            valid: true,
            ..Default::default()
        }];
        let catalog = OfficialCatalog::new();
        let mut rewrite = UpstreamRewrite::new(byok, catalog, [0u8; 32]);
        let first = rewrite.ingest(frame.clone()).unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(slugs_from_frame(&first[0]), vec!["official", "byok-slug"]);
        let unchanged = rewrite.ingest(frame).unwrap();
        assert_eq!(unchanged.len(), 1);
        assert_eq!(unchanged[0][0] & connect::COMPRESSION_FLAG, 0);
    }

    #[test]
    fn decompresses_gzip_run_ready() {
        let frame = gzip_frame(&ready_message());
        let byok = vec![pb::RunInferenceSubagentModelPolicy {
            slug: "byok".into(),
            advertised: true,
            valid: true,
            ..Default::default()
        }];
        let mut rewrite = UpstreamRewrite::new(byok, OfficialCatalog::new(), [0u8; 32]);
        let out = rewrite.ingest(frame).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0][0] & connect::COMPRESSION_FLAG,
            connect::COMPRESSION_FLAG
        );
        assert_eq!(slugs_from_frame(&out[0]), vec!["official", "byok"]);
    }

    #[test]
    fn patches_grok_like_run_ready_when_full_prost_decode_fails() {
        let frame = grok_like_undecodable_server_message();
        let byok = vec![
            pb::RunInferenceSubagentModelPolicy {
                slug: "composer-2.5-fast-tt-switch".into(),
                advertised: true,
                valid: true,
                ..Default::default()
            },
            pb::RunInferenceSubagentModelPolicy {
                slug: "deadbeef".into(),
                advertised: false,
                valid: true,
                ..Default::default()
            },
        ];
        let mut rewrite = UpstreamRewrite::new(byok, OfficialCatalog::new(), [0u8; 32]);
        let out = rewrite.ingest(frame.clone()).unwrap();
        assert_eq!(out.len(), 1, "expected patch instead of silent passthrough");
        assert_ne!(out[0], frame, "frame must be rewritten");

        let output_payload = connect::decompress_payload(out[0][0], out[0].slice(5..)).unwrap();
        let input_payload = connect::decompress_payload(frame[0], frame.slice(5..)).unwrap();
        assert_non_rewritten_siblings_preserved(&input_payload, &output_payload);

        let slugs = slugs_from_frame(&out[0]);
        assert!(slugs.contains(&"official".to_string()));
        assert!(slugs.contains(&"composer-2.5-fast-tt-switch".to_string()));
        assert!(slugs.contains(&"deadbeef".to_string()));
        assert!(fingerprint_from_frame(&out[0]).starts_with(b"upstream-fp+byok:"));
    }

    #[test]
    fn patches_gzip_grok_like_run_ready_when_full_prost_decode_fails() {
        let plain = grok_like_undecodable_server_message();
        let payload = plain.slice(5..);
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&payload).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut frame = bytes::BytesMut::with_capacity(5 + compressed.len());
        frame.put_u8(connect::COMPRESSION_FLAG);
        frame.put_u32(compressed.len() as u32);
        frame.extend_from_slice(&compressed);
        let frame = frame.freeze();

        let byok = vec![pb::RunInferenceSubagentModelPolicy {
            slug: "grok-4.7-tt-switch".into(),
            advertised: true,
            valid: true,
            ..Default::default()
        }];
        let mut rewrite = UpstreamRewrite::new(byok, OfficialCatalog::new(), [0u8; 32]);
        let out = rewrite.ingest(frame.clone()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(
            out[0][0] & connect::COMPRESSION_FLAG,
            connect::COMPRESSION_FLAG
        );
        assert!(slugs_from_frame(&out[0]).contains(&"grok-4.7-tt-switch".to_string()));

        let output_payload = connect::decompress_payload(out[0][0], out[0].slice(5..)).unwrap();
        let input_payload = connect::decompress_payload(frame[0], frame.slice(5..)).unwrap();
        assert_non_rewritten_siblings_preserved(&input_payload, &output_payload);
        assert!(fingerprint_from_frame(&out[0]).starts_with(b"upstream-fp+byok:"));
    }

    #[test]
    fn salts_config_fingerprint_when_byok_models_injected() {
        let byok = vec![pb::RunInferenceSubagentModelPolicy {
            slug: "byok-a".into(),
            advertised: true,
            valid: true,
            ..Default::default()
        }];
        let salted = salt_config_fingerprint(Some(b"abc"), &byok).unwrap();
        let digest = byok_digest(&byok);
        assert_eq!(salted, format!("abc+byok:{digest}").into_bytes());

        let mut rewrite = UpstreamRewrite::new(
            vec![
                pb::RunInferenceSubagentModelPolicy {
                    slug: "byok-a".into(),
                    advertised: true,
                    valid: true,
                    ..Default::default()
                },
                pb::RunInferenceSubagentModelPolicy {
                    slug: "byok-b".into(),
                    advertised: true,
                    valid: true,
                    ..Default::default()
                },
            ],
            OfficialCatalog::new(),
            [0u8; 32],
        );
        let frame = connect::encode_message(&ready_message()).unwrap();
        let out = rewrite.ingest(frame).unwrap();
        assert!(fingerprint_from_frame(&out[0]).starts_with(b"upstream-fp+byok:"));

        let digest_b = byok_digest(&[pb::RunInferenceSubagentModelPolicy {
            slug: "byok-b".into(),
            advertised: true,
            valid: true,
            ..Default::default()
        }]);
        assert_ne!(digest, digest_b);
    }

    #[test]
    fn leaves_config_fingerprint_unchanged_without_byok_models() {
        let unchanged = salt_config_fingerprint(Some(b"abc"), &[]).unwrap();
        assert_eq!(unchanged, b"abc");
        let frame = connect::encode_message(&ready_message()).unwrap();
        let mut rewrite = UpstreamRewrite::new(vec![], OfficialCatalog::new(), [0u8; 32]);
        let out = rewrite.ingest(frame.clone()).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], frame);
        assert_eq!(fingerprint_from_frame(&out[0]), b"upstream-fp");
    }

    #[test]
    fn proto_tags_match_generated_run_ready_wire() {
        let encoded = ready_message().encode_to_vec();
        let outer = scan_fields(&encoded).unwrap();
        assert_eq!(outer.len(), 1);
        assert_eq!(outer[0].number, SERVER_TAG_RUN_READY);
        assert_eq!(SERVER_TAG_RUN_READY, 2);

        let run_fields = scan_fields(outer[0].value).unwrap();
        assert_eq!(RUN_READY_TAG_PROMPT_METADATA, 4);
        assert_eq!(RUN_READY_TAG_CONFIG_FINGERPRINT, 17);
        assert!(run_fields
            .iter()
            .any(|field| field.number == RUN_READY_TAG_PROMPT_METADATA
                && field.wire_type == WIRE_LEN));
        assert!(run_fields.iter().any(|field| {
            field.number == RUN_READY_TAG_CONFIG_FINGERPRINT && field.wire_type == WIRE_LEN
        }));

        let metadata = run_fields
            .iter()
            .find(|field| field.number == RUN_READY_TAG_PROMPT_METADATA)
            .unwrap();
        let meta_fields = scan_fields(metadata.value).unwrap();
        assert_eq!(PROMPT_METADATA_TAG_SUBAGENT_CONFIG, 49);
        assert!(meta_fields.iter().any(|field| {
            field.number == PROMPT_METADATA_TAG_SUBAGENT_CONFIG && field.wire_type == WIRE_LEN
        }));
    }

    #[test]
    fn writes_byok_digest_when_original_fingerprint_is_empty() {
        let byok = vec![pb::RunInferenceSubagentModelPolicy {
            slug: "byok-empty-fp".into(),
            advertised: true,
            valid: true,
            ..Default::default()
        }];
        let digest = byok_digest(&byok);
        assert_eq!(
            salt_config_fingerprint(None, &byok).unwrap(),
            format!("byok:{digest}").into_bytes()
        );
        assert_eq!(
            salt_config_fingerprint(Some(b""), &byok).unwrap(),
            format!("byok:{digest}").into_bytes()
        );

        let mut ready = ready_message();
        if let Some(pb::run_inference_server_message::Message::RunReady(run_ready)) =
            ready.message.as_mut()
        {
            run_ready.config_fingerprint.clear();
        }
        let frame = connect::encode_message(&ready).unwrap();
        let mut rewrite = UpstreamRewrite::new(byok, OfficialCatalog::new(), [0u8; 32]);
        let out = rewrite.ingest(frame).unwrap();
        assert_eq!(
            fingerprint_from_frame(&out[0]),
            format!("byok:{digest}").into_bytes()
        );
    }

    #[test]
    fn sets_force_policy_none_when_upstream_omits_it() {
        let mut ready = ready_message();
        if let Some(pb::run_inference_server_message::Message::RunReady(run_ready)) =
            ready.message.as_mut()
        {
            if let Some(config) = ready_subagent_config_mut(run_ready) {
                config.force_policy.clear();
            }
        }
        let frame = connect::encode_message(&ready).unwrap();
        let mut rewrite = UpstreamRewrite::new(
            vec![pb::RunInferenceSubagentModelPolicy {
                slug: "byok-force".into(),
                advertised: true,
                valid: true,
                ..Default::default()
            }],
            OfficialCatalog::new(),
            [0u8; 32],
        );
        let out = rewrite.ingest(frame).unwrap();
        let payload = connect::decompress_payload(out[0][0], out[0].slice(5..)).unwrap();
        let run_ready = scan_fields(&payload)
            .unwrap()
            .into_iter()
            .find(|field| field.number == SERVER_TAG_RUN_READY)
            .unwrap();
        let metadata = scan_fields(run_ready.value)
            .unwrap()
            .into_iter()
            .find(|field| field.number == RUN_READY_TAG_PROMPT_METADATA)
            .unwrap();
        let config = scan_fields(metadata.value)
            .unwrap()
            .into_iter()
            .find(|field| field.number == PROMPT_METADATA_TAG_SUBAGENT_CONFIG)
            .unwrap();
        let decoded = pb::RunInferenceSubagentModelConfig::decode(config.value).unwrap();
        assert_eq!(decoded.force_policy, "none");
        assert!(decoded
            .models
            .iter()
            .any(|model| model.slug == "byok-force"));
    }

    fn ready_subagent_config_mut(
        ready: &mut pb::RunInferenceRunReady,
    ) -> Option<&mut pb::RunInferenceSubagentModelConfig> {
        ready
            .prompt_model_metadata
            .as_mut()
            .and_then(|metadata| metadata.subagent_model_config.as_mut())
    }

    const OFFICIAL_TEN: [&str; 10] = [
        "composer-1.5",
        "composer-2",
        "grok-4.6",
        "claude-4.5-sonnet-thinking",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gemini-3-flash",
        "o3",
        "claude-4.6-opus-high",
        "gpt-5.2",
    ];

    fn deepseek_byok() -> ModelConfig {
        ModelConfig {
            model_hash: "8c1030decdf858d7".into(),
            sort_order: 0,
            display_name: "Deepseek V4.1 Flash (TT Switch)".into(),
            group_name: None,
            model_type: ModelType::OpenAi,
            base_url: "http://127.0.0.1:15721".into(),
            use_full_url: false,
            api_key: "PROXY_MANAGED".into(),
            tooltip_data: "0.11x".into(),
            model_id: "ttsw-codex-deepseek-v4.1-flash".into(),
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

    fn official_ready_frame() -> Bytes {
        let mut ready = ready_message();
        if let Some(pb::run_inference_server_message::Message::RunReady(run_ready)) =
            ready.message.as_mut()
        {
            if let Some(config) = ready_subagent_config_mut(run_ready) {
                config.models = OFFICIAL_TEN
                    .iter()
                    .map(|slug| pb::RunInferenceSubagentModelPolicy {
                        slug: (*slug).into(),
                        advertised: true,
                        valid: true,
                        ..Default::default()
                    })
                    .collect();
            }
        }
        connect::encode_message(&ready).unwrap()
    }

    fn ingest_with_live_reserved(
        frame: &Bytes,
        catalog: &OfficialCatalog,
        token: TokenKey,
        models: &[ModelConfig],
    ) -> Bytes {
        let reserved = catalog.slug_set(token);
        let byok = byok_policies(models, &reserved);
        let mut rewrite = UpstreamRewrite::new(byok, catalog.clone(), token);
        let mut out = rewrite.ingest(frame.clone()).unwrap();
        assert_eq!(out.len(), 1);
        out.remove(0)
    }

    #[test]
    fn two_upstream_sse_rounds_keep_clean_byok_slug_and_stable_fingerprint() {
        let catalog = OfficialCatalog::new();
        let token = [0u8; 32];
        let models = [deepseek_byok()];
        let frame = official_ready_frame();

        let first = ingest_with_live_reserved(&frame, &catalog, token, &models);
        let second = ingest_with_live_reserved(&frame, &catalog, token, &models);

        let first_slugs = slugs_from_frame(&first);
        let second_slugs = slugs_from_frame(&second);
        assert_eq!(first_slugs, second_slugs);
        assert!(first_slugs.contains(&"deepseek-v4.1-flash-tt-switch".to_string()));
        assert!(!first_slugs.contains(&"deepseek-v4.1-flash-tt-switch-8c1030".to_string()));
        assert_eq!(
            fingerprint_from_frame(&first),
            fingerprint_from_frame(&second)
        );

        let harvested = catalog
            .get(token)
            .into_iter()
            .map(|slug| slug.slug)
            .collect::<HashSet<_>>();
        assert_eq!(harvested.len(), 10);
        for official in OFFICIAL_TEN {
            assert!(harvested.contains(official), "missing official {official}");
        }
        assert!(!harvested.contains("deepseek-v4.1-flash-tt-switch"));
        assert!(!harvested.contains("deepseek-v4.1-flash-tt-switch-8c1030"));
        assert!(!harvested.contains("8c1030decdf858d7"));
        assert!(harvested
            .iter()
            .all(|slug| !slug.ends_with("-8c1030") && slug != "8c1030decdf858d7"));
    }

    #[test]
    fn wire_helpers_round_trip_varint() {
        let encoded = encode_varint(300);
        let (value, end) = decode_varint(&encoded, 0).unwrap();
        assert_eq!(value, 300);
        assert_eq!(end, encoded.len());
    }
}
