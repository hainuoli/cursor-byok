//! Protobuf wire-level scan/patch helpers for upstream RunReady rewrite.

use prost::Message;
use sha2::{Digest, Sha256};

use crate::{cursor::protocol::proto::inference as pb, Error, Result};

pub(crate) const SERVER_TAG_RUN_READY: u32 = 2;
pub(crate) const RUN_READY_TAG_PROMPT_METADATA: u32 = 4;
pub(crate) const RUN_READY_TAG_CONFIG_FINGERPRINT: u32 = 17;
pub(crate) const PROMPT_METADATA_TAG_SUBAGENT_CONFIG: u32 = 49;

const WIRE_VARINT: u8 = 0;
const WIRE_64: u8 = 1;
pub(crate) const WIRE_LEN: u8 = 2;
const WIRE_32: u8 = 5;

#[derive(Debug, Clone, Copy)]
pub(crate) struct WireField<'a> {
    pub number: u32,
    pub wire_type: u8,
    pub raw: &'a [u8],
    pub value: &'a [u8],
}

pub(crate) fn decode_varint(buf: &[u8], mut pos: usize) -> Result<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    while pos < buf.len() {
        let byte = buf[pos];
        pos += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok((value, pos));
        }
        shift += 7;
        if shift > 64 {
            return Err(Error::Protocol("protobuf varint overflow".into()));
        }
    }
    Err(Error::Protocol("truncated protobuf varint".into()))
}

pub(crate) fn encode_varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if value == 0 {
            break;
        }
    }
    out
}

fn field_number_and_wire(tag: u64) -> (u32, u8) {
    ((tag >> 3) as u32, (tag & 7) as u8)
}

pub(crate) fn scan_fields(data: &[u8]) -> Result<Vec<WireField<'_>>> {
    let mut fields = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let field_start = pos;
        let (tag, next) = decode_varint(data, pos)?;
        pos = next;
        let (number, wire_type) = field_number_and_wire(tag);
        if wire_type > WIRE_32 || wire_type == 3 || wire_type == 4 {
            return Err(Error::Protocol(format!(
                "unsupported protobuf wire type {wire_type}"
            )));
        }
        let value_start = pos;
        pos = match wire_type {
            WIRE_VARINT => decode_varint(data, pos).map(|(_, end)| end)?,
            WIRE_64 => pos
                .checked_add(8)
                .ok_or_else(|| Error::Protocol("protobuf field overflow".into()))?,
            WIRE_LEN => {
                let (len, after_len) = decode_varint(data, pos)?;
                after_len
                    .checked_add(len as usize)
                    .ok_or_else(|| Error::Protocol("protobuf length overflow".into()))?
            }
            WIRE_32 => pos
                .checked_add(4)
                .ok_or_else(|| Error::Protocol("protobuf field overflow".into()))?,
            _ => unreachable!(),
        };
        if pos > data.len() {
            return Err(Error::Protocol("truncated protobuf field".into()));
        }
        let value = match wire_type {
            WIRE_LEN => {
                let (len, after_len) = decode_varint(data, value_start)?;
                &data[after_len..after_len + len as usize]
            }
            _ => &data[value_start..pos],
        };
        fields.push(WireField {
            number,
            wire_type,
            raw: &data[field_start..pos],
            value,
        });
    }
    Ok(fields)
}

pub(crate) fn encode_length_delimited_field(number: u32, value: &[u8]) -> Vec<u8> {
    let tag = (u64::from(number) << 3) | u64::from(WIRE_LEN);
    let mut out = encode_varint(tag);
    out.extend(encode_varint(value.len() as u64));
    out.extend_from_slice(value);
    out
}

pub(crate) fn encode_bytes_field(number: u32, value: &[u8]) -> Vec<u8> {
    encode_length_delimited_field(number, value)
}

pub(crate) fn merge_subagent_model_config_bytes(
    chunks: &[&[u8]],
) -> Result<pb::RunInferenceSubagentModelConfig> {
    let mut merged = pb::RunInferenceSubagentModelConfig::default();
    for chunk in chunks {
        merged
            .merge(*chunk)
            .map_err(|error| Error::Protocol(format!("subagent_model_config merge: {error}")))?;
    }
    Ok(merged)
}

pub(crate) fn byok_digest(byok: &[pb::RunInferenceSubagentModelPolicy]) -> String {
    if byok.is_empty() {
        return String::new();
    }
    let mut slugs = byok
        .iter()
        .map(|policy| policy.slug.as_str())
        .collect::<Vec<_>>();
    slugs.sort_unstable();
    let mut hasher = Sha256::new();
    for slug in slugs {
        hasher.update(slug.as_bytes());
        hasher.update([0]);
    }
    hex::encode(&hasher.finalize()[..8])
}

pub(crate) fn salt_config_fingerprint(
    original: Option<&[u8]>,
    byok: &[pb::RunInferenceSubagentModelPolicy],
) -> Option<Vec<u8>> {
    let digest = byok_digest(byok);
    if digest.is_empty() {
        return original.map(|value| value.to_vec());
    }
    let suffix = format!("+byok:{digest}");
    Some(match original {
        Some(value) if !value.is_empty() => {
            let mut text = String::from_utf8_lossy(value).into_owned();
            text.push_str(&suffix);
            text.into_bytes()
        }
        _ => format!("byok:{digest}").into_bytes(),
    })
}

fn fingerprint_label(value: Option<&[u8]>) -> String {
    value
        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
        .unwrap_or_default()
}

pub(crate) struct PatchedRunReady {
    pub body: Vec<u8>,
    pub raw_config: pb::RunInferenceSubagentModelConfig,
    pub fingerprint: String,
    pub appended_byok: usize,
}

pub(crate) fn patch_prompt_model_metadata(
    metadata: &[u8],
    byok: &[pb::RunInferenceSubagentModelPolicy],
) -> Result<(Vec<u8>, pb::RunInferenceSubagentModelConfig, usize)> {
    let fields = scan_fields(metadata)?;
    let mut tag49_chunks = Vec::new();
    let mut out = Vec::new();
    for field in &fields {
        if field.number == PROMPT_METADATA_TAG_SUBAGENT_CONFIG && field.wire_type == WIRE_LEN {
            tag49_chunks.push(field.value);
            continue;
        }
        out.extend_from_slice(field.raw);
    }
    let mut config = if tag49_chunks.is_empty() {
        pb::RunInferenceSubagentModelConfig::default()
    } else {
        merge_subagent_model_config_bytes(&tag49_chunks)?
    };
    let raw_config = config.clone();
    let before = config.models.len();
    merge_byok_subagent_config_into(&mut config, byok);
    let appended = config.models.len().saturating_sub(before);
    let encoded = config.encode_to_vec();
    if !encoded.is_empty() || !tag49_chunks.is_empty() || !byok.is_empty() {
        out.extend(encode_length_delimited_field(
            PROMPT_METADATA_TAG_SUBAGENT_CONFIG,
            &encoded,
        ));
    }
    Ok((out, raw_config, appended))
}

pub(crate) fn patch_run_ready_body(
    run_ready: &[u8],
    byok: &[pb::RunInferenceSubagentModelPolicy],
) -> Result<PatchedRunReady> {
    let fields = scan_fields(run_ready)?;
    let mut out = Vec::new();
    let mut raw_config = pb::RunInferenceSubagentModelConfig::default();
    let mut appended_byok = 0usize;
    let mut original_fingerprint: Option<&[u8]> = None;
    let mut saw_metadata = false;

    for field in &fields {
        if field.number == RUN_READY_TAG_PROMPT_METADATA && field.wire_type == WIRE_LEN {
            saw_metadata = true;
            let (patched, raw, appended) = patch_prompt_model_metadata(field.value, byok)?;
            raw_config = raw;
            appended_byok = appended;
            out.extend(encode_length_delimited_field(
                RUN_READY_TAG_PROMPT_METADATA,
                &patched,
            ));
            continue;
        }
        if field.number == RUN_READY_TAG_CONFIG_FINGERPRINT && field.wire_type == WIRE_LEN {
            original_fingerprint = Some(field.value);
            continue;
        }
        out.extend_from_slice(field.raw);
    }

    if !saw_metadata && !byok.is_empty() {
        let mut config = pb::RunInferenceSubagentModelConfig::default();
        let before = config.models.len();
        merge_byok_subagent_config_into(&mut config, byok);
        appended_byok = config.models.len().saturating_sub(before);
        let metadata_body = encode_length_delimited_field(
            PROMPT_METADATA_TAG_SUBAGENT_CONFIG,
            &config.encode_to_vec(),
        );
        out.extend(encode_length_delimited_field(
            RUN_READY_TAG_PROMPT_METADATA,
            &metadata_body,
        ));
    }

    let salted = salt_config_fingerprint(original_fingerprint, byok);
    if let Some(value) = salted.as_deref() {
        out.extend(encode_bytes_field(RUN_READY_TAG_CONFIG_FINGERPRINT, value));
    }

    let fingerprint = fingerprint_label(salted.as_deref());
    Ok(PatchedRunReady {
        body: out,
        raw_config,
        fingerprint,
        appended_byok,
    })
}

pub(crate) enum ServerPatch {
    Unchanged,
    Patched(PatchedRunReady),
}

pub(crate) fn merge_byok_subagent_config_into(
    config: &mut pb::RunInferenceSubagentModelConfig,
    byok: &[pb::RunInferenceSubagentModelPolicy],
) {
    if byok.is_empty() {
        return;
    }
    if config.force_policy.is_empty() {
        config.force_policy = "none".into();
    }
    let existing = config
        .models
        .iter()
        .map(|model| model.slug.clone())
        .collect::<std::collections::HashSet<_>>();
    for policy in byok {
        if existing.contains(&policy.slug) {
            continue;
        }
        config.models.push(policy.clone());
    }
}

pub(crate) fn patch_run_inference_server_message(
    body: &[u8],
    byok: &[pb::RunInferenceSubagentModelPolicy],
) -> Result<ServerPatch> {
    if byok.is_empty() {
        return Ok(ServerPatch::Unchanged);
    }
    let fields = scan_fields(body)?;
    let has_run_ready = fields
        .iter()
        .any(|field| field.number == SERVER_TAG_RUN_READY && field.wire_type == WIRE_LEN);
    if !has_run_ready {
        return Ok(ServerPatch::Unchanged);
    }

    let mut out = Vec::new();
    let mut patched_ready: Option<PatchedRunReady> = None;
    for field in fields {
        if field.number == SERVER_TAG_RUN_READY && field.wire_type == WIRE_LEN {
            let patched = patch_run_ready_body(field.value, byok)?;
            out.extend(encode_length_delimited_field(
                SERVER_TAG_RUN_READY,
                &patched.body,
            ));
            patched_ready = Some(patched);
        } else {
            out.extend_from_slice(field.raw);
        }
    }
    let Some(mut ready) = patched_ready else {
        return Ok(ServerPatch::Unchanged);
    };
    ready.body = out;
    Ok(ServerPatch::Patched(ready))
}

pub(crate) fn contains_run_ready_field(body: &[u8]) -> bool {
    scan_fields(body).ok().is_some_and(|fields| {
        fields
            .iter()
            .any(|field| field.number == SERVER_TAG_RUN_READY && field.wire_type == WIRE_LEN)
    })
}

const CLIENT_TAG_RUN_REQUEST: u32 = 1;
const CLIENT_TAG_FINISH_RUN: u32 = 4;
const BIDI_REQUEST_TAG_DATA_HEX: u32 = 1;
const BIDI_REQUEST_TAG_REQUEST_ID: u32 = 2;
const BIDI_REQUEST_TAG_DATA_BINARY: u32 = 4;
const BIDI_REQUEST_ID_TAG_ID: u32 = 1;
const RUN_REQUEST_TAG_CONVERSATION_ID_INFERENCE: u32 = 1;
const RUN_REQUEST_TAG_MODEL_PRIMARY: u32 = 3;
const RUN_REQUEST_TAG_CONVERSATION_ID_AGENT: u32 = 5;
const RUN_REQUEST_TAG_MODEL_OVERRIDE: u32 = 9;
const MODEL_TAG_MODEL_ID: u32 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SniffedBidiEnvelope {
    pub request_id: String,
    pub payload: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SniffedClientStart {
    pub model_id: String,
    pub conversation_id: Option<String>,
}

fn unwrap_unary_wire(body: &[u8]) -> Option<std::borrow::Cow<'_, [u8]>> {
    use crate::cursor::protocol::connect::{decompress_payload, COMPRESSION_FLAG, END_STREAM_FLAG};
    if body.len() >= 5 {
        let flags = body[0];
        let length = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
        if flags & END_STREAM_FLAG == 0 && length == body.len() - 5 {
            return match flags {
                0 => Some(std::borrow::Cow::Borrowed(&body[5..])),
                COMPRESSION_FLAG => {
                    decompress_payload(flags, bytes::Bytes::copy_from_slice(&body[5..]))
                        .ok()
                        .map(|bytes| std::borrow::Cow::Owned(bytes.to_vec()))
                }
                _ => None,
            };
        }
    }
    Some(std::borrow::Cow::Borrowed(body))
}

fn non_empty_utf8(bytes: &[u8]) -> Option<String> {
    let text = std::str::from_utf8(bytes).ok()?;
    (!text.is_empty() && !text.chars().any(char::is_control)).then(|| text.to_owned())
}

/// Extracts `BidiRequestId.request_id` via raw TLV scan when full prost decode fails.
pub(crate) fn sniff_bidi_request_id(body: &[u8]) -> Option<String> {
    let raw = unwrap_unary_wire(body)?;
    let fields = scan_fields(&raw).ok()?;
    fields
        .into_iter()
        .rev()
        .find(|field| field.number == BIDI_REQUEST_ID_TAG_ID && field.wire_type == WIRE_LEN)
        .and_then(|field| non_empty_utf8(field.value))
}

/// Extracts `(request_id, payload)` from `BidiAppendRequest` via raw TLV scan.
pub(crate) fn sniff_bidi_append_envelope(body: &[u8]) -> Option<SniffedBidiEnvelope> {
    let raw = unwrap_unary_wire(body)?;
    let fields = scan_fields(&raw).ok()?;
    let request_id = fields
        .iter()
        .rev()
        .find(|field| field.number == BIDI_REQUEST_TAG_REQUEST_ID && field.wire_type == WIRE_LEN)
        .and_then(|outer| {
            scan_fields(outer.value)
                .ok()?
                .into_iter()
                .rev()
                .find_map(|inner| {
                    (inner.number == BIDI_REQUEST_ID_TAG_ID && inner.wire_type == WIRE_LEN)
                        .then(|| non_empty_utf8(inner.value))
                        .flatten()
                })
        })?;
    let binary = fields
        .iter()
        .rev()
        .find(|field| field.number == BIDI_REQUEST_TAG_DATA_BINARY && field.wire_type == WIRE_LEN)
        .map(|field| field.value)
        .filter(|value| !value.is_empty());
    let payload = if let Some(binary) = binary {
        binary.to_vec()
    } else {
        let hex_str = fields
            .iter()
            .rev()
            .find(|field| field.number == BIDI_REQUEST_TAG_DATA_HEX && field.wire_type == WIRE_LEN)
            .and_then(|field| std::str::from_utf8(field.value).ok())
            .filter(|value| !value.is_empty())?;
        hex::decode(hex_str).ok()?
    };
    (!payload.is_empty()).then_some(SniffedBidiEnvelope {
        request_id,
        payload,
    })
}

/// Extracts `model_id` (and optional `conversation_id`) from either `RunInferenceClientMessage`
/// or `AgentClientMessage` `run_request` (tag 1) without decoding sibling fields.
pub(crate) fn sniff_client_run_start(payload: &[u8]) -> Option<SniffedClientStart> {
    let outer = scan_fields(payload).ok()?;
    let run_request = outer
        .into_iter()
        .rev()
        .find(|field| field.number == CLIENT_TAG_RUN_REQUEST && field.wire_type == WIRE_LEN)?;
    let inner = scan_fields(run_request.value).ok()?;
    let extract_model = |tag: u32| -> Option<String> {
        inner
            .iter()
            .rev()
            .filter(|field| field.number == tag && field.wire_type == WIRE_LEN)
            .find_map(|field| {
                scan_fields(field.value)
                    .ok()?
                    .into_iter()
                    .rev()
                    .find(|sub| sub.number == MODEL_TAG_MODEL_ID && sub.wire_type == WIRE_LEN)
                    .and_then(|sub| non_empty_utf8(sub.value))
            })
    };
    let model_id = extract_model(RUN_REQUEST_TAG_MODEL_OVERRIDE)
        .or_else(|| extract_model(RUN_REQUEST_TAG_MODEL_PRIMARY))?;
    let conversation_id = inner
        .iter()
        .rev()
        .find(|field| {
            field.number == RUN_REQUEST_TAG_CONVERSATION_ID_AGENT && field.wire_type == WIRE_LEN
        })
        .and_then(|field| non_empty_utf8(field.value))
        .or_else(|| {
            inner
                .iter()
                .rev()
                .find(|field| {
                    field.number == RUN_REQUEST_TAG_CONVERSATION_ID_INFERENCE
                        && field.wire_type == WIRE_LEN
                })
                .and_then(|field| non_empty_utf8(field.value))
        });
    Some(SniffedClientStart {
        model_id,
        conversation_id,
    })
}

/// Checks whether a client append payload carries `RunInferenceClientMessage.finish_run` (tag 4).
pub(crate) fn sniff_client_finish_run(payload: &[u8]) -> bool {
    scan_fields(payload).ok().is_some_and(|fields| {
        fields
            .iter()
            .any(|field| field.number == CLIENT_TAG_FINISH_RUN && field.wire_type == WIRE_LEN)
    })
}

#[cfg(test)]
pub(crate) fn append_length_delimited_field(buf: &mut Vec<u8>, number: u32, value: &[u8]) {
    buf.extend(encode_length_delimited_field(number, value));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cursor::protocol::{
        connect,
        proto::{agent::v1 as agent, aiserver::v1 as ai, inference as pb},
    };
    use bytes::Bytes;
    use prost::Message;

    fn poison_with_wire_conflict(mut encoded: Vec<u8>) -> Vec<u8> {
        append_length_delimited_field(&mut encoded, 3, b"wire-type-conflict");
        encoded
    }

    fn append_varint_field(buf: &mut Vec<u8>, number: u32, value: u64) {
        let tag = (u64::from(number) << 3) | u64::from(WIRE_VARINT);
        buf.extend(encode_varint(tag));
        buf.extend(encode_varint(value));
    }

    #[test]
    fn sniff_bidi_request_id_after_prost_decode_failure() {
        let mut inner = agent::BidiRequestId {
            request_id: "req-sniff-sse".into(),
        }
        .encode_to_vec();
        append_varint_field(&mut inner, 1, 1);
        let mut frame = vec![0u8];
        frame.extend_from_slice(&(inner.len() as u32).to_be_bytes());
        frame.extend_from_slice(&inner);
        let poisoned = Bytes::from(frame);
        assert!(connect::decode_unary::<agent::BidiRequestId>(&poisoned).is_err());
        assert_eq!(
            sniff_bidi_request_id(&poisoned).as_deref(),
            Some("req-sniff-sse")
        );
    }

    #[test]
    fn sniff_bidi_append_envelope_after_prost_decode_failure() {
        let run = pb::RunInferenceClientMessage {
            message: Some(pb::run_inference_client_message::Message::RunRequest(
                pb::RunInferenceRunRequest {
                    conversation_id: "conv-append".into(),
                    requested_model: Some(pb::InferenceRequestedModel {
                        model_id: "composer-2".into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )),
        };
        let payload = poison_with_wire_conflict(run.encode_to_vec());
        assert!(pb::RunInferenceClientMessage::decode(payload.as_slice()).is_err());
        let append = ai::BidiAppendRequest {
            request_id: Some(ai::BidiRequestId {
                request_id: "req-append-sniff".into(),
            }),
            data_binary: payload.clone(),
            append_seqno: 0,
            ..Default::default()
        };
        let mut body = append.encode_to_vec();
        append_varint_field(&mut body, 2, 1);
        assert!(ai::BidiAppendRequest::decode(body.as_slice()).is_err());
        let sniffed = sniff_bidi_append_envelope(body.as_slice()).expect("sniffed envelope");
        assert_eq!(sniffed.request_id, "req-append-sniff");
        assert_eq!(sniffed.payload, payload);
    }

    #[test]
    fn sniff_client_run_start_after_prost_decode_failure() {
        let message = pb::RunInferenceClientMessage {
            message: Some(pb::run_inference_client_message::Message::RunRequest(
                pb::RunInferenceRunRequest {
                    conversation_id: "conv-run-start".into(),
                    requested_model: Some(pb::InferenceRequestedModel {
                        model_id: "gpt-5.6-sol-medium".into(),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )),
        };
        let payload = poison_with_wire_conflict(message.encode_to_vec());
        assert!(pb::RunInferenceClientMessage::decode(payload.as_slice()).is_err());
        let sniffed = sniff_client_run_start(&payload).expect("run start");
        assert_eq!(sniffed.model_id, "gpt-5.6-sol-medium");
        assert_eq!(sniffed.conversation_id.as_deref(), Some("conv-run-start"));
    }

    #[test]
    fn sniff_client_finish_run_after_prost_decode_failure() {
        let message = pb::RunInferenceClientMessage {
            message: Some(pb::run_inference_client_message::Message::FinishRun(
                pb::RunInferenceFinishRun {
                    aborted: false,
                    turn_completed: Some(true),
                    ..Default::default()
                },
            )),
        };
        let payload = poison_with_wire_conflict(message.encode_to_vec());
        assert!(pb::RunInferenceClientMessage::decode(payload.as_slice()).is_err());
        assert!(sniff_client_finish_run(&payload));
    }
}
