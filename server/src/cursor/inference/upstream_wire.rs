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

#[cfg(test)]
pub(crate) fn append_length_delimited_field(buf: &mut Vec<u8>, number: u32, value: &[u8]) {
    buf.extend(encode_length_delimited_field(number, value));
}
