//! Encodes and decodes Connect protocol frames.
use bytes::{BufMut, Bytes, BytesMut};
use prost::Message;
use serde::Serialize;

use crate::{Error, Result};

pub const END_STREAM_FLAG: u8 = 0x02;
pub const COMPRESSION_FLAG: u8 = 0x01;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnectCode {
    Canceled,
    InvalidArgument,
    NotFound,
    Unavailable,
    Internal,
}

impl ConnectCode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Canceled => "canceled",
            Self::InvalidArgument => "invalid_argument",
            Self::NotFound => "not_found",
            Self::Unavailable => "unavailable",
            Self::Internal => "internal",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ConnectErrorDetail {
    #[serde(rename = "type")]
    pub type_name: String,
    pub value: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConnectStreamError {
    pub code: ConnectCode,
    pub message: String,
    pub details: Vec<ConnectErrorDetail>,
}

#[derive(Serialize)]
struct EndStreamResponse<'a> {
    error: WireError<'a>,
}

#[derive(Serialize)]
struct WireError<'a> {
    code: &'static str,
    #[serde(skip_serializing_if = "str::is_empty")]
    message: &'a str,
    #[serde(skip_serializing_if = "details_are_empty")]
    details: &'a [ConnectErrorDetail],
}

fn details_are_empty(details: &&[ConnectErrorDetail]) -> bool {
    details.is_empty()
}

pub fn encode_message<M: Message>(message: &M) -> Result<Bytes> {
    let len = message.encoded_len();
    let mut output = BytesMut::with_capacity(5 + len);
    output.put_u8(0);
    output.put_u32(len as u32);
    message.encode(&mut output)?;
    Ok(output.freeze())
}

pub fn encode_end_stream() -> Bytes {
    encode_end_stream_payload(b"{}")
}

pub fn encode_error_end_stream(error: &ConnectStreamError) -> Result<Bytes> {
    let payload = serde_json::to_vec(&EndStreamResponse {
        error: WireError {
            code: error.code.as_str(),
            message: &error.message,
            details: &error.details,
        },
    })?;
    Ok(encode_end_stream_payload(&payload))
}

fn encode_end_stream_payload(payload: &[u8]) -> Bytes {
    let mut output = BytesMut::with_capacity(5 + payload.len());
    output.put_u8(END_STREAM_FLAG);
    output.put_u32(payload.len() as u32);
    output.extend_from_slice(payload);
    output.freeze()
}

pub fn decode_unary<M: Message + Default>(body: &[u8]) -> Result<M> {
    if body.len() >= 5 {
        let flags = body[0];
        let length = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
        if flags & END_STREAM_FLAG == 0 && length == body.len() - 5 {
            return match flags {
                0 => Ok(M::decode(&body[5..])?),
                COMPRESSION_FLAG => {
                    let raw = decompress_payload(flags, Bytes::copy_from_slice(&body[5..]))?;
                    Ok(M::decode(raw)?)
                }
                _ => Err(Error::Protocol(format!(
                    "unsupported Connect unary flags: {flags:#x}"
                ))),
            };
        }
    }
    Ok(M::decode(body)?)
}

pub fn decode_frames(mut body: &[u8]) -> Result<Vec<(u8, Bytes)>> {
    let mut frames = Vec::new();
    while !body.is_empty() {
        if body.len() < 5 {
            return Err(Error::Protocol("truncated Connect envelope".into()));
        }
        let flags = body[0];
        let length = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
        body = &body[5..];
        if body.len() < length {
            return Err(Error::Protocol("truncated Connect payload".into()));
        }
        frames.push((flags, Bytes::copy_from_slice(&body[..length])));
        body = &body[length..];
    }
    Ok(frames)
}

/// Incrementally decodes Connect stream frames across arbitrary read boundaries.
pub struct FrameDecoder {
    buf: BytesMut,
}

impl FrameDecoder {
    pub fn new() -> Self {
        Self {
            buf: BytesMut::new(),
        }
    }

    /// Returns `(flags, payload, original_envelope)` for each complete frame.
    pub fn push(&mut self, chunk: Bytes) -> Result<Vec<(u8, Bytes, Bytes)>> {
        self.buf.extend_from_slice(&chunk);
        let mut out = Vec::new();
        loop {
            if self.buf.len() < 5 {
                break;
            }
            let flags = self.buf[0];
            let length =
                u32::from_be_bytes([self.buf[1], self.buf[2], self.buf[3], self.buf[4]]) as usize;
            if self.buf.len() < 5 + length {
                break;
            }
            let envelope = self.buf.split_to(5 + length).freeze();
            let payload = envelope.slice(5..);
            out.push((flags, payload, envelope));
        }
        Ok(out)
    }
}

impl Default for FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn decompress_payload(flags: u8, payload: Bytes) -> Result<Bytes> {
    if flags & COMPRESSION_FLAG == 0 {
        return Ok(payload);
    }
    use flate2::read::GzDecoder;
    use std::io::Read;
    let mut decoder = GzDecoder::new(payload.as_ref());
    let mut decoded = Vec::new();
    decoder
        .read_to_end(&mut decoded)
        .map_err(|error| Error::Protocol(format!("gzip decompress failed: {error}")))?;
    Ok(Bytes::from(decoded))
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use flate2::{write::GzEncoder, Compression};
    use prost::Message;
    use std::io::Write;

    #[derive(Clone, PartialEq, Message)]
    struct Tiny {
        #[prost(string, tag = "1")]
        value: String,
    }

    #[test]
    fn frame_decoder_reassembles_split_frames() {
        let frame = encode_message(&Tiny {
            value: "hello".into(),
        })
        .unwrap();
        let mid = frame.len() / 2;
        let mut decoder = FrameDecoder::new();
        assert!(decoder.push(frame.slice(..mid)).unwrap().is_empty());
        let frames = decoder.push(frame.slice(mid..)).unwrap();
        assert_eq!(frames.len(), 1);
        let decoded: Tiny =
            Message::decode(decompress_payload(frames[0].0, frames[0].1.clone()).unwrap()).unwrap();
        assert_eq!(decoded.value, "hello");
    }

    #[test]
    fn decompresses_gzip_connect_payload() {
        let inner = encode_message(&Tiny { value: "gz".into() }).unwrap();
        let payload = inner.slice(5..);
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&payload).unwrap();
        let compressed = encoder.finish().unwrap();
        let mut envelope = BytesMut::with_capacity(5 + compressed.len());
        envelope.put_u8(COMPRESSION_FLAG);
        envelope.put_u32(compressed.len() as u32);
        envelope.extend_from_slice(&compressed);
        let frame = envelope.freeze();
        let flags = frame[0];
        let body = frame.slice(5..);
        let raw = decompress_payload(flags, body).unwrap();
        let decoded: Tiny = Message::decode(raw).unwrap();
        assert_eq!(decoded.value, "gz");
        let unary: Tiny = decode_unary(&frame).unwrap();
        assert_eq!(unary.value, "gz");
    }

    #[test]
    fn decode_unary_rejects_unknown_envelope_flags_cleanly() {
        let mut envelope = BytesMut::from(&[0x04, 0, 0, 0, 2, 0x0a, 0x00][..]);
        let frame = envelope.split().freeze();
        let err = decode_unary::<Tiny>(&frame).unwrap_err();
        assert!(err.to_string().contains("unsupported Connect unary flags"));
    }
}
