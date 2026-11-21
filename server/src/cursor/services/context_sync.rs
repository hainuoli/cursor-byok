//! Hydrates request context blobs supplied by Cursor.
use std::{sync::Arc, time::Duration};

use prost::Message;
use tokio::sync::{oneshot, Mutex};

use crate::{
    cursor::{
        protocol::proto::agent::v1 as pb, services::blob_sync::BlobSynchronizer,
        transport::TransportHandle,
    },
    store::{BlobId, Store},
    Error, Result,
};

type ContextSender = oneshot::Sender<Result<pb::RequestContext>>;

#[derive(Clone)]
pub(crate) struct RequestContextSynchronizer {
    handle: TransportHandle,
    store: Store,
    blob_sync: BlobSynchronizer,
    pending: Arc<Mutex<Option<ContextSender>>>,
}

impl RequestContextSynchronizer {
    pub(crate) fn new(handle: TransportHandle, store: Store, blob_sync: BlobSynchronizer) -> Self {
        Self {
            handle,
            store,
            blob_sync,
            pending: Arc::new(Mutex::new(None)),
        }
    }

    pub(crate) async fn refresh_if_missing(
        &self,
        references: &pb::RequestContextPartReferences,
        conversation_id: &str,
    ) -> Result<Option<pb::RequestContext>> {
        if !self.has_missing_part(references).await? {
            return Ok(None);
        }
        // Newer Cursor clients send context as blob *references* only and no
        // longer answer `RequestContextArgs`. Pull the missing parts over the
        // KV blob channel first; only fall back to the legacy round trip when
        // that channel cannot supply them.
        if self.hydrate_missing_parts(references).await? {
            return Ok(None);
        }
        let context = self.load(conversation_id).await?;
        self.cache_parts(&context).await?;
        Ok(Some(context))
    }

    /// Pulls every referenced-but-absent context part through the KV blob
    /// channel. Returns `true` when all referenced parts are present afterwards.
    async fn hydrate_missing_parts(
        &self,
        references: &pb::RequestContextPartReferences,
    ) -> Result<bool> {
        for (part, raw_id) in self.referenced_parts(references) {
            if raw_id.is_empty() {
                continue;
            }
            let id = BlobId::from_bytes(raw_id)?;
            if self.store.get_blob(&id).await?.is_some() {
                continue;
            }
            match self.blob_sync.get(&id).await {
                Ok(Some(data)) => {
                    tracing::info!(
                        request_id = self.handle.request_id(),
                        part = part,
                        byte_count = data.len(),
                        "hydrated Cursor context part over KV blob channel"
                    );
                }
                Ok(None) => {
                    tracing::warn!(
                        request_id = self.handle.request_id(),
                        part = part,
                        "Cursor reported context blob as absent"
                    );
                    return Ok(false);
                }
                Err(error) => {
                    tracing::warn!(
                        request_id = self.handle.request_id(),
                        part = part,
                        %error,
                        "Cursor context blob fetch failed"
                    );
                    return Ok(false);
                }
            }
        }
        Ok(!self.has_missing_part(references).await?)
    }

    fn referenced_parts<'a>(
        &self,
        references: &'a pb::RequestContextPartReferences,
    ) -> [(&'static str, &'a [u8]); 4] {
        [
            ("rules", references.rules_blob_id.as_slice()),
            ("skills", references.skills_blob_id.as_slice()),
            ("subagents", references.subagents_blob_id.as_slice()),
            ("mcps", references.mcps_blob_id.as_slice()),
        ]
    }

    pub(crate) async fn get(&self, id: &BlobId) -> Result<Option<Vec<u8>>> {
        self.store.get_blob(id).await
    }

    pub(crate) async fn load(&self, conversation_id: &str) -> Result<pb::RequestContext> {
        let (sender, receiver) = oneshot::channel();
        let mut pending = self.pending.lock().await;
        if pending.is_some() {
            return Err(Error::Protocol(
                "Cursor request context is already being loaded".into(),
            ));
        }
        *pending = Some(sender);
        drop(pending);

        tracing::info!(
            request_id = self.handle.request_id(),
            conversation_id,
            "requesting uncached Cursor context"
        );

        if let Err(error) = self.handle.emit(&pb::AgentServerMessage {
            ttft_breakdown: None,
            message: Some(pb::agent_server_message::Message::ExecServerMessage(
                pb::ExecServerMessage {
                    id: 0,
                    message: Some(pb::exec_server_message::Message::RequestContextArgs(
                        pb::RequestContextArgs {
                            notes_session_id: Some(conversation_id.into()),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                },
            )),
        }) {
            self.pending.lock().await.take();
            return Err(error);
        }

        let cancellation = self.handle.disconnect_token();
        let result = tokio::select! {
            result = receiver => result.map_err(|_| Error::Protocol("request context response channel closed".into()))?,
            _ = cancellation.cancelled() => Err(Error::Cancelled),
            _ = tokio::time::sleep(Duration::from_secs(60)) => Err(Error::Protocol("request context timed out".into())),
        };
        if result.is_err() {
            self.pending.lock().await.take();
        }
        result
    }

    pub(crate) async fn handle_client(&self, message: &pb::ExecClientMessage) -> bool {
        if message.id != 0 {
            return false;
        }
        let Some(pb::exec_client_message::Message::RequestContextResult(result)) =
            message.message.as_ref()
        else {
            return false;
        };
        let Some(sender) = self.pending.lock().await.take() else {
            tracing::warn!(
                request_id = self.handle.request_id(),
                "unexpected Cursor request context result"
            );
            return true;
        };
        use pb::request_context_result::Result as ContextResult;
        let result = match result.result.as_ref() {
            Some(ContextResult::Success(success)) => success
                .request_context
                .clone()
                .ok_or_else(|| Error::Protocol("Cursor returned empty request context".into())),
            Some(ContextResult::Error(error)) => Err(Error::Protocol(format!(
                "Cursor request context failed: {}",
                error.error
            ))),
            Some(ContextResult::Rejected(rejected)) => Err(Error::Protocol(format!(
                "Cursor rejected request context: {}",
                rejected.reason
            ))),
            None => Err(Error::Protocol(
                "Cursor returned no request context result".into(),
            )),
        };
        let _ = sender.send(result);
        true
    }

    pub(crate) async fn handle_stream_close(&self, id: u32) -> bool {
        id == 0 && self.pending.lock().await.is_some()
    }

    pub(crate) async fn handle_throw(&self, id: u32, message: String) -> bool {
        let sender = if id == 0 {
            self.pending.lock().await.take()
        } else {
            None
        };
        let Some(sender) = sender else { return false };
        let _ = sender.send(Err(Error::Protocol(message)));
        true
    }

    async fn has_missing_part(&self, parts: &pb::RequestContextPartReferences) -> Result<bool> {
        for (_part, raw_id) in self.referenced_parts(parts) {
            if raw_id.is_empty() {
                continue;
            }
            let id = BlobId::from_bytes(raw_id)?;
            if self.store.get_blob(&id).await?.is_none() {
                return Ok(true);
            }
        }
        Ok(false)
    }

    async fn cache_parts(&self, context: &pb::RequestContext) -> Result<()> {
        self.cache_part(&pb::RequestContextRulesPart {
            rules: context.rules.clone(),
            non_file_rules: context.non_file_rules.clone(),
            cloud_rule: context.cloud_rule.clone(),
        })
        .await?;
        self.cache_part(&pb::RequestContextSkillsPart {
            agent_skills: context.agent_skills.clone(),
            skill_options: context.skill_options.clone(),
        })
        .await?;
        self.cache_part(&pb::RequestContextSubagentsPart {
            custom_subagents: context.custom_subagents.clone(),
        })
        .await?;
        self.cache_part(&pb::RequestContextMcpsPart {
            tools: context.tools.clone(),
            mcp_instructions: context.mcp_instructions.clone(),
            mcp_file_system_options: context.mcp_file_system_options.clone(),
            mcp_meta_tool_options: context.mcp_meta_tool_options.clone(),
        })
        .await
    }

    async fn cache_part<T: Message>(&self, part: &T) -> Result<()> {
        let data = part.encode_to_vec();
        self.store.put_blob(&data, &[]).await?;
        Ok(())
    }
}
