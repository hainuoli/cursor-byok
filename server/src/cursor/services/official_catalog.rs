//! Caches official Cursor subagent model slugs per auth token.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    body::{Body, Bytes},
    http::{header, HeaderMap, HeaderValue, Request},
};
use parking_lot::Mutex;
use prost::Message;
use sha2::{Digest, Sha256};

use crate::{
    api::cursor::proxy::{self, CursorProxy},
    cursor::protocol::proto::inference as pb,
    Error, Result,
};

pub type TokenKey = [u8; 32];

const CATALOG_TTL: Duration = Duration::from_secs(30 * 60);
const FETCH_TIMEOUT: Duration = Duration::from_millis(2_500);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OfficialSlug {
    pub slug: String,
    pub advertised: bool,
    pub valid: bool,
    pub blocked: bool,
    pub legacy_aliases: Vec<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CatalogSource {
    AvailableModels,
    RunReadyHarvest,
}

#[derive(Clone, Debug)]
struct Bucket {
    models: Vec<OfficialSlug>,
    source: CatalogSource,
    fetched_at: Instant,
}

#[derive(Clone)]
pub struct OfficialCatalog {
    buckets: Arc<Mutex<HashMap<TokenKey, Bucket>>>,
}

impl OfficialCatalog {
    pub fn new() -> Self {
        Self {
            buckets: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn token_key(headers: &HeaderMap) -> TokenKey {
        let Some(value) = headers.get(header::AUTHORIZATION) else {
            return [0; 32];
        };
        token_key_bytes(value.as_bytes())
    }

    pub fn token_key_from_auth(header: Option<&axum::http::HeaderValue>) -> TokenKey {
        match header {
            Some(value) => token_key_bytes(value.as_bytes()),
            None => [0; 32],
        }
    }

    pub fn get(&self, token: TokenKey) -> Vec<OfficialSlug> {
        let guard = self.buckets.lock();
        guard
            .get(&token)
            .filter(|bucket| bucket.fetched_at.elapsed() < CATALOG_TTL)
            .map(|bucket| bucket.models.clone())
            .unwrap_or_default()
    }

    pub fn slug_set(&self, token: TokenKey) -> HashSet<String> {
        self.get(token)
            .into_iter()
            .flat_map(|model| std::iter::once(model.slug).chain(model.legacy_aliases.clone()))
            .collect()
    }

    pub fn is_official_slug(&self, token: TokenKey, slug: &str) -> bool {
        if slug.is_empty() {
            return false;
        }
        self.get(token).iter().any(|model| {
            model.slug == slug || model.legacy_aliases.iter().any(|alias| alias == slug)
        })
    }

    pub fn remember_available_models(&self, token: TokenKey, models: Vec<OfficialSlug>) {
        let mut guard = self.buckets.lock();
        if matches!(
            guard.get(&token).map(|bucket| bucket.source),
            Some(CatalogSource::RunReadyHarvest)
        ) {
            return;
        }
        guard.insert(
            token,
            Bucket {
                models,
                source: CatalogSource::AvailableModels,
                fetched_at: Instant::now(),
            },
        );
    }

    pub fn remember_run_ready(&self, token: TokenKey, models: Vec<OfficialSlug>) {
        self.buckets.lock().insert(
            token,
            Bucket {
                models,
                source: CatalogSource::RunReadyHarvest,
                fetched_at: Instant::now(),
            },
        );
    }

    pub fn harvest_run_ready(&self, token: TokenKey, ready: &pb::RunInferenceRunReady) {
        let Some(models) = ready
            .prompt_model_metadata
            .as_ref()
            .and_then(|metadata| metadata.subagent_model_config.as_ref())
            .map(|config| {
                config
                    .models
                    .iter()
                    .map(|policy| OfficialSlug {
                        slug: policy.slug.clone(),
                        advertised: policy.advertised,
                        valid: policy.valid,
                        blocked: policy.blocked,
                        legacy_aliases: policy.legacy_aliases.clone(),
                    })
                    .collect::<Vec<_>>()
            })
        else {
            return;
        };
        if !models.is_empty() {
            self.remember_run_ready(token, models);
        }
    }

    pub async fn get_or_fetch(
        &self,
        token: TokenKey,
        headers: &HeaderMap,
        proxy: &CursorProxy,
        byok_hashes: &HashSet<String>,
    ) -> Vec<OfficialSlug> {
        let cached = self.get(token);
        if !cached.is_empty() {
            return cached;
        }
        match tokio::time::timeout(FETCH_TIMEOUT, fetch_available_models(proxy, headers)).await {
            Ok(Ok(models)) => {
                let slugs = filter_official_slugs(&models, byok_hashes);
                self.remember_available_models(token, slugs.clone());
                slugs
            }
            Ok(Err(error)) => {
                tracing::warn!(%error, "official AvailableModels fetch failed");
                Vec::new()
            }
            Err(_) => {
                tracing::warn!("official AvailableModels fetch timed out");
                Vec::new()
            }
        }
    }
}

impl Default for OfficialCatalog {
    fn default() -> Self {
        Self::new()
    }
}

pub fn token_key_bytes(authorization: &[u8]) -> TokenKey {
    let digest = Sha256::digest(authorization);
    let mut key = [0u8; 32];
    key.copy_from_slice(&digest);
    key
}

#[derive(Clone, PartialEq, Message)]
struct AvailableModelsView {
    #[prost(string, repeated, tag = "1")]
    model_names: Vec<String>,
    #[prost(message, repeated, tag = "2")]
    models: Vec<AvailableModelView>,
    #[prost(map = "string, message", tag = "16")]
    subagent_model_configs: HashMap<String, FeatureModelConfigView>,
}

#[derive(Clone, PartialEq, Message)]
struct AvailableModelView {
    #[prost(string, tag = "1")]
    name: String,
    #[prost(bool, tag = "2")]
    default_on: bool,
    #[prost(bool, optional, tag = "4")]
    is_chat_only: Option<bool>,
    #[prost(bool, optional, tag = "5")]
    supports_agent: Option<bool>,
    #[prost(int32, optional, tag = "6")]
    degradation_status: Option<i32>,
    #[prost(message, repeated, tag = "30")]
    variants: Vec<VariantView>,
    #[prost(bool, optional, tag = "35")]
    is_hidden: Option<bool>,
    #[prost(string, repeated, tag = "36")]
    legacy_slugs: Vec<String>,
    #[prost(string, repeated, tag = "37")]
    id_aliases: Vec<String>,
}

#[derive(Clone, PartialEq, Message)]
struct VariantView {
    #[prost(string, optional, tag = "11")]
    legacy_slug: Option<String>,
}

#[derive(Clone, PartialEq, Message)]
struct FeatureModelConfigView {
    #[prost(string, tag = "1")]
    default_model: String,
    #[prost(string, repeated, tag = "2")]
    fallback_models: Vec<String>,
    #[prost(string, repeated, tag = "3")]
    best_of_n_default_models: Vec<String>,
}

fn decode_available_models(payload: &[u8]) -> Result<AvailableModelsView> {
    AvailableModelsView::decode(payload).map_err(Error::from)
}

pub fn slugs_from_available_models_payload(
    payload: &[u8],
    byok_hashes: &HashSet<String>,
) -> Result<Vec<OfficialSlug>> {
    let view = decode_available_models(payload)?;
    Ok(filter_official_slugs(&view, byok_hashes))
}

fn filter_official_slugs(
    view: &AvailableModelsView,
    byok_hashes: &HashSet<String>,
) -> Vec<OfficialSlug> {
    let mut slugs = HashSet::new();
    for model in &view.models {
        if model.is_chat_only == Some(true)
            || model.is_hidden == Some(true)
            || model.degradation_status == Some(2)
            || model.supports_agent == Some(false)
        {
            continue;
        }
        slugs.insert(model.name.clone());
        slugs.extend(model.legacy_slugs.iter().cloned());
        slugs.extend(model.id_aliases.iter().cloned());
        for variant in &model.variants {
            if let Some(legacy) = &variant.legacy_slug {
                slugs.insert(legacy.clone());
            }
        }
    }
    for config in view.subagent_model_configs.values() {
        slugs.insert(config.default_model.clone());
        slugs.extend(config.fallback_models.iter().cloned());
        slugs.extend(config.best_of_n_default_models.iter().cloned());
    }
    slugs.retain(|slug| !slug.is_empty() && slug != "default" && !byok_hashes.contains(slug));
    let mut ordered = slugs.into_iter().collect::<Vec<_>>();
    ordered.sort();
    ordered
        .into_iter()
        .map(|slug| OfficialSlug {
            slug,
            advertised: true,
            valid: true,
            blocked: false,
            legacy_aliases: Vec::new(),
        })
        .collect()
}

async fn fetch_available_models(
    proxy: &CursorProxy,
    headers: &HeaderMap,
) -> Result<AvailableModelsView> {
    static CONNECT_PROTOCOL_VERSION: axum::http::HeaderName =
        axum::http::HeaderName::from_static("connect-protocol-version");
    static CONNECT_ACCEPT_ENCODING: axum::http::HeaderName =
        axum::http::HeaderName::from_static("connect-accept-encoding");
    let mut request_headers = HeaderMap::new();
    if let Some(value) = headers.get(header::AUTHORIZATION) {
        request_headers.insert(header::AUTHORIZATION, value.clone());
    }
    if let Some(value) = headers.get(header::COOKIE) {
        request_headers.insert(header::COOKIE, value.clone());
    }
    if let Some(value) = headers.get(&CONNECT_PROTOCOL_VERSION) {
        request_headers.insert(CONNECT_PROTOCOL_VERSION.clone(), value.clone());
    }
    request_headers.insert(
        CONNECT_ACCEPT_ENCODING.clone(),
        HeaderValue::from_static("identity"),
    );
    request_headers.insert(
        header::ACCEPT_ENCODING,
        HeaderValue::from_static("identity"),
    );
    request_headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/proto"),
    );
    let request = Request::builder()
        .method(axum::http::Method::POST)
        .uri("/aiserver.v1.AiService/AvailableModels")
        .body(Body::empty())
        .map_err(|error| Error::Protocol(error.to_string()))?;
    let mut request = request;
    *request.headers_mut() = request_headers;
    let upstream = proxy::forward_buffered(proxy, request).await?;
    if !upstream.status.is_success() {
        return Err(Error::Protocol(format!(
            "AvailableModels upstream status {}",
            upstream.status
        )));
    }
    let payload = unary_payload(&upstream.body)?;
    decode_available_models(payload)
}

fn unary_payload(body: &Bytes) -> Result<&[u8]> {
    if body.len() >= 5 {
        let flags = body[0];
        let length = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
        if flags == 0 && length == body.len() - 5 {
            return Ok(&body[5..]);
        }
    }
    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_view() -> AvailableModelsView {
        AvailableModelsView {
            model_names: vec!["m1".into()],
            models: vec![
                AvailableModelView {
                    name: "gpt-official".into(),
                    default_on: true,
                    is_chat_only: Some(false),
                    supports_agent: Some(true),
                    degradation_status: None,
                    variants: vec![VariantView {
                        legacy_slug: Some("cursor-gpt-variant".into()),
                    }],
                    is_hidden: Some(false),
                    legacy_slugs: vec!["legacy-gpt".into()],
                    id_aliases: vec![],
                },
                AvailableModelView {
                    name: "hidden".into(),
                    default_on: false,
                    is_chat_only: None,
                    supports_agent: None,
                    degradation_status: None,
                    variants: vec![],
                    is_hidden: Some(true),
                    legacy_slugs: vec![],
                    id_aliases: vec![],
                },
                AvailableModelView {
                    name: "chat-only".into(),
                    default_on: false,
                    is_chat_only: Some(true),
                    supports_agent: None,
                    degradation_status: None,
                    variants: vec![],
                    is_hidden: None,
                    legacy_slugs: vec![],
                    id_aliases: vec![],
                },
                AvailableModelView {
                    name: "disabled".into(),
                    default_on: false,
                    is_chat_only: None,
                    supports_agent: None,
                    degradation_status: Some(2),
                    variants: vec![],
                    is_hidden: None,
                    legacy_slugs: vec![],
                    id_aliases: vec![],
                },
            ],
            subagent_model_configs: HashMap::from([(
                "explore".into(),
                FeatureModelConfigView {
                    default_model: "explore-default".into(),
                    fallback_models: vec!["explore-fallback".into()],
                    best_of_n_default_models: vec![],
                },
            )]),
        }
    }

    #[test]
    fn filters_unusable_models_and_collects_slugs() {
        let byok = HashSet::new();
        let slugs = filter_official_slugs(&sample_view(), &byok);
        let names = slugs
            .iter()
            .map(|slug| slug.slug.as_str())
            .collect::<Vec<_>>();
        assert!(names.contains(&"gpt-official"));
        assert!(names.contains(&"legacy-gpt"));
        assert!(names.contains(&"cursor-gpt-variant"));
        assert!(names.contains(&"explore-default"));
        assert!(!names.contains(&"hidden"));
        assert!(!names.contains(&"chat-only"));
        assert!(!names.contains(&"disabled"));
    }

    #[test]
    fn run_ready_harvest_overrides_available_models_bucket() {
        let catalog = OfficialCatalog::new();
        let token = [7u8; 32];
        catalog.remember_available_models(
            token,
            vec![OfficialSlug {
                slug: "approx".into(),
                advertised: true,
                valid: true,
                blocked: false,
                legacy_aliases: vec![],
            }],
        );
        catalog.remember_run_ready(
            token,
            vec![OfficialSlug {
                slug: "harvested".into(),
                advertised: true,
                valid: true,
                blocked: false,
                legacy_aliases: vec![],
            }],
        );
        let slugs = catalog
            .get(token)
            .into_iter()
            .map(|slug| slug.slug)
            .collect::<Vec<_>>();
        assert_eq!(slugs, vec!["harvested"]);
        catalog.remember_available_models(
            token,
            vec![OfficialSlug {
                slug: "should-not-replace".into(),
                advertised: true,
                valid: true,
                blocked: false,
                legacy_aliases: vec![],
            }],
        );
        assert_eq!(catalog.get(token)[0].slug, "harvested");
    }

    #[test]
    fn different_tokens_do_not_share_buckets() {
        let catalog = OfficialCatalog::new();
        catalog.remember_available_models(
            [1u8; 32],
            vec![OfficialSlug {
                slug: "one".into(),
                advertised: true,
                valid: true,
                blocked: false,
                legacy_aliases: vec![],
            }],
        );
        assert!(catalog.get([2u8; 32]).is_empty());
    }
}
