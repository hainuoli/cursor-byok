//! Resolves Task model selections to configured BYOK model identities.

use std::collections::HashSet;

use crate::{cursor::protocol::proto::inference as pb, model::ModelConfig, Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByokModel {
    pub hash: String,
    pub model_id: String,
    pub display_name: String,
    pub advertised_slug: String,
}

pub(crate) fn models_from_configs(models: &[ModelConfig]) -> Vec<ByokModel> {
    models
        .iter()
        .map(|model| ByokModel {
            hash: model.model_hash.clone(),
            model_id: model.model_id.clone(),
            display_name: model.display_name.clone(),
            advertised_slug: String::new(),
        })
        .collect()
}

pub(crate) fn assign_slugs(models: &mut [ByokModel], reserved: &HashSet<String>) {
    models.sort_by(|left, right| left.hash.cmp(&right.hash));
    let mut used = reserved.clone();
    for model in models {
        let base = slug_base(model);
        model.advertised_slug = unique_slug(&base, &model.hash, &mut used);
        used.insert(model.advertised_slug.clone());
    }
}

fn unique_slug(base: &str, hash: &str, used: &mut HashSet<String>) -> String {
    if !used.contains(base) {
        return base.to_owned();
    }
    let short = hash6_slug(base, hash);
    if !used.contains(&short) {
        return short;
    }
    hash.to_owned()
}

/// Collision form emitted by `unique_slug` when `base` is already reserved.
pub(crate) fn hash6_slug(base: &str, hash: &str) -> String {
    format!("{base}-{}", &hash[..hash.len().min(6)])
}

pub(crate) fn slug_base(model: &ByokModel) -> String {
    let base = kebab_display(&model.display_name);
    if base.is_empty() {
        model.hash.clone()
    } else {
        base
    }
}

/// Readable kebab-case slug for Task `model=` arguments.
pub(crate) fn kebab_display(display_name: &str) -> String {
    let source = if has_tt_switch_suffix(display_name) {
        display_name
    } else {
        strip_trailing_parens(display_name)
    };
    kebab_raw(source)
}

fn has_tt_switch_suffix(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    lower.contains("tt switch") || lower.contains("tt-switch")
}

fn strip_trailing_parens(name: &str) -> &str {
    let name = name.trim_end();
    match name.strip_suffix(')').and_then(|rest| rest.rfind('(')) {
        Some(start) => name[..start].trim_end(),
        None => name,
    }
}

fn kebab_raw(value: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in value.to_ascii_lowercase().chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' {
            out.push(if ch == '.' {
                ch
            } else {
                ch.to_ascii_lowercase()
            });
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_owned()
}

pub(crate) fn display_key(name: &str) -> String {
    name.split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
        .to_lowercase()
}

pub(crate) fn without_suffix(name: &str) -> &str {
    strip_trailing_parens(name)
}

pub(crate) fn legacy_aliases(model: &ByokModel) -> Vec<String> {
    let mut aliases = vec![
        model.hash.clone(),
        model.model_id.clone(),
        model.display_name.clone(),
        model.display_name.to_ascii_lowercase(),
        display_key(&model.display_name),
    ];
    let suffix_key = display_key(without_suffix(&model.display_name));
    if !suffix_key.is_empty() {
        aliases.push(suffix_key.clone());
        aliases.push(kebab_display(without_suffix(&model.display_name)));
    }
    aliases.sort();
    aliases.dedup();
    aliases.retain(|alias| alias != &model.advertised_slug);
    aliases
}

pub(crate) fn byok_policies(
    models: &[ModelConfig],
    reserved: &HashSet<String>,
) -> Vec<pb::RunInferenceSubagentModelPolicy> {
    let mut byok = models_from_configs(models);
    assign_slugs(&mut byok, reserved);
    let mut policies = Vec::new();
    for model in &byok {
        policies.push(policy_entry(
            &model.advertised_slug,
            true,
            legacy_aliases(model),
        ));
        policies.push(policy_entry(&model.hash, false, Vec::new()));
    }
    policies
}

fn policy_entry(
    slug: &str,
    advertised: bool,
    legacy_aliases: Vec<String>,
) -> pb::RunInferenceSubagentModelPolicy {
    pb::RunInferenceSubagentModelPolicy {
        slug: slug.into(),
        advertised,
        valid: true,
        blocked: false,
        requires_max_mode: false,
        representative_cost_dollars: 0.0,
        legacy_aliases,
        turned_off_by_user: false,
    }
}

pub(crate) fn resolve<'a>(models: &'a [ByokModel], requested: &'a str) -> Result<&'a str> {
    if let Some(model) = models.iter().find(|model| model.hash == requested) {
        return Ok(&model.hash);
    }
    let key = display_key(requested);
    if requested.contains('(') {
        let exact_display = models
            .iter()
            .filter(|model| display_key(&model.display_name) == key)
            .collect::<Vec<_>>();
        if exact_display.len() == 1 {
            return Ok(&exact_display[0].hash);
        }
    }
    let suffix_key = (requested.contains(' ') || requested.contains('('))
        .then(|| display_key(without_suffix(requested)));
    let mut hashes = std::collections::HashSet::new();
    for model in models {
        if !model.advertised_slug.is_empty() && model.advertised_slug == requested {
            hashes.insert(model.hash.as_str());
        }
        if requested == hash6_slug(&slug_base(model), &model.hash) {
            hashes.insert(model.hash.as_str());
        }
        if model.model_id == requested {
            hashes.insert(model.hash.as_str());
        }
        if display_key(&model.display_name) == key {
            hashes.insert(model.hash.as_str());
        }
        if let Some(suffix_key) = &suffix_key {
            if display_key(without_suffix(&model.display_name)) == *suffix_key {
                hashes.insert(model.hash.as_str());
            }
        }
    }
    match hashes.len() {
        0 => Ok(requested),
        1 => Ok(hashes.into_iter().next().expect("single match")),
        _ => {
            let mut ambiguous = hashes.into_iter().map(str::to_owned).collect::<Vec<_>>();
            ambiguous.sort();
            Err(Error::Protocol(format!(
                "Task model {requested:?} is ambiguous; use a configured model hash: {}",
                ambiguous.join(", ")
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(hash: &str, model_id: &str, display_name: &str) -> ByokModel {
        ByokModel {
            hash: hash.into(),
            model_id: model_id.into(),
            display_name: display_name.into(),
            advertised_slug: String::new(),
        }
    }

    #[test]
    fn kebab_strips_parens_and_appends_tt_switch() {
        assert_eq!(
            kebab_display("Claude 5.5 Sonnet (TT Switch)"),
            "claude-5.5-sonnet-tt-switch"
        );
        assert_eq!(
            kebab_display("GPT 5.6 Sol Medium (TT Switch)"),
            "gpt-5.6-sol-medium-tt-switch"
        );
    }

    #[test]
    fn assign_slugs_avoids_reserved_collision() {
        let mut models = [model(
            "0b9f3dcf3d1bcef5",
            "ttsw-claude",
            "Claude 5.5 Sonnet (TT Switch)",
        )];
        let reserved = HashSet::from(["claude-5.5-sonnet-tt-switch".to_owned()]);
        assign_slugs(&mut models, &reserved);
        assert_ne!(models[0].advertised_slug, "claude-5.5-sonnet-tt-switch");
        assert!(models[0].advertised_slug.contains("0b9f3"));
    }

    #[test]
    fn hashes_take_precedence_over_other_models_aliases() {
        let models = [
            model("hash-a", "hash-b", "Model A"),
            model("hash-b", "other-id", "Model B"),
        ];
        assert_eq!(resolve(&models, "hash-b").unwrap(), "hash-b");
    }

    #[test]
    fn duplicate_ids_or_display_names_require_a_hash() {
        for (models, requested) in [
            (
                [
                    model("a", "gpt-x", "Model A"),
                    model("b", "gpt-x", "Model B"),
                ],
                "gpt-x",
            ),
            (
                [
                    model("a", "id-a", "My Model"),
                    model("b", "id-b", "my-model"),
                ],
                "my-model",
            ),
        ] {
            let error = resolve(&models, requested).unwrap_err().to_string();
            assert!(error.contains("ambiguous"));
            assert!(error.contains("a, b"));
            assert_eq!(resolve(&models, "b").unwrap(), "b");
        }
    }

    #[test]
    fn full_display_names_take_precedence_over_suffix_aliases() {
        let models = [
            model("a", "id-a", "My Model (Provider A)"),
            model("b", "id-b", "My Model (Provider B)"),
        ];
        assert_eq!(resolve(&models, "MY-MODEL (PROVIDER B)").unwrap(), "b");
    }

    #[test]
    fn kebab_slug_resolves_to_hash() {
        let mut models = [model(
            "0b9f3dcf3d1bcef5",
            "ttsw-claude",
            "Claude 5.5 Sonnet (TT Switch)",
        )];
        assign_slugs(&mut models, &HashSet::new());
        assert_eq!(
            resolve(&models, &models[0].advertised_slug).unwrap(),
            "0b9f3dcf3d1bcef5"
        );
    }

    #[test]
    fn hash6_collision_slug_resolves_when_advertised_is_clean() {
        let mut models = [model(
            "8c1030decdf858d7",
            "ttsw-codex-deepseek-v4.1-flash",
            "Deepseek V4.1 Flash (TT Switch)",
        )];
        assign_slugs(&mut models, &HashSet::new());
        assert_eq!(models[0].advertised_slug, "deepseek-v4.1-flash-tt-switch");
        assert_eq!(
            resolve(&models, "deepseek-v4.1-flash-tt-switch-8c1030").unwrap(),
            "8c1030decdf858d7"
        );
    }

    #[test]
    fn hash6_collision_slug_resolves_when_base_is_reserved() {
        let mut models = [model(
            "8c1030decdf858d7",
            "ttsw-codex-deepseek-v4.1-flash",
            "Deepseek V4.1 Flash (TT Switch)",
        )];
        let reserved = HashSet::from(["deepseek-v4.1-flash-tt-switch".to_owned()]);
        assign_slugs(&mut models, &reserved);
        assert_eq!(
            models[0].advertised_slug,
            "deepseek-v4.1-flash-tt-switch-8c1030"
        );
        assert_eq!(
            resolve(&models, "deepseek-v4.1-flash-tt-switch-8c1030").unwrap(),
            "8c1030decdf858d7"
        );
    }

    #[test]
    fn bare_kebab_does_not_match_tt_switch_display_via_suffix() {
        let models = [model(
            "4c3368f26587305a",
            "gpt-sol",
            "GPT 5.6 Sol Medium (TT Switch)",
        )];
        assert_eq!(
            resolve(&models, "gpt-5.6-sol-medium").unwrap(),
            "gpt-5.6-sol-medium"
        );
    }

    #[test]
    fn display_name_without_suffix_resolves_when_unique() {
        let models = [model(
            "0b9f3dcf3d1bcef5",
            "ttsw-claude",
            "Claude 5.5 Sonnet (TT Switch)",
        )];
        assert_eq!(
            resolve(&models, "Claude 5.5 Sonnet").unwrap(),
            "0b9f3dcf3d1bcef5"
        );
    }
}
