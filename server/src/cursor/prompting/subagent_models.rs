//! Adds configured BYOK model identities to the Task tool description.

use crate::{cursor::tools::subagent_models::ByokModel, model::PromptSpec};

pub(crate) fn append_subagent_models(prompt: &mut PromptSpec, models: &[ByokModel]) {
    if models.is_empty() {
        return;
    }
    let Some(task) = prompt.tools.iter_mut().find(|tool| tool.name == "Task") else {
        return;
    };
    let mut models = models.to_vec();
    crate::cursor::tools::subagent_models::assign_slugs(
        &mut models,
        &std::collections::HashSet::new(),
    );
    models.sort_by(|left, right| left.hash.cmp(&right.hash));
    task.description.push_str(
        "\n\nAdditional available subagent models (BYOK)\n\n\
         These configured models are also available for explicit model requests. \
         Pass the exact hash shown as model, rather than the display name. \
         Display names and model IDs can be ambiguous.\n",
    );
    for model in models {
        let slug = if model.advertised_slug.is_empty() {
            model.hash.as_str()
        } else {
            model.advertised_slug.as_str()
        };
        task.description.push_str(&format!(
            "\n- {}: model=\"{}\" (hash=\"{}\")",
            serde_json::json!(model.display_name),
            slug,
            model.hash,
        ));
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        cursor::prompting::{Mode, PromptAssets, PromptCompiler},
        model::ModelSpec,
    };

    use super::*;

    #[test]
    fn lists_unique_hashes_without_changing_instructions_or_other_tools() {
        let compiler = PromptCompiler::new(PromptAssets::embedded().unwrap());
        let base = compiler
            .prompt_spec(Mode::Agent, &ModelSpec::new("parent"), &[], false)
            .unwrap();
        let models = ["b", "a"].map(|hash| ByokModel {
            hash: hash.into(),
            model_id: "gpt-x".into(),
            display_name: "Same\nName".into(),
            advertised_slug: String::new(),
        });
        let mut first = base.clone();
        append_subagent_models(&mut first, &models);
        let mut second = base.clone();
        append_subagent_models(&mut second, &[models[1].clone(), models[0].clone()]);
        let first_task = first
            .tools
            .iter()
            .find(|tool| tool.name == "Task")
            .map(|tool| tool.description.clone())
            .unwrap_or_default();
        let second_task = second
            .tools
            .iter()
            .find(|tool| tool.name == "Task")
            .map(|tool| tool.description.clone())
            .unwrap_or_default();
        assert_eq!(first_task, second_task);
        assert_eq!(first.instructions, base.instructions);
        for (original, rendered) in base.tools.iter().zip(&first.tools) {
            if original.name == "Task" {
                assert!(rendered.description.starts_with(&original.description));
                assert!(rendered.description.contains("(hash=\"a\")"));
                assert!(rendered.description.contains("(hash=\"b\")"));
            } else {
                assert_eq!(original, rendered);
            }
        }
        let mut empty = base.clone();
        append_subagent_models(&mut empty, &[]);
        assert_eq!(empty, base);
        let mut disabled = base;
        disabled.tools.retain(|tool| tool.name != "Task");
        let original = disabled.clone();
        append_subagent_models(&mut disabled, &models);
        assert_eq!(disabled, original);
    }
}
