use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use log::info;
use serde_json::json;
use std::path::Path;

use crate::base_llm::Code2LoRAModel;
use crate::config::EvoTrainConfig;
use crate::dataset::{CodeDataset, EvolutionCommitExample, EvolutionSequence};
use crate::evo::Code2LoRAEvo;
use crate::repo_encoder::RepoEncoder;

/// Training loop for Code2LoRA-Evo.
///
/// The GRU state is initialized from the first repository snapshot embedding,
/// then updated once per commit diff embedding. Loss is computed after each
/// update using the adapter generated from the current state.
pub struct EvoTrainer {
    pub evo: Code2LoRAEvo,
    pub base_model: Code2LoRAModel,
    pub config: EvoTrainConfig,
    pub device: Device,
    pub varmap: VarMap,
    diff_encoder: Option<RepoEncoder>,
}

impl EvoTrainer {
    pub fn new(
        evo: Code2LoRAEvo,
        base_model: Code2LoRAModel,
        varmap: VarMap,
        config: EvoTrainConfig,
        device: Device,
    ) -> Self {
        Self {
            evo,
            base_model,
            config,
            device,
            varmap,
            diff_encoder: None,
        }
    }

    pub fn train(&mut self, dataset: &CodeDataset) -> Result<()> {
        let mut sequences = dataset.evolution_sequences();
        anyhow::ensure!(
            !sequences.is_empty(),
            "Evo training requires commit-indexed rows. Use the commit-joined RepoPeftBench JSONL."
        );
        if let Some(max_sequences) = self.config.max_sequences {
            sequences.truncate(max_sequences.max(1));
        }

        let checkpoint_dir = Path::new(&self.config.output).to_path_buf();
        std::fs::create_dir_all(&checkpoint_dir)?;
        let truncation_steps = self.config.truncation_steps.max(1);
        info!(
            "Evo training config: base_model={}, data_dir={}, rank={}, sequences={}, truncation_steps={}, lr={}",
            self.config.base_model,
            self.config.data_dir,
            self.config.rank,
            sequences.len(),
            truncation_steps,
            self.config.lr
        );

        let vars = self.varmap.all_vars();
        let params = ParamsAdamW {
            lr: self.config.lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let mut opt = AdamW::new(vars, params)?;
        let mut epoch_metrics = Vec::new();

        for epoch in 0..self.config.epochs {
            let mut train_loss_sum = 0.0f64;
            let mut train_steps = 0usize;

            for sequence in &sequences {
                info!(
                    "  Evo sequence repo={} commits={}",
                    sequence.repo_id,
                    sequence.commits.len()
                );
                let (sum, steps) = self.train_sequence(sequence, truncation_steps, &mut opt)?;
                train_loss_sum += sum;
                train_steps += steps;
            }

            let train_loss = average_loss(train_loss_sum, train_steps);
            let eval = self.evaluate_sequences(&sequences)?;
            info!(
                "Evo epoch {}/{} - train_loss={:.6}, eval_loss={}",
                epoch + 1,
                self.config.epochs,
                train_loss,
                format_optional(eval)
            );

            epoch_metrics.push(json!({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_steps": train_steps,
                "eval_loss": eval,
            }));

            if (epoch + 1) % 5 == 0 {
                let ckpt_path =
                    checkpoint_dir.join(format!("evo_epoch_{:04}.safetensors", epoch + 1));
                self.evo.save(&ckpt_path)?;
                info!("Evo checkpoint saved to {ckpt_path:?}");
            }
        }

        let final_path = checkpoint_dir.join("evo_final.safetensors");
        self.evo.save(&final_path)?;
        let metrics_path = checkpoint_dir.join("evo_metrics.json");
        std::fs::write(
            &metrics_path,
            serde_json::to_string_pretty(&json!({
                "epochs": epoch_metrics,
                "sequence_count": sequences.len(),
                "truncation_steps": truncation_steps,
                "final_checkpoint": final_path.display().to_string(),
            }))?,
        )?;
        info!("Final Evo checkpoint saved to {final_path:?}");
        info!("Evo metrics saved to {metrics_path:?}");
        Ok(())
    }

    fn train_sequence(
        &mut self,
        sequence: &EvolutionSequence,
        truncation_steps: usize,
        opt: &mut AdamW,
    ) -> Result<(f64, usize)> {
        let mut state = self
            .evo
            .init_state(&sequence.initial_repo_embedding.to_tensor(&self.device)?)?;
        let mut chunk_loss: Option<Tensor> = None;
        let mut chunk_steps = 0usize;
        let mut loss_sum = 0.0f64;
        let mut loss_steps = 0usize;

        for commit in sequence
            .commits
            .iter()
            .filter(|c| is_training_split(&c.split))
        {
            state = self.update_state_for_commit(&state, commit)?;
            let adapter = self.evo.adapters_from_state(&state)?;
            let loss = self
                .base_model
                .compute_ir_loss_with_lora(&adapter, &commit.code_content)?;
            chunk_loss = match chunk_loss {
                Some(acc) => Some((acc + loss)?),
                None => Some(loss),
            };
            chunk_steps += 1;

            if chunk_steps >= truncation_steps {
                let avg = (chunk_loss.context("missing Evo chunk loss")? / chunk_steps as f64)?;
                let avg_value = avg.to_scalar::<f32>()? as f64;
                opt.backward_step(&avg)?;
                state = state.detach();
                loss_sum += avg_value;
                loss_steps += 1;
                chunk_loss = None;
                chunk_steps = 0;
            }
        }

        if chunk_steps > 0 {
            let avg = (chunk_loss.context("missing Evo final chunk loss")? / chunk_steps as f64)?;
            let avg_value = avg.to_scalar::<f32>()? as f64;
            opt.backward_step(&avg)?;
            loss_sum += avg_value;
            loss_steps += 1;
        }

        Ok((loss_sum, loss_steps))
    }

    fn evaluate_sequences(&mut self, sequences: &[EvolutionSequence]) -> Result<Option<f64>> {
        let mut loss_sum = 0.0f64;
        let mut loss_steps = 0usize;

        for sequence in sequences {
            let mut state = self
                .evo
                .init_state(&sequence.initial_repo_embedding.to_tensor(&self.device)?)?;
            for commit in &sequence.commits {
                state = self.update_state_for_commit(&state, commit)?;
                if is_training_split(&commit.split) {
                    continue;
                }
                let adapter = self.evo.adapters_from_state(&state)?;
                let loss = self
                    .base_model
                    .compute_ir_loss_with_lora(&adapter, &commit.code_content)?;
                loss_sum += loss.to_scalar::<f32>()? as f64;
                loss_steps += 1;
            }
        }

        Ok((loss_steps > 0).then(|| average_loss(loss_sum, loss_steps)))
    }

    fn update_state_for_commit(
        &mut self,
        state: &Tensor,
        commit: &EvolutionCommitExample,
    ) -> Result<Tensor> {
        let diff = match &commit.diff_embedding {
            Some(embedding) => embedding.to_tensor(&self.device)?,
            None => {
                let diff_text = commit.diff_text.as_deref().with_context(|| {
                    format!(
                        "missing diff_embedding and production_code_diff for repo={} commit_index={}",
                        commit.repo_id, commit.commit_index
                    )
                })?;
                if self.diff_encoder.is_none() {
                    info!(
                        "Loading MiniLM diff encoder for commit rows without precomputed diff embeddings"
                    );
                    self.diff_encoder = Some(RepoEncoder::new(&Device::Cpu)?);
                }
                self.diff_encoder
                    .as_ref()
                    .context("diff encoder was not initialized")?
                    .embed_text_as_repo(diff_text)?
                    .to_tensor(&self.device)?
            }
        };
        self.evo.update_state(state, &diff).with_context(|| {
            format!(
                "failed to update Evo state for repo={} commit_index={}",
                commit.repo_id, commit.commit_index
            )
        })
    }
}

fn average_loss(loss_sum: f64, steps: usize) -> f64 {
    if steps == 0 {
        f64::NAN
    } else {
        loss_sum / steps as f64
    }
}

fn format_optional(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn is_training_split(split: &str) -> bool {
    matches!(
        split.to_ascii_lowercase().as_str(),
        "" | "train" | "ir_train" | "in_repo_train"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HypernetworkConfig;
    use crate::dataset::CodeExample;
    use crate::qwen2_lora::LoRAModel;
    use crate::repo_encoder::RepoEmbedding;
    use candle_core::DType;
    use candle_nn::{VarBuilder, VarMap};
    use candle_transformers::models::qwen2 as qwen2_model;
    use tokenizers::Tokenizer;

    #[test]
    fn test_tiny_evo_trainer_runs_commit_sequence() -> Result<()> {
        let device = Device::Cpu;
        let hn_config = HypernetworkConfig {
            hidden_dim: 8,
            rank: 2,
            num_layers: 1,
            repo_embed_dim: 4,
            llm_hidden_dim: 16,
            llm_intermediate_dim: 32,
            kv_proj_dim: 8,
        };
        let qwen_cfg = qwen2_model::Config {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            vocab_size: 32,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            sliding_window: 0,
            max_window_layers: 0,
            tie_word_embeddings: false,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        };

        let model_varmap = VarMap::new();
        let model_vb = VarBuilder::from_varmap(&model_varmap, DType::F32, &device);
        let base_model = LoRAModel::new(&qwen_cfg, model_vb.clone())?;
        let lm_head = candle_nn::linear_no_bias(16, 32, model_vb.pp("lm_head"))?;
        let tokenizer = tiny_tokenizer()?;
        let base_model = Code2LoRAModel {
            base_model,
            lm_head,
            tokenizer,
            device: device.clone(),
            config: qwen_cfg,
        };

        let evo_varmap = VarMap::new();
        let evo_vb = VarBuilder::from_varmap(&evo_varmap, DType::F32, &device);
        let evo = Code2LoRAEvo::new(evo_vb, &hn_config, &evo_varmap)?;
        let dataset = CodeDataset::from_examples(vec![
            CodeExample {
                repo_id: "owner/evo".into(),
                repo_embedding: RepoEmbedding {
                    data: vec![0.1, 0.2, 0.3, 0.4],
                },
                diff_embedding: Some(RepoEmbedding {
                    data: vec![0.2, 0.1, 0.0, 0.3],
                }),
                diff_text: Some("diff --git a/a.py b/a.py".into()),
                code_content: "def test return value".into(),
                language: "python".into(),
                split: "train".into(),
                commit_index: Some(0),
            },
            CodeExample {
                repo_id: "owner/evo".into(),
                repo_embedding: RepoEmbedding {
                    data: vec![0.1, 0.2, 0.3, 0.4],
                },
                diff_embedding: Some(RepoEmbedding {
                    data: vec![0.4, 0.3, 0.2, 0.1],
                }),
                diff_text: Some("diff --git a/b.py b/b.py".into()),
                code_content: "assert value equals value".into(),
                language: "python".into(),
                split: "train".into(),
                commit_index: Some(1),
            },
        ]);
        let output =
            std::env::temp_dir().join(format!("code2lora-evo-trainer-test-{}", std::process::id()));
        std::fs::remove_dir_all(&output).ok();

        let config = EvoTrainConfig {
            data_dir: "inline".into(),
            base_model: "tiny-qwen2".into(),
            output: output.display().to_string(),
            rank: hn_config.rank,
            epochs: 1,
            lr: 1e-3,
            truncation_steps: 1,
            max_sequences: None,
        };
        let mut trainer = EvoTrainer::new(evo, base_model, evo_varmap, config, device);
        trainer.train(&dataset)?;

        assert!(output.join("evo_final.safetensors").exists());
        assert!(output.join("evo_metrics.json").exists());
        std::fs::remove_dir_all(&output).ok();
        Ok(())
    }

    fn tiny_tokenizer() -> Result<Tokenizer> {
        let json = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "[UNK]": 0,
                    "def": 1,
                    "test": 2,
                    "return": 3,
                    "value": 4,
                    "assert": 5,
                    "equals": 6
                },
                "unk_token": "[UNK]"
            }
        }"#;
        Tokenizer::from_bytes(json.as_bytes())
            .map_err(|e| anyhow::anyhow!("tiny tokenizer load failed: {e}"))
    }
}
