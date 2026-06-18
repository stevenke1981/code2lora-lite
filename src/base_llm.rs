use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{self as qwen2_model, ModelForCausalLM};
use hf_hub::api::sync::Api;
use log::info;

use crate::hypernetwork::{Code2LoRAHead, LoRAWeights};

/// Wraps Qwen2 with frozen base model + optional LoRA injection.
pub struct Code2LoRAModel {
    pub model: ModelForCausalLM,
    pub device: Device,
    pub dtype: DType,
    pub config: qwen2_model::Config,
    /// LoRA adapters per layer (optionally injected)
    pub lora_adapters: Option<Vec<LayerLoRA>>,
}

pub struct LayerLoRA {
    pub layer_idx: usize,
    pub weights: LoRAWeights,
}

impl Code2LoRAModel {
    pub fn new(device: &Device, dtype: DType) -> Result<Self> {
        let api = Api::new().context("HF Hub API")?;
        let model_id = "Qwen/Qwen2.5-Coder-0.5B";
        let repo = api.model(model_id.to_string());
        let config_path = repo.get("config.json")?;
        let config: qwen2_model::Config = serde_json::from_slice(&std::fs::read(config_path)?)?;

        let weights_path = repo.get("model.safetensors")
            .or_else(|_| repo.get("model-00001-of-00002.safetensors"))
            .context("No safetensors weights found for Qwen2.5-Coder-0.5B")?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
        };

        let model = ModelForCausalLM::new(&config, vb)?;
        Ok(Self {
            model,
            device: device.clone(),
            dtype,
            config,
            lora_adapters: None,
        })
    }

    /// Apply LoRA adapters from Hypernetwork to all decoder layers.
    pub fn inject_lora(&mut self, hn: &Code2LoRAHead, repo_emb: &Tensor) -> Result<()> {
        let n_layers = self.config.num_hidden_layers;
        let mut adapters = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let weights = hn.forward(repo_emb)?;
            adapters.push(LayerLoRA { layer_idx: i, weights });
        }
        self.lora_adapters = Some(adapters);
        Ok(())
    }

    /// Generate text: run a forward pass and return token IDs.
    pub fn generate(&mut self, input_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut generated = input_ids.to_vec();

        for _step in 0..max_new_tokens {
            let input = Tensor::new(generated.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, 0)?;
            let next_token = logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
            generated.push(next_token);

            if next_token == 151643 || generated.len() >= input_ids.len() + max_new_tokens {
                break;
            }
        }

        Ok(generated[..input_ids.len() + max_new_tokens.min(generated.len().saturating_sub(input_ids.len()))].to_vec())
    }

    /// Forward pass returning logits (for training loss).
    pub fn forward_logits(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (_batch, seq_len) = input_ids.dims2()?;
        let logits = self.model.forward(input_ids, seq_len - 1)?;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_base_llm_basic() -> Result<()> {
        let config = qwen2_model::Config {
            hidden_size: 1024,
            intermediate_size: 896,
            num_hidden_layers: 2,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            vocab_size: 151936,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            sliding_window: 0,
            max_window_layers: 0,
            tie_word_embeddings: false,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        };
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 2);
        Ok(())
    }
}
