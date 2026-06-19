use crate::config::HypernetworkConfig;
use crate::hypernetwork::{save_lora_layers, Code2LoRAHead, LoRAWeights};
use crate::repo_encoder::RepoEmbedding;
use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::rnn::GRUState;
use candle_nn::{gru, layer_norm, GRUConfig, LayerNorm, Linear, VarBuilder, VarMap, GRU, RNN};
use std::collections::HashMap;
use std::path::Path;

/// Recurrent Code2LoRA-Evo hypernetwork.
///
/// The paper's Evo variant initializes a repository state from a snapshot
/// embedding, updates that state with one GRU step per commit diff embedding,
/// then feeds the state into the LoRA projection head. This lightweight version
/// keeps the repo embedding dimension at this project's MiniLM 768-dim shape.
pub struct Code2LoRAEvo {
    init_proj: Linear,
    init_norm: LayerNorm,
    diff_proj: Linear,
    diff_norm: LayerNorm,
    gru: GRU,
    state_norm: LayerNorm,
    head: Code2LoRAHead,
    embedding_dim: usize,
    hidden_dim: usize,
    varmap: VarMap,
}

impl Code2LoRAEvo {
    pub fn new(vb: VarBuilder, config: &HypernetworkConfig, varmap: &VarMap) -> Result<Self> {
        let embedding_dim = config.repo_embed_dim;
        let hidden_dim = config.hidden_dim;
        let init_proj = candle_nn::linear(embedding_dim, hidden_dim, vb.pp("evo.init_proj"))?;
        let init_norm = layer_norm(hidden_dim, 1e-5, vb.pp("evo.init_norm"))?;
        let diff_proj = candle_nn::linear(embedding_dim, hidden_dim, vb.pp("evo.diff_proj"))?;
        let diff_norm = layer_norm(hidden_dim, 1e-5, vb.pp("evo.diff_norm"))?;
        let gru = gru(
            hidden_dim,
            hidden_dim,
            GRUConfig::default(),
            vb.pp("evo.gru"),
        )?;
        let state_norm = layer_norm(hidden_dim, 1e-5, vb.pp("evo.state_norm"))?;

        let mut head_config = config.clone();
        head_config.repo_embed_dim = hidden_dim;
        let head = Code2LoRAHead::new(vb.pp("evo.head"), &head_config, varmap)?;

        Ok(Self {
            init_proj,
            init_norm,
            diff_proj,
            diff_norm,
            gru,
            state_norm,
            head,
            embedding_dim,
            hidden_dim,
            varmap: varmap.clone(),
        })
    }

    pub fn init_state(&self, repo_embedding: &Tensor) -> Result<Tensor> {
        self.ensure_embedding_shape(repo_embedding, "repo_embedding")?;
        let state = self.init_proj.forward(repo_embedding)?.gelu()?;
        Ok(self.init_norm.forward(&state)?)
    }

    pub fn update_state(&self, previous_state: &Tensor, diff_embedding: &Tensor) -> Result<Tensor> {
        self.ensure_state_shape(previous_state)?;
        self.ensure_embedding_shape(diff_embedding, "diff_embedding")?;
        let projected = self.diff_proj.forward(diff_embedding)?;
        let projected = self.diff_norm.forward(&projected)?;
        let state = GRUState {
            h: previous_state.clone(),
        };
        Ok(self.gru.step(&projected, &state)?.h)
    }

    pub fn update_sequence(
        &self,
        initial_state: &Tensor,
        diff_embeddings: &[Tensor],
    ) -> Result<Tensor> {
        let mut state = initial_state.clone();
        for diff in diff_embeddings {
            state = self.update_state(&state, diff)?;
        }
        Ok(state)
    }

    pub fn adapters_from_state(&self, state: &Tensor) -> Result<Vec<LoRAWeights>> {
        self.ensure_state_shape(state)?;
        let normalized = self.state_norm.forward(state)?;
        self.head.forward_all(&normalized)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn load(
        path: &Path,
        config: &HypernetworkConfig,
        dtype: DType,
        device: &Device,
    ) -> Result<(Self, VarMap)> {
        let mut varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, device);
        let evo = Self::new(vb, config, &varmap)?;
        varmap
            .load(path)
            .with_context(|| format!("Failed to load Evo checkpoint {}", path.display()))?;
        Ok((evo, varmap))
    }

    fn ensure_embedding_shape(&self, tensor: &Tensor, name: &str) -> Result<()> {
        anyhow::ensure!(
            tensor.dims() == [1, self.embedding_dim],
            "{name} must have shape [1, {}], got {:?}",
            self.embedding_dim,
            tensor.dims()
        );
        Ok(())
    }

    fn ensure_state_shape(&self, tensor: &Tensor) -> Result<()> {
        anyhow::ensure!(
            tensor.dims() == [1, self.hidden_dim],
            "Evo state must have shape [1, {}], got {:?}",
            self.hidden_dim,
            tensor.dims()
        );
        Ok(())
    }
}

pub fn save_evo_state(state: &Tensor, path: &Path) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    tensors.insert("state".to_string(), state.clone());
    candle_core::safetensors::save(&tensors, path)?;
    Ok(())
}

pub fn load_evo_state(path: &Path, device: &Device) -> Result<Tensor> {
    let tensors = candle_core::safetensors::load(path, device)?;
    tensors
        .get("state")
        .cloned()
        .with_context(|| format!("Missing `state` tensor in {}", path.display()))
}

pub fn load_embedding_tensor(path: &Path, device: &Device) -> Result<Tensor> {
    RepoEmbedding::load(path)?.to_tensor(device)
}

pub fn save_evo_adapter(adapter: &[LoRAWeights], path: &Path) -> Result<()> {
    save_lora_layers(adapter, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hypernetwork::load_lora_layers;

    fn tiny_config() -> HypernetworkConfig {
        HypernetworkConfig {
            hidden_dim: 16,
            rank: 2,
            num_layers: 2,
            repo_embed_dim: 8,
            llm_hidden_dim: 8,
            llm_intermediate_dim: 12,
            kv_proj_dim: 4,
        }
    }

    #[test]
    fn test_evo_gru_updates_state_and_generates_adapters() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let evo = Code2LoRAEvo::new(vb, &config, &varmap)?;

        let repo = Tensor::rand(0f32, 1.0, (1, config.repo_embed_dim), &device)?;
        let diff1 = Tensor::rand(0f32, 1.0, (1, config.repo_embed_dim), &device)?;
        let diff2 = Tensor::rand(0f32, 1.0, (1, config.repo_embed_dim), &device)?;

        let state0 = evo.init_state(&repo)?;
        let state1 = evo.update_state(&state0, &diff1)?;
        let state2 = evo.update_sequence(&state1, &[diff2])?;

        assert_eq!(state0.dims(), &[1, config.hidden_dim]);
        assert_eq!(state2.dims(), &[1, config.hidden_dim]);
        let delta: f32 = state0.sub(&state2)?.abs()?.sum_all()?.to_scalar()?;
        assert!(delta > 0.001, "GRU diff updates should change hidden state");

        let adapters = evo.adapters_from_state(&state2)?;
        assert_eq!(adapters.len(), config.num_layers);
        assert_eq!(
            adapters[0].q.0.dims(),
            &[config.rank, config.llm_hidden_dim]
        );
        assert_eq!(
            adapters[0].down.1.dims(),
            &[config.llm_hidden_dim, config.rank]
        );
        Ok(())
    }

    #[test]
    fn test_evo_checkpoint_state_and_adapter_round_trip() -> Result<()> {
        let device = Device::Cpu;
        let config = tiny_config();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let evo = Code2LoRAEvo::new(vb, &config, &varmap)?;
        let root = std::env::temp_dir().join(format!("code2lora-evo-test-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        let ckpt = root.join("evo.safetensors");
        let state_path = root.join("state.safetensors");
        let adapter_path = root.join("adapter.safetensors");

        evo.save(&ckpt)?;
        let (loaded, _loaded_varmap) = Code2LoRAEvo::load(&ckpt, &config, DType::F32, &device)?;

        let repo = Tensor::rand(0f32, 1.0, (1, config.repo_embed_dim), &device)?;
        let diff = Tensor::rand(0f32, 1.0, (1, config.repo_embed_dim), &device)?;
        let state = loaded.update_state(&loaded.init_state(&repo)?, &diff)?;
        save_evo_state(&state, &state_path)?;
        let loaded_state = load_evo_state(&state_path, &device)?;
        assert_eq!(loaded_state.dims(), &[1, config.hidden_dim]);

        let adapters = loaded.adapters_from_state(&loaded_state)?;
        save_evo_adapter(&adapters, &adapter_path)?;
        let round_trip = load_lora_layers(&adapter_path, &device)?;
        assert_eq!(round_trip.len(), config.num_layers);
        assert_eq!(round_trip[1].k.1.dims(), &[config.kv_proj_dim, config.rank]);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }
}
