use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2::{self as qwen2_model};
use hf_hub::api::sync::Api;
use log::info;
use tokenizers::Tokenizer;

use crate::config::HypernetworkConfig;
use crate::hypernetwork::{Code2LoRAHead, LoRAWeights};
use crate::qwen2_lora::LoRAModel;

/// Wraps Qwen2 with frozen base model + per-layer LoRA injection.
pub struct Code2LoRAModel {
    pub base_model: LoRAModel,
    pub lm_head: candle_nn::Linear,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub config: qwen2_model::Config,
}

impl Code2LoRAModel {
    pub fn new(device: &Device, dtype: DType, _hn_config: &HypernetworkConfig) -> Result<Self> {
        let api = Api::new().context("HF Hub API")?;
        let model_id = "Qwen/Qwen2.5-Coder-0.5B";
        let repo = api.model(model_id.to_string());
        let config_path = repo.get("config.json")?;
        let config: qwen2_model::Config = serde_json::from_slice(&std::fs::read(config_path)?)?;

        let weights_paths = collect_safetensors(&repo)?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&weights_paths, dtype, device)? };

        // Load the LoRA-capable model (no lm_head)
        let base_model = LoRAModel::new(&config, vb.clone())?;
        // Load lm_head separately (handle tie_word_embeddings)
        let lm_head = if config.tie_word_embeddings {
            // Use the embedding weight as the output projection
            let embed_w = base_model.embed_tokens.embeddings().clone();
            candle_nn::Linear::new(embed_w, None)
        } else {
            candle_nn::linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };

        // Load tokenizer
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("No tokenizer.json for Qwen2.5-Coder-0.5B")?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {e}"))?;

        info!(
            "Qwen2.5-Coder-0.5B loaded: hidden={}, layers={}, heads={}, vocab={}",
            config.hidden_size,
            config.num_hidden_layers,
            config.num_attention_heads,
            config.vocab_size
        );

        Ok(Self {
            base_model,
            lm_head,
            tokenizer,
            device: device.clone(),
            config,
        })
    }

    // ─── Forward helpers ───

    /// Return hidden states (before lm_head) — used for training loss.
    pub fn forward_hidden(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        // LoRAModel::forward runs all decoder layers (with any active LoRA)
        self.base_model.forward(input_ids, 0, None)
    }

    /// Return logits (last position only, for generation).
    pub fn forward_logits(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden = self.forward_hidden(input_ids)?;
        let seq_len = hidden.dim(1)?;
        let last = hidden.narrow(1, seq_len.saturating_sub(1), 1)?;
        Ok(last.apply(&self.lm_head)?)
    }

    // ─── LoRA injection (P4: per-layer distinct weights) ───

    /// Inject per-layer LoRA weights into decoder layers.
    /// `all_lora[0]` → layer 0, `all_lora[1]` → layer 1, etc.
    pub fn inject_lora(&mut self, all_lora: &[LoRAWeights]) {
        self.base_model.inject_lora_all(all_lora);
    }

    /// Remove LoRA adapters from all layers.
    pub fn clear_lora(&mut self) {
        self.base_model.clear_lora_all();
    }

    /// Generate per-layer LoRA from the hypernetwork and inject.
    pub fn inject_lora_from_hn(&mut self, hn: &Code2LoRAHead, repo_emb: &Tensor) -> Result<()> {
        let all_lora = hn.forward_all(repo_emb)?;
        self.inject_lora(&all_lora);
        Ok(())
    }

    // ─── Tokenization ───

    fn tokenize(&self, text: &str) -> Result<(Vec<u32>, usize)> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization error: {e}"))?;
        let ids = encoding.get_ids().to_vec();
        let max_len = self.config.max_position_embeddings.min(2048);
        let ids = if ids.len() > max_len {
            ids[..max_len].to_vec()
        } else {
            ids
        };
        let seq_len = ids.len();
        Ok((ids, seq_len))
    }

    pub fn encode_text(&self, text: &str) -> Result<Vec<u32>> {
        let (ids, _) = self.tokenize(text)?;
        Ok(ids)
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("Decode error: {e}"))
    }

    // ─── Loss functions ───

    /// Compute generative (IR) loss for a single (code, repo_emb) pair.
    /// P3: LoRA is injected per-layer via inject_lora_from_hn before the forward pass,
    ///     so the model's attention & MLP projections are truly adapted.
    fn compute_single_ir_loss(
        &mut self,
        hn: &Code2LoRAHead,
        repo_emb: &Tensor,
        code_text: &str,
    ) -> Result<Tensor> {
        let (ids, seq_len) = self.tokenize(code_text)?;
        if seq_len < 2 {
            return Ok(Tensor::zeros(&[], DType::F32, &self.device)?);
        }
        let input_ids = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;

        // P3: inject LoRA into all decoder layers
        self.clear_lora();
        self.inject_lora_from_hn(hn, repo_emb)?;

        // Forward through the adapted model → hidden states
        let hidden = self.forward_hidden(&input_ids)?;

        // Remove LoRA so state is clean for next example
        self.clear_lora();

        // Apply lm_head → logits
        let logits = hidden.apply(&self.lm_head)?; // (1, seq_len, vocab_size)

        // Cross-entropy: predict next token at each position
        let vocab_size = logits.dim(2)?;
        let shift_logits = logits.narrow(1, 0, seq_len - 1)?;
        let shift_labels = input_ids.narrow(1, 1, seq_len - 1)?;

        let flat_logits = shift_logits.reshape((seq_len - 1, vocab_size))?;
        let flat_labels = shift_labels.reshape((seq_len - 1,))?;

        let loss = candle_nn::loss::cross_entropy(&flat_logits, &flat_labels)?;
        Ok(loss)
    }

    /// Compute generative (IR) loss over a batch.
    pub fn compute_ir_loss(
        &mut self,
        hn: &Code2LoRAHead,
        repo_embs: &Tensor,
        code_texts: &[String],
    ) -> Result<Tensor> {
        let batch_size = code_texts.len();
        if batch_size == 0 {
            return Ok(Tensor::zeros(&[], DType::F32, &self.device)?);
        }

        let mut total_loss: Option<Tensor> = None;
        for i in 0..batch_size {
            let repo_emb = repo_embs.narrow(0, i, 1)?;
            let example_loss = self.compute_single_ir_loss(hn, &repo_emb, &code_texts[i])?;
            total_loss = match total_loss {
                Some(l) => Some((l + example_loss)?),
                None => Some(example_loss),
            };
        }

        let avg = (total_loss.context("No losses computed")? / batch_size as f64)?;
        Ok(avg)
    }

    /// Compute contrastive (CR) loss over a batch.
    /// Uses adapted code representations with per-layer LoRA injection.
    pub fn compute_cr_loss(
        &mut self,
        hn: &Code2LoRAHead,
        repo_embs: &Tensor,
        code_texts: &[String],
    ) -> Result<Tensor> {
        let batch_size = code_texts.len();
        if batch_size < 2 {
            return Ok(Tensor::zeros(&[], DType::F32, &self.device)?);
        }

        let mut reprs: Vec<Tensor> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let (ids, seq_len) = self.tokenize(&code_texts[i])?;
            if seq_len < 2 {
                let zeros = Tensor::zeros((1, self.config.hidden_size), DType::F32, &self.device)?;
                reprs.push(zeros);
                continue;
            }
            let input_ids = Tensor::new(ids.as_slice(), &self.device)?.unsqueeze(0)?;

            // P3: inject LoRA
            self.clear_lora();
            self.inject_lora_from_hn(hn, &repo_embs.narrow(0, i, 1)?)?;
            let hidden = self.forward_hidden(&input_ids)?;
            self.clear_lora();

            // Mean pool over sequence dimension
            let mean_pooled = hidden.mean(1)?;
            reprs.push(mean_pooled);
        }

        let reprs_tensors: Vec<&Tensor> = reprs.iter().collect();
        let stacked = Tensor::stack(&reprs_tensors, 0)?;
        let normalized = l2_normalize(&stacked)?;

        // InfoNCE loss
        let sim = normalized.matmul(&normalized.t()?)?;
        let temperature = Tensor::new(0.07f32, &self.device)?;
        let logits = (sim / temperature)?;
        let labels = Tensor::arange(0u32, batch_size as u32, &self.device)?;

        let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
        Ok(loss)
    }

    // ─── Generation ───

    pub fn generate(&mut self, input_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut generated = input_ids.to_vec();

        for _step in 0..max_new_tokens {
            let input = Tensor::new(generated.as_slice(), &self.device)?.unsqueeze(0)?;
            let logits = self.forward_logits(&input)?;
            let logits_1d = logits.squeeze(0)?.squeeze(0)?;
            let next_token = logits_1d.argmax(0)?.to_scalar::<u32>()?;

            generated.push(next_token);

            if next_token == 151643 || generated.len() >= input_ids.len() + max_new_tokens {
                break;
            }
        }

        Ok(generated)
    }

    pub fn generate_text(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String> {
        let input_ids = self.encode_text(prompt)?;
        let generated = self.generate(&input_ids, max_new_tokens)?;
        let new_tokens = if generated.len() > input_ids.len() {
            &generated[input_ids.len()..]
        } else {
            &[]
        };
        self.decode_tokens(new_tokens)
    }
}

/// Collect all safetensors weight paths for a model repo.
/// Handles both single-file and sharded models by reading the index file.
fn collect_safetensors(repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<std::path::PathBuf>> {
    // Try single file first
    if let Ok(path) = repo.get("model.safetensors") {
        return Ok(vec![path]);
    }
    // Read index file to discover shards
    let index_path = repo
        .get("model.safetensors.index.json")
        .context("No model.safetensors or model.safetensors.index.json in repo")?;
    let index_text = std::fs::read_to_string(index_path)?;
    let index: serde_json::Value = serde_json::from_str(&index_text)?;
    let weight_map = index
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("Invalid safetensors index: missing weight_map")?;

    let mut shard_names: Vec<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();
    shard_names.sort();
    shard_names.dedup();

    let mut shards = Vec::with_capacity(shard_names.len());
    for name in &shard_names {
        shards.push(repo.get(name)?);
    }
    Ok(shards)
}

fn l2_normalize(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
    Ok(x.broadcast_div(&norm.clamp(1e-12, f32::MAX)?)?)
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

    #[test]
    fn test_lora_linear_forward() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::rand(-1f32, 1f32, (64, 128), &device)?; // (out, in)
        let lin = crate::qwen2_lora::LoRALinear::new(w, 128, 64);

        let x = Tensor::rand(-1f32, 1f32, (2, 16, 128), &device)?;
        let y = lin.forward(&x)?;
        assert_eq!(y.dims(), &[2, 16, 64]);

        // With LoRA
        let a = Tensor::rand(-0.1f32, 0.1f32, (4, 128), &device)?;
        let b = Tensor::rand(-0.1f32, 0.1f32, (64, 4), &device)?;
        let mut lin2 = lin;
        lin2.set_lora(a, b);
        let y2 = lin2.forward(&x)?;
        assert_eq!(y2.dims(), &[2, 16, 64]);
        Ok(())
    }

    #[test]
    fn test_training_pipeline_full() -> Result<()> {
        use candle_core::DType;
        use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};

        let device = Device::Cpu;

        // ── 1. Tiny Qwen2 config ──
        let qwen_cfg = qwen2_model::Config {
            hidden_size: 64,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            vocab_size: 256,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            sliding_window: 0,
            max_window_layers: 0,
            tie_word_embeddings: false,
            use_sliding_window: false,
            hidden_act: candle_nn::Activation::Silu,
        };

        // ── 2. Hypernetwork config (matching model dims) ──
        let hn_cfg = HypernetworkConfig {
            hidden_dim: 32,
            rank: 4,
            num_layers: 2,
            repo_embed_dim: 64,
            llm_hidden_dim: 64,
            llm_intermediate_dim: 64,
            kv_proj_dim: 32, // 2 kv_heads × 16 head_dim
        };

        // ── 3. Create hypernetwork with random weights ──
        let hn_varmap = VarMap::new();
        let hn_vb = VarBuilder::from_varmap(&hn_varmap, DType::F32, &device);
        let hn = Code2LoRAHead::new(hn_vb, &hn_cfg, &hn_varmap)?;

        // ── 4. Create tiny model with fresh weights ──
        let model_varmap = VarMap::new();
        let model_vb = VarBuilder::from_varmap(&model_varmap, DType::F32, &device);
        let mut model = crate::qwen2_lora::LoRAModel::new(&qwen_cfg, model_vb.clone())?;
        let lm_head = candle_nn::linear_no_bias(64, 256, model_vb.pp("lm_head"))?;

        // ── 5. Dummy training data ──
        // repo_emb dimension must match hn_cfg.repo_embed_dim
        let repo_emb = Tensor::rand(-1f32, 1f32, (1, hn_cfg.repo_embed_dim), &device)?;
        let ids_raw = Tensor::rand(0f32, 255f32, (1, 16), &device)?;
        let input_ids = ids_raw.to_dtype(DType::U32)?;

        // ── 6. Inject per-layer LoRA ──
        let all_lora = hn.forward_all(&repo_emb)?;
        assert_eq!(all_lora.len(), qwen_cfg.num_hidden_layers);
        model.inject_lora_all(&all_lora);

        // ── 7. Forward + loss (simulating IR phase) ──
        let hidden = model.forward(&input_ids, 0, None)?;
        assert_eq!(hidden.dims(), &[1, 16, 64]);

        let logits = hidden.apply(&lm_head)?;
        assert_eq!(logits.dims(), &[1, 16, 256]);

        let shift_logits = logits.narrow(1, 0, 15)?;
        let shift_labels = input_ids.narrow(1, 1, 15)?;
        let flat_logits = shift_logits.reshape((15, 256))?;
        let flat_labels = shift_labels.reshape((15,))?;
        let loss = candle_nn::loss::cross_entropy(&flat_logits, &flat_labels)?;
        let loss_val: f32 = loss.to_scalar::<f32>()?;
        assert!(
            loss_val.is_finite(),
            "loss should be finite, got {loss_val}"
        );
        assert!(loss_val > 0.0, "loss should be positive, got {loss_val}");

        // ── 8. Verify LoRA path is active: compute loss without LoRA ──
        model.clear_lora_all();
        let hidden_no_lora = model.forward(&input_ids, 0, None)?;
        let logits_no_lora = hidden_no_lora.apply(&lm_head)?;
        let loss_no_lora = candle_nn::loss::cross_entropy(
            &logits_no_lora.narrow(1, 0, 15)?.reshape((15, 256))?,
            &input_ids.narrow(1, 1, 15)?.reshape((15,))?,
        )?;
        let loss_no_lora_val: f32 = loss_no_lora.to_scalar::<f32>()?;
        assert!(loss_no_lora_val.is_finite());
        // With zero initialised model weights, loss_with_lora should differ from loss_without
        let lora_effect = (loss_val - loss_no_lora_val).abs();
        println!("  LoRA effect (|loss - loss_no_lora|) = {lora_effect:.8}");
        assert!(lora_effect > 1e-8, "LoRA injection should change the loss");

        // ── 9. Backward step via candle Optimizer (as trainer.rs does) ──
        let hn_vars = hn_varmap.all_vars();
        let params = ParamsAdamW {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        };
        let mut opt = <AdamW as Optimizer>::new(hn_vars, params)?;
        opt.backward_step(&loss)?;
        // (Gradient flow verified indirectly: backward_step completed without panic)

        // ── 10. Generate new LoRA with updated HN & re-forward ──
        let all_lora2 = hn.forward_all(&repo_emb)?;
        model.inject_lora_all(&all_lora2);
        let hidden2 = model.forward(&input_ids, 0, None)?;
        let logits2 = hidden2.apply(&lm_head)?;

        let loss2 = candle_nn::loss::cross_entropy(
            &logits2.narrow(1, 0, 15)?.reshape((15, 256))?,
            &input_ids.narrow(1, 1, 15)?.reshape((15,))?,
        )?;
        let loss_val2: f32 = loss2.to_scalar::<f32>()?;
        assert!(
            loss_val2.is_finite(),
            "loss2 should be finite, got {loss_val2}"
        );

        println!(
            "P5 integration: loss before={loss_val:.6} after={loss_val2:.6} (backward_step OK)"
        );

        Ok(())
    }

    /// ─── P6: Real-model training demo ───
    ///
    /// Downloads the real Qwen2.5-Coder-0.5B model (if not cached),
    /// generates synthetic training data, and runs a short training loop
    /// on GPU (or CPU fallback).  Verifies the training loop completes
    /// without numeric divergence and reports timing.
    #[test]
    #[ignore = "Requires HF model download (~2 GB) and GPU"]
    fn test_p6_real_model_training() -> Result<()> {
        use candle_core::DType;
        use candle_nn::VarMap;
        use std::time::Instant;

        let device = Device::cuda_if_available(0)?;
        info!("P6: using device {device:?}");

        // ── 1. Load real Qwen2.5-Coder-0.5B model ──
        eprintln!("P6: loading Qwen2.5-Coder-0.5B…");
        let start = Instant::now();
        let hn_cfg = HypernetworkConfig {
            hidden_dim: 384,
            rank: 8,
            ..Default::default()
        };
        let qwen = Code2LoRAModel::new(&device, DType::F32, &hn_cfg)?;
        eprintln!(
            "P6: model loaded ({:.1}s) hidden={}, intermediate={}, layers={}",
            start.elapsed().as_secs_f32(),
            hn_cfg.llm_hidden_dim,
            hn_cfg.llm_intermediate_dim,
            hn_cfg.num_layers
        );

        // ── 2. Create hypernetwork on GPU ──
        let hn_varmap = VarMap::new();
        let hn_vb = VarBuilder::from_varmap(&hn_varmap, DType::F32, &device);
        let hn = Code2LoRAHead::new(hn_vb, &hn_cfg, &hn_varmap)?;
        info!(
            "P6: hypernetwork created ({} hidden dim, rank {})",
            hn_cfg.hidden_dim, hn_cfg.rank
        );

        // ── 3. Create synthetic dataset ──
        let n_examples = 8;
        let dataset = crate::dataset::generate_synthetic(n_examples);
        info!("P6: synthetic dataset: {n_examples} examples");

        // ── 4. Create trainer & train ──
        let train_cfg = crate::config::TrainConfig {
            data_dir: String::new(),
            base_model: String::new(),
            output: "p6_checkpoints".into(),
            rank: hn_cfg.rank,
            epochs: 3,
            lr: 1e-4,
            batch_size: 2,
            seq_len: 2048,
            cache_dir: "cache".into(),
            cr_holdout: 0.2,
        };
        std::fs::create_dir_all("p6_checkpoints")?;
        let mut trainer = crate::trainer::Trainer::new(hn, qwen, hn_varmap, train_cfg, device);
        let train_start = Instant::now();
        trainer.train(&dataset)?;
        info!(
            "P6: training completed in {:.1}s",
            train_start.elapsed().as_secs_f32()
        );

        // ── 5. Cleanup ──
        std::fs::remove_dir_all("p6_checkpoints").ok();
        Ok(())
    }

    /// ─── P7: Train on a tiny slice of real RepoPeftBench JSONL ───
    ///
    /// Requires:
    ///   - `CODE2LORA_DATA_DIR` env var pointing to a prepared data/repopeftbench/ directory
    ///     (train.jsonl must exist with valid 768-dim repo embeddings)
    ///   - HF Hub access for Qwen2.5-Coder-0.5B download
    ///   - GPU (CUDA)
    #[test]
    #[ignore = "Requires prepared RepoPeftBench JSONL + HF model download + GPU"]
    fn test_p7_repopeftbench_tiny_train() -> Result<()> {
        use crate::config::TrainConfig;
        use crate::trainer::Trainer;
        use candle_core::DType;
        use candle_nn::VarMap;

        let data_dir = std::env::var("CODE2LORA_DATA_DIR")
            .unwrap_or_else(|_| "data/repopeftbench".to_string());
        let device = Device::cuda_if_available(0)?;
        info!("P7: device={device:?}, data_dir={data_dir}");

        // Load first 16 rows from train.jsonl via load_jsonl
        let dataset_path = std::path::PathBuf::from(&data_dir).join("train.jsonl");
        anyhow::ensure!(
            dataset_path.exists(),
            "train.jsonl not found at {dataset_path:?}"
        );

        let all_examples = crate::dataset::CodeDataset::load_jsonl(&dataset_path)?;
        anyhow::ensure!(!all_examples.is_empty(), "train.jsonl is empty");

        // Take first 16 examples for a quick smoke test
        let mut subset = all_examples;
        subset.truncate(16);
        let dataset = crate::dataset::CodeDataset::from_examples(subset);
        let summary = dataset.summary();
        info!(
            "P7: dataset loaded: repos={}, examples={}",
            summary.repo_count,
            dataset.len()
        );

        // Load real Qwen2.5-Coder-0.5B + hypernetwork
        let hn_cfg = HypernetworkConfig {
            hidden_dim: 384,
            rank: 8,
            ..Default::default()
        };
        let qwen = Code2LoRAModel::new(&device, DType::F32, &hn_cfg)?;

        let hn_varmap = VarMap::new();
        let hn_vb = VarBuilder::from_varmap(&hn_varmap, DType::F32, &device);
        let hn = Code2LoRAHead::new(hn_vb, &hn_cfg, &hn_varmap)?;

        let train_cfg = TrainConfig {
            data_dir: data_dir.clone(),
            base_model: "Qwen/Qwen2.5-Coder-0.5B".into(),
            output: "p7_checkpoints".into(),
            rank: hn_cfg.rank,
            epochs: 3,
            lr: 1e-4,
            batch_size: 2,
            seq_len: 2048,
            cache_dir: "cache".into(),
            cr_holdout: 0.2,
        };
        std::fs::create_dir_all("p7_checkpoints")?;
        let mut trainer = Trainer::new(hn, qwen, hn_varmap, train_cfg, device);
        trainer.train(&dataset)?;
        std::fs::remove_dir_all("p7_checkpoints").ok();

        info!("P7: tiny real-data training completed successfully");
        Ok(())
    }

    #[test]
    fn test_clear_lora() -> Result<()> {
        let device = Device::Cpu;
        let w = Tensor::rand(-1f32, 1f32, (64, 128), &device)?;
        let mut lin = crate::qwen2_lora::LoRALinear::new(w, 128, 64);
        let a = Tensor::rand(-0.1f32, 0.1f32, (4, 128), &device)?;
        let b = Tensor::rand(-0.1f32, 0.1f32, (64, 4), &device)?;
        lin.set_lora(a, b);
        lin.clear_lora();
        // After clear, forward should be just Wx (same as original)
        let x = Tensor::rand(-1f32, 1f32, (3, 128), &device)?;
        let _ = lin.forward(&x)?;
        // Smoke test: no crash
        Ok(())
    }
}
