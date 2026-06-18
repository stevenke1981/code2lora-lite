use crate::config::HypernetworkConfig;
use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;

pub type LoRAPair = (Tensor, Tensor);

#[derive(Debug, Clone)]
pub struct LoRAWeights {
    pub q: LoRAPair,
    pub k: LoRAPair,
    pub v: LoRAPair,
    pub o: LoRAPair,
    pub gate: LoRAPair,
    pub up: LoRAPair,
    pub down: LoRAPair,
}

/// Hypernetwork: repo_embedding → per-layer LoRA adapter.
pub struct Code2LoRAHead {
    linear1: Linear,
    linear2: Linear,
    layer_emb: candle_nn::Embedding,
    heads_a: HashMap<String, Linear>,
    heads_b: HashMap<String, Linear>,
    log_scale_a: HashMap<String, Tensor>,
    log_scale_b: HashMap<String, Tensor>, // stored as learnable scalar tensors
    config: HypernetworkConfig,
    varmap: VarMap,
}

impl Code2LoRAHead {
    pub fn new(vb: VarBuilder, config: &HypernetworkConfig, varmap: &VarMap) -> Result<Self> {
        let hidden_dim = config.hidden_dim;
        let rank = config.rank;
        let input_dim = config.repo_embed_dim;
        let dtype = vb.dtype();

        // Shared MLP
        let linear1 = candle_nn::linear(input_dim, hidden_dim, vb.pp("mlp.0"))?;
        let linear2 = candle_nn::linear(hidden_dim, hidden_dim, vb.pp("mlp.2"))?;

        let mut heads_a = HashMap::new();
        let mut heads_b = HashMap::new();
        let mut log_scale_a = HashMap::new();
        let mut log_scale_b = HashMap::new();

        for &mod_type in HypernetworkConfig::MODULE_TYPES {
            let in_dim = config.lora_in_dim(mod_type);
            let out_dim = config.lora_out_dim(mod_type);
            // A: hidden → rank × in_dim   (A matrix: LoRA projection from in_dim to rank)
            let ha = candle_nn::linear(
                hidden_dim,
                rank * in_dim,
                vb.pp(format!("head_{mod_type}_a")),
            )?;
            // B: hidden → out_dim × rank   (B matrix: LoRA projection from rank to out_dim)
            let hb = candle_nn::linear(
                hidden_dim,
                out_dim * rank,
                vb.pp(format!("head_{mod_type}_b")),
            )?;

            heads_a.insert(mod_type.to_string(), ha);
            heads_b.insert(mod_type.to_string(), hb);

            let sa = vb.get(&[1], &format!("log_scale_{mod_type}_a"))?;
            let sb = vb.get(&[1], &format!("log_scale_{mod_type}_b"))?;
            log_scale_a.insert(mod_type.to_string(), sa.to_dtype(dtype)?);
            log_scale_b.insert(mod_type.to_string(), sb.to_dtype(dtype)?);
        }

        let layer_emb =
            candle_nn::embedding(config.num_layers, config.hidden_dim, vb.pp("layer_emb"))?;

        Ok(Self {
            linear1,
            linear2,
            layer_emb,
            heads_a,
            heads_b,
            log_scale_a,
            log_scale_b,
            config: config.clone(),
            varmap: varmap.clone(),
        })
    }

    /// Generate LoRA weights for layer 0 only (legacy convenience).
    pub fn forward(&self, repo_emb: &Tensor) -> Result<LoRAWeights> {
        let h = self.linear1.forward(repo_emb)?;
        let h = h.gelu()?;
        let h = self.linear2.forward(&h)?;
        // L2Norm + scale
        let norm = h.sqr()?.sum_keepdim(1)?.sqrt()?;
        let h = h.broadcast_div(&norm)?;
        let scale = (self.config.hidden_dim as f64).sqrt();
        let h = (h * scale)?;

        let q = self.gen_pair(&h, "q")?;
        let k = self.gen_pair(&h, "k")?;
        let v = self.gen_pair(&h, "v")?;
        let o = self.gen_pair(&h, "o")?;
        let gate = self.gen_pair(&h, "gate")?;
        let up = self.gen_pair(&h, "up")?;
        let down = self.gen_pair(&h, "down")?;

        Ok(LoRAWeights {
            q,
            k,
            v,
            o,
            gate,
            up,
            down,
        })
    }

    /// Generate distinct LoRA weights for every decoder layer.
    /// Each layer gets its own (A, B) pairs for all 7 projections,
    /// differentiated by a learned layer embedding.
    pub fn forward_all(&self, repo_emb: &Tensor) -> Result<Vec<LoRAWeights>> {
        let dev = repo_emb.device();
        let h = self.linear1.forward(repo_emb)?;
        let h = h.gelu()?;
        let h = self.linear2.forward(&h)?;
        // L2Norm + scale (shared base)
        let norm = h.sqr()?.sum_keepdim(1)?.sqrt()?;
        let base = h.broadcast_div(&norm)?;
        let scale = (self.config.hidden_dim as f64).sqrt();
        let base = (base * scale)?;

        let num_layers = self.config.num_layers;
        let mut all_weights = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            // Per-layer offset from learnable embedding table
            let idx_t = Tensor::new(&[layer_idx as u32], dev)?;
            let layer_emb = self.layer_emb.forward(&idx_t)?; // (1, hidden_dim)
            let h_layer = (&base + layer_emb)?;

            let q = self.gen_pair(&h_layer, "q")?;
            let k = self.gen_pair(&h_layer, "k")?;
            let v = self.gen_pair(&h_layer, "v")?;
            let o = self.gen_pair(&h_layer, "o")?;
            let gate = self.gen_pair(&h_layer, "gate")?;
            let up = self.gen_pair(&h_layer, "up")?;
            let down = self.gen_pair(&h_layer, "down")?;

            all_weights.push(LoRAWeights {
                q,
                k,
                v,
                o,
                gate,
                up,
                down,
            });
        }
        Ok(all_weights)
    }

    fn gen_pair(&self, h: &Tensor, mod_type: &str) -> Result<LoRAPair> {
        let in_dim = self.config.lora_in_dim(mod_type);
        let out_dim = self.config.lora_out_dim(mod_type);
        let rank = self.config.rank;

        let ha = self.heads_a.get(mod_type).context("Missing head A")?;
        let hb = self.heads_b.get(mod_type).context("Missing head B")?;
        let sa = self.log_scale_a.get(mod_type).context("Missing scale A")?;
        let sb = self.log_scale_b.get(mod_type).context("Missing scale B")?;

        let a_raw = ha.forward(h)?;
        let b_raw = hb.forward(h)?;

        // A: (rank, in_dim),  B: (out_dim, rank)
        let a = a_raw.reshape((rank, in_dim))?;
        let b = b_raw.reshape((out_dim, rank))?;

        let a = a.tanh()?.broadcast_mul(&sa.exp()?)?;
        let b = b.tanh()?.broadcast_mul(&sb.exp()?)?;

        Ok((a, b))
    }

    /// Save to safetensors via VarMap
    pub fn save(&self, path: &Path) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }
}

pub fn save_lora_layers(all_lora: &[LoRAWeights], path: &Path) -> Result<()> {
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (layer_idx, lora) in all_lora.iter().enumerate() {
        insert_pair(&mut tensors, layer_idx, "q", &lora.q);
        insert_pair(&mut tensors, layer_idx, "k", &lora.k);
        insert_pair(&mut tensors, layer_idx, "v", &lora.v);
        insert_pair(&mut tensors, layer_idx, "o", &lora.o);
        insert_pair(&mut tensors, layer_idx, "gate", &lora.gate);
        insert_pair(&mut tensors, layer_idx, "up", &lora.up);
        insert_pair(&mut tensors, layer_idx, "down", &lora.down);
    }
    candle_core::safetensors::save(&tensors, path)?;
    Ok(())
}

pub fn load_lora_layers(path: &Path, device: &candle_core::Device) -> Result<Vec<LoRAWeights>> {
    let tensors = candle_core::safetensors::load(path, device)?;
    let layer_count = tensors
        .keys()
        .filter_map(|name| {
            name.strip_prefix("layers.")
                .and_then(|rest| rest.split('.').next())
                .and_then(|idx| idx.parse::<usize>().ok())
        })
        .max()
        .map(|idx| idx + 1)
        .context("Adapter file does not contain any layers.* tensors")?;

    let mut all_lora = Vec::with_capacity(layer_count);
    for layer_idx in 0..layer_count {
        let q = load_pair(&tensors, layer_idx, "q")?;
        let k = load_pair(&tensors, layer_idx, "k")?;
        let v = load_pair(&tensors, layer_idx, "v")?;
        let o = load_pair(&tensors, layer_idx, "o")?;
        let gate = load_pair(&tensors, layer_idx, "gate")?;
        let up = load_pair(&tensors, layer_idx, "up")?;
        let down = load_pair(&tensors, layer_idx, "down")?;
        all_lora.push(LoRAWeights {
            q,
            k,
            v,
            o,
            gate,
            up,
            down,
        });
    }

    Ok(all_lora)
}

fn insert_pair(
    tensors: &mut HashMap<String, Tensor>,
    layer_idx: usize,
    module: &str,
    pair: &LoRAPair,
) {
    tensors.insert(adapter_key(layer_idx, module, "a"), pair.0.clone());
    tensors.insert(adapter_key(layer_idx, module, "b"), pair.1.clone());
}

fn load_pair(
    tensors: &HashMap<String, Tensor>,
    layer_idx: usize,
    module: &str,
) -> Result<LoRAPair> {
    let a_key = adapter_key(layer_idx, module, "a");
    let b_key = adapter_key(layer_idx, module, "b");
    let a = tensors
        .get(&a_key)
        .with_context(|| format!("Missing adapter tensor {a_key}"))?
        .clone();
    let b = tensors
        .get(&b_key)
        .with_context(|| format!("Missing adapter tensor {b_key}"))?
        .clone();
    Ok((a, b))
}

fn adapter_key(layer_idx: usize, module: &str, side: &str) -> String {
    format!("layers.{layer_idx:02}.{module}.{side}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HypernetworkConfig;
    use candle_core::{DType, Device};

    #[test]
    fn test_hypernetwork_shapes() -> Result<()> {
        let device = Device::Cpu;
        let config = HypernetworkConfig::default();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hn = Code2LoRAHead::new(vb, &config, &varmap)?;

        // Test single forward (backward compat)
        let input = Tensor::rand(0f32, 1.0, (1, config.repo_embed_dim), &device)?;
        let w = hn.forward(&input)?;
        // A: (rank, in_dim),  B: (out_dim, rank)
        assert_eq!(w.q.0.dims(), &[config.rank, config.llm_hidden_dim]);
        assert_eq!(w.q.1.dims(), &[config.llm_hidden_dim, config.rank]);
        // gate: in=llm_hidden_dim, out=llm_intermediate_dim
        assert_eq!(w.gate.0.dims(), &[config.rank, config.llm_hidden_dim]);
        assert_eq!(w.gate.1.dims(), &[config.llm_intermediate_dim, config.rank]);
        // k/v: in=llm_hidden_dim, out=kv_proj_dim
        assert_eq!(w.k.0.dims(), &[config.rank, config.llm_hidden_dim]);
        assert_eq!(w.k.1.dims(), &[config.kv_proj_dim, config.rank]);
        // down: in=llm_intermediate_dim, out=llm_hidden_dim
        assert_eq!(w.down.0.dims(), &[config.rank, config.llm_intermediate_dim]);
        assert_eq!(w.down.1.dims(), &[config.llm_hidden_dim, config.rank]);

        // Test per-layer forward
        let all = hn.forward_all(&input)?;
        assert_eq!(
            all.len(),
            config.num_layers,
            "should be one weight set per layer"
        );
        for (i, layer_w) in all.iter().enumerate() {
            assert_eq!(
                layer_w.q.0.dims(),
                &[config.rank, config.llm_hidden_dim],
                "layer {i} q_A shape"
            );
            assert_eq!(
                layer_w.q.1.dims(),
                &[config.llm_hidden_dim, config.rank],
                "layer {i} q_B shape"
            );
            assert_eq!(
                layer_w.k.0.dims(),
                &[config.rank, config.llm_hidden_dim],
                "layer {i} k_A shape"
            );
            assert_eq!(
                layer_w.k.1.dims(),
                &[config.kv_proj_dim, config.rank],
                "layer {i} k_B shape"
            );
        }

        // Verify per-layer weights are actually distinct
        if all.len() >= 2 {
            let diff = all[0].q.0.sub(&all[1].q.0)?;
            let diff_val: f32 = diff.abs()?.sum_all()?.to_scalar::<f32>()?;
            assert!(
                diff_val > 0.001,
                "layers 0 and 1 should have different q_A weights"
            );
        }

        println!("All shapes and per-layer distinctness correct!");
        Ok(())
    }

    #[test]
    fn test_lora_adapter_safetensors_round_trip() -> Result<()> {
        let device = Device::Cpu;
        let config = HypernetworkConfig {
            hidden_dim: 16,
            rank: 2,
            num_layers: 2,
            repo_embed_dim: 8,
            llm_hidden_dim: 8,
            llm_intermediate_dim: 12,
            kv_proj_dim: 4,
        };
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hn = Code2LoRAHead::new(vb, &config, &varmap)?;
        let input = Tensor::rand(0f32, 1.0, (1, config.repo_embed_dim), &device)?;
        let all_lora = hn.forward_all(&input)?;
        let path = std::env::temp_dir().join(format!(
            "code2lora-adapter-test-{}.safetensors",
            std::process::id()
        ));

        save_lora_layers(&all_lora, &path)?;
        let loaded = load_lora_layers(&path, &device)?;
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.len(), config.num_layers);
        assert_eq!(loaded[0].q.0.dims(), &[config.rank, config.llm_hidden_dim]);
        assert_eq!(loaded[0].q.1.dims(), &[config.llm_hidden_dim, config.rank]);
        assert_eq!(
            loaded[1].down.0.dims(),
            &[config.rank, config.llm_intermediate_dim]
        );
        assert_eq!(
            loaded[1].down.1.dims(),
            &[config.llm_hidden_dim, config.rank]
        );
        Ok(())
    }
}
