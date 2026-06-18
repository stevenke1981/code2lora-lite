use anyhow::{Context, Result};
use candle_core::{Module, Tensor};
use candle_nn::{Linear, VarBuilder, VarMap};
use std::collections::HashMap;
use std::path::Path;
use crate::config::HypernetworkConfig;

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

impl LoRAWeights {
    pub fn get(&self, name: &str) -> Option<&LoRAPair> {
        match name {
            "q" => Some(&self.q), "k" => Some(&self.k), "v" => Some(&self.v),
            "o" => Some(&self.o), "gate" => Some(&self.gate),
            "up" => Some(&self.up), "down" => Some(&self.down),
            _ => None,
        }
    }
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
        let input_dim = config.llm_hidden_dim * 2;
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
            let ha = candle_nn::linear(hidden_dim, rank * in_dim, vb.pp(format!("head_{mod_type}_a")))?;
            // B: hidden → out_dim × rank   (B matrix: LoRA projection from rank to out_dim)
            let hb = candle_nn::linear(hidden_dim, out_dim * rank, vb.pp(format!("head_{mod_type}_b")))?;

            heads_a.insert(mod_type.to_string(), ha);
            heads_b.insert(mod_type.to_string(), hb);

            let sa = vb.get(&[1], &format!("log_scale_{mod_type}_a"))?;
            let sb = vb.get(&[1], &format!("log_scale_{mod_type}_b"))?;
            log_scale_a.insert(mod_type.to_string(), sa.to_dtype(dtype)?);
            log_scale_b.insert(mod_type.to_string(), sb.to_dtype(dtype)?);
        }

        let layer_emb = candle_nn::embedding(config.num_layers, config.hidden_dim, vb.pp("layer_emb"))?;

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

        Ok(LoRAWeights { q, k, v, o, gate, up, down })
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

            all_weights.push(LoRAWeights { q, k, v, o, gate, up, down });
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use crate::config::HypernetworkConfig;

    #[test]
    fn test_hypernetwork_shapes() -> Result<()> {
        let device = Device::Cpu;
        let config = HypernetworkConfig::default();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hn = Code2LoRAHead::new(vb, &config, &varmap)?;

        // Test single forward (backward compat)
        let input = Tensor::rand(0f32, 1.0, (1, config.llm_hidden_dim * 2), &device)?;
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
        assert_eq!(all.len(), config.num_layers, "should be one weight set per layer");
        for (i, layer_w) in all.iter().enumerate() {
            assert_eq!(layer_w.q.0.dims(), &[config.rank, config.llm_hidden_dim],
                "layer {i} q_A shape");
            assert_eq!(layer_w.q.1.dims(), &[config.llm_hidden_dim, config.rank],
                "layer {i} q_B shape");
            assert_eq!(layer_w.k.0.dims(), &[config.rank, config.llm_hidden_dim],
                "layer {i} k_A shape");
            assert_eq!(layer_w.k.1.dims(), &[config.kv_proj_dim, config.rank],
                "layer {i} k_B shape");
        }

        // Verify per-layer weights are actually distinct
        if all.len() >= 2 {
            let diff = all[0].q.0.sub(&all[1].q.0)?;
            let diff_val: f32 = diff.abs()?.sum_all()?.to_scalar::<f32>()?;
            assert!(diff_val > 0.001, "layers 0 and 1 should have different q_A weights");
        }

        println!("All shapes and per-layer distinctness correct!");
        Ok(())
    }
}
