//! Custom Qwen2 model with per-layer LoRA injection support.
//!
//! Based on candle-transformers' qwen2 implementation, modified to support
//! optional LoRA adapters on each linear projection within attention and MLP.

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Activation, Embedding, RmsNorm, VarBuilder};
use candle_transformers::utils::repeat_kv;
use std::sync::Arc;

use crate::hypernetwork::LoRAWeights;

// ─── LoRA-able Linear ───

pub struct LoRALinear {
    weight: Tensor,
    lora_a: Option<Tensor>,
    lora_b: Option<Tensor>,
    in_dim: usize,
    out_dim: usize,
}

impl LoRALinear {
    pub fn new(weight: Tensor, in_dim: usize, out_dim: usize) -> Self {
        Self {
            weight,
            lora_a: None,
            lora_b: None,
            in_dim,
            out_dim,
        }
    }

    pub fn set_lora(&mut self, a: Tensor, b: Tensor) {
        self.lora_a = Some(a);
        self.lora_b = Some(b);
    }

    pub fn clear_lora(&mut self) {
        self.lora_a = None;
        self.lora_b = None;
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Candle requires same-dimensionality for matmul.
        // Flatten batch dims, do 2D @ 2D, then reshape back.
        let ndim = x.dims().len();
        let orig_shape = x.shape().dims().to_vec();
        let in_dim = orig_shape[ndim - 1];
        anyhow::ensure!(
            in_dim == self.in_dim,
            "LoRALinear expected input dim {}, got {}",
            self.in_dim,
            in_dim
        );

        // Flatten to (batch_product, in_dim)
        let batch_product: usize = orig_shape[..ndim - 1].iter().product();
        let x_2d = x.reshape((batch_product, in_dim))?;

        let w_t = self.weight.t()?.contiguous()?;
        let base = x_2d.matmul(&w_t)?; // (B', out_dim)

        let result = match (&self.lora_a, &self.lora_b) {
            (Some(a), Some(b)) => {
                let a_t = a.t()?.contiguous()?; // (in_dim, rank)
                let b_t = b.t()?.contiguous()?; // (rank, out_dim)
                let lora = x_2d.matmul(&a_t)?.matmul(&b_t)?;
                (base + lora)?
            }
            _ => base,
        };

        // Reshape back to original batch dims
        let mut new_shape = orig_shape[..ndim - 1].to_vec();
        new_shape.push(self.out_dim);
        Ok(result.reshape(new_shape)?)
    }
}

impl Module for LoRALinear {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        self.forward(x)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))
    }
}

fn load_lora_linear(
    vb: &VarBuilder,
    in_dim: usize,
    out_dim: usize,
    name: &str,
) -> Result<LoRALinear> {
    let ws = vb.get(&[out_dim, in_dim], name)?;
    Ok(LoRALinear::new(ws, in_dim, out_dim))
}

// ─── Rotary Embedding ───

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        cfg: &candle_transformers::models::qwen2::Config,
        dev: &Device,
    ) -> Result<Self> {
        let dim = cfg.hidden_size / cfg.num_attention_heads;
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seqlen_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;
        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ─── LoRA Attention ───

pub struct LoRAAttention {
    q_proj: LoRALinear,
    k_proj: LoRALinear,
    v_proj: LoRALinear,
    o_proj: LoRALinear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl LoRAAttention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &candle_transformers::models::qwen2::Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let num_kv_groups = num_heads / num_kv_heads;
        let head_dim = hidden_sz / num_heads;

        Ok(Self {
            q_proj: load_lora_linear(&vb.pp("q_proj"), hidden_sz, num_heads * head_dim, "weight")?,
            k_proj: load_lora_linear(
                &vb.pp("k_proj"),
                hidden_sz,
                num_kv_heads * head_dim,
                "weight",
            )?,
            v_proj: load_lora_linear(
                &vb.pp("v_proj"),
                hidden_sz,
                num_kv_heads * head_dim,
                "weight",
            )?,
            o_proj: load_lora_linear(&vb.pp("o_proj"), num_heads * head_dim, hidden_sz, "weight")?,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size: hidden_sz,
            rotary_emb,
        })
    }

    pub fn set_lora(&mut self, lora: &LoRAWeights) {
        self.q_proj.set_lora(lora.q.0.clone(), lora.q.1.clone());
        self.k_proj.set_lora(lora.k.0.clone(), lora.k.1.clone());
        self.v_proj.set_lora(lora.v.0.clone(), lora.v.1.clone());
        self.o_proj.set_lora(lora.o.0.clone(), lora.o.1.clone());
    }

    pub fn clear_lora(&mut self) {
        self.q_proj.clear_lora();
        self.k_proj.clear_lora();
        self.v_proj.clear_lora();
        self.o_proj.clear_lora();
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;

        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;

        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (query_states, key_states) =
            self.rotary_emb
                .apply(&query_states, &key_states, seqlen_offset)?;

        let key_states = repeat_kv(key_states, self.num_kv_groups)?.contiguous()?;
        let value_states = repeat_kv(value_states, self.num_kv_groups)?.contiguous()?;

        let attn_output = {
            let scale = 1f64 / f64::sqrt(self.head_dim as f64);
            let attn_weights = (query_states.matmul(&key_states.transpose(2, 3)?)? * scale)?;
            let attn_weights = match attention_mask {
                None => attn_weights,
                Some(mask) => attn_weights.broadcast_add(mask)?,
            };
            let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
            attn_weights.matmul(&value_states)?
        };
        let out = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?;
        Ok(out.apply(&self.o_proj)?)
    }
}

// ─── LoRA MLP ───

pub struct LoRAMLP {
    gate_proj: LoRALinear,
    up_proj: LoRALinear,
    down_proj: LoRALinear,
    act_fn: Activation,
}

impl LoRAMLP {
    fn new(cfg: &candle_transformers::models::qwen2::Config, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        Ok(Self {
            gate_proj: load_lora_linear(&vb.pp("gate_proj"), hidden_sz, intermediate_sz, "weight")?,
            up_proj: load_lora_linear(&vb.pp("up_proj"), hidden_sz, intermediate_sz, "weight")?,
            down_proj: load_lora_linear(&vb.pp("down_proj"), intermediate_sz, hidden_sz, "weight")?,
            act_fn: cfg.hidden_act,
        })
    }

    pub fn set_lora(&mut self, lora: &LoRAWeights) {
        self.gate_proj
            .set_lora(lora.gate.0.clone(), lora.gate.1.clone());
        self.up_proj.set_lora(lora.up.0.clone(), lora.up.1.clone());
        self.down_proj
            .set_lora(lora.down.0.clone(), lora.down.1.clone());
    }

    pub fn clear_lora(&mut self) {
        self.gate_proj.clear_lora();
        self.up_proj.clear_lora();
        self.down_proj.clear_lora();
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(xs)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

impl Module for LoRAMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        self.forward(xs)
            .map_err(|e| candle_core::Error::Msg(format!("{e}")))
    }
}

// ─── LoRA Decoder Layer ───

pub struct LoRALayer {
    self_attn: LoRAAttention,
    mlp: LoRAMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl LoRALayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &candle_transformers::models::qwen2::Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let vb_sa = vb.pp("self_attn");
        let vb_mlp = vb.pp("mlp");
        Ok(Self {
            self_attn: LoRAAttention::new(rotary_emb, cfg, vb_sa)?,
            mlp: LoRAMLP::new(cfg, vb_mlp)?,
            input_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    pub fn set_lora(&mut self, lora: &LoRAWeights) {
        self.self_attn.set_lora(lora);
        self.mlp.set_lora(lora);
    }

    pub fn clear_lora(&mut self) {
        self.self_attn.clear_lora();
        self.mlp.clear_lora();
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        Ok((residual + xs)?)
    }
}

// ─── LoRA Model ───

pub struct LoRAModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<LoRALayer>,
    pub norm: RmsNorm,
    pub config: candle_transformers::models::qwen2::Config,
    pub device: Device,
    pub dtype: DType,
}

impl LoRAModel {
    pub fn new(cfg: &candle_transformers::models::qwen2::Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = LoRALayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            config: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    fn prepare_causal_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let sw = self.config.sliding_window;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || (sw > 0 && j + sw < i) {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), self.dtype, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        Ok(mask
            .expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)?)
    }

    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_size, seq_len) = input_ids.dims2()?;
        let attention_mask: Option<Tensor> = match attn_mask {
            Some(mask) => Some(self.prepare_attention_mask(mask)?),
            None => {
                if seq_len <= 1 {
                    None
                } else {
                    Some(self.prepare_causal_attention_mask(b_size, seq_len, seqlen_offset)?)
                }
            }
        };
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attention_mask.as_ref(), seqlen_offset)?;
        }
        Ok(xs.apply(&self.norm)?)
    }

    fn prepare_attention_mask(&self, attn_mask: &Tensor) -> Result<Tensor> {
        let (b_sz, sql_len) = attn_mask.dims2()?;
        let mut mask: Vec<Tensor> = vec![];
        for b in 0..b_sz {
            mask.push(attn_mask.i((b, ..))?.expand((1, 1, sql_len, sql_len))?);
        }
        let mask = Tensor::cat(&mask, 0)?;
        let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
        let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
            .broadcast_as(mask.shape())?
            .to_dtype(self.dtype)?;
        Ok(mask.where_cond(&on_true, &on_false)?)
    }

    /// Inject per-layer LoRA adapters into decoder layers.
    /// `all_lora` should have length equal to `self.config.num_hidden_layers`.
    /// Each element is the (A, B) pairs for that specific layer.
    pub fn inject_lora_all(&mut self, all_lora: &[LoRAWeights]) {
        for (layer, lora) in self.layers.iter_mut().zip(all_lora.iter()) {
            layer.set_lora(lora);
        }
    }

    pub fn clear_lora_all(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_lora();
        }
    }
}
