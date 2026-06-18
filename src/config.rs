/// Configuration structures for code2lora-lite.

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub data_dir: String,
    pub base_model: String,
    pub output: String,
    pub rank: usize,
    pub epochs: u32,
    pub lr: f64,
    pub batch_size: usize,
    pub seq_len: usize,
    pub cache_dir: String,
    pub cr_holdout: f64,
}

#[derive(Debug, Clone)]
pub struct HypernetworkConfig {
    /// Hidden dimension for the shared MLP
    pub hidden_dim: usize,
    /// LoRA rank
    pub rank: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Input dimension of the repo embedding (e.g., 768 for all-MiniLM-L6-v2)
    pub repo_embed_dim: usize,
    /// Hidden dimension of the base LLM (e.g., 1024 for Qwen2.5-Coder-0.5B)
    pub llm_hidden_dim: usize,
    /// Intermediate dimension of the base LLM (e.g., 896 for Qwen2.5-Coder-0.5B)
    pub llm_intermediate_dim: usize,
    /// K/V projection dimension = num_kv_heads * head_dim
    /// For non-GQA models, this equals llm_hidden_dim.
    /// For Qwen2.5-Coder-0.5B: 8 kv_heads × 32 head_dim = 256
    pub kv_proj_dim: usize,
}

impl Default for HypernetworkConfig {
    fn default() -> Self {
        // Qwen2.5-Coder-0.5B actual dimensions (loaded from HF config.json):
        //   hidden_size=896, intermediate_size=4864, num_attention_heads=14,
        //   num_key_value_heads=2 → kv_proj_dim = 2 * (896/14) = 128
        Self {
            hidden_dim: 384,
            rank: 8,
            num_layers: 24,
            repo_embed_dim: 768,  // 384 mean + 384 max pool from MiniLM
            llm_hidden_dim: 896,
            llm_intermediate_dim: 4864,
            kv_proj_dim: 128,  // 2 kv_heads × (896/14) head_dim = 128
        }
    }
}

impl HypernetworkConfig {
    /// Input dimension for the LoRA A matrix of each module.
    /// q/k/v/o/gate/up: llm_hidden_dim (input is the hidden state)
    /// down: llm_intermediate_dim (input is the intermediate activation)
    pub fn lora_in_dim(&self, module_type: &str) -> usize {
        match module_type {
            "q" | "k" | "v" | "o" | "gate" | "up" => self.llm_hidden_dim,
            "down" => self.llm_intermediate_dim,
            _ => panic!("Unknown module type: {module_type}"),
        }
    }

    /// Output dimension for the LoRA B matrix of each module.
    /// q, o: llm_hidden_dim
    /// k, v: kv_proj_dim (GQA: smaller than hidden_dim)
    /// gate, up: llm_intermediate_dim
    /// down: llm_hidden_dim
    pub fn lora_out_dim(&self, module_type: &str) -> usize {
        match module_type {
            "q" | "o" => self.llm_hidden_dim,
            "k" | "v" => self.kv_proj_dim,
            "gate" | "up" => self.llm_intermediate_dim,
            "down" => self.llm_hidden_dim,
            _ => panic!("Unknown module type: {module_type}"),
        }
    }

    /// All 7 module type names
    pub const MODULE_TYPES: &'static [&'static str] = &["q", "k", "v", "o", "gate", "up", "down"];
}
