/// Configuration structures for code2lora-lite.

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub data_dir: String,
    pub base_model: String,
    pub output: String,
    pub rank: usize,
    pub epochs: u32,
    pub lr: f64,
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
    /// Number of transformer layers (for layer-shared LoRA, this just determines output size)
    pub num_layers: usize,
    /// Hidden dimension of the base LLM (e.g., 1024 for Qwen2.5-Coder-0.5B)
    pub llm_hidden_dim: usize,
    /// Intermediate dimension of the base LLM (e.g., 896 for Qwen2.5-Coder-0.5B)
    pub llm_intermediate_dim: usize,
}

impl Default for HypernetworkConfig {
    fn default() -> Self {
        // Qwen2.5-Coder-0.5B default dimensions
        Self {
            hidden_dim: 384,
            rank: 8,
            num_layers: 24,
            llm_hidden_dim: 1024,
            llm_intermediate_dim: 896,
        }
    }
}

impl HypernetworkConfig {
    /// Returns the output dimension for each module type's projection.
    /// q, k, v, o: hidden_dim
    /// gate, up, down: intermediate_dim
    pub fn proj_dim(&self, module_type: &str) -> usize {
        match module_type {
            "q" | "k" | "v" | "o" => self.llm_hidden_dim,
            "gate" | "up" | "down" => self.llm_intermediate_dim,
            _ => panic!("Unknown module type: {module_type}"),
        }
    }

    /// All 7 module type names
    pub const MODULE_TYPES: &'static [&'static str] = &["q", "k", "v", "o", "gate", "up", "down"];
}
