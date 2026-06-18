use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::api::sync::Api;
use log::info;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;
use walkdir::WalkDir;

/// A single repository embedding vector (768-dim: 384 mean + 384 max pool).
#[derive(Debug, Clone)]
pub struct RepoEmbedding {
    pub data: Vec<f32>,
}

impl RepoEmbedding {
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let tensor = Tensor::from_vec(self.data.clone(), (1, self.data.len()), device)?;
        Ok(tensor)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let parent = path.parent().unwrap_or(Path::new("."));
        fs::create_dir_all(parent)?;
        let bytes: Vec<u8> = self.data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let header = format!("CODE2LORA_EMBED_V1:{}\n", self.data.len());
        fs::write(path, [header.as_bytes(), &bytes].concat())?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let content = fs::read(path)?;
        let header_end = content
            .iter()
            .position(|b| *b == b'\n')
            .ok_or_else(|| anyhow::anyhow!("Invalid header"))?;
        let header = std::str::from_utf8(&content[..header_end])?;
        let dim: usize = header
            .strip_prefix("CODE2LORA_EMBED_V1:")
            .ok_or_else(|| anyhow::anyhow!("Bad header prefix"))?
            .parse()?;
        let byte_start = header_end + 1;
        let bytes = &content[byte_start..];
        anyhow::ensure!(
            bytes.len() % std::mem::size_of::<f32>() == 0,
            "Embedding payload is not f32-aligned"
        );
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        anyhow::ensure!(data.len() == dim, "Dim mismatch");
        Ok(Self { data })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// Repository encoder using sentence-transformers/all-MiniLM-L6-v2.
pub struct RepoEncoder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    embed_dim: usize, // 384
}

impl RepoEncoder {
    pub fn new(device: &Device) -> Result<Self> {
        info!("Loading all-MiniLM-L6-v2 for repo encoding");
        let api = Api::new().context("HF Hub API")?;
        let model_id = "sentence-transformers/all-MiniLM-L6-v2";
        let repo = api.model(model_id.to_string());

        let config_path = repo.get("config.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .context("No safetensors for all-MiniLM-L6-v2")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("No tokenizer.json for all-MiniLM-L6-v2")?;

        let config: BertConfig = serde_json::from_slice(&std::fs::read(config_path)?)?;
        info!(
            "BERT config: hidden={}, layers={}, heads={}",
            config.hidden_size, config.num_hidden_layers, config.num_attention_heads
        );

        let vocab_size = config.vocab_size;
        let max_pos = config.max_position_embeddings;
        let hidden_size = config.hidden_size;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, device)? };

        let model = BertModel::load(vb, &config)?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {e}"))?;

        info!("MiniLM encoder ready — dim={hidden_size}, max_tokens={max_pos}, vocab={vocab_size}");

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            embed_dim: hidden_size,
        })
    }

    /// Embed a repo: scan .py files, chunk, embed, aggregate (weighted mean + max pool).
    pub fn embed_repo(&self, repo_path: &Path) -> Result<RepoEmbedding> {
        let py_files = self.collect_py_files(repo_path);
        if py_files.is_empty() {
            anyhow::bail!("No .py files found in {repo_path:?}");
        }

        let dim = self.embed_dim;
        let mut sum_mean = vec![0.0f32; dim];
        let mut sum_max = vec![f32::NEG_INFINITY; dim];
        let mut total_weight = 0.0f32;

        for file_path in &py_files {
            let content = fs::read_to_string(file_path)
                .map_err(|e| anyhow::anyhow!("Failed to read {file_path:?}: {e}"))?;
            if content.trim().is_empty() {
                continue;
            }

            let weight = self.compute_file_weight(file_path, &content);
            let chunks = self.chunk_text(&content);
            let mut file_mean = vec![0.0f32; dim];
            let mut file_max = vec![f32::NEG_INFINITY; dim];
            let mut chunk_count = 0usize;

            for chunk in &chunks {
                if let Ok(emb) = self.embed_text(chunk) {
                    for i in 0..dim {
                        file_mean[i] += emb[i];
                        if emb[i] > file_max[i] {
                            file_max[i] = emb[i];
                        }
                    }
                    chunk_count += 1;
                }
            }

            if chunk_count > 0 {
                for i in 0..dim {
                    file_mean[i] /= chunk_count as f32;
                    sum_mean[i] += weight * file_mean[i];
                    if file_max[i] > sum_max[i] {
                        sum_max[i] = file_max[i];
                    }
                }
                total_weight += weight;
            }
        }

        if total_weight <= 0.0 {
            anyhow::bail!("No embeddable content found in repo");
        }

        for i in 0..dim {
            sum_mean[i] /= total_weight;
        }

        // Concatenate mean + max → 768-dim
        let mut repo_data = Vec::with_capacity(2 * dim);
        repo_data.extend_from_slice(&sum_mean);
        repo_data.extend_from_slice(&sum_max);

        Ok(RepoEmbedding { data: repo_data })
    }

    pub fn embed_repo_cached(&self, repo_path: &Path, cache_dir: &Path) -> Result<RepoEmbedding> {
        fs::create_dir_all(cache_dir)?;
        let repo_name = repo_path
            .file_name()
            .unwrap_or_else(|| repo_path.as_os_str())
            .to_string_lossy();
        let cache_path = cache_dir.join(format!("{repo_name}.embed"));

        if cache_path.exists() {
            info!("Loading cached embedding for {repo_name}");
            return RepoEmbedding::load(&cache_path);
        }

        info!("Computing embedding for repo: {repo_name}");
        let emb = self.embed_repo(repo_path)?;
        emb.save(&cache_path)?;
        Ok(emb)
    }

    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    // ─── Private BERT helpers ───

    /// Embed a single text chunk using BERT mean pooling + L2 normalize.
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {e}"))?;

        let ids = encoding.get_ids();
        if ids.len() > 512 {
            // Truncate to max 512 (BERT limit)
            let trunc: Vec<u32> = ids[..512.min(ids.len())].to_vec();
            let input_ids = Tensor::from_slice(&trunc, (1, trunc.len()), &self.device)?;
            let token_type_ids = Tensor::zeros((1, trunc.len()), DType::U32, &self.device)?;
            let attn_mask = Tensor::ones((1, trunc.len()), DType::U32, &self.device)?;
            self.forward_pool(&input_ids, &token_type_ids, &attn_mask)
        } else {
            let input_ids = Tensor::from_slice(ids, (1, ids.len()), &self.device)?;
            let token_type_ids = Tensor::zeros((1, ids.len()), DType::U32, &self.device)?;
            let attn_mask = Tensor::ones((1, ids.len()), DType::U32, &self.device)?;
            self.forward_pool(&input_ids, &token_type_ids, &attn_mask)
        }
    }

    /// BERT forward → mean pool → L2 normalize → return Vec<f32>
    fn forward_pool(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attn_mask: &Tensor,
    ) -> Result<Vec<f32>> {
        let hidden = self
            .model
            .forward(input_ids, token_type_ids, Some(attn_mask))?;
        // hidden: (1, seq_len, 384)

        // Mean pool (excluding padding)
        let mask = attn_mask.unsqueeze(2)?.to_dtype(DType::F32)?; // (1, seq_len, 1)
        let masked = hidden.broadcast_mul(&mask)?; // (1, seq_len, 384)
        let sum_hidden = masked.sum(1)?; // (1, 384)
        let count = mask.sum(1)?.clamp(1.0, f32::MAX)?; // (1, 1)
        let mean_pooled = sum_hidden.broadcast_div(&count)?;

        // L2 normalize
        let norm = mean_pooled.sqr()?.sum_keepdim(1)?.sqrt()?;
        let normalized = mean_pooled.broadcast_div(&norm)?;

        let vec: Vec<f32> = normalized.squeeze(0)?.to_vec1()?;
        Ok(vec)
    }

    /// Split text into chunks of max ~512 tokens with ~256 token overlap.
    fn chunk_text(&self, text: &str) -> Vec<String> {
        let max_chars = 512 * 8; // ~512 tokens ≈ ~4000 chars for code
        let overlap_chars = 256 * 8;
        let mut chunks = Vec::new();
        let mut start = 0usize;
        let text_len = text.len();

        while start < text_len {
            let end = (start + max_chars).min(text_len);
            chunks.push(text[start..end].to_string());
            if end >= text_len {
                break;
            }
            start += max_chars - overlap_chars;
        }

        if chunks.is_empty() && !text.is_empty() {
            chunks.push(text.to_string());
        }

        chunks
    }

    // ─── File helpers ───

    fn collect_py_files(&self, repo_path: &Path) -> Vec<std::path::PathBuf> {
        WalkDir::new(repo_path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .filter(|e| e.path().extension().map(|ext| ext == "py").unwrap_or(false))
            .map(|e| e.into_path())
            .filter(|p| {
                let s = p.to_string_lossy();
                !s.contains("/.") && !s.contains("\\.")
            })
            .collect()
    }

    fn compute_file_weight(&self, path: &Path, content: &str) -> f32 {
        let path_str = path.to_string_lossy().to_lowercase();
        let size_weight = (content.len() as f32).ln_1p().min(10.0) / 10.0;
        let path_weight = if path_str.contains("test") {
            0.8
        } else if path_str.contains("__init__") {
            0.6
        } else if path_str.contains("src/") || path_str.contains("/lib/") {
            1.0
        } else {
            0.5
        };
        let name_weight = if path_str.ends_with("main.py") || path_str.ends_with("core.py") {
            1.2
        } else {
            1.0
        };
        0.3 * size_weight + 0.5 * path_weight + 0.2 * name_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_repo_embedding_binary_round_trip() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("code2lora-embed-test-{}.embed", std::process::id()));
        let embedding = RepoEmbedding {
            data: vec![0.0, 1.25, -2.5, 3.75],
        };

        embedding.save(&path)?;
        let loaded = RepoEmbedding::load(&path)?;
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.data, embedding.data);
        Ok(())
    }
}
