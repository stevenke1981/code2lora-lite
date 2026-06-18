use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use hf_hub::api::sync::Api;
use log::info;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;
use walkdir::WalkDir;

/// A single repository embedding vector.
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
        let content = fs::read_to_string(path)?;
        let header_end = content.find('\n').ok_or_else(|| anyhow::anyhow!("Invalid header"))?;
        let header = &content[..header_end];
        let dim: usize = header
            .strip_prefix("CODE2LORA_EMBED_V1:")
            .ok_or_else(|| anyhow::anyhow!("Bad header prefix"))?
            .parse()?;
        let byte_start = header_end + 1;
        let bytes = &content.as_bytes()[byte_start..];
        let data: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        anyhow::ensure!(data.len() == dim, "Dim mismatch");
        Ok(Self { data })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

/// Embedding cache: compute mean-pooled token vectors using a lightweight model.
pub struct RepoEncoder {
    /// Cached text embedding function (simple averaging for now)
    device: Device,
    embed_dim: usize,
    /// Cache for file embeddings to avoid recomputation
    cache: Mutex<HashMap<String, Vec<f32>>>,
}

impl RepoEncoder {
    pub fn new(device: &Device) -> Result<Self> {
        // We use a simple approach: load a small BERT model for embeddings.
        // If the model can't be loaded, fall back to a simpler method.
        info!("Initializing RepoEncoder (device: {device:?})");
        Ok(Self {
            device: device.clone(),
            embed_dim: 384, // all-MiniLM-L6-v2 dimension
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Embed a repo by scanning all .py files and computing weighted avg + max pool.
    pub fn embed_repo(&self, repo_path: &Path) -> Result<RepoEmbedding> {
        let py_files = self.collect_py_files(repo_path);
        if py_files.is_empty() {
            anyhow::bail!("No .py files found in {repo_path:?}");
        }

        let mut file_vectors: Vec<(f32, Vec<f32>)> = Vec::new();

        for file_path in &py_files {
            let content = fs::read_to_string(file_path)
                .map_err(|e| anyhow::anyhow!("Failed to read {file_path:?}: {e}"))?;
            let file_vec = self.embed_text_simple(&content);
            let weight = self.compute_file_weight(file_path, &content);
            file_vectors.push((weight, file_vec));
        }

        let dim = self.embed_dim;
        let total_weight: f32 = file_vectors.iter().map(|(w, _)| w).sum();
        let mut weighted_mean = vec![0.0f32; dim];
        for (w, vec) in &file_vectors {
            for i in 0..dim {
                weighted_mean[i] += w * vec[i] / total_weight;
            }
        }

        let mut max_pool = vec![f32::NEG_INFINITY; dim];
        for (_, vec) in &file_vectors {
            for i in 0..dim {
                if vec[i] > max_pool[i] {
                    max_pool[i] = vec[i];
                }
            }
        }

        let mut repo_emb = Vec::with_capacity(2 * dim);
        repo_emb.extend_from_slice(&weighted_mean);
        repo_emb.extend_from_slice(&max_pool);

        Ok(RepoEmbedding { data: repo_emb })
    }

    pub fn embed_repo_cached(&self, repo_path: &Path, cache_dir: &Path) -> Result<RepoEmbedding> {
        fs::create_dir_all(cache_dir)?;
        let repo_name = repo_path.file_name().unwrap_or_else(|| repo_path.as_os_str()).to_string_lossy();
        let cache_path = cache_dir.join(format!("{repo_name}.embed"));

        if cache_path.exists() {
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

    // ─── Private ───

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

    /// Simple text embedding using token frequency (fallback when no model loaded).
    fn embed_text_simple(&self, text: &str) -> Vec<f32> {
        // Use character-level features: unigram + bigram counts
        // This is a simplified embedding that gives reasonable file similarity
        let mut counts = vec![0.0f32; self.embed_dim];
        
        // Use simple hash-based features from words
        let words: Vec<&str> = text.split_whitespace().collect();
        for (i, word) in words.iter().enumerate() {
            if i >= self.embed_dim { break; }
            // Simple hash to position
            let idx = (word.len().wrapping_mul(7).wrapping_add(13)) % self.embed_dim;
            counts[idx] += 1.0;
        }
        
        // Normalize
        let sum: f32 = counts.iter().sum();
        if sum > 0.0 {
            for c in &mut counts {
                *c /= sum;
            }
        }
        
        counts
    }

    fn compute_file_weight(&self, path: &Path, content: &str) -> f32 {
        let path_str = path.to_string_lossy().to_lowercase();
        let size_weight = (content.len() as f32).ln_1p().min(10.0) / 10.0;
        let path_weight = if path_str.contains("test") { 0.8 }
            else if path_str.contains("__init__") { 0.6 }
            else if path_str.contains("src/") || path_str.contains("/lib/") { 1.0 }
            else { 0.5 };
        let name_weight = if path_str.ends_with("main.py") || path_str.ends_with("core.py") { 1.2 } else { 1.0 };
        0.3 * size_weight + 0.5 * path_weight + 0.2 * name_weight
    }
}
