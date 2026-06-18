use anyhow::Result;
use candle_core::{Device, Tensor};
use log::info;
use std::path::Path;

use crate::repo_encoder::RepoEmbedding;

/// A single code repository example.
pub struct CodeExample {
    pub repo_embedding: RepoEmbedding,
    pub code_content: String,
    pub language: String,
}

/// Dataset: code examples for training the hypernetwork.
pub struct CodeDataset {
    examples: Vec<CodeExample>,
}

impl CodeDataset {
    pub fn new() -> Self {
        Self { examples: Vec::new() }
    }

    /// Load from a directory of .txt files (for now; future: Parquet from HF).
    pub fn load_from_dir(path: &Path, _device: &Device) -> Result<Self> {
        info!("Loading dataset from {path:?}");
        let mut examples = Vec::new();
        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                let p = entry.path();
                if p.extension().map(|e| e == "txt").unwrap_or(false) {
                    let code = std::fs::read_to_string(&p)?;
                    let name = p.file_stem().unwrap_or_default().to_string_lossy().to_string();
                    let emb = RepoEmbedding { data: vec![0.0f32; 768] };
                    examples.push(CodeExample {
                        repo_embedding: emb,
                        code_content: code,
                        language: if name.contains("python") || name.contains(".py") { "python".into() } else { "unknown".into() },
                    });
                }
            }
        }
        info!("Loaded {} examples", examples.len());
        Ok(Self { examples })
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    pub fn get(&self, idx: usize) -> Option<&CodeExample> {
        self.examples.get(idx)
    }

    /// Split into CR (code retrieval) and IR (instruction retrieval) portions.
    pub fn split(&self, cr_ratio: f32) -> (Vec<&CodeExample>, Vec<&CodeExample>) {
        let n = self.examples.len();
        let cr_count = (n as f32 * cr_ratio) as usize;
        let cr_count = cr_count.clamp(1, n.saturating_sub(1));
        (self.examples.iter().take(cr_count).collect(),
         self.examples.iter().skip(cr_count).collect())
    }
}

/// Batch iterator for training.
pub struct BatchIterator<'a> {
    embeddings: Vec<&'a CodeExample>,
    batch_size: usize,
    pos: usize,
}

impl<'a> BatchIterator<'a> {
    pub fn new(examples: &[&'a CodeExample], batch_size: usize) -> Self {
        Self {
            embeddings: examples.to_vec(),
            batch_size,
            pos: 0,
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = (Tensor, Vec<String>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.embeddings.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.embeddings.len());
        let batch = &self.embeddings[self.pos..end];
        self.pos = end;

        let data: Vec<f32> = batch.iter()
            .flat_map(|ex| ex.repo_embedding.as_slice().to_vec())
            .collect();
        let dim = batch[0].repo_embedding.len();
        let device = Device::Cpu;
        let tensor = Tensor::from_vec(data, (batch.len(), dim), &device).ok()?;
        let texts: Vec<String> = batch.iter().map(|ex| ex.code_content.clone()).collect();
        Some((tensor, texts))
    }
}
