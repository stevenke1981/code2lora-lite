use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use log::info;
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use walkdir::WalkDir;

use crate::repo_encoder::RepoEmbedding;

/// A single code repository example.
#[derive(Debug, Clone)]
pub struct CodeExample {
    pub repo_id: String,
    pub repo_embedding: RepoEmbedding,
    pub code_content: String,
    pub language: String,
    pub split: String,
    pub commit_index: Option<i64>,
}

/// JSONL row produced from RepoPeftBench parquet files.
///
/// The official Parquet files are converted by `scripts/download_code2lora_data.ps1`.
/// Field aliases keep the Rust loader tolerant of minor schema drift.
#[derive(Debug, Clone, Deserialize)]
pub struct AssertionRecord {
    #[serde(default)]
    pub repo_id: String,
    #[serde(default)]
    pub commit_index: Option<i64>,
    #[serde(default, alias = "prefix", alias = "prompt", alias = "question")]
    pub input_prefix: Option<String>,
    #[serde(
        default,
        alias = "target",
        alias = "answer",
        alias = "completion",
        alias = "assertion"
    )]
    pub target_value: Option<String>,
    #[serde(default)]
    pub cross_repo_split: Option<String>,
    #[serde(default)]
    pub in_repo_split: Option<String>,
    #[serde(default)]
    pub production_code_diff: Option<String>,
    #[serde(default, alias = "repo_state_embedding", alias = "embedding")]
    pub repo_embedding: Vec<f32>,
    #[serde(default)]
    pub file_path: Option<String>,
}

impl AssertionRecord {
    fn into_example(self, row_idx: usize) -> Result<CodeExample> {
        let repo_id = if self.repo_id.trim().is_empty() {
            format!("unknown_repo_{row_idx}")
        } else {
            self.repo_id
        };
        let split = self
            .cross_repo_split
            .as_deref()
            .filter(|s| !s.trim().is_empty())
            .or(self
                .in_repo_split
                .as_deref()
                .filter(|s| !s.trim().is_empty()))
            .unwrap_or("train")
            .to_string();

        let mut code_content = String::new();
        if let Some(prefix) = self.input_prefix.filter(|s| !s.trim().is_empty()) {
            code_content.push_str(&prefix);
            if !code_content.ends_with('\n') {
                code_content.push('\n');
            }
        }
        if let Some(target) = self.target_value.filter(|s| !s.trim().is_empty()) {
            code_content.push_str(&target);
        }
        if code_content.trim().is_empty() {
            code_content = self
                .production_code_diff
                .filter(|s| !s.trim().is_empty())
                .context("record has no input_prefix/target_value/production_code_diff")?;
        }

        let embedding = if self.repo_embedding.len() == 768 {
            self.repo_embedding
        } else {
            vec![0.0; 768]
        };

        let language = self
            .file_path
            .as_deref()
            .and_then(|p| Path::new(p).extension())
            .and_then(|ext| ext.to_str())
            .map(|ext| match ext {
                "py" => "python",
                other => other,
            })
            .unwrap_or("python")
            .to_string();

        Ok(CodeExample {
            repo_id,
            repo_embedding: RepoEmbedding { data: embedding },
            code_content,
            language,
            split,
            commit_index: self.commit_index,
        })
    }
}

/// Dataset: code examples for training the hypernetwork.
pub struct CodeDataset {
    examples: Vec<CodeExample>,
}

impl CodeDataset {
    /// Load from a directory of JSONL records or .txt/.py files.
    ///
    /// JSONL is the real-data path: convert HF Parquet files first with
    /// `scripts/download_code2lora_data.ps1`, then point `--data-dir` at the output.
    pub fn load_from_dir(path: &Path, _device: &Device) -> Result<Self> {
        info!("Loading dataset from {path:?}");
        let mut examples = Vec::new();
        if path.is_dir() {
            for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
                let p = entry.path();
                if !entry.file_type().is_file() {
                    continue;
                }
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or_default();
                match ext {
                    "jsonl" => examples.extend(Self::load_jsonl(p)?),
                    "txt" | "py" => {
                        let code = std::fs::read_to_string(p)?;
                        let name = p
                            .file_stem()
                            .unwrap_or_default()
                            .to_string_lossy()
                            .to_string();
                        let language = if ext == "py" || name.contains("python") {
                            "python"
                        } else {
                            "unknown"
                        };
                        examples.push(CodeExample {
                            repo_id: name,
                            repo_embedding: RepoEmbedding {
                                data: vec![0.0f32; 768],
                            },
                            code_content: code,
                            language: language.into(),
                            split: "train".into(),
                            commit_index: None,
                        });
                    }
                    _ => {}
                }
            }
        }
        info!("Loaded {} examples", examples.len());
        Ok(Self { examples })
    }

    pub fn load_jsonl(path: &Path) -> Result<Vec<CodeExample>> {
        info!("Loading RepoPeftBench JSONL from {path:?}");
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut examples = Vec::new();

        for (idx, line) in reader.lines().enumerate() {
            let line = line?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let record: AssertionRecord = serde_json::from_str(trimmed).with_context(|| {
                format!("Invalid JSONL record at {}:{}", path.display(), idx + 1)
            })?;
            examples.push(record.into_example(idx)?);
        }

        Ok(examples)
    }

    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    pub fn summary(&self) -> DatasetSummary {
        let mut repo_ids = std::collections::HashSet::new();
        let mut languages = std::collections::HashSet::new();
        let mut commit_rows = 0usize;

        for example in &self.examples {
            repo_ids.insert(example.repo_id.as_str());
            languages.insert(example.language.as_str());
            if example.commit_index.is_some() {
                commit_rows += 1;
            }
        }

        DatasetSummary {
            repo_count: repo_ids.len(),
            language_count: languages.len(),
            commit_rows,
        }
    }

    /// Split into CR (code retrieval) and IR (instruction retrieval) portions.
    pub fn split(&self, cr_ratio: f32) -> (Vec<&CodeExample>, Vec<&CodeExample>) {
        let n = self.examples.len();
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        let mut cr_examples = Vec::new();
        let mut ir_examples = Vec::new();
        let mut has_explicit_split = false;
        for ex in &self.examples {
            let split = ex.split.to_ascii_lowercase();
            has_explicit_split |= split != "train";
            if split.contains("cross")
                || split.contains("ood")
                || split == "test"
                || split == "val"
                || split == "validation"
            {
                cr_examples.push(ex);
            } else {
                ir_examples.push(ex);
            }
        }
        if has_explicit_split {
            return (cr_examples, ir_examples);
        }

        if n == 1 {
            return (Vec::new(), self.examples.iter().collect());
        }

        let cr_count = (n as f32 * cr_ratio) as usize;
        let cr_count = cr_count.clamp(1, n.saturating_sub(1));
        (
            self.examples.iter().take(cr_count).collect(),
            self.examples.iter().skip(cr_count).collect(),
        )
    }
}

/// Generate a synthetic dataset with random embeddings and fabricated code text.
/// Useful for integration testing and demos — no real repo data required.
pub fn generate_synthetic(n_examples: usize) -> CodeDataset {
    let dim = 768; // 384 mean + 384 max pool, matching all-MiniLM-L6-v2 output
    let mut examples = Vec::with_capacity(n_examples);
    for i in 0..n_examples {
        let data: Vec<f32> = (0..dim)
            .map(|_| rand::random::<f32>() * 2.0 - 1.0)
            .collect();
        let emb = RepoEmbedding { data };
        // Fabricated code that references the example index so it differs per example
        let code = format!(
            "def function_{idx}():\n    \"\"\"Generated function {idx}\"\"\"\n    x = {val}\n    return x ** 2\n\nclass Helper{idx}:\n    pass\n",
            idx = i,
            val = (i as i32 * 7 + 3) % 100,
        );
        examples.push(CodeExample {
            repo_id: format!("synthetic_{i}"),
            repo_embedding: emb,
            code_content: code,
            language: "python".into(),
            split: "train".into(),
            commit_index: None,
        });
    }
    CodeDataset { examples }
}

/// Batch iterator for training.
pub struct BatchIterator<'a> {
    embeddings: Vec<&'a CodeExample>,
    batch_size: usize,
    pos: usize,
    device: Device,
}

impl<'a> BatchIterator<'a> {
    pub fn new_on_device(examples: &[&'a CodeExample], batch_size: usize, device: Device) -> Self {
        Self {
            embeddings: examples.to_vec(),
            batch_size,
            pos: 0,
            device,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DatasetSummary {
    pub repo_count: usize,
    pub language_count: usize,
    pub commit_rows: usize,
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

        let mut data = Vec::with_capacity(batch.len() * batch[0].repo_embedding.len());
        for ex in batch {
            data.extend_from_slice(ex.repo_embedding.as_slice());
        }
        let dim = batch[0].repo_embedding.len();
        let tensor = Tensor::from_vec(data, (batch.len(), dim), &self.device).ok()?;
        let texts: Vec<String> = batch.iter().map(|ex| ex.code_content.clone()).collect();
        Some((tensor, texts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jsonl_real_dataset_record_load_and_split() -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("code2lora-jsonl-test-{}.jsonl", std::process::id()));
        let emb = vec![0.5f32; 768];
        let row = serde_json::json!({
            "repo_id": "owner/repo",
            "commit_index": 7,
            "cross_repo_split": "ood_test",
            "in_repo_split": "train",
            "file_path": "tests/test_example.py",
            "input_prefix": "def test_answer():\n    assert answer() ==",
            "target_value": " 42",
            "repo_embedding": emb,
        });
        std::fs::write(&path, format!("{row}\n"))?;

        let examples = CodeDataset::load_jsonl(&path)?;
        std::fs::remove_file(&path).ok();

        let dataset = CodeDataset { examples };
        let summary = dataset.summary();
        let (cr, ir) = dataset.split(0.2);

        assert_eq!(summary.repo_count, 1);
        assert_eq!(summary.language_count, 1);
        assert_eq!(summary.commit_rows, 1);
        assert_eq!(cr.len(), 1);
        assert_eq!(ir.len(), 0);
        assert_eq!(cr[0].repo_embedding.len(), 768);
        assert!(cr[0].code_content.contains("assert answer() =="));
        Ok(())
    }
}
