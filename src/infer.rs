use anyhow::Result;
use candle_core::{DType, Device};
use log::info;
use std::path::Path;

use crate::base_llm::Code2LoRAModel;
use crate::config::HypernetworkConfig;
use crate::evo::{
    load_embedding_tensor, load_evo_state, save_evo_adapter, save_evo_state, Code2LoRAEvo,
};
use crate::hypernetwork::{load_lora_layers, save_lora_layers, Code2LoRAHead};
use crate::repo_encoder::{RepoEmbedding, RepoEncoder};

/// Adapt: encode repo → generate LoRA adapter → save to file.
pub fn adapt(
    repo_path: &Path,
    hypernetwork_path: &Path,
    adapter_output_path: &Path,
    device: &Device,
) -> Result<()> {
    info!("Adapting repo {repo_path:?} → {adapter_output_path:?}");

    let hn_config = HypernetworkConfig::default();
    let (hn, _varmap) = Code2LoRAHead::load(hypernetwork_path, &hn_config, DType::F32, device)?;

    let encoder = RepoEncoder::new(device)?;
    let emb = encoder.embed_repo_cached(repo_path, Path::new(".cache/embeddings"))?;
    let emb_tensor = emb.to_tensor(device)?;

    let all_lora = hn.forward_all(&emb_tensor)?;
    info!(
        "LoRA adapter generated: {} layers, q shape {:?} per layer",
        all_lora.len(),
        all_lora[0].q.0.shape()
    );

    save_lora_layers(&all_lora, adapter_output_path)?;
    info!("Adapter saved to {adapter_output_path:?}");
    Ok(())
}

/// Complete: load adapters and generate assertion.
pub fn complete(
    _repo_path: &Path,
    adapter_path: &Path,
    prefix: &str,
    output_path: &Path,
    device: &Device,
    max_new_tokens: usize,
) -> Result<()> {
    complete_with_max_new_tokens(adapter_path, prefix, output_path, device, max_new_tokens)
}

fn complete_with_max_new_tokens(
    adapter_path: &Path,
    prefix: &str,
    output_path: &Path,
    device: &Device,
    max_new_tokens: usize,
) -> Result<()> {
    info!("Loading adapter from {adapter_path:?}");

    let hn_config = HypernetworkConfig::default();
    let mut base_model = Code2LoRAModel::new(device, DType::F32, &hn_config)?;
    let all_lora = load_lora_layers(adapter_path, device)?;
    anyhow::ensure!(
        all_lora.len() == hn_config.num_layers,
        "Adapter layer count mismatch: expected {}, got {}",
        hn_config.num_layers,
        all_lora.len()
    );
    base_model.inject_lora(&all_lora);

    let output_text = base_model.generate_text(prefix, max_new_tokens)?;
    std::fs::write(output_path, &output_text)?;
    info!("Assertion saved to {output_path:?}");
    Ok(())
}

/// Encode: just compute the embedding and save to file.
pub fn encode(repo_path: &Path, output_path: &Path, device: &Device) -> Result<()> {
    info!("Encoding repo {repo_path:?} → {output_path:?}");
    let encoder = RepoEncoder::new(device)?;
    let emb = encoder.embed_repo_cached(repo_path, Path::new(".cache/embeddings"))?;
    info!("Repository embedding dim={}", encoder.embed_dim() * 2);
    emb.save(output_path)?;
    info!("Embedding saved to {output_path:?}");
    Ok(())
}

/// Evo adapt: update a recurrent hidden state from commit diffs and emit adapter.
pub fn evo_adapt(
    evo_checkpoint_path: &Path,
    repo_path: Option<&Path>,
    repo_embedding_path: Option<&Path>,
    state_in_path: Option<&Path>,
    diff_files: &[String],
    diff_embedding_paths: &[String],
    state_output_path: &Path,
    adapter_output_path: &Path,
    device: &Device,
) -> Result<()> {
    let hn_config = HypernetworkConfig::default();
    let (evo, _varmap) = Code2LoRAEvo::load(evo_checkpoint_path, &hn_config, DType::F32, device)?;

    let mut state = if let Some(path) = state_in_path {
        load_evo_state(path, device)?
    } else {
        let repo_embedding = match (repo_embedding_path, repo_path) {
            (Some(path), _) => load_embedding_tensor(path, device)?,
            (None, Some(path)) => {
                let encoder = RepoEncoder::new(device)?;
                encoder
                    .embed_repo_cached(path, Path::new(".cache/embeddings"))?
                    .to_tensor(device)?
            }
            (None, None) => anyhow::bail!(
                "Either --state-in or one of --repo-embedding / --repo-path is required"
            ),
        };
        evo.init_state(&repo_embedding)?
    };

    let mut diff_tensors = Vec::new();
    for diff_path in diff_embedding_paths {
        diff_tensors.push(load_embedding_tensor(Path::new(diff_path), device)?);
    }

    if !diff_files.is_empty() {
        let encoder = RepoEncoder::new(device)?;
        for diff_file in diff_files {
            let diff_text = std::fs::read_to_string(diff_file)?;
            let diff_embedding: RepoEmbedding = encoder.embed_text_as_repo(&diff_text)?;
            diff_tensors.push(diff_embedding.to_tensor(device)?);
        }
    }

    anyhow::ensure!(
        !diff_tensors.is_empty(),
        "At least one --diff-file or --diff-embedding is required"
    );

    state = evo.update_sequence(&state, &diff_tensors)?;
    let adapter = evo.adapters_from_state(&state)?;
    save_evo_state(&state, state_output_path)?;
    save_evo_adapter(&adapter, adapter_output_path)?;
    info!(
        "Evo adapter saved to {:?}; state saved to {:?}",
        adapter_output_path, state_output_path
    );
    Ok(())
}

/// Initialize an Evo checkpoint. This is useful for smoke tests and for future
/// evolution-track training to overwrite with learned weights.
pub fn evo_init_checkpoint(output_path: &Path, device: &Device) -> Result<()> {
    let hn_config = HypernetworkConfig::default();
    let varmap = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, device);
    let evo = Code2LoRAEvo::new(vb, &hn_config, &varmap)?;
    evo.save(output_path)?;
    info!("Evo checkpoint initialized at {output_path:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "Requires HF downloads for MiniLM + Qwen2.5-Coder and may take several minutes"]
    fn test_p7_full_end_to_end_real_inference() -> Result<()> {
        let root = std::env::temp_dir().join(format!("code2lora-p7-e2e-{}", std::process::id()));
        let repo = root.join("repo");
        std::fs::create_dir_all(&repo)?;
        std::fs::write(
            repo.join("calculator.py"),
            "def add(a, b):\n    return a + b\n\n\ndef test_add():\n    assert add(2, 3) ==",
        )?;

        let device = Device::cuda_if_available(0)?;
        let embedding_path = root.join("repo_embedding.embed");
        let hypernetwork_path = root.join("hypernetwork.safetensors");
        let adapter_path = root.join("adapter.safetensors");
        let output_path = root.join("assertion.txt");
        let hn_config = HypernetworkConfig::default();
        let varmap = candle_nn::VarMap::new();
        let vb = candle_nn::VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let hn = Code2LoRAHead::new(vb, &hn_config, &varmap)?;
        hn.save(&hypernetwork_path)?;

        encode(&repo, &embedding_path, &device)?;
        adapt(&repo, &hypernetwork_path, &adapter_path, &device)?;
        complete_with_max_new_tokens(
            &adapter_path,
            "def test_add():\n    assert add(2, 3) ==",
            &output_path,
            &device,
            4,
        )?;

        let output = std::fs::read_to_string(&output_path)?;
        assert!(embedding_path.exists());
        assert!(adapter_path.exists());
        assert!(output_path.exists());
        std::fs::remove_dir_all(&root).ok();

        assert!(!output.is_empty());
        Ok(())
    }
}
