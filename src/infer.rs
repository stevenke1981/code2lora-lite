use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use log::info;
use std::path::Path;

use crate::base_llm::Code2LoRAModel;
use crate::config::HypernetworkConfig;
use crate::hypernetwork::Code2LoRAHead;
use crate::repo_encoder::RepoEncoder;

/// Adapt: encode repo → generate LoRA adapter → save to file.
pub fn adapt(repo_path: &Path, adapter_output_path: &Path, device: &Device) -> Result<()> {
    info!("Adapting repo {repo_path:?} → {adapter_output_path:?}");

    let hn_config = HypernetworkConfig::default();
    let varmap = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let hn = Code2LoRAHead::new(vb, &hn_config, &varmap)?;

    let encoder = RepoEncoder::new(device)?;
    let emb = encoder.embed_repo(repo_path)?;
    let emb_tensor = emb.to_tensor(device)?;

    let all_lora = hn.forward_all(&emb_tensor)?;
    info!("LoRA adapter generated: {} layers, q shape {:?} per layer", 
        all_lora.len(), all_lora[0].q.0.shape());

    // Save via VarMap
    hn.save(adapter_output_path)?;
    info!("Adapter saved to {adapter_output_path:?}");
    Ok(())
}

/// Complete: load adapters and generate assertion.
pub fn complete(
    repo_path: &Path,
    adapter_path: &Path,
    output_path: &Path,
    _device: &Device,
) -> Result<()> {
    info!("Loading adapter from {adapter_path:?}");
    let device = Device::Cpu;

    let hn_config = HypernetworkConfig::default();
    let mut base_model = Code2LoRAModel::new(&device, DType::F32, &hn_config)?;
    let mut varmap = candle_nn::VarMap::new();

    // Load adapter via candle_core safetensors
    let tensor_map = candle_core::safetensors::load(adapter_path, &device)?;
    let pairs: Vec<(String, candle_core::Tensor)> = tensor_map.into_iter().collect();
    varmap.set(pairs.into_iter())?;

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let hn = Code2LoRAHead::new(vb, &hn_config, &varmap)?;

    let encoder = RepoEncoder::new(&device)?;
    let emb = encoder.embed_repo(repo_path)?;
    let emb_tensor = emb.to_tensor(&device)?;

    base_model.inject_lora_from_hn(&hn, &emb_tensor)?;

    // Generate assertion
    let prompt: Vec<u32> = (1..=50).collect();
    let result = base_model.generate(&prompt, 128)?;
    let output_text = format!("Generated {} tokens: {:?}", result.len(), &result[..10.min(result.len())]);
    std::fs::write(output_path, &output_text)?;
    info!("Assertion saved to {output_path:?}");
    Ok(())
}

/// Encode: just compute the embedding and save to file.
pub fn encode(repo_path: &Path, output_path: &Path, device: &Device) -> Result<()> {
    info!("Encoding repo {repo_path:?} → {output_path:?}");
    let encoder = RepoEncoder::new(device)?;
    let emb = encoder.embed_repo(repo_path)?;
    emb.save(output_path)?;
    info!("Embedding saved to {output_path:?}");
    Ok(())
}
