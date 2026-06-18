use anyhow::Result;
use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use log::info;
use std::path::Path;

use crate::config::TrainConfig;
use crate::dataset::CodeDataset;
use crate::hypernetwork::Code2LoRAHead;

/// Training loop for the hypernetwork.
pub struct Trainer {
    pub model: Code2LoRAHead,
    pub config: TrainConfig,
    pub device: Device,
    pub varmap: VarMap,
}

impl Trainer {
    pub fn new(
        model: Code2LoRAHead,
        varmap: VarMap,
        config: TrainConfig,
        device: Device,
    ) -> Self {
        Self { model, config, device, varmap }
    }

    pub fn train(&mut self, dataset: &CodeDataset) -> Result<()> {
        let n_examples = dataset.len();
        info!("Dataset has {n_examples} examples");

        let n_epochs = self.config.epochs as usize;
        let lr = self.config.lr;
        let checkpoint_dir = Path::new(&self.config.output);

        // Collect trainable variables
        let vars = self.varmap.all_vars();
        let params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        };
        let mut opt = AdamW::new(vars, params)?;

        for epoch in 0..n_epochs {
            // FUTURE: iterate batches, compute loss
            let loss = 0.1f64;

            info!("Epoch {}/{} — avg_loss = {:.6}", epoch + 1, n_epochs, loss);

            if (epoch + 1) % 5 == 0 {
                let ckpt_path = checkpoint_dir.join(format!("epoch_{:04}.safetensors", epoch + 1));
                self.model.save(&ckpt_path)?;
                info!("Checkpoint saved to {ckpt_path:?}");
            }
        }

        Ok(())
    }
}
