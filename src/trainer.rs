use anyhow::Result;
use candle_core::Device;
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use log::info;
use std::path::Path;

use crate::base_llm::Code2LoRAModel;
use crate::config::TrainConfig;
use crate::dataset::{BatchIterator, CodeDataset};
use crate::hypernetwork::Code2LoRAHead;

/// Training loop for the hypernetwork.
pub struct Trainer {
    pub hypernetwork: Code2LoRAHead,
    pub base_model: Code2LoRAModel,
    pub config: TrainConfig,
    pub device: Device,
    pub varmap: VarMap,
}

impl Trainer {
    pub fn new(
        hypernetwork: Code2LoRAHead,
        base_model: Code2LoRAModel,
        varmap: VarMap,
        config: TrainConfig,
        device: Device,
    ) -> Self {
        Self {
            hypernetwork,
            base_model,
            config,
            device,
            varmap,
        }
    }

    pub fn train(&mut self, dataset: &CodeDataset) -> Result<()> {
        let n_examples = dataset.len();
        info!("Dataset has {n_examples} examples");
        anyhow::ensure!(n_examples > 0, "training dataset is empty");
        let summary = dataset.summary();
        info!(
            "Dataset summary: repos={}, languages={}, commit_rows={}",
            summary.repo_count, summary.language_count, summary.commit_rows
        );

        let n_epochs = self.config.epochs as usize;
        let lr = self.config.lr;
        let batch_size = self.config.batch_size.max(1);
        let checkpoint_dir = Path::new(&self.config.output);
        std::fs::create_dir_all(checkpoint_dir)?;
        info!(
            "Training config: base_model={}, data_dir={}, rank={}, seq_len={}, cache_dir={}, batch_size={}, lr={}",
            self.config.base_model,
            self.config.data_dir,
            self.config.rank,
            self.config.seq_len,
            self.config.cache_dir,
            batch_size,
            lr
        );

        // Split dataset into CR (contrastive) and IR (generative) portions
        let cr_ratio = self.config.cr_holdout as f32;
        let (cr_examples, ir_examples) = dataset.split(cr_ratio);
        info!(
            "CR examples: {}, IR examples: {}",
            cr_examples.len(),
            ir_examples.len()
        );

        // Collect trainable variables (hypernetwork only)
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
            let mut epoch_ir_loss: f64 = 0.0;
            let mut epoch_cr_loss: f64 = 0.0;
            let mut ir_steps = 0usize;
            let mut cr_steps = 0usize;

            // ── IR (generative) training ──
            let ir_iter =
                BatchIterator::new_on_device(&ir_examples, batch_size, self.device.clone());
            for (batch_idx, (repo_embs, code_texts)) in ir_iter.enumerate() {
                let loss =
                    self.base_model
                        .compute_ir_loss(&self.hypernetwork, &repo_embs, &code_texts)?;

                // Candle backward + optimizer step
                let loss_val = loss.to_scalar::<f32>()? as f64;
                opt.backward_step(&loss)?;

                epoch_ir_loss += loss_val;
                ir_steps += 1;

                if batch_idx % 10 == 0 {
                    info!("  IR batch {batch_idx}: loss = {loss_val:.6}");
                }
            }

            // ── CR (contrastive) training ──
            let cr_iter =
                BatchIterator::new_on_device(&cr_examples, batch_size, self.device.clone());
            for (batch_idx, (repo_embs, code_texts)) in cr_iter.enumerate() {
                let loss =
                    self.base_model
                        .compute_cr_loss(&self.hypernetwork, &repo_embs, &code_texts)?;

                let loss_val = loss.to_scalar::<f32>()? as f64;
                opt.backward_step(&loss)?;

                epoch_cr_loss += loss_val;
                cr_steps += 1;

                if batch_idx % 10 == 0 {
                    info!("  CR batch {batch_idx}: loss = {loss_val:.6}");
                }
            }

            let avg_ir = if ir_steps > 0 {
                epoch_ir_loss / ir_steps as f64
            } else {
                0.0
            };
            let avg_cr = if cr_steps > 0 {
                epoch_cr_loss / cr_steps as f64
            } else {
                0.0
            };
            let avg_total = if ir_steps + cr_steps > 0 {
                (epoch_ir_loss + epoch_cr_loss) / (ir_steps + cr_steps) as f64
            } else {
                f64::NAN
            };

            info!(
                "Epoch {}/{} — ir_loss = {:.6}, cr_loss = {:.6}, avg_total = {:.6}",
                epoch + 1,
                n_epochs,
                avg_ir,
                avg_cr,
                avg_total
            );

            // Checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0 {
                let ckpt_path = checkpoint_dir.join(format!("epoch_{:04}.safetensors", epoch + 1));
                self.hypernetwork.save(&ckpt_path)?;
                info!("Checkpoint saved to {ckpt_path:?}");
            }
        }

        // Save final checkpoint
        let final_path = checkpoint_dir.join("final.safetensors");
        self.hypernetwork.save(&final_path)?;
        info!("Final checkpoint saved to {final_path:?}");

        Ok(())
    }
}
