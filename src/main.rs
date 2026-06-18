use anyhow::Result;
use candle_core::Device;
use clap::{Parser, Subcommand};
use log::info;

mod base_llm;
mod config;
mod dataset;
mod hypernetwork;
mod infer;
mod repo_encoder;
mod trainer;

#[derive(Parser)]
#[command(name = "code2lora-lite", about = "Code2LoRA hypernetwork for repo-specific code LLM adapters")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the hypernetwork
    Train {
        /// Path to the training dataset
        #[arg(short, long, default_value = "data")]
        data_dir: String,
        /// Output directory for checkpoints
        #[arg(short, long, default_value = "checkpoints")]
        output: String,
        /// Number of epochs
        #[arg(short, long, default_value_t = 10)]
        epochs: u32,
        /// Learning rate
        #[arg(short, long, default_value_t = 1e-4)]
        lr: f64,
    },
    /// Generate LoRA adapter for a repo
    Adapt {
        /// Path to the repository
        repo_path: String,
        /// Output adapter path
        #[arg(short, long, default_value = "adapter.safetensors")]
        output: String,
    },
    /// Run adapted model and output assertion
    Complete {
        /// Path to the repository
        repo_path: String,
        /// Path to the adapter weights
        adapter: String,
        /// Output path for the assertion
        #[arg(short, long, default_value = "assertion.txt")]
        output: String,
    },
    /// Encode a repo to embedding
    Encode {
        /// Path to the repository
        repo_path: String,
        /// Output path
        #[arg(short, long, default_value = "repo_embedding.embed")]
        output: String,
    },
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();
    let device = Device::cuda_if_available(0)?;
    info!("Using device: {device:?}");

    match cli.command {
        Commands::Train { data_dir, output, epochs, lr } => {
            let train_config = config::TrainConfig {
                rank: 8,
                base_model: "Qwen/Qwen2.5-Coder-0.5B".into(),
                data_dir,
                output,
                epochs,
                lr,
                seq_len: 2048,
                cache_dir: "cache".into(),
                cr_holdout: 0.2,
            };
            let hn_config = config::HypernetworkConfig::default();
            let varmap = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
            let hn = hypernetwork::Code2LoRAHead::new(vb, &hn_config, &varmap)?;
            let mut trainer = trainer::Trainer::new(hn, varmap, train_config, device);
            let dataset = dataset::CodeDataset::new();
            trainer.train(&dataset)?;
        }
        Commands::Adapt { repo_path, output } => {
            infer::adapt(
                &std::path::PathBuf::from(repo_path),
                &std::path::PathBuf::from(output),
                &device,
            )?;
        }
        Commands::Complete { repo_path, adapter, output } => {
            infer::complete(
                &std::path::PathBuf::from(repo_path),
                &std::path::PathBuf::from(adapter),
                &std::path::PathBuf::from(output),
                &device,
            )?;
        }
        Commands::Encode { repo_path, output } => {
            infer::encode(
                &std::path::PathBuf::from(repo_path),
                &std::path::PathBuf::from(output),
                &device,
            )?;
        }
    }

    Ok(())
}
