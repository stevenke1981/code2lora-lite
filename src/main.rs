use anyhow::Result;
use candle_core::Device;
use clap::{Parser, Subcommand};
use log::info;

mod base_llm;
mod config;
mod dataset;
mod hypernetwork;
mod infer;
mod qwen2_lora;
mod repo_encoder;
mod trainer;

#[derive(Parser)]
#[command(
    name = "code2lora-lite",
    about = "Code2LoRA hypernetwork for repo-specific code LLM adapters"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the hypernetwork
    Train {
        /// Path to the training dataset (directory of .jsonl, .txt, or .py files)
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
        /// Batch size
        #[arg(short, long, default_value_t = 4)]
        batch_size: usize,
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
        Commands::Train {
            data_dir,
            output,
            epochs,
            lr,
            batch_size,
        } => {
            let hn_config = config::HypernetworkConfig::default();

            // Load dataset before touching the large base model so bad data paths fail fast.
            let dataset_path = std::path::PathBuf::from(&data_dir);
            anyhow::ensure!(
                dataset_path.exists(),
                "Data directory {data_dir:?} not found. For real data, run scripts/download_code2lora_data.ps1 first."
            );
            let dummy_device = candle_core::Device::Cpu;
            let dataset = dataset::CodeDataset::load_from_dir(&dataset_path, &dummy_device)?;
            anyhow::ensure!(
                !dataset.is_empty(),
                "No training examples found in {data_dir:?}"
            );

            let train_config = config::TrainConfig {
                rank: 8,
                base_model: "Qwen/Qwen2.5-Coder-0.5B".into(),
                data_dir: data_dir.clone(),
                output,
                epochs,
                lr,
                batch_size,
                seq_len: 2048,
                cache_dir: "cache".into(),
                cr_holdout: 0.2,
            };

            // Load base model (frozen Qwen2) with tokenizer
            let dtype = candle_core::DType::F32;
            let base_model = base_llm::Code2LoRAModel::new(&device, dtype, &hn_config)?;
            info!("Base model loaded");

            // Create hypernetwork (trainable)
            let varmap = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&varmap, dtype, &device);
            let hn = hypernetwork::Code2LoRAHead::new(vb, &hn_config, &varmap)?;
            info!("Hypernetwork created");

            // Create trainer
            let mut trainer = trainer::Trainer::new(hn, base_model, varmap, train_config, device);

            trainer.train(&dataset)?;
        }
        Commands::Adapt { repo_path, output } => {
            infer::adapt(
                &std::path::PathBuf::from(repo_path),
                &std::path::PathBuf::from(output),
                &device,
            )?;
        }
        Commands::Complete {
            repo_path,
            adapter,
            output,
        } => {
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
