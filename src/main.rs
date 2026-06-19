use anyhow::Result;
use candle_core::Device;
use clap::{Parser, Subcommand};
use log::info;

mod agent_context;
mod base_llm;
mod config;
mod dataset;
mod evo;
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
        /// Path to a trained hypernetwork checkpoint
        #[arg(short = 'm', long)]
        hypernetwork: String,
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
        /// Assertion/code prefix used as the generation prompt
        #[arg(short, long)]
        prefix: String,
        /// Maximum number of new tokens to generate
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
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
    /// Build a compact Codex/OpenCode context pack with token-savings metrics
    AgentContext {
        /// Path to the repository
        repo_path: String,
        /// Output directory, relative to the repository when not absolute
        #[arg(short, long, default_value = ".code2lora/agent-context")]
        output_dir: String,
        /// Maximum number of high-signal files to include
        #[arg(long, default_value_t = 24)]
        max_files: usize,
    },
    /// Initialize a Code2LoRA-Evo checkpoint
    EvoInit {
        /// Output Evo checkpoint path
        #[arg(short, long, default_value = "evo.safetensors")]
        output: String,
    },
    /// Incrementally update an Evo adapter from commit diff embeddings/files
    EvoAdapt {
        /// Path to a trained Code2LoRA-Evo checkpoint
        #[arg(short = 'm', long)]
        evo_checkpoint: String,
        /// Initial repository path, used when --state-in is absent
        #[arg(long)]
        repo_path: Option<String>,
        /// Initial repository embedding file, used when --state-in is absent
        #[arg(long)]
        repo_embedding: Option<String>,
        /// Previous Evo hidden state
        #[arg(long)]
        state_in: Option<String>,
        /// Output Evo hidden state after applying diffs
        #[arg(long, default_value = "evo_state.safetensors")]
        state_out: String,
        /// Commit diff text/patch file; may be repeated
        #[arg(long)]
        diff_file: Vec<String>,
        /// Commit diff embedding file; may be repeated
        #[arg(long)]
        diff_embedding: Vec<String>,
        /// Output adapter path
        #[arg(short, long, default_value = "adapter.safetensors")]
        output: String,
    },
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Train {
            data_dir,
            output,
            epochs,
            lr,
            batch_size,
        } => {
            let device = Device::cuda_if_available(0)?;
            info!("Using device: {device:?}");
            let hn_config = config::HypernetworkConfig::default();

            // Load dataset before touching the large base model so bad data paths fail fast.
            let dataset_path = std::path::PathBuf::from(&data_dir);
            anyhow::ensure!(
                dataset_path.exists(),
                "Data directory {data_dir:?} not found. For real data, run scripts/prepare_repopeftbench.ps1 first."
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
        Commands::Adapt {
            repo_path,
            hypernetwork,
            output,
        } => {
            let device = Device::cuda_if_available(0)?;
            info!("Using device: {device:?}");
            infer::adapt(
                &std::path::PathBuf::from(repo_path),
                &std::path::PathBuf::from(hypernetwork),
                &std::path::PathBuf::from(output),
                &device,
            )?;
        }
        Commands::Complete {
            repo_path,
            adapter,
            prefix,
            max_tokens,
            output,
        } => {
            let device = Device::cuda_if_available(0)?;
            info!("Using device: {device:?}");
            infer::complete(
                &std::path::PathBuf::from(repo_path),
                &std::path::PathBuf::from(adapter),
                &prefix,
                &std::path::PathBuf::from(output),
                &device,
                max_tokens,
            )?;
        }
        Commands::Encode { repo_path, output } => {
            let device = Device::cuda_if_available(0)?;
            info!("Using device: {device:?}");
            infer::encode(
                &std::path::PathBuf::from(repo_path),
                &std::path::PathBuf::from(output),
                &device,
            )?;
        }
        Commands::AgentContext {
            repo_path,
            output_dir,
            max_files,
        } => {
            let report = agent_context::write_agent_context(
                &std::path::PathBuf::from(repo_path),
                &std::path::PathBuf::from(output_dir),
                max_files,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Commands::EvoInit { output } => {
            let device = Device::cuda_if_available(0)?;
            info!("Using device: {device:?}");
            infer::evo_init_checkpoint(&std::path::PathBuf::from(output), &device)?;
        }
        Commands::EvoAdapt {
            evo_checkpoint,
            repo_path,
            repo_embedding,
            state_in,
            state_out,
            diff_file,
            diff_embedding,
            output,
        } => {
            let device = Device::cuda_if_available(0)?;
            info!("Using device: {device:?}");
            infer::evo_adapt(
                &std::path::PathBuf::from(evo_checkpoint),
                repo_path.as_deref().map(std::path::Path::new),
                repo_embedding.as_deref().map(std::path::Path::new),
                state_in.as_deref().map(std::path::Path::new),
                &diff_file,
                &diff_embedding,
                &std::path::PathBuf::from(state_out),
                &std::path::PathBuf::from(output),
                &device,
            )?;
        }
    }

    Ok(())
}
