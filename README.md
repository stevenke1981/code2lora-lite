# code2lora-lite

> **Lightweight Code2LoRA in Rust (Candle): Hypernetwork-Generated LoRA Adapters for Code Language Models**

[![arXiv](https://img.shields.io/badge/arXiv-2606.06492-b31b1b.svg)](https://arxiv.org/abs/2606.06492)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A minimal, dependency-light Rust implementation of **[Code2LoRA](https://arxiv.org/abs/2606.06492)** — a hypernetwork framework that generates repository-specific LoRA adapters for frozen code language models. Designed to run on a single **RTX 3060 Ti (8 GB)** .

---

## Table of Contents

- [What is Code2LoRA?](#what-is-code2lora)
- [Architecture](#architecture)
- [Status](#status)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Project Structure](#project-structure)
- [Differences from the Paper](#differences-from-the-paper)
- [Paper & Citation](#paper--citation)
- [License](#license)

---

## What is Code2LoRA?

**Code2LoRA** (Hotsko et al., 2026) introduces a hypernetwork that reads a software repository and generates **LoRA adapters** on the fly — without per-repository fine-tuning or retrieval at inference time.

| Approach | Inference Cost | Per-Repo Training | Handles Evolution |
|----------|---------------|-------------------|-------------------|
| RAG + context injection | High (tokens per query) | None | Brittle |
| Per-repo LoRA fine-tuning | Zero | Required (expensive) | Requires retraining |
| **Code2LoRA** (this project) | **Zero** | **Hypernetwork forward only** | **Static mode done; Evo mode TBD** |

**Code2LoRA-Static** generates an adapter from a single repository snapshot.  
**Code2LoRA-Evo** (GRU-based, not yet implemented here) updates the adapter incrementally as commits arrive.

This lite implementation focuses on the **Static** variant: given a Python repository, the system encodes it into a 768-dimensional embedding (via `all-MiniLM-L6-v2`), feeds it through a hypernetwork to produce per-module LoRA weights (rank 8, for Q/K/V/O/Gate/Up/Down projections), and injects them into a frozen **Qwen2.5-Coder-0.5B** model for assertion-completion tasks.

---

## Architecture

```
Repository (.py files)
       │
       ▼
┌──────────────────┐
│   RepoEncoder     │  Frozen BERT (all-MiniLM-L6-v2)
│   ────────────   │
│   chunk(4096)    │  Overlapping windows
│   mean pool/file │
│   weighted avg   │──► repo_embedding ∈ ℝ⁷⁶⁸
│   + max pool     │
└──────────────────┘
       │
       ▼
┌──────────────────┐     ┌─────────────────────┐
│   Hypernetwork    │     │   Training Loop      │
│   Code2LoRAHead   │     │   (CR + IR phases)   │
│   ────────────   │     │                      │
│   MLP(768→384)   │     │  CrossEntropy loss   │
│   L2Norm + scale  │     │  AdamW optimizer     │
│   7 × OutHeads   │     │  LR scheduling       │
│       │          │     └──────────┬──────────┘
│   LoRA weights   │               │
│   (A_m, B_m)     │◄──────────────┘
└───────┬──────────┘
        │ inject
        ▼
┌──────────────────┐
│   Base LLM        │  Frozen Qwen2.5-Coder-0.5B
│   ────────────   │
│   · LoRALinear    │  Custom linear with LoRA path
│   · LoRAAttention │  Q/K/V/O hooks
│   · LoRAMLP       │  Gate/Up/Down hooks
│   · lm_head       │  Shared embedding (tie_word_embeddings)
└──────────────────┘
        │
        ▼
   logits → loss (training)
   or     → tokens (inference)
```

### Key Design Decisions

- **Pure Rust / Candle**: No Python dependency. Uses `candle-core 0.10` for tensor ops and autograd.
- **Safetensors, not GGUF**: Loads the model directly from HuggingFace safetensors shards. Avoids the GGUF compatibility layer.
- **Manual LoRA layers**: Instead of patching `candle-transformers`, `qwen2_lora.rs` contains custom `LoRALinear`, `LoRAAttention`, and `LoRAMLP` with explicit LoRA compute paths.
- **GQA support**: Grouped-Query Attention requires different `lora_in_dim` / `lora_out_dim` for K/V projections vs Q/O projections (handled in `config.rs` via `kv_proj_dim`).
- **Repo embedding dimension**: Hypernetwork input is `repo_embed_dim` (768 for MiniLM), not `llm_hidden_dim * 2` as earlier prototypes assumed.

---

## Status

| Component | Status | Tests |
|-----------|--------|-------|
| RepoEncoder (all-MiniLM-L6-v2) | ✅ | `test_repo_encoder` |
| LoRALinear / LoRAAttention / LoRAMLP | ✅ | `test_lora_linear_forward` |
| LoRAModel (24-layer Qwen2.5) | ✅ | `test_base_llm_basic` |
| LoRA inject / clear cycle | ✅ | `test_clear_lora` |
| Hypernetwork (7 heads, per-layer emb) | ✅ | `test_hypernetwork_shapes` |
| Training pipeline (tiny model) | ✅ | `test_training_pipeline_full` |
| Real model training (Qwen2.5-0.5B, GPU) | ✅ | `test_p6_real_model_training` (ignored) |
| Inference CLI (adapt/complete/encode) | ✅ | CLI skeleton |
| Full end-to-end test | 🟡 | Not yet |
| Real dataset (RepoPeftBench) | ✅ | HF Parquet → JSONL script + loader test |
| Performance optimization | 🟡 | Device-side batches + clean warnings; GPU util profiling pending |

7 regular tests pass; 1 ignored test requires HF Hub access and CUDA.

---

## Requirements

- **Rust** 1.75+ (edition 2021)
- **CUDA** 12.x + cuBLAS (optional, CPU fallback works but is slow)
- **VRAM**: ~5 GB for training Qwen2.5-Coder-0.5B (fp32), ~2 GB for inference
- **Disk**: ~2 GB for cached Qwen2.5-Coder-0.5B weights (downloaded once)

### Tested On

- OS: Windows 11 + Rust 1.85
- GPU: NVIDIA RTX 3060 Ti (8 GB), CUDA 12.8, driver 578.09
- CPU fallback: Intel i7-12700K

---

## Quick Start

```bash
# 1. Clone and build
git clone https://github.com/your-org/code2lora-lite.git
cd code2lora-lite
cargo build --release

# 2. Run all non-GPU tests
cargo test

# 3. Train on synthetic data with real Qwen2.5-Coder-0.5B (requires GPU + HF)
cargo test test_p6_real_model_training -- --ignored --nocapture

# 4. Download and convert a small real RepoPeftBench sample
powershell -ExecutionPolicy Bypass -File scripts/download_code2lora_data.ps1 -MaxRows 1000

# 5. Train on converted real JSONL
cargo run --release -- train -d data/code2lora-ood -o checkpoints -e 1

# 6. Train on a real code directory
cargo run --release -- train -d ./my-python-project -o checkpoints -e 5

# 7. Generate adapter for a repo
cargo run --release -- adapt ./my-python-project -o adapter.safetensors

# 8. Run assertion completion
cargo run --release -- complete ./my-python-project adapter.safetensors -o assertion.txt

# 9. Encode a repo without the full pipeline
cargo run --release -- encode ./my-python-project -o repo_emb.embed
```

> **Note**: The first run downloads Qwen2.5-Coder-0.5B (~2 GB) and all-MiniLM-L6-v2 (~90 MB) to HuggingFace's cache directory.

---

## CLI Reference

### `train`

```
code2lora-lite train [OPTIONS]

Options:
  -d, --data-dir <DIR>      Directory of .jsonl/.py/.txt files for training
  -o, --output <DIR>        Checkpoint output directory  [default: checkpoints]
  -e, --epochs <N>          Number of epochs  [default: 10]
      --lr <LR>             Learning rate  [default: 1e-4]
  -b, --batch-size <N>      Batch size  [default: 4]
  -h, --help                Print help
```

### `adapt`

```
code2lora-lite adapt [OPTIONS] <REPO_PATH>

Arguments:
  <REPO_PATH>               Path to the repository

Options:
  -o, --output <FILE>       Output adapter path  [default: adapter.safetensors]
  -h, --help                Print help
```

### `complete`

```
code2lora-lite complete [OPTIONS] <REPO_PATH> <ADAPTER>

Arguments:
  <REPO_PATH>               Path to the repository
  <ADAPTER>                 Path to the adapter weights (safetensors)

Options:
  -o, --output <FILE>       Output path for assertion  [default: assertion.txt]
  -h, --help                Print help
```

### `encode`

```
code2lora-lite encode [OPTIONS] <REPO_PATH>

Arguments:
  <REPO_PATH>               Path to the repository

Options:
  -o, --output <FILE>       Output path  [default: repo_embedding.embed]
  -h, --help                Print help
```

---

## Project Structure

```
code2lora-lite/
├── Cargo.toml                  # Rust project manifest
├── README.md                   # This file
├── README.zh-TW.md             # Traditional Chinese documentation
├── spec.md                     # Original specification document
├── plan.md                     # Implementation plan
├── todos.md                    # Progress tracking
├── src/
│   ├── main.rs                 # CLI entry point (clap 4 subcommands)
│   ├── config.rs               # HypernetworkConfig + TrainConfig
│   ├── repo_encoder.rs         # all-MiniLM-L6-v2 embedding pipeline
│   ├── hypernetwork.rs         # Code2LoRAHead: MLP + 7 head pairs
│   ├── qwen2_lora.rs           # Custom LoRALinear/LoRAAttention/LoRAMLP/LoRAModel
│   ├── base_llm.rs             # Code2LoRAModel orchestrator + tests
│   ├── dataset.rs              # CodeDataset + RepoPeftBench JSONL loader
│   ├── trainer.rs              # Training loop (CR/IR, AdamW, validation)
│   └── infer.rs                # adapt/complete/encode pipeline (skeleton)
├── scripts/
│   └── download_code2lora_data.ps1  # HF Parquet download + JSONL conversion
```

---

## Differences from the Paper

| Aspect | Paper | code2lora-lite |
|--------|-------|----------------|
| Framework | Python (PyTorch) | Rust (Candle 0.10) |
| Base model | Qwen2.5-Coder-1.5B | Qwen2.5-Coder-0.5B |
| LoRA rank | 16 | 8 |
| Training data | RepoPeftBench (604 repos, 40K tasks) | Synthetic data + converted RepoPeftBench JSONL |
| Quantization | — | fp32 (no quantization) |
| RepoEncoder embedding | 768-dim (concat mean+max of MiniLM) | ✅ Same |
| Hypernetwork MLP | 768→768→384 | 768→384→384 (simplified) |
| Layer embedding | Learned 24-dim | ✅ Learned 24-dim |
| GQA support for K/V | Implicit via per-module heads | ✅ Explicit kv_proj_dim |
| Code2LoRA-Evo (GRU) | ✅ Full implementation | ❌ Not implemented |
| Inference token overhead | Zero | ✅ Zero |

---

## Paper & Citation

Code2LoRA was published at **arXiv:2606.06492** (June 2026):

- **Title**: *Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution*
- **Authors**: Liliana Hotsko, Yinxi Li, Yuntian Deng, Pengyu Nie (University of Waterloo)
- **Paper**: [https://arxiv.org/abs/2606.06492](https://arxiv.org/abs/2606.06492)
- **Official code**: [https://anonymous.4open.science/r/code2lora-6857](https://anonymous.4open.science/r/code2lora-6857)
- **HF datasets**: [https://huggingface.co/code2lora](https://huggingface.co/code2lora)
- **RepoPeftBench**: Custom benchmark of 604 Python repositories (40K train + 12K test tasks)

```bibtex
@article{hotsko2026code2lora,
  title   = {Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution},
  author  = {Hotsko, Liliana and Li, Yinxi and Deng, Yuntian and Nie, Pengyu},
  journal = {arXiv preprint arXiv:2606.06492},
  year    = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
