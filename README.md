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
- [Human + Agents Usage Guide](USAGE.md)
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
| Inference CLI (adapt/complete/encode) | ✅ | LoRA adapter safetensors |
| Full end-to-end test | ✅ | `test_p7_full_end_to_end_real_inference` (ignored) |
| Real dataset (RepoPeftBench) | ✅ | HF Parquet → JSONL script + real-data smoke test |
| Performance optimization | 🟡 | Device-side batches + clean warnings; GPU util profiling pending |

10 regular tests pass; 4 ignored tests require HF Hub/model access, prepared
RepoPeftBench data, or longer GPU runs.

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

# 4. Run the full real inference E2E test (downloads MiniLM + Qwen2.5-Coder)
cargo test test_p7_full_end_to_end_real_inference -- --ignored --nocapture

# 5. Download and convert RepoPeftBench snapshots/QnA data
powershell -ExecutionPolicy Bypass -File scripts/prepare_repopeftbench.ps1 `
  -OutputDir data/repopeftbench `
  -SkipCloneRepos

# 6. Verify the converted real dataset with the Rust loader
$env:CODE2LORA_REAL_DATA_DIR="data/repopeftbench"
cargo test test_real_repopeftbench_jsonl_smoke -- --ignored --nocapture

# 7. Train on converted real JSONL
cargo run --release -- train -d data/repopeftbench -o checkpoints -e 1

# 8. Train on a real code directory
cargo run --release -- train -d ./my-python-project -o checkpoints -e 5

# 9. Generate adapter for a repo using the trained hypernetwork checkpoint
cargo run --release -- adapt ./my-python-project -m checkpoints/final.safetensors -o adapter.safetensors

# 10. Run assertion completion from a real prompt/prefix
cargo run --release -- complete ./my-python-project adapter.safetensors `
  --prefix "def test_answer():`n    assert answer() ==" `
  --max-tokens 64 `
  -o assertion.txt

# 11. Build a compact Codex/OpenCode context pack with token-savings metrics
cargo run --release -- agent-context ./my-python-project -o .code2lora/agent-context --max-files 24

# Or use the agent-friendly PowerShell wrapper
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-context.ps1 -RepoPath ./my-python-project

# Optional: require at least 80% estimated reduction (default)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-context.ps1 -RepoPath ./my-python-project -MinReduction 0.80

# Audit a real agent session after recording opened raw files
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-open.ps1 `
  -RepoPath ./my-python-project `
  -Files AGENTS.md,src/lib.rs
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-session-audit.ps1 `
  -RepoPath ./my-python-project `
  -OpenedFilesPath .code2lora/agent-context/opened-files.txt

# Run the MCP wrapper smoke test
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/mcp-smoke.ps1 -RepoPath .

# Install the MCP server into local Codex/OpenCode config (backs up first)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install-mcp-config.ps1 -RepoPath . -Target All -Apply

# 12. Encode a repo without the full pipeline
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
  -m, --hypernetwork <FILE>  Trained hypernetwork checkpoint
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
  -p, --prefix <TEXT>        Assertion/code prefix used as the generation prompt
      --max-tokens <N>       Maximum number of new tokens to generate  [default: 64]
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

### `agent-context`

```
code2lora-lite agent-context [OPTIONS] <REPO_PATH>

Arguments:
  <REPO_PATH>               Path to the repository

Options:
  -o, --output-dir <DIR>    Output directory, relative to the repo when not absolute
                            [default: .code2lora/agent-context]
      --max-files <N>       Maximum high-signal files in the context pack [default: 24]
  -h, --help                Print help
```

This command writes:

- `context.md`: compact repository context for Codex/OpenCode to read first
- `metrics.json`: raw-token estimate, compact-context estimate, and saved-token ratio
- `audit.json`: pass/fail gate for the required token-reduction ratio
- `session-audit.json`: pass/fail estimate for context pack plus raw files the
  agent actually opened
- `opened-files.txt`: raw files recorded by `scripts/agent-open.ps1`
- `codex-prompt.md`: prompt stub for Codex sessions
- `opencode-prompt.md`: prompt stub for OpenCode sessions
- `Symbol Map`: Rust/PowerShell entry points so agents can navigate without
  opening broad source files first

The token metric is a deterministic `chars / 4` estimate. It is not a billing
counter, but it gives a repeatable before/after signal for whether the agent is
reading a compact pack instead of broad source dumps.

Project-level `AGENTS.md` tells Codex/OpenCode to run
`scripts/agent-context.ps1` at session start, then read
`.code2lora/agent-context/context.md` before opening broad source files.
The wrapper fails non-zero when `-MinReduction` is not met, so token savings are
enforced instead of being only informational.
For end-of-task evidence, `scripts/agent-session-audit.ps1` compares the raw
repository estimate with `context.md` plus the raw files listed in
`.code2lora/agent-context/opened-files.txt`.
Use `scripts/agent-open.ps1` when reading raw files so the opened-files log is
maintained automatically.

MCP-compatible clients can run `scripts/code2lora-mcp.ps1` as a stdio server.
Repo-local config examples live in `mcp/codex.example.toml` and
`mcp/opencode.example.jsonc`.
Use `scripts/install-mcp-config.ps1` to merge the server entry into local
Codex/OpenCode config files with backups and a smoke-test gate.

---

## Project Structure

```
code2lora-lite/
├── Cargo.toml                  # Rust project manifest
├── AGENTS.md                   # Codex/OpenCode compact-context startup rule
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
│   ├── infer.rs                # adapt/complete/encode pipeline
│   └── agent_context.rs        # Codex/OpenCode context pack + token metrics
├── scripts/
│   ├── agent-context.ps1           # Codex/OpenCode context-pack wrapper
│   ├── agent-open.ps1              # Open raw files and record session usage
│   ├── agent-session-audit.ps1     # Audit actual session token savings
│   ├── code2lora-mcp.ps1           # MCP stdio server wrapper
│   ├── install-mcp-config.ps1       # Merge MCP entry into Codex/OpenCode config
│   ├── mcp-smoke.ps1               # MCP JSON-RPC smoke test
│   └── prepare_repopeftbench.ps1   # HF Parquet download + JSONL conversion
├── mcp/
│   ├── codex.example.toml          # Codex MCP config example
│   └── opencode.example.jsonc      # OpenCode MCP config example
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
