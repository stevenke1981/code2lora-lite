# Plan: code2lora-lite

## Goal
Implement a lightweight Code2LoRA system in Rust (Candle) that trains a hypernetwork to generate repository-specific LoRA adapters for Qwen2.5-Coder-0.5B, targeting RTX 3060 Ti 8GB.

## Complexity
L3 — Multi-file, dependencies between modules (encoder → hypernetwork → trainer → inference)

## Sub-tasks

### P0: Scaffold
1. [ ] Init Cargo project with dependencies → file: `Cargo.toml` → output: compileable project skeleton
2. [ ] CLI entry point with clap subcommands → file: `src/main.rs` → output: `code2lora-lite --help` works
3. [ ] Config loader → file: `src/config.rs` → output: TOML/YAML config parsed
4. [ ] Setup CUDA device detection → file: `src/main.rs` → output: logs "Using CUDA" or falls back to CPU

### P1: RepoEncoder
1. [ ] all-MiniLM-L6-v2 embedding model loader → file: `src/repo_encoder.rs` → output: can embed a single string
2. [ ] File chunking (4096 token window, 512 overlap) → file: `src/repo_encoder.rs` → output: correct chunk boundaries
3. [ ] File-level mean pooling → file: `src/repo_encoder.rs` → output: per-file vector
4. [ ] Repository-level weighted avg + max pool aggregation → file: `src/repo_encoder.rs` → output: repo embedding
5. [ ] NPZ cache system → file: `src/repo_encoder.rs` → output: cached embeddings reloaded correctly
6. [ ] Unit tests with sample repo → file: `tests/test_repo_encoder.rs` → output: test passes

### P2: BaseLLM + LoRA Injection
1. [ ] GGUF Qwen2.5-Coder-0.5B loader → file: `src/base_llm.rs` → output: model generates text without LoRA
2. [ ] Copy Qwen2 attention forward, add LoRA compute path → file: `src/base_llm.rs` → output: LoRA-injected logits
3. [ ] Layer-shared LoRA storage (all layers share one set) → file: `src/base_llm.rs` → output: same adapter applied everywhere
4. [ ] LoRA injection/removal API → file: `src/base_llm.rs` → output: per-batch inject + remove cycle works
5. [ ] Tokenizer (Qwen2 tokenizer via tokenizers crate) → file: `src/base_llm.rs` → output: encode/decode round-trips
6. [ ] Generate function (simple greedy + top-k) → file: `src/base_llm.rs` → output: completes a prompt
7. [ ] Test: compare logits with/without LoRA → file: `tests/test_base_llm.rs` → output: LoRA changes distribution

### P3: Hypernetwork
1. [ ] Code2LoRAHead struct with shared MLP → file: `src/hypernetwork.rs` → output: compiles
2. [ ] 7 output head pairs (q/k/v/o/gate/up/down) → file: `src/hypernetwork.rs` → output: generates LoRAWeights
3. [ ] Learnable log-scales s_m → file: `src/hypernetwork.rs` → output: scales integrated into output
4. [ ] L2Norm + scaling factor ✓d_h → file: `src/hypernetwork.rs` → output: matches paper formula
5. [ ] Save/load checkpoints (safetensors) → file: `src/hypernetwork.rs` → output: round-trip preserves weights
6. [ ] Integration test: repo_emb → hypernetwork → LoRAWeights → type-check → file: `tests/test_hypernetwork.rs`

### P4: Dataset
1. [ ] Download snapshot dataset from HF → file: `scripts/download_data.sh` → output: parquet files exist
2. [ ] Parquet reader for AssertionRecord → file: `src/dataset.rs` → output: deserializes all columns
3. [ ] CR split (hold out repos) → file: `src/dataset.rs` → output: train/val disjoint repos
4. [ ] IR split (chronological 8:1:1) → file: `src/dataset.rs` → output: no temporal leakage
5. [ ] Batch iterator → file: `src/dataset.rs` → output: yields (repo_emb, input_ids, target_ids) triples
6. [ ] Pre-compute repo embeddings for all repos in dataset → file: `src/dataset.rs` → output: cached .npz per repo

### P5: Trainer
1. [ ] Training loop skeleton → file: `src/trainer.rs` → output: one epoch completes without error
2. [ ] Loss computation (cross-entropy on target tokens) → file: `src/trainer.rs` → output: loss decreases over epochs
3. [ ] AdamW optimizer + cosine lr schedule → file: `src/trainer.rs` → output: loss curve matches expected shape
4. [ ] Gradient accumulation (accum=4) → file: `src/trainer.rs` → output: loss same as bs=4 (approximately)
5. [ ] Validation loop (CR eval after each epoch) → file: `src/trainer.rs` → output: Exact Match + EditSim reported
6. [ ] Checkpoint saving (best val loss) → file: `src/trainer.rs` → output: checkpoint written each epoch
7. [ ] Overfit test: train on single repo, verify memorization → file: `tests/end_to_end.rs` → output: loss → 0

### P6: Inference Pipeline
1. [ ] `adapt` command implementation → file: `src/infer.rs` → output: produces adapter.safetensors for any repo
2. [ ] `complete` command implementation → file: `src/infer.rs` → output: assertion completion via adapted model
3. [ ] `encode` command implementation → file: `src/infer.rs` → output: cached .npz file
4. [ ] Full end-to-end test: train tiny → adapt tiny → complete → verify output format → file: `tests/end_to_end.rs`

### P7: Polish
1. [ ] Error handling with thiserror → file: throughout → output: informative error messages
2. [ ] Logging (env_logger, info/warn/error levels) → file: throughout → output: training progress visible
3. [ ] Performance pass: tensor shape checks, unnecessary clones removed → file: throughout → output: GPU util > 60%
4. [ ] README with quick-start example → file: `README.md` → output: user can run from scratch

## Risks

| Risk | Mitigation |
|------|------------|
| Candle Qwen2 GGUF support missing | Patch candle-transformers or use safetensors format |
| Candle autograd too complex for training | Start with inference-only, add training iteratively |
| 8GB VRAM insufficient for seq=2048 | Reduce to seq=1024, lower rank to 4, enable checkpointing |
| HF Parquet format incompatible with arrow-rs | Convert to JSONL via Python script |
| CUDA 13.2 + candle compatibility | Use candle from source, pin commpatible commit |

## Definition of Done
- [ ] `code2lora-lite train` converges on snapshot dataset
- [ ] `code2lora-lite adapt` produces a valid adapter.safetensors
- [ ] `code2lora-lite complete` returns meaningful assertion completions
- [ ] All unit tests pass under `cargo test`
- [ ] End-to-end test on a small repo runs in < 30 min on RTX 3060 Ti

## Assumptions
- Candle 0.9+ has Qwen2 model support (confirmed via current source)
- GGUF 4-bit of Qwen2.5-Coder-0.5B is available on HF Hub
- HF code2lora datasets are accessible via hf-hub + arrow-rs
- 8GB VRAM is sufficient for training with the stated configuration
