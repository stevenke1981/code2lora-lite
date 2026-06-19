# Todos — code2lora-lite

> 追蹤實作進度。使用格式：`[ ]` 未開始、`[/]` 進行中、`[x]` 完成。

---

## P0: Scaffold

- [x] Init Cargo 專案，設定依賴
- [x] CLI 入口（clap 4 子命令：train/adapt/complete/encode）
- [x] Config 載入（HypernetworkConfig + TrainConfig）
- [x] CUDA device detection

## P1: RepoEncoder

- [x] all-MiniLM-L6-v2 模型載入（Candle BERT）
- [x] 檔案 chunking（4096 window, 512 overlap）
- [x] File-level mean pooling
- [x] Repository-level weighted avg + max pool → 768-dim
- [x] NPZ 快取系統
- [x] 單元測試

## P2: BaseLLM + LoRA 注入

- [x] Qwen2.5-Coder-0.5B safetensors 載入（非 GGUF，Candle 原生）
- [x] LoRALinear: 自訂線性層含 LoRA 前向路徑（flatten → 2D matmul → reshape）
- [x] LoRAAttention: Q/K/V/O 個別 LoRA hook
- [x] LoRAMLP: Gate/Up/Down 個別 LoRA hook
- [x] LoRAModel: 完整 24 層 transformer 支援
- [x] Layer-shared LoRA 注入/清除 API（inject_lora_from_hn / clear_lora）
- [x] Tokenizer 整合（Qwen2 tokenizer via tokenizers crate）
- [x] 分片 safetensors 載入支援（collect_safetensors via index.json）
- [x] tie_word_embeddings 處理
- [x] 單元測試

## P3: Hypernetwork

- [x] Code2LoRAHead struct + shared MLP（768→384→384）
- [x] 7 對輸出頭（q/k/v/o/gate/up/down）
- [x] Learnable log-scales s_m（初始 -3.5）
- [x] L2Norm + scaling（√d_h）
- [x] Per-layer embedding（Embedding(24, 384) + forward_all）
- [x] lora_in_dim / lora_out_dim 分離（GQA 支援）
- [x] Checkpoint save/load（safetensors）
- [x] 整合測試（shape + forward + backward）

## P4: Dataset

- [x] CodeDataset struct（load_from_dir 讀取 .py/.txt）
- [x] RepoEmbedding 資料結構（768-dim f32 vec）
- [x] BatchIterator（依 batch_size 分割）
- [x] generate_synthetic() 整合測試用合成資料
- [x] 下載/轉換 HF 真實 RepoPeftBench 資料集（`scripts/prepare_repopeftbench.ps1`）
- [x] JSONL 真實資料 loader（支援 `input_prefix` / `target_value` / split 欄位）
- [x] 完整 CR/IR 分割（優先使用官方 split，否則 ratio fallback）
- [x] 真實資料 smoke 驗證（100 筆 OOD QnA JSONL + Rust loader ignored test）

## P5: Trainer

- [x] 訓練循環 skeleton（Trainer struct + train 方法）
- [x] Cross-entropy loss（Code2LoRAModel::forward + compute_loss）
- [x] AdamW optimizer（支援 VarMap）
- [x] CR/IR 訓練模式（CR with repo_emb, IR without）
- [x] Validation loop
- [x] Checkpoint saving
- [x] Overfit test（test_training_pipeline_full）

## P6: 真實模型端到端測試

- [x] 真實 Qwen2.5-Coder-0.5B 下載與載入
- [x] 合成資料訓練迴圈（3 epochs, batch=2, 8 筆範例）
- [x] GPU 訓練驗證（RTX 3060 Ti, ~4.25s 完成）
- [x] Bug 修復：分片載入 / tie_word_embeddings / repo_embed_dim

## P7: Inference Pipeline

- [x] `adapt` 命令實現（main.rs → infer::adapt）
- [x] `complete` 命令實現（main.rs → infer::complete）
- [x] `encode` 命令實現（main.rs → infer::encode）
- [x] `adapt` 載入已訓練 hypernetwork checkpoint（不再產生隨機 adapter）
- [x] `complete` 接受真實 prefix 與 max token 參數
- [x] `evo-init` / `evo-adapt` 實作 Code2LoRA-Evo GRU hidden-state adapter 更新 primitive
- [x] Repo embedding cache round-trip 修正（binary-safe load）
- [x] Adapter safetensors round-trip 測試（fast test）
- [x] RepoEncoder → adapt → complete 完整端到端測試（`test_p7_full_end_to_end_real_inference`, ignored）

## P8: Polish

- [x] Error handling（thiserror + anyhow）
- [x] Logging（env_logger, info 層級）
- [x] README 與專案說明（中英文）
- [x] Codex/OpenCode compact context pack（`agent-context`）
- [x] 專案級 `AGENTS.md` 要求 agent session 先使用 compact context
- [x] PowerShell wrapper（`scripts/agent-context.ps1`）一鍵產生 context + metrics
- [x] Token 減量 metrics（raw repo estimate vs compact context estimate）
- [x] Token reduction gate（`audit.json` + `-MinReduction` non-zero fail）
- [x] Session token audit（`session-audit.json`，量測 context + 實際開檔）
- [x] Agent open wrapper（`scripts/agent-open.ps1` 自動維護 opened-files log）
- [x] MCP wrapper（`scripts/code2lora-mcp.ps1` + `scripts/mcp-smoke.ps1`）
- [x] MCP config installer（`scripts/install-mcp-config.ps1` 實際寫入 Codex/OpenCode config）
- [x] Linux/macOS MCP config installer（`scripts/install-mcp-config.sh` 使用 `pwsh` + Python stdlib）
- [x] Symbol Map（Rust/PowerShell 入口摘要，降低 agent 導航成本）
- [x] 效能調優：BatchIterator 直接在訓練 device 建 tensor，減少 CPU→GPU batch 搬移
- [x] 效能調優：adapt / complete / encode 使用 repo embedding cache
- [x] Code2LoRA-Evo：GRU state/update/adapter persistence 已完成
- [x] Code2LoRA-Evo：commit sequence dataset + truncated-BPTT `evo-train` 已完成
- [x] RepoPeftBench Evo prepare script（datasets-server parquet → commit-joined JSONL）
- [ ] Code2LoRA-Evo：真實 evolution-track CR/IR/OOD exact-match metrics 長跑驗證
- [ ] 效能量測：GPU util > 60%（需真實 GPU profiling）
- [x] 移除 dead code warnings（`cargo test --no-default-features` 無 warnings）
