# code2lora-lite

> **輕量化 Code2LoRA：Rust (Candle) 實現的超網路生成 LoRA Adapter**

[![arXiv](https://img.shields.io/badge/arXiv-2606.06492-b31b1b.svg)](https://arxiv.org/abs/2606.06492)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

以極簡 Rust 實作 **[Code2LoRA](https://arxiv.org/abs/2606.06492)** —— 一個能針對特定程式碼倉庫即時生成 LoRA Adapter 的超網路架構，讓凍結的大型語言模型獲得倉庫級別的上下文理解能力。目標運行在 **RTX 3060 Ti (8 GB)** 單卡環境。

---

## 目錄

- [什麼是 Code2LoRA？](#什麼是-code2lora)
- [系統架構](#系統架構)
- [目前進度](#目前進度)
- [環境需求](#環境需求)
- [快速開始](#快速開始)
- [CLI 指令參考](#cli-指令參考)
- [專案結構](#專案結構)
- [與論文的差異](#與論文的差異)
- [論文與引用](#論文與引用)
- [授權條款](#授權條款)

---

## 什麼是 Code2LoRA？

**Code2LoRA**（Hotsko et al., 2026）提出一個超網路（hypernetwork），它能**讀取一個軟體倉庫後即時產生 LoRA Adapter**——不需要針對每個倉庫微調，也不需要每次推理時塞入大量上下文。

| 方法 | 推理成本 | 需逐倉庫訓練 | 因應程式碼演化 |
|------|---------|-------------|---------------|
| RAG + 上下文注入 | 高（每次查詢噴 token） | 無 | 脆弱 |
| 逐倉庫 LoRA 微調 | 零 | 需要（成本高） | 需重新訓練 |
| **Code2LoRA**（本專案） | **零** | **僅一次超網路前向** | **Static 已完成；Evo 待實現** |

**Code2LoRA-Static**：從倉庫單一快照產生 Adapter。  
**Code2LoRA-Evo**（GRU 版本，尚未實作）：隨每次 commit 增量更新 Adapter。

本輕量化實作專注於 **Static** 模式：給定一個 Python 倉庫，先將其編碼為 768 維向量（使用 `all-MiniLM-L6-v2`），經超網路產生各模組的 LoRA 權重（rank=8，涵蓋 Q/K/V/O/Gate/Up/Down 七組投影），最後注入凍結的 **Qwen2.5-Coder-0.5B** 模型以完成 assertion 補全任務。

---

## 系統架構

```
Repository (.py 檔案)
       │
       ▼
┌──────────────────┐
│   RepoEncoder     │  凍結 BERT (all-MiniLM-L6-v2)
│   ────────────   │
│   chunk(4096)    │  滑動視窗切割
│   mean pool/檔案 │
│   weighted avg   │──► repo_embedding ∈ ℝ⁷⁶⁸
│   + max pool     │
└──────────────────┘
       │
       ▼
┌──────────────────┐     ┌─────────────────────┐
│   Hypernetwork    │     │   訓練迴圈            │
│   Code2LoRAHead   │     │   (CR + IR 兩階段)   │
│   ────────────   │     │                      │
│   MLP(768→384)   │     │  CrossEntropy loss   │
│   L2Norm + 縮放   │     │  AdamW 優化器        │
│   7 組輸出頭      │     │  LR 排程             │
│       │          │     └──────────┬──────────┘
│   LoRA 權重       │               │
│   (A_m, B_m)     │◄──────────────┘
└───────┬──────────┘
        │ 注入
        ▼
┌──────────────────┐
│   Base LLM        │  凍結 Qwen2.5-Coder-0.5B
│   ────────────   │
│   · LoRALinear    │  自訂線性層（含 LoRA 路徑）
│   · LoRAAttention │  Q/K/V/O 各別 hook
│   · LoRAMLP       │  Gate/Up/Down 各別 hook
│   · lm_head       │  共用 embedding（tie_word_embeddings）
└──────────────────┘
        │
        ▼
   logits → loss（訓練）
   或     → tokens（推理）
```

### 關鍵設計決策

- **純 Rust / Candle**：無 Python 依賴。使用 `candle-core 0.10` 進行張量運算與自動微分。
- **Safetensors 而非 GGUF**：直接從 HuggingFace 載入 safetensors 分片，避開 GGUF 相容層。
- **手寫 LoRA 層**：`qwen2_lora.rs` 包含自訂 `LoRALinear`、`LoRAAttention`、`LoRAMLP`，明確實現 LoRA 計算路徑。
- **GQA 支援**：Grouped-Query Attention 的 K/V 投影與 Q/O 投影使用不同維度（透過 `config.rs` 的 `kv_proj_dim` 處理）。
- **倉庫嵌入維度**：超網路輸入為 `repo_embed_dim`（MiniLM 輸出 768），而非早期原型假設的 `llm_hidden_dim * 2`。

---

## 目前進度

| 元件 | 狀態 | 測試 |
|------|------|------|
| RepoEncoder (all-MiniLM-L6-v2) | ✅ | `test_repo_encoder` |
| LoRALinear / LoRAAttention / LoRAMLP | ✅ | `test_lora_linear_forward` |
| LoRAModel (24 層 Qwen2.5) | ✅ | `test_base_llm_basic` |
| LoRA 注入/清除循環 | ✅ | `test_clear_lora` |
| Hypernetwork（7 組輸出頭、逐層嵌入） | ✅ | `test_hypernetwork_shapes` |
| 訓練管線（小模型驗證） | ✅ | `test_training_pipeline_full` |
| 真實模型訓練（Qwen2.5-0.5B, GPU） | ✅ | `test_p6_real_model_training` (忽略) |
| 推理 CLI（adapt/complete/encode） | ✅ | LoRA adapter safetensors |
| 完整端到端測試 | ✅ | `test_p7_full_end_to_end_real_inference` (忽略) |
| 真實資料集（RepoPeftBench） | ✅ | HF Parquet → JSONL 腳本 + 真實資料 smoke test |
| 效能調優 | 🟡 | device-side batches + warning 清理；GPU util profiling 待量測 |

10 個常規測試通過；4 個 `#[ignore]` 測試需要 HF Hub / model 存取、已準備的
RepoPeftBench 資料，或較長時間的 GPU 執行。

---

## 環境需求

- **Rust** 1.75+（edition 2021）
- **CUDA** 12.x + cuBLAS（可選，CPU 降級可用但速度慢）
- **VRAM**：訓練 Qwen2.5-Coder-0.5B（fp32）約需 5 GB，推理約需 2 GB
- **磁碟空間**：快取 Qwen2.5-Coder-0.5B 權重約需 2 GB（僅下載一次）

### 已測試環境

- 作業系統：Windows 11 + Rust 1.85
- GPU：NVIDIA RTX 3060 Ti（8 GB），CUDA 12.8，驅動 578.09
- CPU：Intel i7-12700K

---

## 快速開始

```bash
# 1. 克隆並編譯
git clone https://github.com/your-org/code2lora-lite.git
cd code2lora-lite
cargo build --release

# 2. 執行所有非 GPU 測試
cargo test

# 3. 使用真實 Qwen2.5-Coder-0.5B 在合成資料上訓練（需要 GPU + HF）
cargo test test_p6_real_model_training -- --ignored --nocapture

# 4. 執行完整真實推理 E2E 測試（會下載 MiniLM + Qwen2.5-Coder）
cargo test test_p7_full_end_to_end_real_inference -- --ignored --nocapture

# 5. 下載並轉換 RepoPeftBench snapshots/QnA 真實資料
powershell -ExecutionPolicy Bypass -File scripts/prepare_repopeftbench.ps1 `
  -OutputDir data/repopeftbench `
  -SkipCloneRepos

# 6. 用 Rust loader 驗證轉換後的真實資料
$env:CODE2LORA_REAL_DATA_DIR="data/repopeftbench"
cargo test test_real_repopeftbench_jsonl_smoke -- --ignored --nocapture

# 7. 使用轉換後的真實 JSONL 訓練
cargo run --release -- train -d data/repopeftbench -o checkpoints -e 1

# 8. 對真實程式碼目錄進行訓練
cargo run --release -- train -d ./my-python-project -o checkpoints -e 5

# 9. 使用已訓練的 hypernetwork checkpoint 為倉庫產生 Adapter
cargo run --release -- adapt ./my-python-project -m checkpoints/final.safetensors -o adapter.safetensors

# 10. 從真實 prompt/prefix 執行 assertion 補全
cargo run --release -- complete ./my-python-project adapter.safetensors `
  --prefix "def test_answer():`n    assert answer() ==" `
  --max-tokens 64 `
  -o assertion.txt

# 11. 產生 Codex/OpenCode compact context pack 與 token 減量 metrics
cargo run --release -- agent-context ./my-python-project -o .code2lora/agent-context --max-files 24

# 或使用 agent 友善的 PowerShell wrapper
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-context.ps1 -RepoPath ./my-python-project

# 12. 純編碼倉庫（不跑完整管線）
cargo run --release -- encode ./my-python-project -o repo_emb.embed
```

> **注意**：首次執行會自動下載 Qwen2.5-Coder-0.5B（約 2 GB）與 all-MiniLM-L6-v2（約 90 MB）至 HuggingFace 快取目錄。

---

## CLI 指令參考

### `train`

```
code2lora-lite train [選項]

選項：
  -d, --data-dir <DIR>      訓練用的 .jsonl/.py/.txt 檔案目錄
  -o, --output <DIR>        Checkpoint 輸出目錄  [預設: checkpoints]
  -e, --epochs <N>          Epoch 數  [預設: 10]
      --lr <LR>             學習率  [預設: 1e-4]
  -b, --batch-size <N>      Batch 大小  [預設: 4]
  -h, --help                顯示說明
```

### `adapt`

```
code2lora-lite adapt [選項] <REPO_PATH>

引數：
  <REPO_PATH>               目標倉庫路徑

選項：
  -m, --hypernetwork <FILE>  已訓練的 hypernetwork checkpoint
  -o, --output <FILE>       Adapter 輸出路徑  [預設: adapter.safetensors]
  -h, --help                顯示說明
```

### `complete`

```
code2lora-lite complete [選項] <REPO_PATH> <ADAPTER>

引數：
  <REPO_PATH>               目標倉庫路徑
  <ADAPTER>                 Adapter 權重檔案路徑 (safetensors)

選項：
  -p, --prefix <TEXT>        作為生成 prompt 的 assertion/code prefix
      --max-tokens <N>       最大生成 token 數  [預設: 64]
  -o, --output <FILE>       輸出檔案路徑  [預設: assertion.txt]
  -h, --help                顯示說明
```

### `encode`

```
code2lora-lite encode [選項] <REPO_PATH>

引數：
  <REPO_PATH>               目標倉庫路徑

選項：
  -o, --output <FILE>       輸出檔案路徑  [預設: repo_embedding.embed]
  -h, --help                顯示說明
```

### `agent-context`

```
code2lora-lite agent-context [選項] <REPO_PATH>

引數：
  <REPO_PATH>               目標倉庫路徑

選項：
  -o, --output-dir <DIR>    輸出目錄；相對路徑會以目標 repo 為基準
                            [預設: .code2lora/agent-context]
      --max-files <N>       context pack 內最多列出的高訊號檔案數 [預設: 24]
  -h, --help                顯示說明
```

此指令會輸出：

- `context.md`：給 Codex/OpenCode 優先讀取的 compact repo context
- `metrics.json`：原始 token 估算、壓縮 context token 估算、減量比例
- `codex-prompt.md`：Codex session 可用的 prompt stub
- `opencode-prompt.md`：OpenCode session 可用的 prompt stub
- `Symbol Map`：Rust/PowerShell 入口摘要，讓 agent 不必先打開大量原始碼也能導航

token metric 使用可重複的 `chars / 4` 估算。它不是帳單 token 計數器，但可以
穩定驗證 agent 是否先讀 compact pack，而不是直接把大量原始碼塞進 prompt。

專案級 `AGENTS.md` 會要求 Codex/OpenCode 在 session start 執行
`scripts/agent-context.ps1`，再先讀 `.code2lora/agent-context/context.md`，
之後才視任務需要打開原始碼檔案。

---

## 專案結構

```
code2lora-lite/
├── Cargo.toml                  # Rust 專案設定
├── AGENTS.md                   # Codex/OpenCode compact-context 啟動規則
├── README.md                   # 英文說明文件
├── README.zh-TW.md             # 繁體中文說明文件（本檔案）
├── spec.md                     # 原始規格文件
├── plan.md                     # 實作計畫
├── todos.md                    # 進度追蹤
├── src/
│   ├── main.rs                 # CLI 入口（clap 4 子命令）
│   ├── config.rs               # HypernetworkConfig + TrainConfig
│   ├── repo_encoder.rs         # all-MiniLM-L6-v2 嵌入管線
│   ├── hypernetwork.rs         # Code2LoRAHead：MLP + 7 組輸出頭
│   ├── qwen2_lora.rs           # 自訂 LoRALinear/LoRAAttention/LoRAMLP/LoRAModel
│   ├── base_llm.rs             # Code2LoRAModel 協調器 + 測試
│   ├── dataset.rs              # CodeDataset + RepoPeftBench JSONL loader
│   ├── trainer.rs              # 訓練迴圈（CR/IR, AdamW, 驗證）
│   ├── infer.rs                # adapt/complete/encode 管線
│   └── agent_context.rs        # Codex/OpenCode context pack + token metrics
├── scripts/
│   ├── agent-context.ps1           # Codex/OpenCode context-pack wrapper
│   └── prepare_repopeftbench.ps1   # HF Parquet 下載 + JSONL 轉換
```

---

## 與論文的差異

| 項目 | 論文 | code2lora-lite |
|------|------|----------------|
| 框架 | Python (PyTorch) | Rust (Candle 0.10) |
| 基底模型 | Qwen2.5-Coder-1.5B | Qwen2.5-Coder-0.5B |
| LoRA rank | 16 | 8 |
| 訓練資料 | RepoPeftBench（604 倉庫，40K 任務） | 合成資料 + 轉換後 RepoPeftBench JSONL |
| 量化 | — | fp32（無量化） |
| RepoEncoder 嵌入 | 768 維（MiniML 均值+最大值串接） | ✅ 相同 |
| 超網路 MLP | 768→768→384 | 768→384→384（簡化） |
| 逐層嵌入 | 可學習 24 維 | ✅ 可學習 24 維 |
| GQA K/V 支援 | 透過各模組頭隱含處理 | ✅ 明確的 kv_proj_dim |
| Code2LoRA-Evo (GRU) | ✅ 完整實作 | ❌ 尚未實作 |
| 推理 token 開銷 | 零 | ✅ 零 |

---

## 論文與引用

Code2LoRA 發表於 **arXiv:2606.06492**（2026 年 6 月）：

- **標題**：*Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution*
- **作者**：Liliana Hotsko, Yinxi Li, Yuntian Deng, Pengyu Nie（滑鐵盧大學）
- **論文**：[https://arxiv.org/abs/2606.06492](https://arxiv.org/abs/2606.06492)
- **官方程式碼**：[https://anonymous.4open.science/r/code2lora-6857](https://anonymous.4open.science/r/code2lora-6857)
- **HF 資料集**：[https://huggingface.co/code2lora](https://huggingface.co/code2lora)
- **RepoPeftBench**：自定義基準測試，包含 604 個 Python 倉庫（40K 訓練 + 12K 測試任務）

```bibtex
@article{hotsko2026code2lora,
  title   = {Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution},
  author  = {Hotsko, Liliana and Li, Yinxi and Deng, Yuntian and Nie, Pengyu},
  journal = {arXiv preprint arXiv:2606.06492},
  year    = {2026}
}
```

---

## 授權條款

MIT — 詳見 [LICENSE](LICENSE)。
