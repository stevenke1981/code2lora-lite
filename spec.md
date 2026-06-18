# spec.md — code2lora-lite

> 基於論文 _Code2LoRA: Hypernetwork-Generated Adapters for Code Language Models under Software Evolution_ (arXiv: 2606.06492)
> 的輕量化 Rust 實作，目標在 RTX 3060 Ti 8GB 上可訓練與推理。

---

## 1. 專案概述

### 1.1 目的

在 GPU 資源有限的環境下（RTX 3060 Ti 8GB），用純 Rust（Candle 框架）實作一個可運作的 Code2LoRA 系統，能夠：

1. **訓練**一個 hypernetwork，學會從倉庫程式碼產生 repository-specific LoRA adapter
2. **推理**時對任意 Python 倉庫快速產生 adapter（零 token 開銷）
3. 用產生的 adapter 輔助 Qwen2.5-Coder-0.5B 完成 assertion completion 任務

### 1.2 論文核心概念保留

| 論文概念 | code2lora-lite |
|---------|----------------|
| Hypernetwork 生成 LoRA adapter | ✅ 完全保留 |
| Layer-shared LoRA（全層共享） | ✅ 保留 |
| 7 種 projection modules (q,k,v,o,gate,up,down) | ✅ 保留 |
| Repository encoder（weighted avg + max pool） | ✅ 保留 |
| Code2LoRA-Static（單次投影） | ✅ Phase 1 實作 |
| Code2LoRA-Evo（GRU 遞迴） | ❌ Phase 2，暫不納入 |
| LoRA rank 16 | ⚡ 縮為 rank 8 |
| Base LLM Qwen2.5-Coder-1.5B | ⚡ 縮為 Qwen2.5-Coder-0.5B |

### 1.3 非目標

- 不追求複製論文 SOTA 分數（硬體限制）
- 不實作 Code2LoRA-Evo（GRU 遞迴版本）
- 不支援 Python 以外的程式語言（Phase 1）
- 不實作完整的 RepoPeftBench（僅訓練子集，支援 CR 評估）

---

## 2. 系統架構

### 2.1 整體資料流

```
Repository (.py files)
       │
       ▼
┌──────────────────┐
│   RepoEncoder    │  (凍結 embedding model, 無 grad)
│   ────────────   │
│   chunk(4096)    │
│   embed(chunk)   │
│   mean pool/file │
│   weighted avg   │──► repo_embedding ∈ ℝ⁷⁶⁸
│   + max pool     │
└──────────────────┘
       │
       ▼
┌──────────────────┐     ┌─────────────────────┐
│   Hypernetwork   │     │   Training Data      │
│   Code2LoRAHead  │     │   (input, target)    │
│   ────────────   │     │                      │
│   MLP(768→384)   │     │  repo_id             │
│   L2Norm         │     │  input_prefix        │
│   7 × OutHeads   │     │  target_value        │
│       │          │     └──────────┬──────────┘
│   LoRA weights   │               │
│   (A_m, B_m)     │◄──────────────┘
└───────┬──────────┘
        │ inject
        ▼
┌──────────────────┐
│   Base LLM        │  (凍結, 4-bit GGUF)
│   Qwen2.5-Coder   │
│   ────────────   │
│   · LoRA 注入     │
│   · Forward pass  │──► logits → loss → backward → AdamW
└──────────────────┘
```

### 2.2 模組依賴圖

```
main.rs
  ├── config.rs        ← serde::Deserialize (TOML/YAML)
  ├── dataset.rs       ← arrow::parquet
  ├── repo_encoder.rs  ← candle, hf-hub
  ├── hypernetwork.rs  ← candle, candle-nn
  ├── base_llm.rs      ← candle, candle-transformers, candle-quantized
  ├── trainer.rs       ← candle, candle-optim, dataset, hypernetwork, base_llm
  └── infer.rs         ← base_llm, hypernetwork, repo_encoder
```

---

## 3. 元件設計

### 3.1 RepoEncoder (`repo_encoder.rs`)

**用途**：將倉庫中所有 Python 檔案編碼為單一固定長度向量。

**演算法**（忠於論文 §3.1）：

```
Step 1: file-level embedding
  for each .py file f_i:
    chunks = split_into_chunks(f_i, chunk_size=4096, overlap=512)
    file_vec = mean_pool([embed_model.encode(c) for c in chunks])
    // file_vec ∈ ℝ³⁸⁴ (all-MiniLM-L6-v2)

Step 2: repository-level aggregation
  importance weight w_i = α·content_distinctiveness + β·file_size + γ·path_importance
  repo_emb = concat(weighted_mean([w_i·f_i]), max_pool([f_i]))
  // repo_emb ∈ ℝ⁷⁶⁸
```

**嵌入模型**：使用 `sentence-transformers/all-MiniLM-L6-v2`（透過 Candle 載入），原因：
- 僅 80MB，VRAM 佔用極低
- 輸出 384-dim，適合降維聚合
- Apache 2.0 license

**快取機制**：編碼結果存為 `.npz`（numpy 格式），避免重複編碼。

**介面**：
```rust
pub struct RepoEncoder {
    model: CandleEmbeddingModel,  // all-MiniLM-L6-v2
}

impl RepoEncoder {
    pub fn new(device: &Device) -> Result<Self>;
    pub fn embed_repo(&self, repo_path: &Path) -> Result<RepoEmbedding>;
    pub fn embed_repo_cached(&self, repo_path: &Path, cache_dir: &Path) -> Result<RepoEmbedding>;
}
```

### 3.2 Hypernetwork (`hypernetwork.rs`)

**用途**：將 repo embedding 轉換為 LoRA adapter 權重。

**架構**（忠於論文 §3.2）：

```
Input: repo_emb ∈ ℝ⁷⁶⁸
  │
  ▼
Linear(768 → 384) + GELU
  │
  ▼
L2Norm + scale(√384)
  │  h ∈ ℝ³⁸⁴
  ├──► Head_q_A → tanh × exp(s_q) → A_q ∈ ℝ⁸×ˢ
  ├──► Head_q_B → tanh × exp(s_q) → B_q ∈ ℝˢ×⁸
  ├──► Head_k_A → tanh × exp(s_k) → A_k
  ├──► Head_k_B → ...              → B_k
  ├──► Head_v_A ...                → A_v, B_v
  ├──► Head_o_A ...                → A_o, B_o
  ├──► Head_gate_A ...             → A_g, B_g
  ├──► Head_up_A ...               → A_up, B_up
  └──► Head_down_A ...             → A_down, B_down
```

**細節**：
- 共享 2-layer MLP（768→384）+ GELU
- 7 對獨立輸出頭（每 module type 一對 A/B）
- 每輸出頭：Linear(384 → hidden_dim × rank) 或 Linear(384 → rank × hidden_dim)
  - hidden_dim 取決於 backbone 的投影維度（Qwen2.5-Coder-0.5B: intermediate=896, hidden=1024）
  - rank=8
- Learnable log-scales s_m 初始 -3.5（控制 adapter 強度）

**參數估算**：
```
Shared MLP: 768*384 + 384*384 + 2*384 = ~442K
Per head (A): 384 * (hidden_dim * rank) = 384 * 1024*8 = 3.15M
Per head (B): 384 * (rank * hidden_dim)  = 3.15M
7 heads: 7 * (3.15M + 3.15M) = ~44.1M
Log-scales: 7 * 2 = 14
────────────────────────────
Total: ~44.5M params (bf16: ~89 MB)
```

**介面**：
```rust
pub struct Code2LoRAHead {
    shared_mlp: Sequential,
    heads_a: Vec<Linear>,
    heads_b: Vec<Linear>,
    log_scales_a: Vec<Packed1D<f32>>,
    log_scales_b: Vec<Packed1D<f32>>,
}

impl Code2LoRAHead {
    pub fn new(vb: VarBuilder, hidden_dim: usize, rank: usize) -> Result<Self>;
    pub fn forward(&self, repo_emb: &Tensor) -> Result<LoRAWeights>;
    pub fn save(&self, path: &Path) -> Result<()>;
    pub fn load(vb: VarBuilder, path: &Path) -> Result<Self>;
}

pub struct LoRAWeights {
    pub q: (Tensor, Tensor),   // (A_q, B_q)
    pub k: (Tensor, Tensor),
    pub v: (Tensor, Tensor),
    pub o: (Tensor, Tensor),
    pub gate: (Tensor, Tensor),
    pub up: (Tensor, Tensor),
    pub down: (Tensor, Tensor),
}
```

### 3.3 Base LLM (`base_llm.rs`)

**用途**：載入 Qwen2.5-Coder-0.5B GGUF 量化模型，提供 LoRA 注入機制。

**關鍵決策**：
- 使用 `candle-quantized` 載入 GGUF 格式
- 從 `candle-transformers` 複製 `Qwen2Model` 的 forward 程式碼，在 Attention 層加入 LoRA 項
- Base LLM 參數設定 `requires_grad(false)`（凍結）

**LoRA 注入**（在 Attention forward 中）：

```rust
// 原始計算
let q = q_proj.forward(&x)?;        // x @ W_q^T
let k = k_proj.forward(&x)?;
let v = v_proj.forward(&x)?;

// 加入 LoRA（當 adapter 存在時）
if let Some((lora_a, lora_b)) = &self.lora_q {
    // ΔW = B @ A, 注入: W' = W + (α/r) * B @ A
    // 等效: out' = x @ (W + α/r·BA)^T = x @ W^T + α/r · (x @ A^T) @ B^T
    let lora_out = (x.matmul(&lora_a.t()?)?  // x @ A^T ∈ ℝ^{seq×r}
        .matmul(&lora_b.t()?)?               // ... @ B^T ∈ ℝ^{seq×hidden}
        .broadcast_mul(&self.lora_scale)?);   // × α/r
    q = (q + &lora_out)?;
}
```

**Layer-shared 機制**：
- 所有層共用同一組 LoRA 權重（論文關鍵設計）
- 儲存方式：`Vec<Option<LoRALayer>>`（`size = num_layers`），所有層指向同一組

**介面**：
```rust
pub struct Code2LoRAModel {
    model: Qwen2Model,     // 從 candle-transformers 複製修改
    lora_layers: Vec<LoRALayer>,  // layer-shared（所有層指向同一組）
}

impl Code2LoRAModel {
    pub fn new_quantized(model_id: &str, device: &Device) -> Result<Self>;
    pub fn inject_lora(&mut self, weights: &LoRAWeights);
    pub fn remove_lora(&mut self);
    pub fn forward(&self, input_ids: &Tensor, seq_len: usize) -> Result<Tensor>;
    pub fn generate(&self, prompt: &str, max_tokens: usize) -> Result<String>;
}
```

### 3.4 Dataset (`dataset.rs`)

**用途**：載載 HF datasets 中的 code2lora Parquet 資料。

**支援的資料集**：
- `code2lora/code2lora-data-snapshots`（靜態訓練 & 評估）
- `code2lora/code2lora-data-smartcap`（可選，資料更多）

**資料結構**：
```rust
#[derive(Deserialize)]
pub struct AssertionRecord {
    pub repo_id: String,
    pub commit_sha: String,
    pub file_path: String,
    pub test_name: String,
    pub assertion_type: String,  // "assert", "self.assert*", "pytest.raises", etc.
    pub input_prefix: String,    // 程式碼前綴
    pub target_value: String,    // 預期輸出
    pub split: String,           // "train", "val", "test"
    pub token_count: Option<u32>,
}

pub struct Dataset {
    pub records: Vec<AssertionRecord>,
    pub repo_embeddings: HashMap<String, RepoEmbedding>,  // 預先編碼
}

impl Dataset {
    pub fn from_parquet(path: &Path) -> Result<Self>;
    pub fn split_cr(&self, held_out_repos: &[&str]) -> (Dataset, Dataset);
    pub fn split_ir(&self, train_ratio: f64) -> (Dataset, Dataset);
    pub fn batches(&self, batch_size: usize) -> Vec<Batch>;
}
```

### 3.5 Trainer (`trainer.rs`)

**用途**：訓練 hypernetwork。

**訓練循環**：
```rust
pub fn train(config: &TrainConfig) -> Result<()> {
    // 1. 載入資料集
    let dataset = Dataset::from_parquet(&config.data_path)?;

    // 2. 初始化模型
    let device = Device::Cuda(0);
    let base_llm = Code2LoRAModel::new_quantized(&config.base_model, &device)?;
    let mut hypernetwork = Code2LoRAHead::new(vb, hidden_dim, rank)?;
    let mut optimizer = AdamW::new(hypernetwork.params(), config.lr, Default::default())?;

    // 3. 預先編碼所有 repo embedding
    let encoder = RepoEncoder::new(&device)?;
    let repo_embs = dataset.cache_repo_embeddings(&encoder, &config.cache_dir)?;

    // 4. 訓練循環
    for epoch in 0..config.epochs {
        for batch in dataset.batches(config.batch_size) {
            let repo_emb = repo_embs[&batch.repo_id];
            let lora_weights = hypernetwork.forward(&repo_emb)?;
            base_llm.inject_lora(&lora_weights);

            let input_ids = tokenizer.encode(&batch.input_prefix, &device)?;
            let target_ids = tokenizer.encode(&batch.target_value, &device)?;
            let logits = base_llm.forward(&input_ids, batch.seq_len)?;
            let loss = cross_entropy(&logits, &target_ids)?;

            optimizer.backward_step(&loss)?;
        }
    }
    Ok(())
}
```

**VRAM 管理**：
- Base LLM 全程 4-bit GGUF（不參與 backward）
- 每個 batch 結束後 `remove_lora()` 清除暫態 LoRA 權重
- 每 N 步呼叫 `cuda_device.synchronize()` + 顯式 drop 暫態 tensor
- Gradient checkpointing：在 LLM forward 中啟用

**CR/IR 分割**：
- CR：hold out 約 15-20% 倉庫做 validation/test
- IR：剩餘倉庫內 8:1:1 分割

### 3.6 CLI (`main.rs`)

使用 `clap` 框架，4 個子命令（詳見 §5）。

### 3.7 Inference (`infer.rs`)

**adapt 流程**：
1. 載入 hypernetwork checkpoint
2. RepoEncoder → repo embedding
3. Hypernetwork forward → LoRAWeights
4. 存為 safetensors（單檔，~2-3 MB）

**complete 流程**：  
1. 載入 adapter.safetensors → LoRAWeights
2. 注入 base LLM
3. Tokenize + generate → completion

---

## 4. 記憶體管理策略

### 4.1 VRAM 預算（RTX 3060 Ti 8GB）

| 組件 | 訓練時 | 推理時 |
|------|--------|--------|
| Base LLM (4-bit GGUF) | ~400 MB | ~400 MB |
| RepoEncoder (MinLM, fp32) | — | ~160 MB |
| Hypernetwork (bf16) | ~90 MB | ~90 MB |
| AdamW states | ~180 MB | — |
| Gradients | ~90 MB | — |
| KV Cache (seq=2048) | ~200 MB | ~400 MB (generate) |
| Activations | ~500 MB | ~200 MB |
| Tensors + overhead | ~500 MB | ~300 MB |
| **總計** | **~1.96 GB** | **~1.55 GB** |
| **可用** | **6 GB 餘裕** | **6.4 GB 餘裕** |

### 4.2 最佳化技術

1. **4-bit GGUF quantization**：主要 VRAM 節省來源
2. **Gradient checkpointing**：LLM forward 不保留中間 activation
3. **LoRA 暫態注入**：每次 batch 後移除，不持久保留
4. **bfloat16**：Hypernetwork 使用 bf16
5. **序列長度裁剪**：`max_seq_len=2048`（原始論文 8192）
6. **Gradient accumulation**：bs=1, accum=4

---

## 5. CLI 命令參考

### 5.1 `train`

```
code2lora-lite train [OPTIONS]

Options:
  -d, --data-dir <DIR>          訓練資料目錄 (HF cached datasets)
  -b, --base-model <MODEL>      Base LLM 名稱 [default: Qwen/Qwen2.5-Coder-0.5B]
  -o, --output <FILE>           輸出 checkpoint 路徑
  -r, --rank <RANK>             LoRA rank [default: 8]
  -e, --epochs <N>              訓練 epoch 數 [default: 3]
      --lr <LR>                 學習率 [default: 1e-4]
      --seq-len <N>             最大序列長度 [default: 2048]
      --cache-dir <DIR>         Repo embedding 快取目錄 [default: ./.cache/embeddings]
      --cr-holdout <RATIO>      Cross-repo holdout 比例 [default: 0.2]
  -h, --help                    顯示說明
```

### 5.2 `adapt`

```
code2lora-lite adapt [OPTIONS]

Options:
  -r, --repo-path <DIR>         目標倉庫路徑
  -m, --hypernetwork <FILE>     訓練好的 hypernetwork checkpoint
  -o, --output <FILE>           輸出 adapter 路徑 [default: ./code2lora-adapter.safetensors]
      --cache-dir <DIR>         Repo embedding 快取目錄 [default: ./.cache/embeddings]
  -h, --help                    顯示說明
```

### 5.3 `complete`

```
code2lora-lite complete [OPTIONS]

Options:
  -r, --repo-path <DIR>         目標倉庫路徑
  -a, --adapter <FILE>          LoRA adapter 檔案
  -p, --prefix <TEXT>           assertion 前綴文字
      --max-tokens <N>          最多產生 token 數 [default: 64]
  -h, --help                    顯示說明
```

### 5.4 `encode`

```
code2lora-lite encode [OPTIONS]

Options:
  -r, --repo-path <DIR>         目標倉庫路徑
  -o, --output <FILE>           輸出 .npz 檔案路徑
      --cache-dir <DIR>         預設快取目錄 [default: ./.cache/embeddings]
  -h, --help                    顯示說明
```

---

## 6. 專案結構

```
D:\code2lora-lite/
├── Cargo.toml
├── spec.md                        # 本檔案
├── plan.md                        # 實作計畫
├── todos.md                       # 追蹤清單
├── src/
│   ├── main.rs                    # CLI 入口
│   ├── config.rs                  # 設定檔結構 + 解析
│   ├── repo_encoder.rs            # RepoEncoder
│   ├── hypernetwork.rs            # Code2LoRAHead
│   ├── base_llm.rs                # Code2LoRAModel (Qwen2 + LoRA)
│   ├── trainer.rs                 # 訓練循環
│   ├── dataset.rs                 # Parquet 資料集載入
│   └── infer.rs                   # adapt + complete pipeline
├── tests/
│   ├── test_repo_encoder.rs
│   ├── test_hypernetwork.rs
│   ├── test_base_llm.rs
│   └── end_to_end.rs
├── examples/
│   └── quickstart.rs              # 快速入門範例
└── scripts/
    └── download_data.sh           # 下載資料集腳本
```

---

## 7. 相依套件 (Cargo.toml)

```toml
[package]
name = "code2lora-lite"
version = "0.1.0"
edition = "2024"

[dependencies]
candle-core = "0.9"
candle-nn = "0.9"
candle-transformers = "0.9"
candle-optim = "0.9"
candle-quantized = "0.9"
candle-datasets = "0.9"
hf-hub = "0.4"
tokenizers = "0.21"
anyhow = "1"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
serde_yaml = "0.9"
parquet = "54"
half = "2"
rand = "0.8"
log = "0.4"
env_logger = "0.11"
thiserror = "2"
walkdir = "2"
rayon = "1"

[features]
default = ["cuda"]
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda",
        "candle-optim/cuda", "candle-quantized/cuda"]
```

---

## 8. 風險與緩解

| 風險 | 影響 | 緩解 |
|------|------|------|
| Candle 對 Qwen2 GGUF 支援不完全 | 無法載入 base LLM | 預留降級方案：使用 candle 直接載入 safetensors（非量化） |
| training autograd 在 Candle 中過於底層 | 開發時間拉長 | 先以 inference-only 為 MVP，訓練視需要調整 |
| 8GB VRAM 仍不足 | OOM | 開啓 gradient checkpointing + 縮小 seq_len + 降 rank |
| HF datasets 無法直接用 arrow-rs 讀取 | 資料集載入失敗 | 降級方案：先用 Python 轉 JSONL，Rust 讀 JSONL |
| CUDA 13.2 與 candle 相容性 | 編譯失敗 | 確認 candle 0.9+ 支援，必要時用 CPU backup |

---

## 9. Phase 2 展望

| 功能 | 說明 |
|------|------|
| Code2LoRa-Evo | GRU 遞迴版本，需 commit diff 資料集 |
| Multi-language | 支援 JS/TS/Java/Rust 等 |
| IDE 整合 | VSCode extension + LSP |
| Cloud inference | WASM / ONNX 匯出 |
| 更多 backbone | CodeGemma, DeepSeek-Coder, StarCoder2 |
