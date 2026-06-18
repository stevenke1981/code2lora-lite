# Spec: RepoPeftBench 真實資料整合

> 將 HuggingFace RepoPeftBench 資料集整合至 code2lora-lite 訓練管線的設計文件。

---

## 1. 目標

讓 `code2lora-lite train` 能夠使用 **真實的 assertion-completion 資料**（而非合成資料）進行訓練與評估。涵蓋：

- 從 HF 下載 `code2lora/code2lora-data-snapshots` 的 QnA Parquet 檔案
- 對資料集中 ~400 個 Python 倉庫執行 `git clone`
- 透過現有 `RepoEncoder`（all-MiniLM-L6-v2）產生 768-dim repo embedding
- 合併為 JSONL 格式供訓練管線使用
- 保留完整的 CR/IR/OOD 分割供評估

---

## 2. 資料來源

### 2.1 RepoPeftBench Snapshots 資料集

| 屬性 | 值 |
|------|-----|
| HF 路徑 | `code2lora/code2lora-data-snapshots` |
| 格式 | Parquet (QnA rows) |
| 總行數 | 2.32M |
| 唯一 repo 數 | ~400 |
| License | MIT |

### 2.2 檔案對應

HF 上的 Parquet split：

| Split | Parquet 路徑 | 用途 | 預估行數 |
|-------|-------------|------|---------|
| train | `qna/train.parquet` | CR train (`in_repo_split=train`) | ~44k |
| ir_val | `qna/ir_val.parquet` | IR validation | ~437k |
| ir_test | `qna/ir_test.parquet` | IR test | ~535k |
| cr_val | `qna/cr_val.parquet` | CR validation | ~642k |
| cr_test | `qna/cr_test.parquet` | CR test | ~476k |
| ood_test | `qna/ood_test.parquet` | OOD test | ~182k |

### 2.3 QnA 資料行

關鍵行名（從 HF viewer 確認）：

| 行名 | 型別 | 用途 |
|------|------|------|
| `repo_id` | string | `"owner/repo"` 格式 |
| `commit_index` | int32 | 對應 commits 表的索引 |
| `prefix` | string | assertion 前綴程式碼 |
| `target` | string | 斷言目標值 |
| `cross_repo_split` | string | `train`, `cr_val`, `cr_test`, `ood_test` |
| `in_repo_split` | string | `train`, `val`, `test` |
| `test_file` | string | 測試檔案路徑 |

---

## 3. 系統架構與資料流

### 3.1 一鍵準備腳本

```powershell
scripts/prepare_repopeftbench.ps1
```

```
Step 1: 下載所有 QnA Parquet (hf_hub 直接 HTTP 下載)
          │
          ▼
Step 2: Python + pyarrow 讀取 Parquet → 按 split 寫出 JSONL
          │  (不含 embedding, 保留 repo_id, prefix, target)
          │  + 同時提取唯一 repo_id 列表
          │
          ▼
Step 3: 對每個 repo_id: git clone --depth 1 → repos/{owner}__{repo}/
          │  (若已存在則跳過)
          │
          ▼
Step 4: code2lora-lite encode 每個 repo → cached_embeddings/{hash}.npz
          │  (若已快取則跳過)
          │
          ▼
Step 5: Python 腳本合併 JSONL + NPZ → 寫回完整 JSONL (含 embedding)
          │
          ▼
完成: data/repopeftbench/ 目錄結構就緒
```

### 3.2 目錄結構

```
code2lora-lite/
└── data/
    └── repopeftbench/               # 由 prepare 腳本建立
        ├── repos/                   # git clone 的原始碼
        │   ├── owner__repo1/
        │   └── owner__repo2/
        ├── cached_embeddings/       # NPZ 快取
        │   ├── abc123.npz
        │   └── def456.npz
        ├── train.jsonl              # 訓練集
        ├── ir_val.jsonl             # IR 驗證
        ├── ir_test.jsonl            # IR 測試
        ├── cr_val.jsonl             # CR 驗證
        ├── cr_test.jsonl            # CR 測試
        └── ood_test.jsonl           # OOD 測試
```

### 3.3 JSONL 格式

每行一個 JSON 物件，與現有 `dataset.rs` 的 `AssertionRecord` 結構相容：

```json
{
  "repo_id": "owner/repo",
  "cross_repo_split": "train",
  "in_repo_split": "train",
  "file_path": "tests/test_example.py",
  "input_prefix": "def test_answer():\n    assert answer() ==",
  "target_value": "42",
  "repo_embedding": [0.0123, -0.0456, ...]
}
```

---

## 4. 檔案變更清單

### 4.1 新增檔案

| 檔案 | 說明 |
|------|------|
| `spec-data.md` | 本設計文件 |
| `scripts/prepare_repopeftbench.ps1` | 一鍵準備腳本（取代舊 `download_code2lora_data.ps1`） |

### 4.2 修改檔案

| 檔案 | 變更 |
|------|------|
| `scripts/download_code2lora_data.ps1` | 刪除（功能由新腳本取代） |
| `src/dataset.rs` | 極小幅（或無變更）：現有 `AssertionRecord` + `load_jsonl` 已正確處理 |

> 現有 `AssertionRecord::into_example` 中，當 `repo_embedding.len() != 768` 時自動 fallback 為零向量（第 90-94 行）。因此 JSONL 格式可直接相容。
| `todos.md` | 更新進度 |

### 4.3 不變的檔案

| 檔案 | 理由 |
|------|------|
| `src/trainer.rs` | 已接收 `CodeDataset` 作為輸入，不需變更 |
| `src/hypernetwork.rs` | `repo_embed_dim=768` 不變 |
| `src/base_llm.rs` | 載入邏輯不變 |
| `src/repo_encoder.rs` | 已正確輸出 768-dim |
| `src/config.rs` | `TrainConfig` + `HypernetworkConfig` 參數不需變更 |
| `src/qwen2_lora.rs` | LoRA 層邏輯不變 |

---

## 5. 風險與緩解

| 風險 | 影響 | 緩解 |
|------|------|------|
| git clone 400 repos 耗時 | 首次準備 ~30 分鐘 | `--depth 1` 加速；嵌入結果快取後永久有效 |
| 部分 repo 已不存在或 private | 缺少部分 repo embedding | 跳過失敗 repo，用零向量代替 |
| Python + pyarrow 未安裝 | 腳本中斷 | 腳本開頭檢查依賴，提示 `pip install` |
| OOM 在 parquet 讀取時 | 記憶體不足 | 使用 `iter_batches` 而非一次載入全部 |
| repo 編碼耗時 | ~1-3 秒/repo | 批次編碼 + NPZ 快取；可中斷續傳 |
| JSONL 檔案過大 (2-3 GB) | 載入慢 | 現有 `BufReader` + 逐行讀取不受影響 |

---

## 6. 測試計畫

| 測試 | 類型 | 說明 |
|------|------|------|
| `test_repopeftbench_jsonl_load` | 單元 | 載入一小段真實 JSONL（從 OOD 子集） |
| `test_repopeftbench_tiny_train` | 整合 (ignore) | 從 OOD 抽 5 個 repo 做 2 epochs 訓練 |
| P6 (`test_p6_real_model_training`) | 整合 (ignore) | 既有合成資料測試不變 |

---

## 7. 成功標準

- [ ] `scripts/prepare_repopeftbench.ps1` 一鍵執行從無到有準備資料
- [ ] 產生的 JSONL 能被 `CodeDataset::load_from_dir` 正確載入
- [ ] `cargo run --release -- train -d data/repopeftbench/` 在真實資料上收斂
- [ ] `data/repopeftbench/` 總大小在 12 GB 以內
