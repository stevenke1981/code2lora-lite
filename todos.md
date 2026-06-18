# Todos — code2lora-lite

追蹤實作進度。使用格式：`[ ]` 未開始、`[/]` 進行中、`[x]` 完成。

---

## P0: Scaffold

- [ ] Init Cargo 專案，設定依賴
- [ ] CLI 入口（clap 4 子命令）
- [ ] Config 載入（TOML/YAML）
- [ ] CUDA device detection

## P1: RepoEncoder

- [ ] all-MiniLM-L6-v2 模型載入
- [ ] 檔案 chunking（4096 window, 512 overlap）
- [ ] File-level mean pooling
- [ ] Repository-level weighted avg + max pool
- [ ] NPZ 快取系統
- [ ] 單元測試

## P2: BaseLLM + LoRA Injection

- [ ] GGUF Qwen2.5-Coder-0.5B 載入
- [ ] Attention forward 加 LoRA 計算路徑
- [ ] Layer-shared LoRA 儲存
- [ ] LoRA injection/removal API
- [ ] Tokenizer 整合
- [ ] Generate function（greedy + top-k）
- [ ] 單元測試

## P3: Hypernetwork

- [ ] Code2LoRAHead struct + shared MLP
- [ ] 7 對輸出頭（q/k/v/o/gate/up/down）
- [ ] Learnable log-scales s_m
- [ ] L2Norm + scaling
- [ ] Checkpoint save/load（safetensors）
- [ ] 整合測試

## P4: Dataset

- [ ] 下載 HF snapshot dataset
- [ ] Parquet reader
- [ ] CR split
- [ ] IR split
- [ ] Batch iterator
- [ ] Pre-compute repo embeddings

## P5: Trainer

- [ ] 訓練循環 skeleton
- [ ] Cross-entropy loss
- [ ] AdamW + cosine schedule
- [ ] Gradient accumulation（accum=4）
- [ ] Validation loop + metrics（EM, EditSim）
- [ ] Checkpoint saving
- [ ] Overfit test

## P6: Inference Pipeline

- [ ] `adapt` 命令
- [ ] `complete` 命令
- [ ] `encode` 命令
- [ ] End-to-end test

## P7: Polish

- [ ] Error handling（thiserror）
- [ ] Logging（env_logger）
- [ ] Performance pass
- [ ] README quick-start
