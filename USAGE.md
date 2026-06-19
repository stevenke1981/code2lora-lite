# code2lora-lite 使用說明：Human + Agents

這份文件是 `code2lora-lite` 的日常操作入口。目標是讓人類使用者可以照著跑，也讓
Codex / OpenCode agents 可以用 MCP 或 wrapper 先讀 compact context，真實減少任務
中的原始碼 token 使用量。

## 先看這裡

如果你是 human，只想確認專案能跑：

```powershell
cargo fmt --check
cargo check --no-default-features
cargo test --no-default-features
```

如果你是 agent，先不要直接打開整個 repo。照這個順序：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-context.ps1 -RepoPath .
Get-Content .code2lora\agent-context\context.md
```

如果 Codex / OpenCode 已掛上 MCP server，優先使用 MCP tools：

1. `code2lora_agent_context`
2. `code2lora_read_context`
3. `code2lora_agent_open`
4. `code2lora_session_audit`

## 專案能做什麼

`code2lora-lite` 有三條主要路線：

1. Code2LoRA runtime：把 repo 編碼成 embedding，透過 hypernetwork 產生 LoRA
   adapter，再用 adapter 做 assertion/code completion。
2. Code2LoRA-Evo runtime：從初始 repo embedding 建立 hidden state，之後每個
   commit diff 只跑一個 GRU step 產生新版 adapter。
3. Code2LoRA-Evo training：把 commit-joined RepoPeftBench JSONL 依 repo +
   commit_index 排序，使用 truncated BPTT 訓練 GRU Evo hypernetwork。
4. Agent token-saving workflow：為 Codex/OpenCode 產生 compact context pack，讓
   agent 先讀小摘要和 Symbol Map，只在必要時打開原始檔，最後用 session audit
   量測是否真的省 token。

## Human Workflow

### 1. 建置與基本驗證

```powershell
cargo build --release
cargo test --no-default-features
```

常規測試不會下載大型 HuggingFace models。被標記為 `ignored` 的測試才會跑真實
model / real dataset / GPU-heavy path。

### 2. 準備 RepoPeftBench 真實資料

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_repopeftbench.ps1 `
  -OutputDir data/repopeftbench `
  -SkipCloneRepos
```

驗證轉換後資料：

```powershell
$env:CODE2LORA_REAL_DATA_DIR="data/repopeftbench"
cargo test test_real_repopeftbench_jsonl_smoke -- --ignored --nocapture
```

### 3. 訓練 hypernetwork

```powershell
cargo run --release -- train -d data/repopeftbench -o checkpoints -e 1
```

輸出重點：

- `checkpoints/final.safetensors`：後續 `adapt` 需要的 hypernetwork checkpoint。

### 4. 為目標 repo 產生 adapter

```powershell
cargo run --release -- adapt .\my-python-project `
  -m checkpoints\final.safetensors `
  -o adapter.safetensors
```

### 5. 執行 completion

```powershell
cargo run --release -- complete .\my-python-project adapter.safetensors `
  --prefix "def test_answer():`n    assert answer() ==" `
  --max-tokens 64 `
  -o assertion.txt
```

### 6. 單純產生 repo embedding

```powershell
cargo run --release -- encode .\my-python-project -o repo_embedding.embed
```

### 7. 用 Code2LoRA-Evo 隨 commit 更新 adapter

先準備 commit-joined Evo JSONL，再訓練 Evo checkpoint：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/prepare_repopeftbench_evo.ps1 `
  -OutputDir data/repopeftbench-evo `
  -MaxRows 2000

cargo run --release -- evo-train -d data/repopeftbench-evo `
  -o checkpoints-evo `
  -e 1 `
  --truncation-steps 8 `
  --max-sequences 4
```

第一次 commit update 可用 repo path 初始化 state：

```powershell
cargo run --release -- evo-adapt -m checkpoints-evo\evo_final.safetensors `
  --repo-path .\my-python-project `
  --diff-file .\commit-001.patch `
  --state-out evo_state.safetensors `
  -o adapter.safetensors
```

下一個 commit 用上一輪 state：

```powershell
cargo run --release -- evo-adapt -m evo.safetensors `
  --state-in evo_state.safetensors `
  --diff-file .\commit-002.patch `
  --state-out evo_state.safetensors `
  -o adapter.safetensors
```

如果你已經有預先算好的 diff embedding，也可以用 `--diff-embedding` 取代
`--diff-file`。`evo-train` 會輸出：

- `checkpoints-evo/evo_final.safetensors`
- `checkpoints-evo/evo_metrics.json`

如果 JSONL rows 已含 `diff_embedding`，`evo-train` 會直接使用；若只有
`production_code_diff`，它會載入 MiniLM 並即時計算 diff embedding，因此首次訓練會
多下載/載入 encoder。

`evo-init` 仍可產生未訓練 checkpoint，適合 smoke/dev；要得到有意義的 adapter
品質，請使用 `evo-train` 產生的 checkpoint。

## Agent Token-Saving Workflow

這條路線不需要 GPU，目標是讓 Codex/OpenCode 任務先讀 compact context，少打開
原始檔。

### 1. 產生 compact context pack

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-context.ps1 -RepoPath .
```

會產生：

- `.code2lora/agent-context/context.md`
- `.code2lora/agent-context/metrics.json`
- `.code2lora/agent-context/audit.json`
- `.code2lora/agent-context/codex-prompt.md`
- `.code2lora/agent-context/opencode-prompt.md`

`audit.json` 是 token reduction gate。預設 `-MinReduction 0.80`，低於門檻會
non-zero fail。

### 2. Agent 讀 context，不先讀整個 repo

```powershell
Get-Content .code2lora\agent-context\context.md
```

Agent 應該先用 `context.md` 內的 Symbol Map 找入口，再決定要打開哪些 raw files。

### 3. 打開 raw files 時要記錄

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-open.ps1 `
  -RepoPath . `
  -Files AGENTS.md,src\agent_context.rs
```

這會更新：

- `.code2lora/agent-context/opened-files.txt`

如果只想記錄、不輸出檔案內容：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-open.ps1 `
  -RepoPath . `
  -Files AGENTS.md,src\agent_context.rs `
  -NoContent
```

### 4. 任務結束前跑 session audit

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-session-audit.ps1 `
  -RepoPath . `
  -OpenedFilesPath .code2lora\agent-context\opened-files.txt
```

輸出：

- `.code2lora/agent-context/session-audit.json`

看這些欄位：

- `passed`
- `raw_token_estimate`
- `context_token_estimate`
- `opened_file_token_estimate`
- `session_token_estimate`
- `saved_token_estimate`
- `reduction_ratio`

目前本 repo 的 smoke evidence 約為：session `12.4k` tokens、saved `51k`
tokens、reduction 約 `80%`。實際數字以
`.code2lora/agent-context/session-audit.json` 為準。

## OpenCode Autoload Hook

本 repo 也提供 project-local OpenCode hook，讓支援 project config 的 OpenCode
client 在 chat 開始時自動載入 Code2LoRA compact context。

已啟用的 repo-local config：

```text
opencode.jsonc
hooks/code2lora-autoload.mjs
```

預設流程：

1. OpenCode 載入 `opencode.jsonc` 的 `plugin` 設定。
2. hook 檢查 `.code2lora/agent-context/context.md`。
3. 如果 context 不存在，hook 會呼叫 `scripts/agent-context.ps1` 自動產生。
4. hook 把 compact context 注入 `experimental.chat.system.transform` 的
   `output.system`。
5. Agent 仍然用 `scripts/agent-open.ps1` 或 MCP `code2lora_agent_open`
   記錄後續打開的 raw files，任務結束前跑 session audit。

如果要把同樣設定複製到其他 OpenCode config，使用：

```text
mcp/opencode.autoload.example.jsonc
```

常用調整：

```jsonc
{
  "plugin": [
    [
      "./hooks/code2lora-autoload.mjs",
      {
        "refresh": "missing",
        "contextDir": ".code2lora/agent-context",
        "maxChars": 24000,
        "maxFiles": 24,
        "minReduction": 0.8
      }
    ]
  ]
}
```

`refresh` 可設為 `always`，讓每次 chat system context transform 都重建 context；
`strict: true` 可讓 hook 在 context 產生或讀取失敗時直接報錯，適合 CI 或嚴格
驗收環境。

## MCP Workflow for Codex / OpenCode

### 1. 本機安裝 MCP config

Dry run：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install-mcp-config.ps1 `
  -RepoPath . `
  -Target All
```

Linux/macOS + PowerShell 7 dry run：

```bash
bash scripts/install-mcp-config.sh --repo-path . --target all
```

實際寫入 Codex / OpenCode config：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install-mcp-config.ps1 `
  -RepoPath . `
  -Target All `
  -Apply
```

Linux/macOS + PowerShell 7 實際寫入：

```bash
bash scripts/install-mcp-config.sh --repo-path . --target all --apply
```

installer 會先備份：

- `C:\Users\<you>\.codex\config.toml`
- `C:\Users\<you>\.config\opencode\opencode.jsonc`
- Linux/macOS: `~/.codex/config.toml`
- Linux/macOS: `${XDG_CONFIG_HOME:-~/.config}/opencode/opencode.jsonc`

並先跑 MCP smoke test。OpenCode config 以 UTF-8 讀寫，避免中文描述被 Windows
PowerShell 預設編碼破壞。Linux installer 需要 `python3` 與 `pwsh`，且會把 MCP
server command 寫成 `pwsh -NoProfile -File scripts/code2lora-mcp.ps1 ...`。

### 2. 確認 Codex 看得到 MCP server

```powershell
codex mcp list
```

應看到：

```text
code2lora-lite ... enabled
```

### 3. 確認 OpenCode 連得上 MCP server

```powershell
opencode mcp list
```

應看到：

```text
code2lora-lite connected
```

### 4. 手動 smoke MCP server

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/mcp-smoke.ps1 -RepoPath .
```

這會透過 stdio JSON-RPC 呼叫：

- `initialize`
- `tools/list`
- `code2lora_agent_context`
- `code2lora_read_context`
- `code2lora_agent_open`
- `code2lora_session_audit`

輸出：

- `.code2lora/mcp-smoke-context/mcp-smoke.json`

## Agent Operating Rules

Agents 在這個 repo 內工作時，遵守以下規則：

1. 先跑 `scripts/agent-context.ps1`、MCP `code2lora_agent_context`，或確認
   OpenCode autoload hook 已注入 compact context。
2. 先讀 `.code2lora/agent-context/context.md` 或 MCP `code2lora_read_context`。
3. 優先使用 Symbol Map 找入口，不要掃整個 repo。
4. 需要 raw file 時，使用 `scripts/agent-open.ps1` 或 MCP `code2lora_agent_open`。
5. 最後跑 `scripts/agent-session-audit.ps1` 或 MCP `code2lora_session_audit`。
6. final answer 要回報 `session-audit.json` 的 token reduction evidence。

## 驗收清單

一般程式變更：

```powershell
cargo fmt --check
cargo check --no-default-features
cargo test --no-default-features
```

Agent/MCP 變更：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/mcp-smoke.ps1 -RepoPath .
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install-mcp-config.ps1 -RepoPath . -Target All
bash scripts/install-mcp-config.sh --repo-path . --target all --skip-smoke
codex mcp list
opencode mcp list
```

Real dataset / GPU-heavy 變更：

```powershell
$env:CODE2LORA_REAL_DATA_DIR="data/repopeftbench"
cargo test test_real_repopeftbench_jsonl_smoke -- --ignored --nocapture
cargo test test_p7_repopeftbench_tiny_train -- --ignored --nocapture
cargo test test_p7_full_end_to_end_real_inference -- --ignored --nocapture
```

## 常見問題

### `agent-context.ps1` 失敗

先看 `.code2lora/agent-context/audit.json` 是否低於 `min_reduction`。若只是測試
失敗路徑，可以調高 `-MaxFiles` 或降低 `-MinReduction`，但正式驗收不應關掉 gate。

### `opencode mcp list` 很慢

OpenCode 會嘗試連線所有已啟用 MCP servers，不只 `code2lora-lite`。如果其他 server
卡住，先看輸出中 `code2lora-lite` 是否為 `connected`。

### `opencode.jsonc` 中文變亂碼

不要用 Windows PowerShell 預設 `Set-Content` 重寫整份 OpenCode config。
`scripts/install-mcp-config.ps1` 已使用 UTF-8 safe read/write；若 config 曾被破壞，
用 installer 產生的 `.code2lora-*.bak` 備份還原。

### 第一次跑 model 測試很慢

第一次執行會下載 Qwen2.5-Coder-0.5B 和 all-MiniLM-L6-v2。這是正常行為。
