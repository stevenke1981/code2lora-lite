# RepoPeftBench 資料整合 — Implementation Plan

> **For agentic workers:** Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate real HuggingFace RepoPeftBench assertion-completion data into the code2lora-lite training pipeline.

**Architecture:** A PowerShell + Python preprocessing pipeline downloads QnA parquet files from HF, git clones ~400 repos, encodes them via the existing MiniLM RepoEncoder, and merges everything into JSONL files (one per evaluation split) that the existing Rust `CodeDataset::load_jsonl` can consume directly.

**Tech Stack:** PowerShell 7+, Python 3 + pyarrow, Rust (code2lora-lite CLI for encoding), HuggingFace Hub (direct HTTP download)

---

### Task 1: Rewrite download script — parquet → JSONL per split

**Files:**
- Create: `scripts/prepare_repopeftbench.ps1`
- Delete: `scripts/download_code2lora_data.ps1`

The script is the preprocessing entry point. It downloads the 6 QnA parquet files from the `code2lora-data-snapshots` dataset, uses Python + pyarrow to read them, and writes one JSONL file per split WITHOUT embeddings (embedded later in Task 3). It also extracts the unique repo_id list.

- [ ] **Step 1: Write the script header + download logic**

```powershell
param(
    [string]$Dataset = "code2lora/code2lora-data-snapshots",
    [string]$OutputDir = "data/repopeftbench",
    [switch]$SkipClone,
    [switch]$SkipEncode,
    [switch]$ForceRedownload
)

$ErrorActionPreference = "Stop"
$baseUrl = "https://huggingface.co/datasets/$Dataset/resolve/main"

# Splits exactly as they appear in the dataset
$splits = @(
    @{Name="train"; Parquet="qna/train.parquet"}
    @{Name="ir_val"; Parquet="qna/ir_val.parquet"}
    @{Name="ir_test"; Parquet="qna/ir_test.parquet"}
    @{Name="cr_val"; Parquet="qna/cr_val.parquet"}
    @{Name="cr_test"; Parquet="qna/cr_test.parquet"}
    @{Name="ood_test"; Parquet="qna/ood_test.parquet"}
)

# Create output directories
New-Item -ItemType Directory -Force "$OutputDir/repos" | Out-Null
New-Item -ItemType Directory -Force "$OutputDir/cached_embeddings" | Out-Null
New-Item -ItemType Directory -Force "$OutputDir/raw_jsonl" | Out-Null

# Download each split's parquet (if missing)
foreach ($split in $splits) {
    $url = "$baseUrl/$($split.Parquet)"
    $localPath = "$OutputDir/raw_jsonl/$($split.Name).parquet"
    if (-not (Test-Path $localPath) -or $ForceRedownload) {
        Write-Host "Downloading $url → $localPath"
        Invoke-WebRequest -Uri $url -OutFile $localPath
    } else {
        Write-Host "exists: $localPath"
    }
}
```

- [ ] **Step 2: Add Python inline script for parquet → JSONL conversion**

Embed an inline Python script (like the existing `download_code2lora_data.ps1` does) that:
- Reads each parquet file
- Extracts columns: `repo_id`, `commit_index`, `cross_repo_split`, `in_repo_split`, `file_path`/`test_file`, `prefix`/`input_prefix`, `target`/`target_value`
- Writes one JSONL per split (matching the parquet split name)
- Collects unique repo_ids into a sidecar JSON file
- Uses `iter_batches` to avoid OOM on large splits

```powershell
$pythonConverter = @'
import argparse, json, os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--repos-out", required=True)
parser.add_argument("--splits", nargs="+", required=True)
args = parser.parse_args()

try:
    import pyarrow.parquet as pq
except ImportError:
    raise SystemExit("pip install pyarrow")

all_repo_ids = set()

for split_name in args.splits:
    parquet_path = Path(args.input_dir) / f"{split_name}.parquet"
    if not parquet_path.exists():
        print(f"SKIP {split_name}: {parquet_path} not found")
        continue

    pf = pq.ParquetFile(parquet_path)
    names = pf.schema_arrow.names
    out_path = Path(args.output_dir) / f"{split_name}.jsonl"

    def first(row, *keys):
        for k in keys:
            v = row.get(k)
            if v is not None and v != "":
                return v
        return ""

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for batch in pf.iter_batches(batch_size=4096):
            for row in batch.to_pylist():
                repo_id = row.get("repo_id", "")
                if repo_id:
                    all_repo_ids.add(repo_id)
                record = {
                    "repo_id": repo_id,
                    "commit_index": row.get("commit_index"),
                    "cross_repo_split": first(row, "cross_repo_split"),
                    "in_repo_split": first(row, "in_repo_split"),
                    "file_path": first(row, "file_path", "test_file", "path"),
                    "input_prefix": first(row, "prefix", "input_prefix", "prompt", "question"),
                    "target_value": first(row, "target", "target_value", "answer", "completion", "assertion"),
                    "repo_embedding": [],  # placeholder — filled in Task 3
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{split_name}: {pf.metadata.num_rows} rows → {out_path}")

with open(args.repos_out, "w") as f:
    json.dump(sorted(all_repo_ids), f)
print(f"Unique repos: {len(all_repo_ids)} → {args.repos_out}")
'@
```

- [ ] **Step 3: Wire up the Python invocation in the script**

```powershell
# Write Python script to temp file and execute
$pythonScript = Join-Path $OutputDir "convert_parquet.py"
Set-Content -Path $pythonScript -Value $pythonConverter -Encoding UTF8

$splitNames = $splits | ForEach-Object { $_.Name }
python $pythonScript `
    --input-dir "$OutputDir/raw_jsonl" `
    --output-dir "$OutputDir/raw_jsonl" `
    --repos-out "$OutputDir/repo_ids.json" `
    --splits $splitNames
```

- [ ] **Step 4: Verify by running on a small split first**

Run: `.\scripts\prepare_repopeftbench.ps1 -Dataset "code2lora/code2lora-data-ood" -OutputDir "data/repopeftbench-test"`

Expected output:
```
Downloading ... → data/repopeftbench-test/raw_jsonl/train.parquet
exists: data/repopeftbench-test/raw_jsonl/ir_val.parquet
...
ood_test: 182000 rows → data/repopeftbench-test/raw_jsonl/ood_test.jsonl
Unique repos: 92 → data/repopeftbench-test/repo_ids.json
```

---

### Task 2: Git clone repos + encode via RepoEncoder

- [ ] **Step 1: Add clone logic to the script**

After the parquet conversion, clone each unique repo (if not already cloned). Use `--depth 1 --single-branch` for minimal disk usage. Repo names use `__` instead of `/` for filesystem safety.

```powershell
if (-not $SkipClone) {
    $repoIds = Get-Content "$OutputDir/repo_ids.json" | ConvertFrom-Json
    # Build first to avoid compile overhead per encode
    Write-Host "Building release binary..."
    Push-Location (Split-Path $PSScriptRoot -Parent)  # project root
    cargo build --release
    Pop-Location

    $total = $repoIds.Count
    $i = 0
    foreach ($repoId in $repoIds) {
        $i++
        $safeName = $repoId -replace '/', '__'
        $repoDir = "$OutputDir/repos/$safeName"
        $embedFile = "$OutputDir/cached_embeddings/$safeName.embed"

        Write-Host "[$i/$total] Processing $repoId"

        # Clone if not exists (with retry)
        if (-not (Test-Path "$repoDir/.git")) {
            $cloneUrl = "https://github.com/$repoId.git"
            try {
                git clone --depth 1 --single-branch $cloneUrl $repoDir 2>&1 | Out-Null
            } catch {
                Write-Warning "Failed to clone $repoId: $_"
                continue
            }
        }

        # Encode if not cached
        if (-not (Test-Path $embedFile) -and (Test-Path $repoDir)) {
            & "./target/release/code2lora-lite" encode $repoDir -o $embedFile 2>&1 | Out-Null
        }
    }
}
```

- [ ] **Step 2: Verify encoding**

Manually inspect one `.embed` file:
```powershell
$bytes = [System.IO.File]::ReadAllBytes("data/repopeftbench-test/cached_embeddings/owner__repo.embed")
$header = [System.Text.Encoding]::UTF8.GetString($bytes[0..$bytes.IndexOf([byte]10)])
# Expected: "CODE2LORA_EMBED_V1:768"
```

---

### Task 3: Python merge — inject embeddings into JSONL

- [ ] **Step 1: Add merge Python inline script**

After encoding is done, this script reads the raw JSONL, loads the `.embed` binary file for each repo_id, and writes the final merged JSONL with `repo_embedding` filled in.

```powershell
$pythonMerger = @'
import json, struct, os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True)
parser.add_argument("--embed-dir", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--splits", nargs="+", required=True)
args = parser.parse_args()

def load_embed(path):
    """Read code2lora-lite's custom .embed binary format."""
    with open(path, "rb") as f:
        content = f.read()
    header_end = content.index(b'\n')
    dim_str = content[:header_end].decode().split(":")[1]
    dim = int(dim_str)
    payload = content[header_end+1:]
    values = struct.unpack(f"{dim}f", payload)
    return list(values)

embed_cache = {}

for split_name in args.splits:
    in_path = Path(args.input_dir) / f"{split_name}.jsonl"
    out_path = Path(args.output_dir) / f"{split_name}.jsonl"
    if not in_path.exists():
        continue

    written = 0
    missing_embeds = 0
    with in_path.open("r", encoding="utf-8") as fin, \
         out_path.open("w", encoding="utf-8", newline="\n") as fout:
        for line in fin:
            row = json.loads(line)
            repo_id = row.get("repo_id", "")

            if repo_id not in embed_cache:
                safe = repo_id.replace("/", "__")
                embed_path = Path(args.embed_dir) / f"{safe}.embed"
                if embed_path.exists():
                    embed_cache[repo_id] = load_embed(str(embed_path))
                else:
                    embed_cache[repo_id] = [0.0] * 768
                    missing_embeds += 1

            row["repo_embedding"] = embed_cache[repo_id]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"{split_name}: {written} rows, {missing_embeds} missing embeds (zero-filled)")
'@
```

- [ ] **Step 2: Wire up the merger in the script**

```powershell
if (-not $SkipEncode) {
    $mergeScript = Join-Path $OutputDir "merge_embeddings.py"
    Set-Content -Path $mergeScript -Value $pythonMerger -Encoding UTF8

    $splitNames = $splits | ForEach-Object { $_.Name }
    python $mergeScript `
        --input-dir "$OutputDir/raw_jsonl" `
        --embed-dir "$OutputDir/cached_embeddings" `
        --output-dir "$OutputDir" `
        --splits $splitNames

    Write-Host "Merged JSONL ready in $OutputDir"
    Write-Host "Train:     $OutputDir/train.jsonl"
    Write-Host "IR val:    $OutputDir/ir_val.jsonl"
    Write-Host "IR test:   $OutputDir/ir_test.jsonl"
    Write-Host "CR val:    $OutputDir/cr_val.jsonl"
    Write-Host "CR test:   $OutputDir/cr_test.jsonl"
    Write-Host "OOD test:  $OutputDir/ood_test.jsonl"

    # Cleanup temp files
    Remove-Item -Recurse -Force "$OutputDir/raw_jsonl" -ErrorAction SilentlyContinue
    Remove-Item -Force "$OutputDir/convert_parquet.py" -ErrorAction SilentlyContinue
    Remove-Item -Force "$OutputDir/merge_embeddings.py" -ErrorAction SilentlyContinue
}
```

---

### Task 4: Dataset smoke test with real JSONL

**Files:**
- Modify: `src/dataset.rs` (add test)

- [ ] **Step 1: Write a smoke test that loads a tiny slice of real JSONL**

Add a test that creates a miniature real-data JSONL, loads it, and verifies structure. This test uses the existing `load_jsonl` and `CodeDataset` — validating that the format we produce in Tasks 1-3 actually works.

```rust
#[test]
fn test_repopeftbench_jsonl_with_embeddings() -> Result<()> {
    // Build a 3-row snippet that mimics what the prepare script outputs
    let emb = vec![0.1f32; 768];
    let rows = vec![
        serde_json::json!({
            "repo_id": "owner/repo1",
            "cross_repo_split": "train",
            "in_repo_split": "train",
            "file_path": "tests/test_a.py",
            "input_prefix": "def test_a():\n    assert answer() ==",
            "target_value": "42",
            "repo_embedding": emb,
        }),
        serde_json::json!({
            "repo_id": "owner/repo1",
            "cross_repo_split": "train",
            "in_repo_split": "val",
            "file_path": "tests/test_b.py",
            "input_prefix": "def test_b():\n    assert result ==",
            "target_value": "True",
            "repo_embedding": emb,
        }),
        serde_json::json!({
            "repo_id": "owner/repo2",
            "cross_repo_split": "cr_val",
            "in_repo_split": "train",
            "file_path": "tests/test_c.py",
            "input_prefix": "def test_c():\n    assert foo() ==",
            "target_value": "\"hello\"",
            "repo_embedding": vec![0.2f32; 768],
        }),
    ];

    let tmp = std::env::temp_dir().join(format!("repopeftbench-test-{}.jsonl", std::process::id()));
    let jsonl: String = rows.iter().map(|r| r.to_string() + "\n").collect();
    std::fs::write(&tmp, &jsonl)?;

    let examples = CodeDataset::load_jsonl(&tmp)?;
    std::fs::remove_file(&tmp).ok();

    assert_eq!(examples.len(), 3);
    assert_eq!(examples[0].repo_id, "owner/repo1");
    assert_eq!(examples[0].repo_embedding.len(), 768);
    assert!(examples[0].code_content.contains("def test_a()"));
    assert!(examples[0].code_content.contains("42"));
    assert_eq!(examples[0].split, "train");
    assert_eq!(examples[1].split, "val");

    // Build dataset and verify split logic
    let dataset = CodeDataset { examples };
    let (cr, ir) = dataset.split(0.2);
    // repo2 is cr_val → goes to CR, repo1 train/val → goes to IR
    assert_eq!(cr.len(), 1, "repo2 is CR val");
    assert_eq!(cr[0].repo_id, "owner/repo2");
    assert_eq!(ir.len(), 2, "repo1 rows are training");

    Ok(())
}
```

- [ ] **Step 2: Run the test**

Run: `cargo test test_repopeftbench_jsonl_with_embeddings -- --nocapture`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add scripts/prepare_repopeftbench.ps1 src/dataset.rs
git rm scripts/download_code2lora_data.ps1
git commit -m "feat(data): RepoPeftBench real-data preprocessing pipeline"
```

---

### Task 5: Integration test (ignored, requires HF + GPU)

**Files:**
- Modify: `src/base_llm.rs` (add integration test alongside P6)

- [ ] **Step 1: Add test that trains on a tiny real-data slice**

```rust
/// P7: Train on a tiny subset of real RepoPeftBench data.
///
/// Requires:
///   - CODE2LORA_DATA_DIR env var pointing to the prepared JSONL directory
///   - GPU + HF Hub access for model download
#[test]
#[ignore = "Requires prepared RepoPeftBench JSONL + HF model download + GPU"]
fn test_p7_repopeftbench_tiny_train() -> Result<()> {
    let data_dir = std::env::var("CODE2LORA_DATA_DIR")
        .unwrap_or_else(|_| "data/repopeftbench".to_string());
    let device = Device::cuda_if_available(0)?;
    info!("P7: device={device:?}, data_dir={data_dir}");

    // Load a tiny subset: take first N rows from train.jsonl
    let dataset_path = std::path::PathBuf::from(&data_dir).join("train.jsonl");
    anyhow::ensure!(dataset_path.exists(), "train.jsonl not found at {dataset_path:?}");

    let mut all_examples = CodeDataset::load_jsonl(&dataset_path)?;
    anyhow::ensure!(!all_examples.is_empty(), "train.jsonl is empty");

    // Take first 16 examples for the tiny test
    all_examples.truncate(16);
    let dataset = CodeDataset { examples: all_examples };
    let summary = dataset.summary();
    info!("P7: dataset loaded: repos={}, examples={}", summary.repo_count, dataset.len());

    // Load real model + hypernetwork (same as P6)
    let hn_cfg = HypernetworkConfig {
        hidden_dim: 384,
        rank: 8,
        ..Default::default()
    };
    let qwen = Code2LoRAModel::new(&device, DType::F32, &hn_cfg)?;

    let hn_varmap = VarMap::new();
    let hn_vb = VarBuilder::from_varmap(&hn_varmap, DType::F32, &device);
    let hn = Code2LoRAHead::new(hn_vb, &hn_cfg, &hn_varmap)?;

    let train_cfg = TrainConfig {
        data_dir: data_dir.clone(),
        base_model: "Qwen/Qwen2.5-Coder-0.5B".into(),
        output: "p7_checkpoints".into(),
        rank: hn_cfg.rank,
        epochs: 3,
        lr: 1e-4,
        batch_size: 2,
        seq_len: 2048,
        cache_dir: "cache".into(),
        cr_holdout: 0.2,
    };
    std::fs::create_dir_all("p7_checkpoints")?;
    let mut trainer = Trainer::new(hn, qwen, hn_varmap, train_cfg, device);
    trainer.train(&dataset)?;
    std::fs::remove_dir_all("p7_checkpoints").ok();

    info!("P7: tiny real-data training completed successfully");
    Ok(())
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check`

Expected: clean compile

- [ ] **Step 3: Commit**

```bash
git add src/base_llm.rs todos.md
git commit -m "feat(test): add integration test for real RepoPeftBench training"
```

---

### Self-Review Checklist

- [ ] **Spec coverage:** Every spec requirement maps to a task:
  - ✓ Parquet download → Task 1
  - ✓ Unique repo extraction → Task 1, Step 2 (Python inline)
  - ✓ Git clone → Task 2
  - ✓ MiniLM encoding → Task 2 (via `code2lora-lite encode`)
  - ✓ JSONL merge → Task 3
  - ✓ CR/IR/OOD split preservation → Task 1 (per-split output)
  - ✓ Fallback for missing repos → Task 2 (try/catch + skip) + Task 3 (zero-fill)
- [ ] **Placeholder scan:** No TBD/TODO/filler code — each step has complete code content
- [ ] **Type consistency:** `repo_embedding: Vec<f32>` (768-dim) used consistently throughout — matches existing `AssertionRecord` / `RepoEmbedding` types
