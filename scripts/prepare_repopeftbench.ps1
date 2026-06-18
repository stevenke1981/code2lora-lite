param(
    [string]$Dataset = "code2lora/code2lora-data-snapshots",
    [string]$OutputDir = "data/repopeftbench",
    [switch]$SkipClone,
    [switch]$SkipEncode,
    [switch]$ForceRedownload
)

$ErrorActionPreference = "Stop"

# ── Config ───────────────────────────────────────────────────────────
$baseUrl = "https://huggingface.co/datasets/$Dataset/resolve/main"
$splits = @(
    @{Name="train";   Parquet="qna/train.parquet"}
    @{Name="ir_val";  Parquet="qna/ir_val.parquet"}
    @{Name="ir_test"; Parquet="qna/ir_test.parquet"}
    @{Name="cr_val";  Parquet="qna/cr_val.parquet"}
    @{Name="cr_test"; Parquet="qna/cr_test.parquet"}
    @{Name="ood_test";Parquet="qna/ood_test.parquet"}
)

# ── Create output directories ───────────────────────────────────────
foreach ($d in @("repos", "cached_embeddings", "raw_jsonl")) {
    New-Item -ItemType Directory -Force "$OutputDir/$d" | Out-Null
}

$scriptDir = Split-Path -Parent $PSScriptRoot

# ══════════════════════════════════════════════════════════════════════
# PART 1:  Download QnA Parquet files from HuggingFace
# ══════════════════════════════════════════════════════════════════════
Write-Host "=== Part 1: Downloading QnA Parquet files ==="
foreach ($split in $splits) {
    $url = "$baseUrl/$($split.Parquet)"
    $localPath = "$OutputDir/raw_jsonl/$($split.Name).parquet"
    if (-not (Test-Path $localPath) -or $ForceRedownload) {
        Write-Host "  Downloading $url → $localPath"
        Invoke-WebRequest -Uri $url -OutFile $localPath
    } else {
        Write-Host "  exists: $localPath"
    }
}

# ══════════════════════════════════════════════════════════════════════
# PART 2:  Convert Parquet → JSONL (via Python + pyarrow)
# ══════════════════════════════════════════════════════════════════════
Write-Host "=== Part 2: Converting Parquet → JSONL ==="

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
    raise SystemExit(
        "pyarrow is required for Parquet conversion. Install with: python -m pip install pyarrow"
    )

all_repo_ids = set()

def first(row, *keys):
    for k in keys:
        v = row.get(k)
        if v is not None and v != "":
            return v
    return ""

for split_name in args.splits:
    parquet_path = Path(args.input_dir) / f"{split_name}.parquet"
    if not parquet_path.exists():
        print(f"SKIP {split_name}: {parquet_path} not found")
        continue

    pf = pq.ParquetFile(parquet_path)
    out_path = Path(args.output_dir) / f"{split_name}.jsonl"

    written = 0
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
                    "repo_embedding": [],  # placeholder — filled later
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
        print(f"  {split_name}: {written} rows -> {out_path}")

with open(args.repos_out, "w") as f:
    json.dump(sorted(all_repo_ids), f)
print(f"  Unique repo_ids: {len(all_repo_ids)} -> {args.repos_out}")
'@

$pythonScript = Join-Path $OutputDir "convert_parquet.py"
Set-Content -Path $pythonScript -Value $pythonConverter -Encoding UTF8

$splitNames = $splits | ForEach-Object { $_.Name }
python $pythonScript `
    --input-dir "$OutputDir/raw_jsonl" `
    --output-dir "$OutputDir/raw_jsonl" `
    --repos-out "$OutputDir/repo_ids.json" `
    --splits $splitNames

if ($LASTEXITCODE -ne 0) {
    throw "Python conversion failed (exit code $LASTEXITCODE)"
}

# ══════════════════════════════════════════════════════════════════════
# PART 3:  Git clone each unique repo + encode with RepoEncoder
# ══════════════════════════════════════════════════════════════════════
if (-not $SkipClone) {
    Write-Host "=== Part 3: Cloning repos + encoding ==="
    if (-not (Test-Path "$OutputDir/repo_ids.json")) {
        throw "repo_ids.json not found — did Part 2 complete?"
    }

    # Build release binary once to avoid repeated compile overhead
    Write-Host "  Building release binary..."
    Push-Location $scriptDir
    cargo build --release 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        throw "cargo build --release failed"
    }
    Pop-Location

    $repoIds = Get-Content "$OutputDir/repo_ids.json" | ConvertFrom-Json
    $total = $repoIds.Count
    $i = 0
    $cloned = 0
    $encoded = 0
    $failed = 0

    foreach ($repoId in $repoIds) {
        $i++
        $safeName = $repoId -replace '/', '__'
        $repoDir = "$OutputDir/repos/$safeName"
        $embedFile = "$OutputDir/cached_embeddings/$safeName.embed"

        Write-Host "  [$i/$total] $repoId" -NoNewline

        # ── Clone (if not already cloned) ──
        if (-not (Test-Path "$repoDir/.git")) {
            $cloneUrl = "https://github.com/$repoId.git"
            try {
                git clone --depth 1 --single-branch $cloneUrl $repoDir 2>&1 | Out-Null
                $cloned++
                Write-Host " cloned" -NoNewline
            } catch {
                Write-Host " CLONE FAILED: $_"
                $failed++
                continue
            }
        } else {
            Write-Host " cached" -NoNewline
        }

        # ── Encode (if not already cached) ──
        if (-not (Test-Path $embedFile)) {
            try {
                & "$scriptDir/target/release/code2lora-lite" encode $repoDir -o $embedFile 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    $encoded++
                    Write-Host " encoded"
                } else {
                    Write-Host " ENCODE FAILED (exit $LASTEXITCODE)"
                    $failed++
                }
            } catch {
                Write-Host " ENCODE FAILED: $_"
                $failed++
            }
        } else {
            Write-Host " embedded"
        }
    }

    Write-Host "  Done: $cloned cloned, $encoded encoded, $failed failed (of $total repos)"
}

# ══════════════════════════════════════════════════════════════════════
# PART 4:  Merge embeddings into JSONL (via Python)
# ══════════════════════════════════════════════════════════════════════
if (-not $SkipEncode) {
    Write-Host "=== Part 4: Merging embeddings into JSONL ==="

    $pythonMerger = @'
import argparse, json, struct
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", required=True)
parser.add_argument("--embed-dir", required=True)
parser.add_argument("--output-dir", required=True)
parser.add_argument("--splits", nargs="+", required=True)
args = parser.parse_args()


def load_embed(path):
    """Read code2lora-lite's custom .embed binary format:
    Header: CODE2LORA_EMBED_V1:<dim>\n
    Body: f32 little-endian bytes
    """
    with open(path, "rb") as f:
        content = f.read()
    header_end = content.index(b"\n")
    dim_str = content[:header_end].decode().split(":")[1]
    dim = int(dim_str)
    payload = content[header_end + 1:]
    values = struct.unpack(f"{dim}f", payload)
    return list(values)


embed_cache = {}
missing_count = 0

for split_name in args.splits:
    in_path = Path(args.input_dir) / f"{split_name}.jsonl"
    out_path = Path(args.output_dir) / f"{split_name}.jsonl"
    if not in_path.exists():
        print(f"  SKIP {split_name}: input not found")
        continue

    written = 0
    split_missing = 0
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
                    missing_count += 1
                    split_missing += 1

            row["repo_embedding"] = embed_cache[repo_id]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"  {split_name}: {written} rows, {split_missing} missing embeds (zero-filled)")

print(f"  Total missing embeddings (zero-filled): {missing_count}")
'@

    $mergeScript = Join-Path $OutputDir "merge_embeddings.py"
    Set-Content -Path $mergeScript -Value $pythonMerger -Encoding UTF8

    $splitNames = $splits | ForEach-Object { $_.Name }
    python $mergeScript `
        --input-dir "$OutputDir/raw_jsonl" `
        --embed-dir "$OutputDir/cached_embeddings" `
        --output-dir "$OutputDir" `
        --splits $splitNames

    if ($LASTEXITCODE -ne 0) {
        throw "Python merge failed (exit code $LASTEXITCODE)"
    }
}

# ══════════════════════════════════════════════════════════════════════
# Cleanup temp files
# ══════════════════════════════════════════════════════════════════════
Remove-Item -Recurse -Force "$OutputDir/raw_jsonl" -ErrorAction SilentlyContinue
Remove-Item -Force "$OutputDir/convert_parquet.py" -ErrorAction SilentlyContinue
Remove-Item -Force "$OutputDir/merge_embeddings.py" -ErrorAction SilentlyContinue

# ══════════════════════════════════════════════════════════════════════
Write-Host "=== Done ==="
Write-Host "Output directory: $OutputDir"
Write-Host ""
if (Test-Path "$OutputDir/train.jsonl") {
    Write-Host "Train:    $OutputDir/train.jsonl"
}
if (Test-Path "$OutputDir/ir_val.jsonl") {
    Write-Host "IR val:   $OutputDir/ir_val.jsonl"
}
if (Test-Path "$OutputDir/ir_test.jsonl") {
    Write-Host "IR test:  $OutputDir/ir_test.jsonl"
}
if (Test-Path "$OutputDir/cr_val.jsonl") {
    Write-Host "CR val:   $OutputDir/cr_val.jsonl"
}
if (Test-Path "$OutputDir/cr_test.jsonl") {
    Write-Host "CR test:  $OutputDir/cr_test.jsonl"
}
if (Test-Path "$OutputDir/ood_test.jsonl") {
    Write-Host "OOD test: $OutputDir/ood_test.jsonl"
}
if (Test-Path "$OutputDir/repo_ids.json") {
    $count = (Get-Content "$OutputDir/repo_ids.json" | ConvertFrom-Json).Count
    Write-Host "Repos: $count unique (in repo_ids.json)"
}
