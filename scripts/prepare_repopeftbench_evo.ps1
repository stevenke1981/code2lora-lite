param(
    [string]$Dataset = "code2lora/code2lora-data-ood",
    [string]$OutputDir = "data/repopeftbench-evo",
    [int]$MaxRows = 0,
    [int]$MaxFiles = 0,
    [switch]$ForceRedownload
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force "$OutputDir/raw_parquet" | Out-Null

Write-Host "=== Discovering Evo Parquet files ==="
$endpoint = "https://datasets-server.huggingface.co/parquet?dataset=$Dataset"
$listing = Invoke-RestMethod -Uri $endpoint
$files = @($listing.parquet_files)
if ($files.Count -eq 0) {
    throw "No parquet files found for $Dataset"
}
if ($MaxFiles -gt 0) {
    $files = @($files | Select-Object -First $MaxFiles)
}

Write-Host "  dataset: $Dataset"
Write-Host "  parquet files: $($files.Count)"

$localFiles = @()
foreach ($file in $files) {
    $name = "$($file.split)-$($file.filename)"
    $localPath = Join-Path "$OutputDir/raw_parquet" $name
    if (-not (Test-Path $localPath) -or $ForceRedownload) {
        Write-Host "  downloading $($file.url)"
        Invoke-WebRequest -Uri $file.url -OutFile $localPath
    } else {
        Write-Host "  exists: $localPath"
    }
    $localFiles += $localPath
}

Write-Host "=== Converting joined commit/QnA rows to JSONL ==="

$pythonConverter = @'
import argparse, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
parser.add_argument("--max-rows", type=int, default=0)
parser.add_argument("parquet", nargs="+")
args = parser.parse_args()

try:
    import pyarrow.parquet as pq
except ImportError:
    raise SystemExit(
        "pyarrow is required for Evo Parquet conversion. Install with: python -m pip install pyarrow"
    )

def first(row, *keys):
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return ""

def list_first(row, *keys):
    value = first(row, *keys)
    if value is None or value == "":
        return []
    if hasattr(value, "as_py"):
        value = value.as_py()
    return list(value)

out_dir = Path(args.output_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "train.jsonl"
written = 0

with out_path.open("w", encoding="utf-8", newline="\n") as out:
    for parquet_path in args.parquet:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                record = {
                    "repo_id": first(row, "repo_id"),
                    "commit_index": row.get("commit_index"),
                    "commit_sha": first(row, "commit_sha"),
                    "commit_timestamp": first(row, "commit_timestamp"),
                    "cross_repo_split": first(row, "cross_repo_split"),
                    "in_repo_split": first(row, "in_repo_split"),
                    "file_path": first(row, "file_path", "test_file", "path"),
                    "input_prefix": first(row, "prefix", "input_prefix", "prompt", "question"),
                    "target_value": first(row, "target", "target_value", "answer", "completion", "assertion"),
                    "production_code_diff": first(row, "production_code_diff", "diff", "patch"),
                    "repo_embedding": list_first(row, "repo_embedding", "repo_state_embedding", "state_embedding", "embedding"),
                    "diff_embedding": list_first(row, "diff_embedding", "delta_embedding", "commit_diff_embedding", "production_code_diff_embedding"),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if args.max_rows > 0 and written >= args.max_rows:
                    print(f"  wrote {written} rows -> {out_path}")
                    raise SystemExit(0)

print(f"  wrote {written} rows -> {out_path}")
'@

$converterPath = Join-Path $OutputDir "convert_evo_parquet.py"
Set-Content -Path $converterPath -Value $pythonConverter -Encoding UTF8

python $converterPath --output-dir "$OutputDir" --max-rows $MaxRows $localFiles
if ($LASTEXITCODE -ne 0) {
    throw "Evo conversion failed (exit code $LASTEXITCODE)"
}

Remove-Item -Force $converterPath -ErrorAction SilentlyContinue

Write-Host "=== Done ==="
Write-Host "Evo JSONL: $OutputDir/train.jsonl"
Write-Host ""
Write-Host "Train with:"
Write-Host "  cargo run --release -- evo-train -d $OutputDir -o checkpoints-evo -e 1 --truncation-steps 8"
