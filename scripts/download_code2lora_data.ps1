param(
    [string]$Dataset = "code2lora/code2lora-data-ood",
    [string]$OutputDir = "data/code2lora-ood",
    [int]$MaxRows = 0
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force $OutputDir | Out-Null

$baseUrl = "https://huggingface.co/datasets/$Dataset/resolve/main"
$commitsPath = Join-Path $OutputDir "commits.parquet"
$qnaPath = Join-Path $OutputDir "qna.parquet"
$jsonlPath = Join-Path $OutputDir "train.jsonl"

function Download-IfMissing {
    param(
        [string]$Url,
        [string]$Path
    )

    if (Test-Path $Path) {
        Write-Host "exists: $Path"
        return
    }

    Write-Host "download: $Url"
    Invoke-WebRequest -Uri $Url -OutFile $Path
}

Download-IfMissing "$baseUrl/commits.parquet" $commitsPath
Download-IfMissing "$baseUrl/qna.parquet" $qnaPath

$python = @'
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--commits", required=True)
parser.add_argument("--qna", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--max-rows", type=int, default=0)
args = parser.parse_args()

try:
    import pyarrow.parquet as pq
except Exception as exc:
    raise SystemExit(
        "pyarrow is required for Parquet conversion. Install with: python -m pip install pyarrow\n"
        f"Original error: {exc}"
    )

commit_table = pq.read_table(args.commits)
qna_table = pq.read_table(args.qna)
commit_rows = commit_table.to_pylist()
qna_rows = qna_table.to_pylist()

commit_by_key = {}
for row in commit_rows:
    key = (row.get("repo_id"), row.get("commit_index"), row.get("commit_sha"))
    commit_by_key[key] = row

def first_present(row, names, default=None):
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return default

def find_embedding(row):
    for key, value in row.items():
        lowered = key.lower()
        if "embedding" not in lowered:
            continue
        if isinstance(value, list) and len(value) in (384, 768):
            if len(value) == 384:
                return value + value
            return value
    return []

limit = args.max_rows if args.max_rows and args.max_rows > 0 else None
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

written = 0
with out_path.open("w", encoding="utf-8", newline="\n") as out:
    for row in qna_rows:
        if limit is not None and written >= limit:
            break

        key = (row.get("repo_id"), row.get("commit_index"), row.get("commit_sha"))
        commit = commit_by_key.get(key, {})
        merged = {**commit, **row}

        record = {
            "repo_id": merged.get("repo_id", ""),
            "commit_index": merged.get("commit_index"),
            "commit_sha": merged.get("commit_sha"),
            "cross_repo_split": merged.get("cross_repo_split") or "",
            "in_repo_split": merged.get("in_repo_split") or "",
            "file_path": first_present(merged, ["file_path", "path"], ""),
            "input_prefix": first_present(merged, ["input_prefix", "prefix", "prompt", "question"], ""),
            "target_value": first_present(merged, ["target_value", "target", "answer", "completion", "assertion"], ""),
            "production_code_diff": merged.get("production_code_diff") or "",
            "repo_embedding": find_embedding(merged),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1

print(f"wrote {written} records to {out_path}")
'@

$tmp = Join-Path $OutputDir "convert_code2lora_parquet.py"
Set-Content -Path $tmp -Value $python -Encoding UTF8
python $tmp --commits $commitsPath --qna $qnaPath --output $jsonlPath --max-rows $MaxRows

Write-Host "ready: $jsonlPath"
