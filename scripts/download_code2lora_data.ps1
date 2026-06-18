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
import hashlib
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

commit_file = pq.ParquetFile(args.commits)
qna_file = pq.ParquetFile(args.qna)

commit_columns = set(commit_file.schema_arrow.names)
qna_columns = set(qna_file.schema_arrow.names)

print(f"commits rows={commit_file.metadata.num_rows} columns={sorted(commit_columns)}")
print(f"qna rows={qna_file.metadata.num_rows} columns={sorted(qna_columns)}")

commit_read_columns = [
    name for name in [
        "repo_id",
        "commit_index",
        "commit_sha",
        "cross_repo_split",
        "in_repo_split",
        "production_code_diff",
    ]
    if name in commit_columns
]

commit_by_key = {}
for batch in commit_file.iter_batches(columns=commit_read_columns, batch_size=4096):
    for row in batch.to_pylist():
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

def hash_embedding(text, dim=768):
    values = [0.0] * dim
    if not text:
        text = "empty"

    tokens = text.replace("\n", " ").split()
    if not tokens:
        tokens = [text]

    for token in tokens:
        digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=16).digest()
        idx = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] & 1 else -1.0
        values[idx] += sign

    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]

limit = args.max_rows if args.max_rows and args.max_rows > 0 else None
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)

qna_read_columns = [
    name for name in [
        "repo_id",
        "commit_index",
        "commit_sha",
        "file_path",
        "path",
        "test_file",
        "input_prefix",
        "prefix",
        "prompt",
        "question",
        "target_value",
        "target",
        "answer",
        "completion",
        "assertion",
    ]
    if name in qna_columns
]
embedding_columns = [name for name in (commit_columns | qna_columns) if "embedding" in name.lower()]
qna_read_columns.extend([name for name in embedding_columns if name in qna_columns and name not in qna_read_columns])
commit_embedding_columns = [name for name in embedding_columns if name in commit_columns]

written = 0
with out_path.open("w", encoding="utf-8", newline="\n") as out:
    for batch in qna_file.iter_batches(columns=qna_read_columns, batch_size=1024):
        if limit is not None and written >= limit:
            break

        for row in batch.to_pylist():
            if limit is not None and written >= limit:
                break

            key = (row.get("repo_id"), row.get("commit_index"), row.get("commit_sha"))
            commit = commit_by_key.get(key, {})
            merged = {**commit, **row}

            embedding = find_embedding(merged)
            embedding_source = "parquet"
            if not embedding:
                embedding = hash_embedding(
                    f"{merged.get('repo_id', '')}\n{merged.get('production_code_diff') or ''}"
                )
                embedding_source = "hash_fallback"

            record = {
                "repo_id": merged.get("repo_id", ""),
                "commit_index": merged.get("commit_index"),
                "commit_sha": merged.get("commit_sha"),
                "cross_repo_split": merged.get("cross_repo_split") or "",
                "in_repo_split": merged.get("in_repo_split") or "",
                "file_path": first_present(merged, ["file_path", "path", "test_file"], ""),
                "input_prefix": first_present(merged, ["input_prefix", "prefix", "prompt", "question"], ""),
                "target_value": first_present(merged, ["target_value", "target", "answer", "completion", "assertion"], ""),
                "production_code_diff": merged.get("production_code_diff") or "",
                "repo_embedding": embedding,
                "embedding_source": embedding_source,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

print(f"wrote {written} records to {out_path}")
'@

$tmp = Join-Path $OutputDir "convert_code2lora_parquet.py"
Set-Content -Path $tmp -Value $python -Encoding UTF8
python $tmp --commits $commitsPath --qna $qnaPath --output $jsonlPath --max-rows $MaxRows

Write-Host "ready: $jsonlPath"
