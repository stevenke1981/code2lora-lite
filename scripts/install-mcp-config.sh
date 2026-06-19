#!/usr/bin/env bash
set -euo pipefail

target="all"
repo_path="."
codex_config="${CODEX_CONFIG_PATH:-$HOME/.codex/config.toml}"
opencode_config="${OPENCODE_CONFIG_PATH:-${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.jsonc}"
apply=0
skip_smoke=0

usage() {
  cat <<'USAGE'
Install the code2lora-lite MCP server into Linux Codex/OpenCode configs.

Usage:
  scripts/install-mcp-config.sh [options]

Options:
  --target all|codex|opencode   Config target [default: all]
  --repo-path PATH              Repository path served by MCP [default: .]
  --codex-config PATH           Codex config.toml path
  --opencode-config PATH        OpenCode opencode.jsonc path
  --apply                       Write changes; otherwise dry-run only
  --skip-smoke                  Skip MCP JSON-RPC smoke test
  -h, --help                    Show this help

Requires:
  python3
  pwsh (PowerShell 7+) unless --skip-smoke is used for dry-run only
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="${2:?missing value for --target}"
      shift 2
      ;;
    --repo-path)
      repo_path="${2:?missing value for --repo-path}"
      shift 2
      ;;
    --codex-config)
      codex_config="${2:?missing value for --codex-config}"
      shift 2
      ;;
    --opencode-config)
      opencode_config="${2:?missing value for --opencode-config}"
      shift 2
      ;;
    --apply)
      apply=1
      shift
      ;;
    --skip-smoke)
      skip_smoke=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$target" in
  all|All) target="all" ;;
  codex|Codex) target="codex" ;;
  opencode|OpenCode) target="opencode" ;;
  *)
    echo "--target must be all, codex, or opencode" >&2
    exit 2
    ;;
esac

command -v python3 >/dev/null 2>&1 || {
  echo "python3 is required" >&2
  exit 1
}

if [[ "$apply" -eq 1 || "$skip_smoke" -eq 0 ]]; then
  command -v pwsh >/dev/null 2>&1 || {
    echo "pwsh (PowerShell 7+) is required. Install it or rerun dry-run with --skip-smoke." >&2
    exit 1
  }
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
resolved_repo="$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$repo_path")"
mcp_script="$project_root/scripts/code2lora-mcp.ps1"
summary_path="$resolved_repo/.code2lora/agent-context/mcp-install-summary.json"

if [[ ! -f "$mcp_script" ]]; then
  echo "MCP script not found: $mcp_script" >&2
  exit 1
fi

if [[ "$skip_smoke" -eq 0 ]]; then
  python3 - "$resolved_repo" "$mcp_script" <<'PY'
import json
import os
import shutil
import subprocess
import sys

repo, mcp_script = sys.argv[1], sys.argv[2]
context_dir = ".code2lora/mcp-smoke-context"
context_path = os.path.join(repo, context_dir)
if os.path.isdir(context_path):
    shutil.rmtree(context_path)

def req(req_id, method, params):
    return json.dumps(
        {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params},
        separators=(",", ":"),
    )

requests = [
    req(1, "initialize", {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "code2lora-linux-mcp-smoke", "version": "0"},
    }),
    req(2, "tools/list", {}),
    req(3, "tools/call", {
        "name": "code2lora_agent_context",
        "arguments": {
            "repoPath": repo,
            "outputDir": context_dir,
            "minReduction": 0.80,
            "maxFiles": 24,
        },
    }),
    req(4, "tools/call", {
        "name": "code2lora_read_context",
        "arguments": {"repoPath": repo, "contextDir": context_dir},
    }),
    req(5, "tools/call", {
        "name": "code2lora_agent_open",
        "arguments": {
            "repoPath": repo,
            "contextDir": context_dir,
            "files": ["AGENTS.md", "scripts/code2lora-mcp.ps1"],
            "noContent": True,
        },
    }),
    req(6, "tools/call", {
        "name": "code2lora_session_audit",
        "arguments": {
            "repoPath": repo,
            "contextDir": context_dir,
            "openedFilesPath": f"{context_dir}/opened-files.txt",
            "minReduction": 0.70,
        },
    }),
]

proc = subprocess.run(
    ["pwsh", "-NoProfile", "-File", mcp_script, "-RepoPath", repo],
    input="\n".join(requests) + "\n",
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    check=False,
)
if proc.returncode != 0:
    raise SystemExit(f"MCP smoke process failed ({proc.returncode}):\n{proc.stderr}")

responses = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
if len(responses) != 6:
    raise SystemExit(f"Expected 6 MCP responses, got {len(responses)}")
for response in responses:
    if "error" in response:
        raise SystemExit(f"MCP response {response.get('id')} failed: {response['error']}")

tools = responses[1]["result"]["tools"]
tool_names = {tool["name"] for tool in tools}
required = {
    "code2lora_agent_context",
    "code2lora_read_context",
    "code2lora_agent_open",
    "code2lora_session_audit",
}
missing = sorted(required - tool_names)
if missing:
    raise SystemExit(f"Missing MCP tools: {missing}")

context_text = responses[3]["result"]["content"][0]["text"]
if "Code2LoRA Agent Context" not in context_text:
    raise SystemExit("code2lora_read_context returned unexpected content")

audit_path = os.path.join(context_path, "session-audit.json")
with open(audit_path, "r", encoding="utf-8") as handle:
    audit = json.load(handle)
if not audit.get("passed"):
    raise SystemExit("Session audit did not pass")

summary_path = os.path.join(context_path, "mcp-smoke.json")
summary = {
    "passed": True,
    "response_count": len(responses),
    "tools_count": len(tools),
    "session_reduction_ratio": audit.get("reduction_ratio"),
    "session_token_estimate": audit.get("session_token_estimate"),
    "saved_token_estimate": audit.get("saved_token_estimate"),
    "audit_path": audit_path,
}
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)
print(f"MCP smoke passed: {summary_path}")
PY
fi

python3 - \
  "$target" \
  "$resolved_repo" \
  "$mcp_script" \
  "$codex_config" \
  "$opencode_config" \
  "$summary_path" \
  "$apply" \
  "$skip_smoke" <<'PY'
import json
import os
import re
import shutil
import sys
import time

target, repo, mcp_script, codex_path, opencode_path, summary_path = sys.argv[1:7]
apply = sys.argv[7] == "1"
skip_smoke = sys.argv[8] == "1"
backups = []

def backup(path):
    if not os.path.isfile(path):
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dst = f"{path}.code2lora-{stamp}.bak"
    shutil.copy2(path, dst)
    backups.append(dst)
    return dst

def write_if_apply(path, content):
    if not apply:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)

def toml_string(value):
    return json.dumps(value)

def configure_codex(path):
    args = ["-NoProfile", "-File", mcp_script, "-RepoPath", repo]
    block = (
        '[mcp_servers."code2lora-lite"]\n'
        'command = "pwsh"\n'
        f'args = [{", ".join(toml_string(arg) for arg in args)}]\n'
        "startup_timeout_sec = 120\n"
    )
    content = ""
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
    pattern = re.compile(
        r'(?ms)^\[mcp_servers\.(?:"code2lora-lite"|code2lora-lite)\]\n.*?(?=^\[|\Z)'
    )
    if pattern.search(content):
        new_content = pattern.sub(block + "\n", content)
    else:
        sep = "\n\n" if content.strip() else ""
        new_content = content.rstrip() + sep + block
    if apply:
        backup(path)
        write_if_apply(path, new_content)
    return {
        "path": path,
        "configured": '[mcp_servers."code2lora-lite"]' in new_content,
        "applied": apply,
    }

def count_braces(line):
    return line.count("{") - line.count("}")

def remove_opencode_entry(lines):
    start = -1
    for i, line in enumerate(lines):
        if re.match(r'^\s*"code2lora-lite"\s*:\s*\{', line):
            start = i
            break
    if start < 0:
        return lines
    depth = 0
    end = start
    for i in range(start, len(lines)):
        depth += count_braces(lines[i])
        if depth == 0:
            end = i
            break
    if lines[end].rstrip().endswith(","):
        lines[end] = lines[end].rstrip().rstrip(",")
    elif end + 1 < len(lines) and lines[end + 1].lstrip().startswith(","):
        lines[end + 1] = lines[end + 1].lstrip()[1:]
    return [line for i, line in enumerate(lines) if i < start or i > end]

def opencode_entry_lines():
    return [
        '    "code2lora-lite": {',
        '      "type": "local",',
        '      "command": [',
        '        "pwsh",',
        '        "-NoProfile",',
        '        "-File",',
        f'        {json.dumps(mcp_script)},',
        '        "-RepoPath",',
        f'        {json.dumps(repo)}',
        '      ],',
        '      "enabled": true,',
        '      "timeout": 120000',
        '    }',
    ]

def ensure_mcp_object(lines):
    for i, line in enumerate(lines):
        if re.match(r'^\s*"mcp"\s*:\s*\{', line):
            return lines
    if not lines or not "".join(lines).strip():
        return ["{", '  "mcp": {', "  }", "}"]
    close = -1
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "}":
            close = i
            break
    if close < 0:
        raise SystemExit(f"OpenCode config has no top-level object: {opencode_path}")
    prev = close - 1
    while prev >= 0 and not lines[prev].strip():
        prev -= 1
    if prev >= 0 and not lines[prev].rstrip().endswith(("{", ",")):
        lines[prev] = lines[prev].rstrip() + ","
    return lines[:close] + ['  "mcp": {', "  }"] + lines[close:]

def configure_opencode(path):
    lines = []
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    lines = remove_opencode_entry(lines)
    lines = ensure_mcp_object(lines)

    mcp_start = -1
    for i, line in enumerate(lines):
        if re.match(r'^\s*"mcp"\s*:\s*\{', line):
            mcp_start = i
            break
    if mcp_start < 0:
        raise SystemExit(f"OpenCode config has no top-level mcp object: {path}")

    depth = 0
    mcp_end = -1
    for i in range(mcp_start, len(lines)):
        depth += count_braces(lines[i])
        if i > mcp_start and depth == 0:
            mcp_end = i
            break
    if mcp_end < 0:
        raise SystemExit(f"Could not find end of OpenCode mcp object: {path}")

    prev = mcp_end - 1
    while prev > mcp_start and not lines[prev].strip():
        prev -= 1
    if prev > mcp_start and not lines[prev].rstrip().endswith(("{", ",")):
        lines[prev] = lines[prev].rstrip() + ","

    new_lines = lines[:mcp_end] + opencode_entry_lines() + lines[mcp_end:]
    new_content = "\n".join(new_lines) + "\n"
    if apply:
        backup(path)
        write_if_apply(path, new_content)
    return {
        "path": path,
        "configured": '"code2lora-lite"' in new_content,
        "applied": apply,
    }

summary = {
    "repo_path": repo,
    "mcp_script": mcp_script,
    "target": target,
    "applied": apply,
    "smoke": None if skip_smoke else "passed",
    "codex": None,
    "opencode": None,
    "backups": backups,
}

if target in ("all", "codex"):
    summary["codex"] = configure_codex(codex_path)
if target in ("all", "opencode"):
    summary["opencode"] = configure_opencode(opencode_path)

summary["backups"] = backups
os.makedirs(os.path.dirname(summary_path), exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

if apply:
    print("Code2LoRA MCP config installed")
else:
    print("Code2LoRA MCP config dry-run passed; rerun with --apply to write config")
print(f"Summary: {summary_path}")
print(json.dumps(summary, indent=2))
PY
