# Code2LoRA MCP Wrapper

`scripts/code2lora-mcp.ps1` exposes the token-saving agent workflow as MCP
stdio tools for Codex/OpenCode-compatible clients.

## Tools

- `code2lora_agent_context`: generate `context.md`, `metrics.json`, and
  `audit.json`.
- `code2lora_read_context`: read `context.md` before opening broad source files.
- `code2lora_agent_open`: open raw files and record them in `opened-files.txt`.
- `code2lora_session_audit`: write `session-audit.json` for compact context plus
  files actually opened by the agent.

## Server Command

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/code2lora-mcp.ps1 -RepoPath .
```

## Smoke Test

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/mcp-smoke.ps1 -RepoPath .
```

The smoke test calls the MCP server over stdio, runs the context gate, reads the
compact context through MCP, records opened files, runs session audit, and writes
`.code2lora/mcp-smoke-context/mcp-smoke.json`. The smoke test uses an isolated
context directory so a real agent session's opened-files log cannot pollute the
MCP gate.

## Config Examples

Use the files in this directory as repo-local examples. They are intentionally
not installed into global Codex/OpenCode config automatically.

To install the MCP server into the current Windows user's Codex/OpenCode config
with backups:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install-mcp-config.ps1 -RepoPath . -Target All -Apply
```

Omit `-Apply` for a dry run. The installer runs the MCP smoke test first unless
`-SkipSmoke` is passed.
