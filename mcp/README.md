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
`.code2lora/agent-context/mcp-smoke.json`.

## Config Examples

Use the files in this directory as repo-local examples. They are intentionally
not installed into global Codex/OpenCode config automatically.
