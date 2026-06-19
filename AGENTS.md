# AGENTS.md - code2lora-lite

This repository is designed to be used by Codex and OpenCode with a compact
Code2LoRA context pack before broad source inspection.

For a human-readable and agent-readable operating guide, see `USAGE.md`.

OpenCode project config `opencode.jsonc` installs the local
`hooks/code2lora-autoload.mjs` hook. When supported by the client, it refreshes
the compact context if missing, injects it into chat system context, and writes
`.code2lora/agent-context/autoload-status.json`. Verify with
`scripts/opencode-autoload-smoke.ps1` when changing hook behavior.

## Required Startup Flow

1. Generate or refresh the compact context pack:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-context.ps1 -RepoPath .
   ```

2. Read `.code2lora/agent-context/context.md` before opening broad source files.
3. Use the `Symbol Map` section to find likely Rust/PowerShell entry points.
4. Use `.code2lora/agent-context/audit.json` as the pass/fail token gate.
5. Use `.code2lora/agent-context/metrics.json` as the token-reduction evidence.
6. Open raw files only when the compact context does not contain enough evidence
   for the current task. Use `scripts/agent-open.ps1` so opened files are logged:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-open.ps1 -RepoPath . -Files src/agent_context.rs
   ```

7. `scripts/agent-open.ps1` tracks raw files in
   `.code2lora/agent-context/opened-files.txt`.
8. Before final delivery, run session audit:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-session-audit.ps1 -RepoPath . -OpenedFilesPath .code2lora/agent-context/opened-files.txt
   ```

## Token Budget Rule

Prefer this order:

1. `AGENTS.md`
2. `.code2lora/agent-context/context.md`
3. Direct source files named by the context pack
4. Additional source files discovered by `rg`

Do not paste or summarize the whole repository into the prompt. The context pack
exists to keep Codex/OpenCode runs focused and measurable.

## MCP Workflow

Codex/OpenCode clients that support MCP can use the repo-local stdio server:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/code2lora-mcp.ps1 -RepoPath .
```

Use the MCP tools in this order:

1. `code2lora_agent_context`
2. `code2lora_read_context`
3. `code2lora_agent_open`
4. `code2lora_session_audit`

## Code2LoRA Runtime Flow

- Train hypernetwork:
  `cargo run --release -- train -d data/repopeftbench -o checkpoints -e 1`
- Generate adapter:
  `cargo run --release -- adapt <repo> -m checkpoints/final.safetensors -o adapter.safetensors`
- Complete assertion/code prefix:
  `cargo run --release -- complete <repo> adapter.safetensors --prefix "<code>" --max-tokens 64 -o assertion.txt`
- Prepare Evo commit sequence data:
  `powershell -ExecutionPolicy Bypass -File scripts/prepare_repopeftbench_evo.ps1 -OutputDir data/repopeftbench-evo -MaxRows 2000`
- Train Evo GRU checkpoint:
  `cargo run --release -- evo-train -d data/repopeftbench-evo -o checkpoints-evo -e 1 --truncation-steps 8 --max-sequences 4`
- Incrementally update Evo adapter:
  `cargo run --release -- evo-adapt -m checkpoints-evo/evo_final.safetensors --state-in evo_state.safetensors --diff-file commit.patch --state-out evo_state.safetensors -o adapter.safetensors`
- Build agent context only:
  `cargo run --no-default-features -- agent-context <repo> -o .code2lora/agent-context --max-files 24`

## Verification

For ordinary changes, run:

```powershell
cargo fmt --check
cargo check --no-default-features
cargo test --no-default-features
```

For agent-context changes, also run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-context.ps1 -RepoPath .
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { & .\scripts\agent-context.ps1 -RepoPath . -MinReduction 0.999; exit 10 } catch { Write-Host 'Expected token gate failure verified' }"
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-open.ps1 -RepoPath . -NoContent -Files AGENTS.md,scripts/agent-context.ps1,scripts/agent-session-audit.ps1,scripts/agent-open.ps1,src/agent_context.rs
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-session-audit.ps1 -RepoPath . -OpenedFilesPath .code2lora/agent-context/opened-files.txt
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/mcp-smoke.ps1 -RepoPath .
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install-mcp-config.ps1 -RepoPath . -Target All
bash scripts/install-mcp-config.sh --repo-path . --target all --skip-smoke
```
