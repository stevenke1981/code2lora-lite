# AGENTS.md - code2lora-lite

This repository is designed to be used by Codex and OpenCode with a compact
Code2LoRA context pack before broad source inspection.

## Required Startup Flow

1. Generate or refresh the compact context pack:

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-context.ps1 -RepoPath .
   ```

2. Read `.code2lora/agent-context/context.md` before opening broad source files.
3. Use `.code2lora/agent-context/metrics.json` as the token-reduction evidence.
4. Open raw files only when the compact context does not contain enough evidence
   for the current task.

## Token Budget Rule

Prefer this order:

1. `AGENTS.md`
2. `.code2lora/agent-context/context.md`
3. Direct source files named by the context pack
4. Additional source files discovered by `rg`

Do not paste or summarize the whole repository into the prompt. The context pack
exists to keep Codex/OpenCode runs focused and measurable.

## Code2LoRA Runtime Flow

- Train hypernetwork:
  `cargo run --release -- train -d data/repopeftbench -o checkpoints -e 1`
- Generate adapter:
  `cargo run --release -- adapt <repo> -m checkpoints/final.safetensors -o adapter.safetensors`
- Complete assertion/code prefix:
  `cargo run --release -- complete <repo> adapter.safetensors --prefix "<code>" --max-tokens 64 -o assertion.txt`
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
```
