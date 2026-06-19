# code2lora-lite Usability Roadmap

This file tracks the remaining work required to make the project usable as a real
RepoPeftBench-driven Code2LoRA prototype, not only a compile/test scaffold.

## Current Usable Flow

1. Prepare real RepoPeftBench JSONL:
   `powershell -ExecutionPolicy Bypass -File scripts/prepare_repopeftbench.ps1 -OutputDir data/repopeftbench -SkipCloneRepos`
2. Train the hypernetwork:
   `cargo run --release -- train -d data/repopeftbench -o checkpoints -e 1`
3. Generate a repository adapter from the trained hypernetwork:
   `cargo run --release -- adapt ./my-python-project -m checkpoints/final.safetensors -o adapter.safetensors`
4. Complete from a real assertion/code prefix:
   `cargo run --release -- complete ./my-python-project adapter.safetensors --prefix "def test_answer():`n    assert answer() ==" --max-tokens 64 -o assertion.txt`
5. Build a compact Codex/OpenCode context pack:
   `cargo run --release -- agent-context ./my-python-project -o .code2lora/agent-context --max-files 24`
6. Agent-friendly wrapper:
   `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-context.ps1 -RepoPath ./my-python-project`
7. End-of-task session audit:
   `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-session-audit.ps1 -RepoPath ./my-python-project -OpenedFilesPath .code2lora/agent-context/opened-files.txt`
8. Logged raw-file opening:
   `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/agent-open.ps1 -RepoPath ./my-python-project -Files src/lib.rs`
9. MCP stdio wrapper:
   `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/code2lora-mcp.ps1 -RepoPath ./my-python-project`
10. MCP smoke test:
   `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/mcp-smoke.ps1 -RepoPath .`
11. Install MCP config for local Codex/OpenCode:
   `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/install-mcp-config.ps1 -RepoPath . -Target All -Apply`

## Fixed Blocking Gaps

- `adapt` now requires and loads a trained hypernetwork checkpoint instead of
  producing random LoRA adapter weights.
- `complete` now accepts a user-supplied prefix and decodes generated tokens back
  to text instead of generating from placeholder token IDs.
- README quick-start commands now point to the existing RepoPeftBench preparation
  script and the real checkpoint-driven inference path.
- `agent-context` writes Codex/OpenCode prompt stubs plus deterministic
  before/after token estimates so token reduction is measurable.
- Project-level `AGENTS.md` tells Codex/OpenCode sessions to refresh and read
  the compact context pack before opening broad source files.
- `scripts/agent-context.ps1` is the one-command Windows wrapper for humans,
  Codex, and OpenCode.
- The wrapper writes `audit.json` and fails non-zero when `-MinReduction` is not met.
- `scripts/agent-session-audit.ps1` writes `session-audit.json`, comparing the
  raw repository baseline against compact context plus files actually opened by
  Codex/OpenCode.
- `scripts/agent-open.ps1` opens raw files and records them in
  `opened-files.txt`, making session audit reproducible instead of manual.
- `scripts/code2lora-mcp.ps1` exposes context, open, and audit operations as
  MCP tools for Codex/OpenCode-compatible clients.
- `scripts/mcp-smoke.ps1` verifies MCP initialize, tools/list, context,
  read-context, open, and session audit calls through stdio JSON-RPC.
- `scripts/install-mcp-config.ps1` merges the MCP server into local
  Codex/OpenCode config files with backups and a smoke-test gate.
- Current machine install evidence: `C:\Users\eda\.codex\config.toml` and
  `C:\Users\eda\.config\opencode\opencode.jsonc` contain `code2lora-lite`
  MCP server entries pointing at this repo.
- Latest measured self-run evidence for this repo: raw estimate ~63k tokens,
  compact context estimate ~1.7k tokens, 176 symbols included, estimated
  reduction ~97.1%; the exact run output is in
  `.code2lora/agent-context/metrics.json`.
- Current MCP session-audit evidence for this repo with 6 opened files:
  session estimate ~12.4k tokens, saved estimate ~50.9k tokens, estimated
  reduction ~80%; the exact run output is in
  `.code2lora/agent-context/session-audit.json`.

## P7: Real Dataset Acceptance

- [x] Run `scripts/prepare_repopeftbench.ps1` on a small real RepoPeftBench slice.
- [x] Run `test_real_repopeftbench_jsonl_smoke` against the prepared JSONL.
- [ ] Run `test_p7_repopeftbench_tiny_train` with `CODE2LORA_DATA_DIR`.
- [ ] Save the resulting `checkpoints/final.safetensors` and verify `adapt`
      creates a non-empty adapter from it.
- [x] Run `complete` with a real assertion prefix and inspect output text in
      `test_p7_full_end_to_end_real_inference`.

## P8: Performance Acceptance

- [x] Provide a no-GPU `agent-context` path for Codex/OpenCode token reduction.
- [x] Write `metrics.json` with raw/context token estimates and reduction ratio.
- [x] Write `audit.json` and fail the wrapper when the configured token-reduction gate is missed.
- [x] Write `session-audit.json` comparing compact context plus actual opened files.
- [x] Provide `scripts/agent-open.ps1` to auto-record raw files opened by agents.
- [x] Provide MCP stdio wrapper and smoke test for Codex/OpenCode-compatible clients.
- [x] Provide an installer that writes Codex/OpenCode MCP config entries with backups.
- [ ] Measure prepare/train/adapt/complete wall time on CPU and CUDA.
- [ ] Capture GPU utilization during a real tiny-train run.
- [ ] Confirm repo embedding cache hits on repeated `adapt` / `encode` runs.
- [ ] Add a tiny benchmark command or documented profiling recipe.
- [ ] Keep `cargo fmt --check`, `cargo check --no-default-features`, and
      `cargo test --no-default-features` warning-free.

## Known Limits

- Quality is not proven until the real RepoPeftBench train/eval path reports
  assertion-completion metrics.
- Code2LoRA-Evo is still out of scope; the current implementation is Static.
- The default model path uses Qwen2.5-Coder-0.5B and fp32, so training remains
  heavier than an optimized production pipeline.
