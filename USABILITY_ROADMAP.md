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
- Current self-run evidence for this repo: raw estimate 52,982 tokens,
  compact context estimate 1,551 tokens, 160 symbols included, estimated
  reduction 97.1%.

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
