---
## Lesson #1 - 2026-06-19
**Trigger:** Loading a freshly saved Candle `VarMap` checkpoint failed with `cannot find head_o_a.weight in VarMap`.
**Rule:** When loading Candle `VarMap` checkpoints, first build the model with the same config to create variables, then call `varmap.load(path)`; do not call `varmap.set()` on an empty map.
**Source:** code2lora-lite usability pass

---
## Lesson #2 - 2026-06-19
**Trigger:** MCP wrapper smoke test initially listed all tools but did not call `code2lora_read_context`.
**Rule:** For MCP wrappers, smoke tests must call every advertised tool at least once and validate one meaningful output from each required workflow step.
**Source:** code2lora-lite MCP wrapper

---
## Lesson #3 - 2026-06-19
**Trigger:** Local OpenCode `opencode.jsonc` could not be parsed by PowerShell `ConvertFrom-Json` during MCP config installer validation.
**Rule:** When updating OpenCode config, treat it as relaxed JSONC and perform minimal targeted edits with backups instead of rewriting it through a strict JSON parser.
**Source:** code2lora-lite MCP config installer

---
## Lesson #4 - 2026-06-19
**Trigger:** The OpenCode autoload hook could import in Node but was not proven usable until it refreshed context through the real OpenCode resolved config path.
**Rule:** For OpenCode hooks, add a smoke test that checks `opencode debug config`, calls the registered hook transform, verifies a visible marker/status file, and isolates Cargo refresh builds with a temp `CARGO_TARGET_DIR`.
**Source:** code2lora-lite OpenCode autoload hook
