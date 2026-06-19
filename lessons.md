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
