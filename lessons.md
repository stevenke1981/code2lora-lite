---
## Lesson #1 - 2026-06-19
**Trigger:** Loading a freshly saved Candle `VarMap` checkpoint failed with `cannot find head_o_a.weight in VarMap`.
**Rule:** When loading Candle `VarMap` checkpoints, first build the model with the same config to create variables, then call `varmap.load(path)`; do not call `varmap.set()` on an empty map.
**Source:** code2lora-lite usability pass
