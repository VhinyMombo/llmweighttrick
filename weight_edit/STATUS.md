# Completed

Original/control: 65/100. One-weight edit: 69/100 (9 wins, 5 losses). FFN row edit: 69/100 (8 wins, 4 losses). All runs have 3 truncated answers. Coherence checks passed.

Final: qwen3-edit-best, output_norm.weight[1960] changed from 10.3125 to 9.28125. Exactly one F32 scalar changed; the equally scoring FFN candidate is not combined with it.

15/50 passes conservatively charged; no further inference. All temporary model tags removed and original tag preserved. Full report: REPORT.md. Pass log: budget_log.md. Rebuild instructions: RUN_COMMANDS.md and rebuild_best.py.

The four-point practice gain is inconclusive. Held-out performance is unknown and has not been evaluated. No held-out JSON was parsed.

Rebuild verification completed: best.gguf and selected norm10.gguf have equal metadata and 311/311 raw-identical tensors. No additional forward passes.
