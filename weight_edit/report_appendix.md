

## Compact results and rebuild verification

| Variant | Practice accuracy | Paired wins / losses | Parameters changed | Evaluation passes |
|---|---:|---:|---:|---:|
| Unchanged imported control | 65/100 | — | 0 | 2 |
| Output normalization gain | 69/100 | 9 / 5 | 1 | 2 + 1 coherence |
| Block 27 FFN output row | 69/100 | 8 / 4 | 5952 nonzero values | 2 + 1 coherence |

Additional passes cover the inherited/interrupted original baseline, NumPy validation, activation contrast, layer capture, and roundtrip coherence. Total remains 15/50. Median generated tokens: control 504; output-gain edit 561; FFN edit 524.

Rebuild-only command was executed successfully. Rebuilt `best.gguf` has metadata equal and 311/311 tensors raw-identical to selected `norm10.gguf`; see rebuild_check.log and rebuild_verify.log. Four GGUF files are retained, within the requested limit. Register using the exact `rebuild_best.py --register` command above; its Modelfile preserves original TEMPLATE/PARAMETER values, independently checked against original_show.json. No inference was used for rebuild verification.
