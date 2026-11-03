# Forward-pass budget: 15 / 50 conservatively charged

One pass is up to 64 sequences. Numbering follows reservations; layer pass 12 overlapped generation passes 7–10. No further inference after pass 15.

| # | What ran | Model | Sequences | Learned |
|---|---|---|---|---|
| 1 | inherited interrupted original baseline, scheduled first group | qwen3:1.7b | 64 | Inherited reservation retained; original completed baseline ultimately 65/100 |
| 2 | inherited interrupted original baseline, scheduled second group | qwen3:1.7b | 36 | Inherited reservation retained; incomplete records and duplicates preserved |
| 3 | inherited NumPy validation, teacher force 194-token response | original GGUF NumPy | 1 | Reported top1 agreement .993, top5 1.0, NLL .039 |
| 4 | resume interrupted baseline with 55 remaining unique IDs | qwen3:1.7b | 55 | Original baseline 65/100, 32 wrong answers, 3 truncated; extra conservative charge |
| 5 | contrast final residuals of 8 failures and 8 successes, max 640 tokens | original GGUF NumPy | 16 | Selected coordinate 1960; normalized RMS 5.386 vs 5.071; fixed 10% output gain attenuation |
| 6 | load-check and non-math coherence response | qwen3-edit-roundtrip | 1 | Loads and answers coherently; response saved |
| 7 | unchanged imported control, first group | qwen3-edit-roundtrip | 64 | Full control run 65/100; same settings as original |
| 8 | unchanged imported control, second group | qwen3-edit-roundtrip | 36 | Full control run 65/100; all rows practice only |
| 9 | one F32 output gain edit, first group | qwen3-edit-norm10 | 64 | Full candidate run 69/100, 9 paired wins and 5 losses |
| 10 | one F32 output gain edit, second group | qwen3-edit-norm10 | 36 | Full candidate run 69/100 |
| 11 | non-math coherence response | qwen3-edit-norm10 | 1 | Response saved in edit_coherence.json |
| 12 | layer-localized residual and attention/FFN captures, same 16 response prefixes | original GGUF NumPy | 16 | Block 27 FFN largest measured writer to coordinate 1960; RMS 100.58 failures vs 92.89 successes |
| 13 | separate block 27 FFN row edit, first group | qwen3-edit-ffnrow10 | 64 | Full candidate run 69/100, 8 paired wins and 4 losses |
| 14 | separate block 27 FFN row edit, second group | qwen3-edit-ffnrow10 | 36 | Full candidate run 69/100 |
| 15 | non-math coherence response | qwen3-edit-ffnrow10 | 1 | Response saved in ffn_coherence.json |

Passes 1–3 are inherited log entries. Before resumption, the baseline JSONL contained 53 records with 45 unique IDs. The entire original 100-sequence reservation is retained, plus an extra pass for 55 resumed sequences, so interrupted work is not silently excluded. Duplicate records are preserved and aggregated by unique_id, last record wins. Only local model execution was used. Static dequantization, tokenizer calls, and GGUF writing are not model forward passes.
