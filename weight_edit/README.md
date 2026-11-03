# Direct weight editing: qwen3:1.7b on MATH-500

Make `qwen3:1.7b` (Ollama, Q4_K_M) better at math by editing its weights by hand. No training, no gradients, no internet.

## Results (100 practice problems, non-thinking, seed 1)

| Model | Score | Fixed / broken vs. unchanged | Weights changed |
|---|---:|---:|---:|
| Unchanged | 65/100 | — | 0 |
| **Edit 1:** `output_norm.weight[1960]` × 0.9 | 69/100 | 9 / 5 | 1 |
| Edit 2: `blk.27.ffn_down.weight` row 1960 × 0.9 | 69/100 | 8 / 4 | 5,952 |

Neither gain is statistically clear (sign test p ≈ 0.42 and 0.39). Held-out results on the other 400 problems: pending.

`qwen3-edit-best` is edit 1: a single F32 value changed from 10.3125 to 9.28125. All other 310 tensors are byte-identical to the original.

## Rules

- Practice set: MATH-500 rows with `index % 5 == 0` (100 problems). The other 400 are held out.
- Budget: 50 forward passes of up to 64 sequences. 15 used (see `budget_log.md`).
- Sampling: T=0.7, top_p 0.8, top_k 20, seed 1, max 4,096 tokens, `think=false`.

## Rebuild the edited model

Requires the original `qwen3:1.7b` in Ollama and a venv with `gguf`, `numpy`, `math-verify`, `requests`.

```sh
export SCRATCH=~/llmweighttrick-scratch          # where GGUF files are written
python weight_edit/rebuild_best.py --register    # builds and registers qwen3-edit-best, no inference
```

Evaluate on the held-out problems from the project root:

```sh
python bench_math.py --model qwen3-edit-best --split heldout --seed 1 --workers 1
```

## Files

| File | Purpose |
|---|---|
| `gguf_edit.py` | Read the GGUF, replace tensors, write a new GGUF with metadata and tokenizer intact |
| `qwen3_np.py` | NumPy forward pass of Qwen3, used to read activations |
| `trace_edit.py` | Compares activations on failed vs. correct answers and picks coordinate 1960 |
| `trace_layers.py` | Records which layers write to that coordinate |
| `ffn_edit.py` | Builds edit 2 |
| `eval_practice.py` | Runs and grades the 100 practice problems |
| `rebuild_best.py` | Rebuilds `qwen3-edit-best` from the saved edit spec |
| `REPORT.md` | Full write-up |
| `budget_log.md` | Every forward pass used |
| `layer_trace.json`, `trace_edit.json`, `ffn_edit.json`, `summary.json` | Data behind the results |
