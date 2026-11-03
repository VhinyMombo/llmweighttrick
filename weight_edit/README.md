# Direct weight editing: qwen3:1.7b on MATH-500

Make `qwen3:1.7b` (Ollama, Q4_K_M) better at math by editing its weights by hand. No training, no gradients, no internet.

The weight analysis, both edits and the practice evaluations were done by Codex (GPT-6.1-Sol), working offline under the rules below. It built on an earlier session's setup: the GGUF read/write pipeline, the NumPy forward pass and the first 3 forward passes.

## Results (100 practice problems, non-thinking, seed 1)

| Model | Score | Fixed / broken vs. unchanged | Weights changed |
|---|---:|---:|---:|
| Unchanged | 65/100 | — | 0 |
| **Edit 1:** `output_norm.weight[1960]` × 0.9 | 69/100 | 9 / 5 | 1 |
| Edit 2: `blk.27.ffn_down.weight` row 1960 × 0.9 | 69/100 | 8 / 4 | 5,952 |

Neither gain is statistically clear (sign test p ≈ 0.42 and 0.39). Held-out results on the other 400 problems: pending.

`qwen3-edit-best` is edit 1: a single F32 value changed from 10.3125 to 9.28125. All other 310 tensors are byte-identical to the original.

## Process

1. **Check the pipeline.** Read the original GGUF and write it back unchanged. All 311 tensors and all metadata came out byte-identical, and the copy matched the original on 99/100 practice answers with the same seed.
2. **Build a NumPy copy of the model** (`qwen3_np.py`) to read internal activations. It agreed with llama.cpp on 99.3% of next-token predictions.
3. **Run the unchanged model** on the 100 practice problems: 65 correct, 32 wrong answers, 3 cut off at the token limit.
4. **Compare failures with successes.** Feed 8 failed and 8 correct answers back through the NumPy model (first 640 tokens each) and measure each coordinate of the final residual stream. Coordinate 1960 had the largest excess energy in failed answers, weighted by how strongly it feeds the output layer.
5. **Edit 1.** Scale the final normalisation gain for that coordinate by 0.9 (`output_norm.weight[1960]`, 10.3125 → 9.28125). One weight changed, stored as F32, so no re-quantization.
6. **Find who writes it.** Record every layer's writes to coordinate 1960 (`trace_layers.py`). The block 27 MLP is by far the largest writer.
7. **Edit 2.** Scale row 1960 of `blk.27.ffn_down.weight` by 0.9. The tensor is Q6_K, so only the 24 FP16 block scales of that row change; the integer codes stay the same.
8. **Evaluate each edit** on the same 100 practice problems with the same seed, and check it still answers a non-math prompt normally.
9. **Keep the best.** Both scored 69/100; the tie keeps edit 1 as `qwen3-edit-best` because it changes fewer weights.

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
