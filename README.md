# llmweighttrick

Experiments with local LLM weights (Ollama / GGUF), starting with `qwen3:1.7b`.

## MATH-500 benchmark against a local Ollama model

`bench_math.py` runs the [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) subset
against any Ollama model and grades answers with Hugging Face's
[`math-verify`](https://github.com/huggingface/Math-Verify) (equivalence-based, so `1/2` == `0.5`).

### Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data/math500
curl -sSL -o data/math500/test.jsonl \
  https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl
ollama pull qwen3:1.7b
```

### Run

```bash
.venv/bin/python bench_math.py --mode nothink            # all 500, non-thinking
.venv/bin/python bench_math.py --mode think --limit 50   # thinking mode, sample
.venv/bin/python bench_math.py --model llama3 --mode nothink
.venv/bin/python bench_math.py --split heldout --seed 1 --workers 1   # 400 held-out problems, reproducible
```

- `--split practice` selects every 5th problem (index % 5 == 0, 100 problems); `--split heldout` selects the other 400.
- `--seed` fixes Ollama's sampling seed so two models can be compared on identical settings.
- `--tag` adds a suffix to the results file name.

- Sampling follows Qwen3's recommendations (think: T=0.6, top_p=0.95; non-think: T=0.7, top_p=0.8; top_k=20).
- Results are appended to `results/<model>_<mode>.jsonl`; re-running resumes where it stopped.
- The summary prints overall accuracy plus breakdowns by difficulty level and subject.

Note: Ollama's `qwen3:1.7b` is mostly Q4_K quantized, so scores are expected to be a bit
below the full-precision numbers in the Qwen3 technical report.

## Results

Full MATH-500, unchanged `qwen3:1.7b`, non-thinking: **336/500 = 67.2%**. Breakdown in [`results/qwen3_1.7b_nothink_summary.md`](results/qwen3_1.7b_nothink_summary.md).

## Direct weight editing

Can hand-editing the weights, with no training and no gradients, make `qwen3:1.7b` better at math? Code and notes are in [`weight_edit/`](weight_edit/).

The weight analysis, both edits and the practice evaluations were done by Codex (GPT-6.1-Sol), working offline under the rules below. It built on an earlier session's setup: the GGUF read/write pipeline, the NumPy forward pass and the first 3 forward passes.

| Model | Practice score (100 problems) | Fixed / broken | Weights changed |
|---|---:|---:|---:|
| Unchanged | 65/100 | — | 0 |
| **Edit 1:** `output_norm.weight[1960]` × 0.9 | 69/100 | 9 / 5 | 1 |
| Edit 2: `blk.27.ffn_down.weight` row 1960 × 0.9 | 69/100 | 8 / 4 | 5,952 |

Neither gain is statistically clear yet. Held-out results on the other 400 problems (same seed and settings):

| Model | Held-out score | Fixed / broken | Sign test p |
|---|---:|---:|---:|
| Unchanged | 271/400 = 67.8% | — | — |
| Edit 1 | 277/400 = 69.2% | 35 / 29 | 0.53 |
| Edit 2 | 270/400 = 67.5% | 23 / 24 | 1.00 |

Neither edit beats the unchanged model on unseen problems. Breakdown by level and subject: [`results/heldout_weight_edits_summary.md`](results/heldout_weight_edits_summary.md).

Rebuild the edited model (`qwen3-edit-best`, edit 1):

```bash
export SCRATCH=~/llmweighttrick-scratch
.venv/bin/python weight_edit/rebuild_best.py --register
.venv/bin/python bench_math.py --model qwen3-edit-best --split heldout --seed 1 --workers 1
```

## Reading the raw weights

Ollama stores each model as a GGUF blob under `~/.ollama/models/blobs/`.
Find the path with `ollama show <model> --modelfile | grep ^FROM`, then:

```python
from gguf import GGUFReader
from gguf.quants import dequantize
r = GGUFReader(path)
t = next(t for t in r.tensors if t.name == "blk.0.attn_q.weight")
w = dequantize(t.data, t.tensor_type)   # float32 numpy array
```
