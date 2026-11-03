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
