# Commands used (working directory: project root)

Python executable (`$WEIGHT_PY` below):
`$SCRATCH/venv/bin/python`

```sh
WEIGHT_PY=$SCRATCH/venv/bin/python
"$WEIGHT_PY" weight_edit/eval_practice.py --model qwen3:1.7b --name baseline_seed1
"$WEIGHT_PY" weight_edit/trace_edit.py
"$WEIGHT_PY" weight_edit/weight_evidence.py
"$WEIGHT_PY" weight_edit/trace_layers.py
"$WEIGHT_PY" weight_edit/finish_experiment.py
"$WEIGHT_PY" weight_edit/report.py
```

finish_experiment.py runs the following in order, always with think=false:

```sh
ollama create qwen3-edit-norm10 -f weight_edit/Modelfile.qwen3-edit-norm10
# 1 roundtrip coherence generation (logged separately)
"$WEIGHT_PY" weight_edit/eval_practice.py --model qwen3-edit-roundtrip --name roundtrip_seed1
"$WEIGHT_PY" weight_edit/eval_practice.py --model qwen3-edit-norm10 --name norm10_seed1
# 1 edited coherence generation (logged separately)
ollama create qwen3-edit-best -f weight_edit/Modelfile.qwen3-edit-best
ollama rm qwen3-edit-norm10
ollama rm qwen3-edit-roundtrip
```

Rebuild the selected weights and register best without any forward pass:

```sh
"$WEIGHT_PY" weight_edit/rebuild_best.py --register
```

The trace script selects from the saved partial baseline snapshot it sees at execution time. Exact IDs and scalar are saved in trace_edit.json, so rebuild_best.py is the reproducible way to reconstruct the selected edit after the baseline file has grown. Do not rerun the experiments without updating budget_log.md. Original pipeline command before this session: gguf_edit.py wrote roundtrip.gguf and verified raw equality; this session independently reran verify_identical.

Second, independently defined FFN-row intervention and evaluation:

```sh
"$WEIGHT_PY" weight_edit/ffn_edit.py
"$WEIGHT_PY" weight_edit/finish_ffn.py
"$WEIGHT_PY" weight_edit/final_budget.py
"$WEIGHT_PY" weight_edit/report.py
```

finish_ffn.py waits for the first comparison, registers qwen3-edit-ffnrow10 from its saved Modelfile, evaluates 100 practice problems as ffnrow10_seed1, checks one non-math response, retains the highest practice score as qwen3-edit-best, and removes qwen3-edit-ffnrow10. ffn_edit.py is rebuild-only and performs no model forward pass. It starts from the original GGUF, preserves Q6_K integer codes, and changes 24 FP16 outer scales in row 1960 of block 27 FFN down projection. The exact stored-value error is saved in ffn_edit.json.
