import json,os
from gguf_edit import GGUF_DIR,Model
import numpy as np
h=os.path.dirname(os.path.abspath(__file__));s=json.load(open(h+'/summary.json'));t=s['trace']
s['forward_passes']=15
json.dump(s,open(h+'/summary.json','w'),indent=2)
a=Model();b=Model(GGUF_DIR+'/norm10.gguf'); diffs={}
for n in a.names():
    if not np.array_equal(a.raw(n),b.raw(n)):
        x=a.deq(n);y=b.deq(n);diffs[n]={'changed_scalars':int(np.count_nonzero(x!=y)),'max_abs_change':float(abs(x-y).max())}
assert diffs.keys()=={'output_norm.weight'} and diffs['output_norm.weight']['changed_scalars']==1
json.dump(diffs,open(h+'/tensor_diff.json','w'),indent=2)
original={r['unique_id']:r for r in map(json.loads,open(h+'/runs/baseline_seed1.jsonl'))}
control={r['unique_id']:r for r in map(json.loads,open(h+'/runs/roundtrip_seed1.jsonl'))}
with open(h+'/REPORT.md','w') as f:
    f.write(f'''# Direct weight editing result

Original baseline: {s['original_baseline_correct']}/100. Unchanged imported control: {s['baseline_correct']}/100. One-weight edit: {s['edit_correct']}/100. Paired wins: {len(s['wins'])}; paired losses: {len(s['losses'])}. Original/control exact response text matches on {sum(original[i]["response"]==control[i]["response"] for i in original)}/100 problems; answer correctness disagrees on {sum(original[i]['correct']!=control[i]['correct'] for i in original)} problems.

No clear generalization improvement is established. This is an adaptive 100-problem practice set, with a small biased activation sample and stochastic decoding. A few percentage points can be noise. The held-out 400 problems were never parsed or evaluated.

## Failure modes

Original: {dict(__import__('collections').Counter(r['mode'] for r in original.values()))}. Control: {s['baseline_modes']}. Edit: {s['edit_modes']}.

Observed concrete errors included double-counting the (5,5) dice outcome (13/36 instead of 11/36), forgetting the >200-coin constraint (35 instead of 203), and dropping terms/signs during symbolic expansion. Some long answers repeated conclusions or exhausted 4096 tokens. Two control responses lacked a boxed marker. The coarse loop metric flagged zero responses above 0.2, although qualitative repetition of conclusions was visible. Formatting is not the main failure mode. Subject/level counts and truncated IDs are saved in baseline_diagnostics.json. Full responses are retained in runs/.

## Weight evidence and exact candidate

trace_edit.py teacher-forced 16 saved practice responses, maximum 640 tokens each, with an empty think block. At the final residual (after block 27), RMS-normalized coordinate {t['dimension']} had RMS {t['wrong_rms']:.6f} on 8 failures versus {t['right_rms']:.6f} on 8 successes. The prespecified score multiplies the energy difference by absolute output gain and unembedding-column norm. This identifies a final-readout intervention, not an arithmetic circuit or a causal explanation of individual errors. layer_trace.json records actual attention/FFN writes and residual magnitudes at 32 response positions per sequence across every block. These are observational and are not layer-ablation results. weight_evidence.json contains the unembedding token loadings and the norms of rows writing this coordinate in all 28 attention/FFN output tensors. Static norms alone cannot assign causal blame to a layer.

Only output_norm.weight[{t['dimension']}] changed: {t['old']} to {t['new']} (multiply by 0.9). One F32 scalar of {t['total_parameters']:,}: {100/t['total_parameters']:.12f}% of model parameters. 310 other tensors are byte-identical; only one scalar differs in the remaining tensor. No quantized block was requantized; untouched quantization roundtrip error is exactly zero. See tensor_diff.json. Metadata and tokenizer are copied unchanged by GGUFWriter.

## Second targeted edit: final FFN writer

Actual layer tracing found block 27 FFN writes to coordinate 1960 had mean per-sequence RMS 100.58 on failures versus 92.89 on successes, much larger than other FFN writes. A separate candidate scales only row 1960 of blk.27.ffn_down.weight by 0.9, starting from the original GGUF. It is not combined with the normalization edit. Its Q6_K integer codes and internal scales remain unchanged: only the final FP16 scale in each of the row's 24 blocks changes. This affects up to 6144 dequantized weights. Exact byte changes and FP16 rounding error are in ffn_edit.json; all other rows and tensors were verified identical.

FFN-row candidate: {s['ffn_correct']}/100, paired wins {len(s['ffn_wins'])}, paired losses {len(s['ffn_losses'])} against the {s['baseline_correct']}/100 control. Failure modes: {s['ffn_modes']}. Its non-math response is in ffn_coherence.json.

## Final selection

qwen3-edit-best uses {s['best_gguf']}. Final changed parameter count: {s['final_changed_parameters']} ({100*s['final_changed_parameters']/t['total_parameters']:.12f}% of model). The FFN candidate changes 30 raw bytes, 24 FP16 block scales, and 5952 nonzero dequantized weights; row relative rounding error versus exact scaling is {100*s['ffn_edit']['row_relative_rounding_error']:.6f}% and max absolute error is {s['ffn_edit']['max_error_from_exact_scaling']:.9f}. The candidate with highest practice accuracy above the imported control is selected; ties retain the earlier selection, and if neither improves, best remains unchanged. Neither selection establishes held-out benefit. All three saved coherence responses are identical, coherent, and stop normally at 64 tokens. They are in roundtrip_coherence.json, edit_coherence.json and ffn_coherence.json. Temporary variant tags were removed; final_tags.json confirms only qwen3-edit-best remains, alongside the original qwen3:1.7b.

## Budget and reproduction

Total: {s['forward_passes']}/50 passes conservatively charged, including inherited interrupted work. Each complete 100-problem generation run is charged two passes; activation contrast has 16 sequences, one pass. budget_log.md lists every pass. No gradients, numerical gradients, training, internet, installs, commits, or held-out evaluation were used. Scripts never write results/ or change qwen3:1.7b or Ollama configuration.

Rebuild and register without inference:

```sh
$SCRATCH/venv/bin/python weight_edit/rebuild_best.py --register
```

Primary scripts: gguf_edit.py (GGUF metadata/raw copy), eval_practice.py (practice-only generation and main-thread grading), qwen3_np.py (NumPy forward), trace_edit.py (observational selection and fixed edit), weight_evidence.py (static weights), finish_experiment.py (evaluations/registration/temporary model cleanup), rebuild_best.py (no-inference rebuild). Do not rerun inference scripts without charging the budget. Sampling: T=0.7, top_p=0.8, top_k=20, seed=1, 4096 answer tokens, context 5120, think=false. The complete commands are also in RUN_COMMANDS.md.
''')
with open(h+'/REPORT.md','a') as f:f.write(open(h+'/report_appendix.md').read())
print(open(h+'/REPORT.md').read())
