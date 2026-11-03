# Direct weight editing result

Original baseline: 65/100. Unchanged imported control: 65/100. One-weight edit: 69/100. Paired wins: 9; paired losses: 5. Original/control exact response text matches on 99/100 problems; answer correctness disagrees on 0 problems.

No clear generalization improvement is established. This is an adaptive 100-problem practice set, with a small biased activation sample and stochastic decoding. A few percentage points can be noise. The held-out 400 problems were never parsed or evaluated.

## Failure modes

Original: {'wrong_answer': 32, 'correct': 65, 'truncated': 3}. Control: {'wrong_answer': 32, 'correct': 65, 'truncated': 3}. Edit: {'correct': 69, 'wrong_answer': 28, 'truncated': 3}.

Observed concrete errors included double-counting the (5,5) dice outcome (13/36 instead of 11/36), forgetting the >200-coin constraint (35 instead of 203), and dropping terms/signs during symbolic expansion. Some long answers repeated conclusions or exhausted 4096 tokens. Two control responses lacked a boxed marker. The coarse loop metric flagged zero responses above 0.2, although qualitative repetition of conclusions was visible. Formatting is not the main failure mode. Subject/level counts and truncated IDs are saved in baseline_diagnostics.json. Full responses are retained in runs/.

## Weight evidence and exact candidate

trace_edit.py teacher-forced 16 saved practice responses, maximum 640 tokens each, with an empty think block. At the final residual (after block 27), RMS-normalized coordinate 1960 had RMS 5.385631 on 8 failures versus 5.071316 on 8 successes. The prespecified score multiplies the energy difference by absolute output gain and unembedding-column norm. This identifies a final-readout intervention, not an arithmetic circuit or a causal explanation of individual errors. layer_trace.json records actual attention/FFN writes and residual magnitudes at 32 response positions per sequence across every block. These are observational and are not layer-ablation results. weight_evidence.json contains the unembedding token loadings and the norms of rows writing this coordinate in all 28 attention/FFN output tensors. Static norms alone cannot assign causal blame to a layer.

Only output_norm.weight[1960] changed: 10.3125 to 9.28125 (multiply by 0.9). One F32 scalar of 2,031,739,904: 0.000000049219% of model parameters. 310 other tensors are byte-identical; only one scalar differs in the remaining tensor. No quantized block was requantized; untouched quantization roundtrip error is exactly zero. See tensor_diff.json. Metadata and tokenizer are copied unchanged by GGUFWriter.

## Second targeted edit: final FFN writer

Actual layer tracing found block 27 FFN writes to coordinate 1960 had mean per-sequence RMS 100.58 on failures versus 92.89 on successes, much larger than other FFN writes. A separate candidate scales only row 1960 of blk.27.ffn_down.weight by 0.9, starting from the original GGUF. It is not combined with the normalization edit. Its Q6_K integer codes and internal scales remain unchanged: only the final FP16 scale in each of the row's 24 blocks changes. This affects up to 6144 dequantized weights. Exact byte changes and FP16 rounding error are in ffn_edit.json; all other rows and tensors were verified identical.

FFN-row candidate: 69/100, paired wins 8, paired losses 4 against the 65/100 control. Failure modes: {'correct': 69, 'wrong_answer': 28, 'truncated': 3}. Its non-math response is in ffn_coherence.json.

## Final selection

qwen3-edit-best uses $SCRATCH/ggufs/norm10.gguf. Final changed parameter count: 1 (0.000000049219% of model). The FFN candidate changes 30 raw bytes, 24 FP16 block scales, and 5952 nonzero dequantized weights; row relative rounding error versus exact scaling is 0.062523% and max absolute error is 0.000122070. The candidate with highest practice accuracy above the imported control is selected; ties retain the earlier selection, and if neither improves, best remains unchanged. Neither selection establishes held-out benefit. All three saved coherence responses are identical, coherent, and stop normally at 64 tokens. They are in roundtrip_coherence.json, edit_coherence.json and ffn_coherence.json. Temporary variant tags were removed; final_tags.json confirms only qwen3-edit-best remains, alongside the original qwen3:1.7b.

## Budget and reproduction

Total: 15/50 passes conservatively charged, including inherited interrupted work. Each complete 100-problem generation run is charged two passes; activation contrast has 16 sequences, one pass. budget_log.md lists every pass. No gradients, numerical gradients, training, internet, installs, commits, or held-out evaluation were used. Scripts never write results/ or change qwen3:1.7b or Ollama configuration.

Rebuild and register without inference:

```sh
$SCRATCH/venv/bin/python weight_edit/rebuild_best.py --register
```

Primary scripts: gguf_edit.py (GGUF metadata/raw copy), eval_practice.py (practice-only generation and main-thread grading), qwen3_np.py (NumPy forward), trace_edit.py (observational selection and fixed edit), weight_evidence.py (static weights), finish_experiment.py (evaluations/registration/temporary model cleanup), rebuild_best.py (no-inference rebuild). Do not rerun inference scripts without charging the budget. Sampling: T=0.7, top_p=0.8, top_k=20, seed=1, 4096 answer tokens, context 5120, think=false. The complete commands are also in RUN_COMMANDS.md.


## Compact results and rebuild verification

| Variant | Practice accuracy | Paired wins / losses | Parameters changed | Evaluation passes |
|---|---:|---:|---:|---:|
| Unchanged imported control | 65/100 | — | 0 | 2 |
| Output normalization gain | 69/100 | 9 / 5 | 1 | 2 + 1 coherence |
| Block 27 FFN output row | 69/100 | 8 / 4 | 5952 nonzero values | 2 + 1 coherence |

Additional passes cover the inherited/interrupted original baseline, NumPy validation, activation contrast, layer capture, and roundtrip coherence. Total remains 15/50. Median generated tokens: control 504; output-gain edit 561; FFN edit 524.

Rebuild-only command was executed successfully. Rebuilt `best.gguf` has metadata equal and 311/311 tensors raw-identical to selected `norm10.gguf`; see rebuild_check.log and rebuild_verify.log. Four GGUF files are retained, within the requested limit. Register using the exact `rebuild_best.py --register` command above; its Modelfile preserves original TEMPLATE/PARAMETER values, independently checked against original_show.json. No inference was used for rebuild verification.
