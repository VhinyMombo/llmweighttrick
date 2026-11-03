# Direct weight-edit experiment

Only the 100 practice rows are available to the scripts. The practice cache was checked using an index filter before JSON parsing; none of the other 400 rows were parsed. Do not use bench_math.py: its default loader does not respect this boundary.

Inherited work: a roundtrip GGUF, 53 JSONL records (45 unique), a prior interrupted baseline, and a NumPy validation. The log retains its three charged passes and conservatively charges the resumed 55-sequence baseline separately. Duplicate records are aggregated by unique_id, last record wins. This means the original baseline is not a perfectly clean new run.

Pipeline: independently verified every metadata value and all 311 raw tensor arrays equal between original and roundtrip. No dequantization or requantization for untouched tensors. The tokenizer is preserved. The baseline evaluation uses think=false. NumPy teacher forcing uses the explicit empty think block.

Hypothesis: a residual coordinate disproportionately active in failed responses may distort final token preferences. Measure mean squared RMS-normalized final-layer residual across response positions (up to token 640) for the 8 shortest saved failures and 8 shortest successes. Select the coordinate with largest (error energy - success energy) times output-column L2 norm times absolute output RMSNorm gain. Apply a fixed multiplicative 0.9 to that one output_norm.weight gain. This is an observational heuristic followed by an intervention, not a claim that the coordinate implements arithmetic. No derivative, numerical gradient, backpropagation, fitting, or training is performed.

Limitations: length, subject and response content can confound the contrast. Selected responses are an adaptive, small practice subset. Successes and failures have different length distributions. An edit changes sampling trajectories; a few changed answers do not establish improved mathematical reasoning. Imported-model runner labels differ from the original, so the primary paired comparison uses the unchanged imported roundtrip control.

Reproduction: use the offline venv under $SCRATCH/venv/bin/python. trace_edit.py performs model inference and must not be rerun without charging a pass. A rebuild-only script will reconstruct the final GGUF from the saved scalar specification without inference.

Second candidate, fixed independently of candidate-1 results: actual layer captures identify block 27 FFN as the largest writer to coordinate 1960. Scale only blk.27.ffn_down.weight row 1960 by 0.9. Change Q6_K FP16 outer scales only; integer codes are preserved. This is a separate intervention from the output normalization change, not a combination. Neither layer energy nor an accuracy change proves a math-specific mechanism. Captures stop at token 640 and can precede the actual error.
