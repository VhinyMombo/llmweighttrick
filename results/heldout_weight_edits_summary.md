# MATH-500 held-out comparison: weight edits on qwen3:1.7b

400 held-out problems (MATH-500 rows with index % 5 != 0), never used while designing the edits. Non-thinking mode, T=0.7, top_p 0.8, top_k 20, seed 1, max 4,096 tokens, graded with math-verify.

| Model | Correct | Accuracy | Fixed / broken vs. unchanged | Sign test p | Hit token cap |
|---|---:|---:|---:|---:|---:|
| Unchanged `qwen3:1.7b` | 271/400 | 67.8% | — | — | 9 |
| Edit 1: `output_norm.weight[1960]` × 0.9 | 277/400 | 69.2% | 35 / 29 | 0.53 | 7 |
| Edit 2: `blk.27.ffn_down.weight` row 1960 × 0.9 | 270/400 | 67.5% | 23 / 24 | 1.00 | 6 |

Neither edit is distinguishable from the unchanged model. Edit 1 is +6 answers (p = 0.53); edit 2 is -1 (p = 1.00).

## By difficulty level

| Level | Unchanged | Edit 1 | Edit 2 |
|---|---:|---:|---:|
| 1 | 35/38 | 36/38 | 35/38 |
| 2 | 57/69 | 57/69 | 57/69 |
| 3 | 70/88 | 73/88 | 70/88 |
| 4 | 64/98 | 62/98 | 65/98 |
| 5 | 45/107 | 49/107 | 43/107 |

## By subject

| Subject | Unchanged | Edit 1 | Edit 2 |
|---|---:|---:|---:|
| Algebra | 88/104 | 88/104 | 87/104 |
| Counting & Probability | 21/29 | 19/29 | 17/29 |
| Geometry | 18/35 | 20/35 | 20/35 |
| Intermediate Algebra | 42/79 | 43/79 | 43/79 |
| Number Theory | 30/46 | 34/46 | 31/46 |
| Prealgebra | 51/64 | 50/64 | 52/64 |
| Precalculus | 21/43 | 23/43 | 20/43 |
