# MATH-500 — qwen3:1.7b (Ollama, non-thinking)

**Accuracy: 336 / 500 = 67.2%**

## Setup
- Model: Ollama `qwen3:1.7b` (GGUF, Q4_K_M), Ollama 0.40.11, Apple M3 Pro
- Dataset: [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500), all 500 problems
- Mode: non-thinking (`think: false`)
- Sampling: temperature 0.7, top_p 0.8, top_k 20 (Qwen3 non-thinking recommendation), 1 sample per problem
- Max output tokens: 4096
- Prompt: `{problem}` + "Please reason step by step, and put your final answer within \boxed{}."
- Grading: [math-verify](https://github.com/huggingface/Math-Verify) (equivalence-based)
- Script: `bench_math.py --mode nothink`

## By difficulty level
| Level | Correct | Accuracy |
|---|---|---|
| 1 | 40 / 43 | 93.0% |
| 2 | 78 / 90 | 86.7% |
| 3 | 82 / 105 | 78.1% |
| 4 | 82 / 128 | 64.1% |
| 5 | 54 / 134 | 40.3% |

## By subject
| Subject | Correct | Accuracy |
|---|---|---|
| Algebra | 105 / 124 | 84.7% |
| Prealgebra | 65 / 82 | 79.3% |
| Number Theory | 45 / 62 | 72.6% |
| Counting & Probability | 27 / 38 | 71.1% |
| Precalculus | 29 / 56 | 51.8% |
| Geometry | 21 / 41 | 51.2% |
| Intermediate Algebra | 44 / 97 | 45.4% |

## Other observations
- Median response length: 550 tokens.
- 11 responses hit the 4096-token cap (likely repetition loops); none of them were correct.
- Every response contained an extractable final answer.
