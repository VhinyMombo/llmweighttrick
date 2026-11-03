# Budget pass #3: validate the numpy forward pass by teacher-forcing one real baseline response.
import json, time, numpy as np
from qwen3_np import Qwen3NP, tokenize, chat_text
rows = {r["unique_id"]: r for r in json.load(open("practice.json"))}
res = [json.loads(l) for l in open("runs/baseline_seed1.jsonl")]
r = min(res, key=lambda x: x["tokens"])  # shortest response
prob = rows[r["unique_id"]]["problem"]
prompt_toks = tokenize(chat_text(prob))
full_toks = tokenize(chat_text(prob, r["response"]))
assert full_toks[:len(prompt_toks)] == prompt_toks
t0 = time.time(); m = Qwen3NP(); print("load", round(time.time()-t0,1), "s", flush=True)
t0 = time.time(); logits, _ = m.forward(full_toks); print("forward", len(full_toks), "tokens", round(time.time()-t0,1), "s", flush=True)
pred = logits[:-1].argmax(-1); tgt = np.array(full_toks[1:])
P = len(prompt_toks)
print("teacher-forced top1 acc on response tokens: %.3f (%d tokens)" % ((pred[P-1:] == tgt[P-1:]).mean(), len(tgt)-P+1))
# top-5
top5 = np.argsort(-logits[:-1], axis=-1)[:, :5]
print("top5 acc: %.3f" % np.mean([t in top5[i] for i, t in enumerate(tgt)][P-1:]))
lp = logits[:-1] - logits[:-1].max(-1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
print("mean NLL of response tokens: %.3f" % -lp[np.arange(P-1, len(tgt)), tgt[P-1:]].mean())
