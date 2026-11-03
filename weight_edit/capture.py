"""Capture activations (numpy forward) on baseline responses. One sequence per practice problem.

Saves per sequence: residual stream after every layer (float16, all positions), attention argmax /
max-weight per head per layer, token ids, prompt length, teacher-forced logprob of each token.
Usage: capture.py <run_name> <start> <end> <outdir>     (end-start <= 64 -> one budget pass)
"""
import sys, os, json, time, numpy as np
from qwen3_np import Qwen3NP, tokenize, chat_text, rmsnorm, apply_rope, rope_cache, NH, NKV, HD, D

run, a, b, outdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(outdir, exist_ok=True)
rows = {r["unique_id"]: r for r in json.load(open("practice.json"))}
res = [json.loads(l) for l in open(f"runs/{run}.jsonl")]
res = sorted(res, key=lambda x: x["unique_id"])[a:b]
m = Qwen3NP()


def forward_capture(tokens):
    T = len(tokens)
    x = m.embd[np.asarray(tokens)].astype(np.float32)
    resid = [x.astype(np.float16)]
    cos, sin = rope_cache(T)
    mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
    scale = 1.0 / np.sqrt(HD)
    amax, awt = [], []
    for i, Lw in enumerate(m.L):
        h = rmsnorm(x, Lw["attn_norm"])
        q = rmsnorm((h @ Lw["q"].T).reshape(T, NH, HD), Lw["qn"]); q = apply_rope(q, cos, sin)
        k = rmsnorm((h @ Lw["k"].T).reshape(T, NKV, HD), Lw["kn"]); k = apply_rope(k, cos, sin)
        v = (h @ Lw["v"].T).reshape(T, NKV, HD)
        kk = np.repeat(k, NH // NKV, axis=1); vv = np.repeat(v, NH // NKV, axis=1)
        att = np.matmul(q.transpose(1, 0, 2), kk.transpose(1, 2, 0)) * scale + mask[None]
        att -= att.max(-1, keepdims=True); att = np.exp(att); att /= att.sum(-1, keepdims=True)
        amax.append(att.argmax(-1).astype(np.int16).T)            # (T, H)
        awt.append(att.max(-1).astype(np.float16).T)
        o = np.matmul(att, vv.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(T, NH * HD)
        x = x + o @ Lw["o"].T
        h = rmsnorm(x, Lw["ffn_norm"])
        act = h @ Lw["gate"].T; act = act / (1.0 + np.exp(-act)) * (h @ Lw["up"].T)
        x = x + act @ Lw["down"].T
        resid.append(x.astype(np.float16))
    hN = rmsnorm(x, m.out_norm)
    logits = hN @ m.out_w.T
    lp = logits - logits.max(-1, keepdims=True); lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    tgt = np.asarray(tokens[1:])
    tok_lp = lp[np.arange(T - 1), tgt]
    top1 = lp[:-1].argmax(-1)
    ent = -(np.exp(lp) * lp).sum(-1)
    return np.stack(resid), np.stack(amax), np.stack(awt), tok_lp, top1, ent


for r in res:
    uid = r["unique_id"].replace("/", "_").replace(".json", "")
    fn = os.path.join(outdir, uid + ".npz")
    if os.path.exists(fn):
        continue
    prob = rows[r["unique_id"]]["problem"]
    ptoks = tokenize(chat_text(prob)); toks = tokenize(chat_text(prob, r["response"]))
    toks = toks[:2200]  # cap very long loops
    t0 = time.time()
    resid, amax, awt, tok_lp, top1, ent = forward_capture(toks)
    np.savez(fn, resid=resid, amax=amax, awt=awt, tok_lp=tok_lp, top1=top1, ent=ent,
             toks=np.array(toks), plen=len(ptoks), correct=r["correct"], mode=r["mode"])
    print(f"{uid} T={len(toks)} plen={len(ptoks)} {r['mode']} tf_acc={np.mean(top1[len(ptoks)-1:]==np.array(toks[len(ptoks):])):.3f} {time.time()-t0:.1f}s", flush=True)
