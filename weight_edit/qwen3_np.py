"""Pure-numpy forward pass of Qwen3-1.7B from the (dequantized) GGUF, exposing the residual stream.

Used only for activation capture (counts against the forward-pass budget: 1 pass = up to 64 sequences).
"""
import os, json, subprocess, tempfile
import numpy as np
from gguf_edit import Model, ORIG

GGUF_PATH = ORIG
N_LAYER, D, NH, NKV, HD, FF = 28, 2048, 16, 8, 128, 6144
EPS = 1e-6
ROPE_BASE = 1e6


def rmsnorm(x, w):
    v = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(v + EPS)) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_cache(T):
    inv = ROPE_BASE ** (-np.arange(0, HD, 2, dtype=np.float64) / HD)   # (64,)
    ang = np.arange(T, dtype=np.float64)[:, None] * inv[None, :]          # (T, 64)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope(x, cos, sin):
    """x: (T, H, 128), NEOX style (rotate halves)."""
    x1, x2 = x[..., : HD // 2], x[..., HD // 2:]
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1)


class Qwen3NP:
    def __init__(self, path=GGUF_PATH, replacements=None, layers=None):
        """replacements: {tensor_name: float32 array} applied on top of the file's weights (for testing edits)."""
        M = Model(path)
        rep = replacements or {}
        g = lambda n: rep[n] if n in rep else M.deq(n)
        self.embd = g("token_embd.weight")             # (V, D)
        self.out_norm = g("output_norm.weight")
        self.out_w = g("output.weight")                # (V, D)
        self.L = []
        for i in range(N_LAYER):
            p = f"blk.{i}."
            self.L.append(dict(
                attn_norm=g(p + "attn_norm.weight"), ffn_norm=g(p + "ffn_norm.weight"),
                q=g(p + "attn_q.weight"), k=g(p + "attn_k.weight"), v=g(p + "attn_v.weight"),
                o=g(p + "attn_output.weight"), qn=g(p + "attn_q_norm.weight"), kn=g(p + "attn_k_norm.weight"),
                gate=g(p + "ffn_gate.weight"), up=g(p + "ffn_up.weight"), down=g(p + "ffn_down.weight"),
            ))
        del M

    def forward(self, tokens, capture=None, want_logits=True, hooks=None):
        """tokens: list[int]. capture: dict with keys
             'resid': list of layer indices l -> residual stream AFTER block l is stored (l=-1: embeddings)
             'attn_out': layers -> attention block output (what attn_output writes)
             'ffn_out': layers -> ffn block output
             'ffn_act': layers -> gated activation (silu(gate)*up), (T, FF)  [big]
           positions: optional index array of positions to keep.
           hooks: optional {layer: fn(x)->x} applied to the residual after that block (steering experiments)."""
        cap = capture or {}
        pos = cap.get("positions")
        keep = (lambda a: a if pos is None else a[pos])
        store = {}
        T = len(tokens)
        x = self.embd[np.asarray(tokens)].astype(np.float32)          # (T, D)
        if -1 in cap.get("resid", []):
            store[("resid", -1)] = keep(x).astype(np.float16)
        cos, sin = rope_cache(T)
        mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
        scale = 1.0 / np.sqrt(HD)
        for i, Lw in enumerate(self.L):
            h = rmsnorm(x, Lw["attn_norm"])
            q = (h @ Lw["q"].T).reshape(T, NH, HD)
            k = (h @ Lw["k"].T).reshape(T, NKV, HD)
            v = (h @ Lw["v"].T).reshape(T, NKV, HD)
            q = rmsnorm(q, Lw["qn"]); k = rmsnorm(k, Lw["kn"])
            q = apply_rope(q, cos, sin); k = apply_rope(k, cos, sin)
            # GQA: q head j uses kv head j // (NH//NKV)
            kk = np.repeat(k, NH // NKV, axis=1); vv = np.repeat(v, NH // NKV, axis=1)
            att = np.matmul(q.transpose(1, 0, 2), kk.transpose(1, 2, 0)) * scale + mask[None]   # (H,T,S)
            att -= att.max(-1, keepdims=True)
            att = np.exp(att); att /= att.sum(-1, keepdims=True)
            o = np.matmul(att, vv.transpose(1, 0, 2)).transpose(1, 0, 2).reshape(T, NH * HD)
            a_out = o @ Lw["o"].T
            if i in cap.get("attn_out", []):
                store[("attn_out", i)] = keep(a_out).astype(np.float16)
            x = x + a_out
            h = rmsnorm(x, Lw["ffn_norm"])
            act = silu(h @ Lw["gate"].T) * (h @ Lw["up"].T)
            if i in cap.get("ffn_act", []):
                store[("ffn_act", i)] = keep(act).astype(np.float16)
            f_out = act @ Lw["down"].T
            if i in cap.get("ffn_out", []):
                store[("ffn_out", i)] = keep(f_out).astype(np.float16)
            x = x + f_out
            if hooks and i in hooks:
                x = hooks[i](x)
            if i in cap.get("resid", []):
                store[("resid", i)] = keep(x).astype(np.float16)
        logits = None
        if want_logits:
            hN = rmsnorm(x, self.out_norm)
            if pos is not None:
                hN = hN[pos]
            logits = hN @ self.out_w.T
        return logits, store


def tokenize(text, path=GGUF_PATH):
    """Tokenize with llama.cpp's tokenizer (no forward pass). Parses special tokens."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text); fn = f.name
    try:
        out = subprocess.run(["/opt/homebrew/bin/llama-tokenize", "-m", path, "-f", fn, "--ids",
                              "--no-bos", "--no-escape", "--log-disable"],
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(fn)
    return json.loads(out.strip().splitlines()[-1])


PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def chat_text(problem, response=""):
    """Exactly what Ollama's template renders for think=false (non-thinking mode)."""
    return ("<|im_start|>user\n" + PROMPT.format(problem=problem) + " /no_think<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n" + response)


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    m = Qwen3NP()
    print("loaded in", round(time.time() - t0, 1), "s")
    toks = tokenize(chat_text("What is 2+2?"))
    print(toks)
    t0 = time.time()
    logits, _ = m.forward(toks)
    print("forward", logits.shape, round(time.time() - t0, 2), "s; argmax last:", int(logits[-1].argmax()))
