import numpy as np, time
from gguf_edit import Model, requant
from gguf.quants import dequantize
M = Model()
# 2. norm vectors: outlier dims
on = M.deq("output_norm.weight")
print("output_norm: mean %.3f min %.3f max %.3f; smallest dims %s largest %s" % (on.mean(), on.min(), on.max(), np.argsort(on)[:5].tolist(), np.argsort(on)[-5:].tolist()))
A = np.stack([M.deq(f"blk.{i}.attn_norm.weight") for i in range(28)]); F = np.stack([M.deq(f"blk.{i}.ffn_norm.weight") for i in range(28)])
print("attn_norm per-layer mean:", np.round(A.mean(1),3).tolist())
print("ffn_norm per-layer mean:", np.round(F.mean(1),3).tolist())
# dims that are consistently tiny (massive-activation dims)
small = np.argsort(A[4:].mean(0))[:8]
print("attn_norm smallest dims (layers 4+ avg):", small.tolist(), np.round(A[4:].mean(0)[small],4).tolist())
small = np.argsort(F[4:].mean(0))[:8]
print("ffn_norm smallest dims:", small.tolist(), np.round(F[4:].mean(0)[small],4).tolist())
# 3. per-layer weight norms
for i in range(28):
    row = [np.linalg.norm(M.deq(f"blk.{i}.{n}.weight")) for n in ("attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down")]
    print(i, " ".join(f"{x:7.1f}" for x in row))
