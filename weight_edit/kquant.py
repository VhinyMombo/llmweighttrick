"""Fixed-scale re-encoding of Q4_K / Q6_K blocks.

gguf's python package can dequantize but not quantize K-quants. For small edits we do not want a
fresh quantization anyway: we keep every block's original scales/mins (so every untouched weight
stays bit-identical) and only re-pick the nearest 4-bit / 6-bit code for the edited values.
"""
import numpy as np
from gguf.quants import dequantize
from gguf import GGMLQuantizationType as T

QK_K = 256


def _q4k_scales(scales):  # (n,12) uint8 -> sc (n,8), mn (n,8)
    n = scales.shape[0]
    s = scales.reshape(n, 3, 4)
    d, m, m_d = s[:, 0], s[:, 1], s[:, 2]
    sc = np.concatenate([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
    mn = np.concatenate([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], axis=-1)
    return sc, mn


def requant_q4k_fixed(raw, W):
    """raw: original uint8 (rows, 144*ncols/256); W: float32 (rows, cols). Returns new raw uint8."""
    rows = raw.shape[0]
    blocks = raw.reshape(-1, 144)
    n = blocks.shape[0]
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float32)        # (n,1)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, mn = _q4k_scales(blocks[:, 4:16])
    dsc = (d * sc.astype(np.float32)).reshape(n, 8, 1)                   # per sub-block scale
    dmn = (dmin * mn.astype(np.float32)).reshape(n, 8, 1)
    w = np.ascontiguousarray(W, dtype=np.float32).reshape(n, 8, 32)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(dsc > 0, np.rint((w + dmn) / dsc), 0.0)
    q = np.clip(q, 0, 15).astype(np.uint8)                               # (n,8,32)
    # pack: sub-blocks (2l, 2l+1) -> bytes [32l, 32l+32): low nibble = 2l, high nibble = 2l+1
    q = q.reshape(n, 4, 2, 32)
    packed = (q[:, :, 0, :] | (q[:, :, 1, :] << 4)).reshape(n, 128)
    out = blocks.copy()
    out[:, 16:] = packed
    return out.reshape(raw.shape)


def requant_q6k_fixed(raw, W):
    blocks = raw.reshape(-1, 210)
    n = blocks.shape[0]
    scales = blocks[:, 192:208].copy().view(np.int8).astype(np.float32)  # (n,16)
    d = blocks[:, 208:210].copy().view(np.float16).astype(np.float32)    # (n,1)
    dsc = (d * scales).reshape(n, 16, 1)
    w = np.ascontiguousarray(W, dtype=np.float32).reshape(n, 16, 16)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(dsc != 0, np.rint(w / dsc), 0.0)
    q = (np.clip(q, -32, 31) + 32).astype(np.uint8).reshape(n, 2, 4, 32)  # halves of 128: 4 groups of 32
    # within each half: y[l]=g0, y[l+32]=g1, y[l+64]=g2, y[l+96]=g3 ;  ql[l] = g0 | g2<<4 ; ql[l+32] = g1 | g3<<4
    # qh[l] = (g0>>4) | (g1>>4)<<2 | (g2>>4)<<4 | (g3>>4)<<6
    g = q
    ql = np.concatenate([(g[:, :, 0] & 0xF) | ((g[:, :, 2] & 0xF) << 4),
                         (g[:, :, 1] & 0xF) | ((g[:, :, 3] & 0xF) << 4)], axis=-1)   # (n,2,64)
    qh = ((g[:, :, 0] >> 4) | ((g[:, :, 1] >> 4) << 2) | ((g[:, :, 2] >> 4) << 4) | ((g[:, :, 3] >> 4) << 6))  # (n,2,32)
    out = blocks.copy()
    out[:, 0:128] = ql.reshape(n, 128)
    out[:, 128:192] = qh.reshape(n, 64)
    return out.reshape(raw.shape)


def requant_fixed(raw, qtype, W):
    if qtype == T.Q4_K:
        return requant_q4k_fixed(raw, W)
    if qtype == T.Q6_K:
        return requant_q6k_fixed(raw, W)
    raise ValueError(qtype)


if __name__ == "__main__":
    from gguf_edit import Model
    M = Model()
    for name in ["blk.10.attn_output.weight", "blk.3.ffn_down.weight", "blk.27.ffn_down.weight", "output.weight"]:
        raw = M.raw(name); q = M.qtype(name); W = M.deq(name)
        raw2 = requant_fixed(raw, q, W)
        print(name, q.name, "identity re-encode bit-identical:", raw2.tobytes() == raw.tobytes())
        # small edit: rank-1 perturbation of 1e-3 rms relative
        u = np.random.default_rng(0).standard_normal(W.shape[0]).astype(np.float32); v = np.random.default_rng(1).standard_normal(W.shape[1]).astype(np.float32)
        W2 = W + 0.02 * np.sqrt((W**2).mean()) * np.outer(u, v) / 1.0
        raw3 = requant_fixed(raw, q, W2)
        W3 = dequantize(raw3, q).reshape(W.shape)
        print("   edit rms=%.3g  realized rms=%.3g  corr(edit,realized)=%.3f  frac codes changed=%.3f" % (
            np.sqrt(((W2-W)**2).mean()), np.sqrt(((W3-W)**2).mean()),
            np.corrcoef((W2-W).ravel(), (W3-W).ravel())[0,1], np.mean(W3 != W)))
