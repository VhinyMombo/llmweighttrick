"""Write a copy of the qwen3:1.7b GGUF with selected tensors replaced.

Unchanged tensors are copied as raw bytes (no dequant/requant, bit-identical).
Replaced tensors are given as float32 numpy arrays in dequantized layout
(shape = (out_features, in_features), i.e. the shape gguf.quants.dequantize returns)
and are re-quantized to the tensor's original type (F32 / F16 lossless; Q4_K / Q6_K
block-quantized -> only the touched blocks change if you pass the full tensor with
untouched rows left bit-identical; see requant_rows()).

Usage as a library:
    from gguf_edit import load, write_variant
    R = load()
    W = R.deq("blk.5.attn_output.weight")      # float32 (out, in)
    ... edit W ...
    write_variant("/path/out.gguf", {"blk.5.attn_output.weight": W})
"""
import os, sys, json, subprocess
import numpy as np
from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType
from gguf.constants import GGUFValueType
from gguf.quants import dequantize, quantize

ORIG = "/Users/vhinymombo/.ollama/models/blobs/sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"
GGUF_DIR = os.path.join(os.environ.get("SCRATCH", os.path.expanduser("~/llmweighttrick-scratch")), "ggufs")


class Model:
    def __init__(self, path=ORIG):
        self.r = GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}

    def names(self):
        return list(self.t)

    def qtype(self, name):
        return self.t[name].tensor_type

    def raw(self, name):
        return self.t[name].data

    def deq(self, name):
        """float32 array, shape (out_features, in_features) for matrices, (n,) for vectors."""
        t = self.t[name]
        if t.tensor_type == GGMLQuantizationType.F32:
            return np.array(t.data, dtype=np.float32).reshape(np.flipud(t.shape))
        if t.tensor_type == GGMLQuantizationType.F16:
            return t.data.astype(np.float32).reshape(np.flipud(t.shape))
        return dequantize(t.data, t.tensor_type).reshape(np.flipud(t.shape)).astype(np.float32)


def load(path=ORIG):
    return Model(path)


def requant(arr, qtype):
    """float32 (out,in) -> raw bytes array in the layout GGUFReader gives for that qtype."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    if qtype == GGMLQuantizationType.F32:
        return arr
    if qtype == GGMLQuantizationType.F16:
        return arr.astype(np.float16)
    return quantize(arr, qtype)


def write_variant(out_path, replacements, src=ORIG, verbose=True):
    """replacements: {tensor_name: float32 array (dequantized layout)}.
    Everything else (all KV metadata, tokenizer, untouched tensors) is copied byte-for-byte."""
    reader = GGUFReader(src)
    arch = None
    for f in reader.fields.values():
        if f.name == "general.architecture":
            arch = bytes(f.parts[f.data[0]]).decode()
    writer = GGUFWriter(out_path, arch)
    # --- KV metadata ---
    for f in reader.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == GGUFValueType.ARRAY else None
        writer.add_key_value(f.name, f.contents(), vt, sub_type=sub)
    # --- tensor infos ---
    new_raw = {}
    for t in reader.tensors:
        if t.name in replacements:
            arr = replacements[t.name]
            exp = tuple(int(x) for x in np.flipud(t.shape))
            assert tuple(arr.shape) == exp, (t.name, arr.shape, exp)
            raw = requant(arr, t.tensor_type)
            assert raw.nbytes == t.data.nbytes, (t.name, raw.nbytes, t.data.nbytes)
            new_raw[t.name] = raw
            if verbose:
                print(f"replaced {t.name} ({t.tensor_type.name})", flush=True)
        else:
            raw = t.data
        # for quantized (uint8) data the writer wants the raw byte shape and converts it itself
        shape = list(raw.shape) if raw.dtype == np.uint8 else np.flipud(t.shape).tolist()
        writer.add_tensor_info(t.name, shape, raw.dtype, raw.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        raw = new_raw.get(t.name, t.data)
        writer.write_tensor_data(raw)
    writer.close()
    if verbose:
        print("wrote", out_path, os.path.getsize(out_path), flush=True)


def verify_identical(a_path, b_path, deq_check=True):
    """Compare two GGUFs: metadata keys/values and tensors (raw bytes; optionally dequantized)."""
    A, B = GGUFReader(a_path), GGUFReader(b_path)
    ok = True
    for k, fa in A.fields.items():
        if k.startswith("GGUF."):
            continue
        fb = B.fields.get(k)
        if fb is None:
            print("missing field", k); ok = False; continue
        ca, cb = fa.contents(), fb.contents()
        if ca != cb:
            print("field differs", k); ok = False
    ta = {t.name: t for t in A.tensors}
    tb = {t.name: t for t in B.tensors}
    assert set(ta) == set(tb)
    nraw_same = 0
    for n in ta:
        x, y = ta[n], tb[n]
        assert x.tensor_type == y.tensor_type and list(x.shape) == list(y.shape), n
        same = x.data.tobytes() == y.data.tobytes()
        nraw_same += same
        if deq_check and not same:
            dx = dequantize(x.data, x.tensor_type) if x.tensor_type.value > 1 else x.data
            dy = dequantize(y.data, y.tensor_type) if y.tensor_type.value > 1 else y.data
            print(f"tensor {n}: raw differs, max|diff|={np.abs(dx.astype(np.float32)-dy.astype(np.float32)).max():.4g}")
    print(f"fields ok={ok}; tensors raw-identical: {nraw_same}/{len(ta)}")
    return ok and nraw_same == len(ta)


def register(name, gguf_path):
    """Create an Ollama model `name` from gguf_path with the original template/params."""
    mf = os.path.join(GGUF_DIR, f"Modelfile.{name}")
    with open(mf, "w") as f:
        f.write(f"FROM {gguf_path}\n")
        f.write(open(os.path.join(os.path.dirname(__file__), "template.txt")).read())
    subprocess.run(["ollama", "create", name, "-f", mf], check=True)


if __name__ == "__main__":
    # pipeline check: write back unchanged and verify identical
    out = os.path.join(GGUF_DIR, "roundtrip.gguf")
    write_variant(out, {})
    print("identical:", verify_identical(ORIG, out))
