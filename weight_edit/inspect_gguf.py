import sys, numpy as np
from gguf import GGUFReader
from gguf.constants import GGUFValueType
P="/Users/vhinymombo/.ollama/models/blobs/sha256-3d0b790534fe4b79525fc3692950408dca41171676ed7e21db57af5c65ef6ab6"
r=GGUFReader(P)
for k,f in r.fields.items():
    if k.startswith("tokenizer.ggml.") and k not in ("tokenizer.ggml.model","tokenizer.ggml.pre","tokenizer.ggml.bos_token_id","tokenizer.ggml.eos_token_id","tokenizer.ggml.padding_token_id","tokenizer.ggml.add_bos_token"):
        print(k, "len", len(f.data)); continue
    if k=="tokenizer.chat_template":
        print(k, "len", len(f.parts[-1])); continue
    t=f.types[0]
    if t==GGUFValueType.STRING:
        v=bytes(f.parts[f.data[0]]).decode()
    elif t==GGUFValueType.ARRAY:
        v=f"array len {len(f.data)}"
    else:
        v=f.parts[f.data[0]][0]
    print(k, "=", v)
from collections import Counter
c=Counter(); 
for t in r.tensors: c[t.tensor_type.name]+=1
print(c)
for t in r.tensors:
    if not t.name.startswith("blk.") or t.name.startswith("blk.0.") or t.name.startswith("blk.27."):
        print(f"{t.name:32s} {str(list(t.shape)):16s} {t.tensor_type.name:6s} {int(t.n_elements)}")
