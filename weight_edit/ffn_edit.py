"""Scale one Q6_K output row through its FP16 block scales; no requantization."""
import json,os,numpy as np
from gguf import GGUFReader,GGUFWriter,GGMLQuantizationType
from gguf.constants import GGUFValueType
from gguf_edit import ORIG,GGUF_DIR,Model
r=GGUFReader(ORIG);name='blk.27.ffn_down.weight';j=1960
T=next(t for t in r.tensors if t.name==name)
assert T.tensor_type==GGMLQuantizationType.Q6_K
raw=T.data.copy();blocks=raw[j].reshape(-1,210)
d=blocks[:,-2:].copy().view(np.float16).reshape(-1)
new=(d.astype(np.float32)*.9).astype(np.float16)
blocks[:,-2:]=new.view(np.uint8).reshape(-1,2)
out=GGUF_DIR+'/ffnrow10.gguf';w=GGUFWriter(out,'qwen3')
for f in r.fields.values():
 if f.name=='general.architecture' or f.name.startswith('GGUF.'):continue
 vt=f.types[0];w.add_key_value(f.name,f.contents(),vt,sub_type=f.types[-1] if vt==GGUFValueType.ARRAY else None)
for t in r.tensors:
 a=raw if t.name==name else t.data
 shape=list(a.shape) if a.dtype==np.uint8 else np.flipud(t.shape).tolist()
 w.add_tensor_info(t.name,shape,a.dtype,a.nbytes,t.tensor_type)
w.write_header_to_file();w.write_kv_data_to_file();w.write_ti_data_to_file()
for t in r.tensors:w.write_tensor_data(raw if t.name==name else t.data)
w.close()
a=Model();b=Model(out);x=a.deq(name);y=b.deq(name)
assert np.array_equal(x[:j],y[:j]) and np.array_equal(x[j+1:],y[j+1:])
for n in a.names():
 if n!=name:assert np.array_equal(a.raw(n),b.raw(n))
report={'tensor':name,'row':j,'intended_scale':.9,'changed_parameters':int(np.count_nonzero(x!=y)),'touched_parameters':6144,'changed_raw_bytes':int(np.count_nonzero(T.data!=raw)),'block_scales_changed':len(d),'max_error_from_exact_scaling':float(abs(y[j]-.9*x[j]).max()),'row_relative_rounding_error':float(np.linalg.norm(y[j]-.9*x[j])/np.linalg.norm(.9*x[j])),'gguf':out}
json.dump(report,open('weight_edit/ffn_edit.json','w'),indent=2);print(report)
