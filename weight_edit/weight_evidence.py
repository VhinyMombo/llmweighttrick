import json,numpy as np
from gguf_edit import Model
m=Model(); j=json.load(open('weight_edit/trace_edit.json'))['dimension']
tokens=m.r.fields['tokenizer.ggml.tokens'].contents()
w=m.deq('output.weight')[:,j]
e={'dimension':j,'unembedding_column_norm':float(np.linalg.norm(w)),'positive_token_loadings':[(int(i),tokens[i],float(w[i])) for i in np.argsort(-w)[:20]],'negative_token_loadings':[(int(i),tokens[i],float(w[i])) for i in np.argsort(w)[:20]],'layers':[]}
for l in range(28):
    r={'layer':l}
    for suffix in ['attn_output','ffn_down']:
        a=m.deq(f'blk.{l}.{suffix}.weight'); norms=np.linalg.norm(a,axis=1)
        r[suffix+'_row_norm']=float(norms[j]);r[suffix+'_row_percentile']=float(np.mean(norms<=norms[j]))
    e['layers'].append(r)
json.dump(e,open('weight_edit/weight_evidence.json','w'),indent=2)
print(json.dumps(e,indent=2))
