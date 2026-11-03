"""Observe layer writes to the selected coordinate. No gradients or fitting."""
import json,numpy as np
from qwen3_np import Qwen3NP,tokenize,chat_text
h='weight_edit/';spec=json.load(open(h+'trace_edit.json'));j=spec['dimension']
rows={r['unique_id']:r for r in json.load(open(h+'practice.json'))}
runs={r['unique_id']:r for r in map(json.loads,open(h+'runs/baseline_seed1.jsonl'))}
m=Qwen3NP(); records=[]
for selected in spec['sequences']:
    uid=selected['id'];r=runs[uid];problem=rows[uid]['problem']
    p=tokenize(chat_text(problem));tokens=tokenize(chat_text(problem,r['response']))[:640]
    pos=np.unique(np.linspace(len(p),len(tokens)-1,min(32,len(tokens)-len(p))).astype(int))
    _,c=m.forward(tokens,capture={'resid':list(range(28)),'attn_out':list(range(28)),'ffn_out':list(range(28)),'positions':pos},want_logits=False)
    values={}
    for kind in ['resid','attn_out','ffn_out']:
        values[kind+'_rms']=[float(np.sqrt(np.mean(c[(kind,l)][:,j].astype(np.float32)**2))) for l in range(28)]
        values[kind+'_mean']=[float(np.mean(c[(kind,l)][:,j].astype(np.float32))) for l in range(28)]
    records.append({'id':uid,'correct':r['correct'],**values});print(uid,flush=True)
result={'dimension':j,'sequences':records,'group_means':{}}
for correct in [False,True]:
    result['group_means'][str(correct)]={k:np.mean([r[k] for r in records if r['correct']==correct],axis=0).tolist() for k in values}
json.dump(result,open(h+'layer_trace.json','w'),indent=2)
