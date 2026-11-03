"""One observational pass; no derivatives or parameter fitting."""
import json, os, time
import numpy as np
from qwen3_np import Qwen3NP, tokenize, chat_text
from gguf_edit import Model, write_variant, GGUF_DIR
rows={r['unique_id']:r for r in json.load(open('weight_edit/practice.json'))}
runs={r['unique_id']:r for r in map(json.loads,open('weight_edit/runs/baseline_seed1.jsonl'))}
selected=[]
for correct in [False,True]:
    selected += sorted([r for r in runs.values() if r['correct']==correct],key=lambda r:r['tokens'])[:8]
m=Qwen3NP(); stats=[]; evidence=[]
for r in selected:
    p=tokenize(chat_text(rows[r['unique_id']]['problem']))
    t=tokenize(chat_text(rows[r['unique_id']]['problem'],r['response']))[:640]
    positions=np.arange(len(p),len(t))
    _,cap=m.forward(t,capture={'resid':[27],'positions':positions},want_logits=False)
    x=cap[('resid',27)].astype(np.float32); x/=np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)
    stats.append(np.mean(x*x,axis=0)); evidence.append({'id':r['unique_id'],'correct':r['correct'],'tokens':len(t)})
    print(evidence[-1],flush=True)
a=np.array(stats); wrong=a[:8].mean(0); right=a[8:].mean(0)
# Suppress one residual coordinate disproportionately active on errors.
# Fixed 10% intervention; no search, numerical gradients, or fitting.
w=m.out_norm.copy(); impact=np.linalg.norm(m.out_w,axis=0)*abs(w)
score=(wrong-right)*impact
j=int(np.argmax(score)); w[j]*=np.float32(.9)
M=Model(); original=M.deq('output_norm.weight')
report={'sequences':evidence,'dimension':j,'wrong_rms':float(np.sqrt(wrong[j])),'right_rms':float(np.sqrt(right[j])),'old':float(original[j]),'new':float(w[j]),'top_dimensions':np.argsort(-score)[:12].tolist(),'changed_parameters':1,'total_parameters':sum(int(np.prod(t.shape)) for t in M.t.values())}
os.makedirs(GGUF_DIR,exist_ok=True)
write_variant(GGUF_DIR+'/norm10.gguf',{'output_norm.weight':w})
json.dump(report,open('weight_edit/trace_edit.json','w'),indent=2)
np.savez('weight_edit/trace_stats.npz',stats=a,wrong=wrong,right=right,score=score)
print(report,flush=True)
