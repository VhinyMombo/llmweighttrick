"""Run after trace_edit.py and baseline complete; local-only model calls."""
import json, os, subprocess, sys, time
from collections import Counter
from gguf_edit import GGUF_DIR,ORIG,Model,verify_identical
HERE=os.path.dirname(os.path.abspath(__file__)); PY=sys.executable

def records(name):
    return {r['unique_id']:r for r in map(json.loads,open(f'{HERE}/runs/{name}.jsonl'))}
def log(n,what,variant,seqs,learned):
    with open(f'{HERE}/budget_log.md','a') as f:f.write(f'\n| {n} | {what} | {variant} | {seqs} | {learned} |\n')
def register(name,path):
    mf=f'{HERE}/Modelfile.{name}'
    open(mf,'w').write(f'FROM {path}\n'+open(f'{HERE}/template.txt').read())
    subprocess.run(['ollama','create',name,'-f',mf],check=True)
while len(records('baseline_seed1'))<100:
    time.sleep(5)
assert len(records('baseline_seed1'))==100
trace=json.load(open(f'{HERE}/trace_edit.json'))
register('qwen3-edit-norm10',GGUF_DIR+'/norm10.gguf')
# Explicitly test roundtrip loading through a no-thinking generation.
import requests
log(6,'roundtrip load and non-math coherence check','qwen3-edit-roundtrip',1,'pending')
from eval_practice import ask
reply,nt,why=ask('qwen3-edit-roundtrip','Write one sentence explaining why leaves look green.',128,1)
json.dump({'response':reply,'tokens':nt,'done_reason':why},open(f'{HERE}/roundtrip_coherence.json','w'),indent=2)
log('7-8','100 roundtrip control practice generations','qwen3-edit-roundtrip',100,'pending')
subprocess.run([PY,f'{HERE}/eval_practice.py','--model','qwen3-edit-roundtrip','--name','roundtrip_seed1'],check=True)
log('9-10','100 practice generations, seed1, unchanged harness','qwen3-edit-norm10',100,'pending')
subprocess.run([PY,f'{HERE}/eval_practice.py','--model','qwen3-edit-norm10','--name','norm10_seed1'],check=True)
log(11,'non-math coherence check','qwen3-edit-norm10',1,'pending')
reply,nt,why=ask('qwen3-edit-norm10','Write one sentence explaining why leaves look green.',128,1)
json.dump({'response':reply,'tokens':nt,'done_reason':why},open(f'{HERE}/edit_coherence.json','w'),indent=2)
original=records('baseline_seed1'); b=records('roundtrip_seed1'); e=records('norm10_seed1')
bs=sum(r['correct'] for r in b.values()); es=sum(r['correct'] for r in e.values())
wins=[i for i in b if not b[i]['correct'] and e[i]['correct']]; losses=[i for i in b if b[i]['correct'] and not e[i]['correct']]
best=GGUF_DIR+'/norm10.gguf' if es>bs else GGUF_DIR+'/roundtrip.gguf'
register('qwen3-edit-best',best)
for name in ['qwen3-edit-norm10','qwen3-edit-roundtrip']:
    subprocess.run(['ollama','rm',name],check=True)
summary={'original_baseline_correct':sum(r['correct'] for r in original.values()),'baseline_correct':bs,'edit_correct':es,'n':100,'wins':wins,'losses':losses,'baseline_modes':dict(Counter(r['mode'] for r in b.values())),'edit_modes':dict(Counter(r['mode'] for r in e.values())),'best_gguf':best,'final_changed_parameters':1 if es>bs else 0,'forward_passes':11,'trace':trace}
json.dump(summary,open(f'{HERE}/summary.json','w'),indent=2)
print(json.dumps(summary,indent=2),flush=True)
