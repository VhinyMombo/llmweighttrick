import json,os,sys,time,subprocess
from collections import Counter
from gguf_edit import GGUF_DIR
from eval_practice import ask
h=os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(h+'/summary.json'):time.sleep(5)
s=json.load(open(h+'/summary.json'));t=json.load(open(h+'/ffn_edit.json'))
def register(name,path):
 mf=h+'/Modelfile.'+name;open(mf,'w').write('FROM '+path+'\n'+open(h+'/template.txt').read());subprocess.run(['ollama','create',name,'-f',mf],check=True)
register('qwen3-edit-ffnrow10',t['gguf'])
with open(h+'/budget_log.md','a') as f:f.write('\n| 13-14 | full 100 practice problems seed1, FFN row scaling | qwen3-edit-ffnrow10 | 100 | pending |\n')
subprocess.run([sys.executable,h+'/eval_practice.py','--model','qwen3-edit-ffnrow10','--name','ffnrow10_seed1'],check=True)
with open(h+'/budget_log.md','a') as f:f.write('\n| 15 | non-math coherence check | qwen3-edit-ffnrow10 | 1 | pending |\n')
reply,nt,why=ask('qwen3-edit-ffnrow10','Write one sentence explaining why leaves look green.',128,1)
json.dump({'response':reply,'tokens':nt,'done_reason':why},open(h+'/ffn_coherence.json','w'),indent=2)
def records(name):return {r['unique_id']:r for r in map(json.loads,open(h+'/runs/'+name+'.jsonl'))}
b=records('roundtrip_seed1');e=records('ffnrow10_seed1');score=sum(r['correct'] for r in e.values())
s['ffn_edit']=t;s['ffn_correct']=score;s['ffn_modes']=dict(Counter(r['mode'] for r in e.values()));s['ffn_wins']=[i for i in b if not b[i]['correct'] and e[i]['correct']];s['ffn_losses']=[i for i in b if b[i]['correct'] and not e[i]['correct']]
if score>max(s['baseline_correct'],s['edit_correct']):
 s['best_gguf']=t['gguf'];s['final_changed_parameters']=t['changed_parameters'];register('qwen3-edit-best',t['gguf'])
s['forward_passes']=15
json.dump(s,open(h+'/summary.json','w'),indent=2)
subprocess.run(['ollama','rm','qwen3-edit-ffnrow10'],check=True)
print(json.dumps(s,indent=2),flush=True)
