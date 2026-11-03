"""Rebuild the selected final GGUF without inference. --register creates only best."""
import json,os,sys,subprocess
from gguf_edit import Model,write_variant,GGUF_DIR
here=os.path.dirname(os.path.abspath(__file__))
s=json.load(open(here+'/summary.json')); t=s['trace']; replacements={}
if s['final_changed_parameters']==1:
    w=Model().deq('output_norm.weight');assert float(w[t['dimension']])==t['old']
    w[t['dimension']]=t['new'];replacements['output_norm.weight']=w
if s['best_gguf'].endswith('ffnrow10.gguf'):
    subprocess.run([sys.executable,here+'/ffn_edit.py'],cwd=os.path.dirname(here),check=True)
    out=GGUF_DIR+'/ffnrow10.gguf'
else:
    out=GGUF_DIR+'/best.gguf';write_variant(out,replacements)
mf=here+'/Modelfile.qwen3-edit-best'
open(mf,'w').write('FROM '+out+'\n'+open(here+'/template.txt').read())
if '--register' in sys.argv:subprocess.run(['ollama','create','qwen3-edit-best','-f',mf],check=True)
print(out)
