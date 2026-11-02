"""Paired uninstrumented full-frame measurements, ordered forward then reverse."""
import argparse,csv,json,struct,hashlib
from pathlib import Path
import numpy as np
from model_b_mixed import ROOT,DATA,OUT,run

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',default='static');p.add_argument('--rounds',type=int,default=2);a=p.parse_args()
 exe={'release':ROOT/'build/model-ab/model_ab_mixed.exe','fp16':ROOT/'build/model-ab/model_ab_fp16.exe',a.candidate:ROOT/f'build/model-b-mixed/{a.candidate}/model_ab_mixed.exe'}
 exe.update({key:ROOT/f'build/model-b-mixed/{key}/model_ab_mixed.exe' for key in ['dynamic','balanced']})
 results={}
 for scene in ['landscape540','street','portrait','chloe']:
  dest=OUT/'paired'/scene;dest.mkdir(parents=True,exist_ok=True)
  for model in ['a','b']:
   with (DATA/f'ab/{scene}/{model}.bin').open('rb') as f:
    magic,n,h,w,nh,nw=struct.unpack('<8s5I',f.read(28));record=f.read(48+h*w*20)
   with (dest/f'{model}.bin').open('wb') as f:
    f.write(struct.pack('<8s5I',magic,24,h,w,nh,nw))
    for _ in range(24):f.write(record)
  samples={key:[] for key in exe};runs={key:[] for key in exe}
  for round in range(a.rounds):
   for i,key in enumerate([*exe,*reversed(exe)]):
    folder=dest/f'{round}-{i}-{key}';model='a' if key=='release' else 'b'
    run(exe[key],model,dest/f'{model}.bin',folder,['--first-frame-only']+([] if key=='release' else ['--images-only']))
    reference=(DATA/f'ab/{scene}/release/0.rgba.f16') if key=='release' else (DATA/f'ab/{scene}/b/0.rgba.f16') if key=='fp16' else (OUT/scene/key/'0.rgba.f16')
    assert (folder/'0.rgba.f16').read_bytes()==reference.read_bytes(),f'Output drift: {scene}/{key}'
    rows=list(csv.DictReader((folder/'timings.csv').open()))[4:]
    assert all(int(r['profile'])==0 for r in rows)
    ms=[float(r['stream_ms']) for r in rows];samples[key]+=ms;runs[key].append(float(np.median(ms)))
  results[scene]={key:{'median_ms':float(np.median(v)),'p95_ms':float(np.percentile(v,95)),'samples':len(v),'run_medians_ms':runs[key],'exe_sha256':hashlib.sha256(exe[key].read_bytes()).hexdigest()} for key,v in samples.items()}
  (OUT/'paired/results.json').write_text(json.dumps(results,indent=2))
  print(json.dumps({scene:results[scene]},indent=2),flush=True)
if __name__=='__main__':main()
