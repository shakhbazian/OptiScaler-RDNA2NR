"""Quality checks target the NR edit as well as the final RGB image."""
import argparse,json,struct
from pathlib import Path
import numpy as np
from model_b_mixed import ROOT,DATA,OUT
from model_ab_controls import CASES

def rgb(path,h=None,w=None):
 x=np.fromfile(path,'<f2').astype('f4').reshape(-1,4)
 assert np.isfinite(x).all(),path
 return x[:,:3]
def correlation(a,b):
 a=a.ravel();b=b.ravel();am=float(np.mean(a));bm=float(np.mean(b))
 cov=float(np.mean(a*b))-am*bm
 den=max((float(np.mean(a*a))-am*am)*(float(np.mean(b*b))-bm*bm),1e-30)
 return cov/np.sqrt(den)
def metrics(ref,out,src):
 error=out-ref;edit=ref-src;other=out-src
 mse=float(np.mean(error**2));energy=float(np.sum(edit.astype('f8')**2))
 return {'mae_255':float(np.abs(error).mean()*255),'psnr_db':float(-10*np.log10(max(mse,1e-30))),
         'edit_rms_255':float(np.sqrt(np.mean(other**2))*255),
         'edit_projection':float(np.sum(other.astype('f8')*edit)/max(energy,1e-30)),
         'edit_correlation':float(correlation(edit,other)) if energy else 1.}
def read_source(corpus,frame):
 with corpus.open('rb') as f:
  _,n,h,w,_,_=struct.unpack('<8s5I',f.read(28));f.seek(28+frame*(48+h*w*20)+48)
  src=np.frombuffer(f.read(h*w*8),'<f2').astype('f4').reshape(-1,4)[:,:3]
 return src

def main():
 p=argparse.ArgumentParser();p.add_argument('--variants',nargs='+',default=['dynamic','static-large','static']);a=p.parse_args();result=json.loads((OUT/'quality.json').read_text()) if (OUT/'quality.json').exists() else {}
 for variant in a.variants:
  result[variant]={}
  for scene in ['portrait','chloe','street','landscape540','holdout','dialogue','controls1080']:
   folder=OUT/scene/variant
   if not folder.exists():continue
   reference=(OUT/scene/'fp16') if scene in ['holdout','controls1080'] else DATA/f'ab/{scene}/b'
   corpus=OUT/scene/'b.bin' if scene in ['holdout','controls1080'] else DATA/f'ab/{scene}/b.bin'
   files=sorted(folder.glob('*.rgba.f16'),key=lambda p:int(p.name.split('.')[0]));results=[]
   for path in files:
    frame=int(path.name.split('.')[0]);src=read_source(corpus,frame);ref=rgb(reference/path.name);out=rgb(path)
    results.append(dict(frame=frame,**metrics(ref,out,src)))
   if scene=='controls1080':
    outputs=[rgb(folder/f'{i}.rgba.f16') for i in range(8)];refs=[rgb(reference/f'{i}.rgba.f16') for i in range(8)]
    assert np.array_equal(outputs[0],outputs[7]),'reset must restore first-frame result'
    assert np.array_equal(outputs[5],read_source(corpus,5)),'zero intensity must bypass exactly'
    responses={name:metrics(refs[0],outputs[i]-outputs[0]+refs[0],refs[0]) for i,(name,_,_) in enumerate(CASES)}
    for i in [1,2,3,4,6]:
     ref=refs[i]-refs[0];delta=outputs[i]-outputs[0]
     responses[CASES[i][0]]={'rms_255':float(np.sqrt(np.mean(delta**2))*255),'reference_rms_255':float(np.sqrt(np.mean(ref**2))*255),'direction_correlation':float(correlation(ref,delta))}
     assert np.any(delta)
    result[variant]['control_checks']={'reset_exact':True,'zero_exact':True,'responses':responses}
   if scene=='dialogue':
    temporal=[]
    for frame in range(1,60):
     ro=rgb(reference/f'{frame}.rgba.f16')-rgb(reference/f'{frame-1}.rgba.f16')
     mo=rgb(folder/f'{frame}.rgba.f16')-rgb(folder/f'{frame-1}.rgba.f16')
     temporal.append(float(np.abs(ro-mo).mean()*255))
    result[variant]['temporal_change_mae_255']={'median':float(np.median(temporal)),'max':max(temporal)}
   result[variant][scene]=results
 (OUT/'quality.json').write_text(json.dumps(result,indent=2))
 for variant,data in result.items():
  print(variant)
  for scene in ['portrait','chloe','street','landscape540','holdout','dialogue']:
   rows=data.get(scene,[])
   if rows:print(scene,'MAE range',min(r['mae_255'] for r in rows),max(r['mae_255'] for r in rows),'edit projection',min(r['edit_projection'] for r in rows),max(r['edit_projection'] for r in rows))
if __name__=='__main__':main()
