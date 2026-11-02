"""Independent scenes and extended controls for B mixed quality validation."""
import argparse,json,struct,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from model_b_mixed import ROOT,DATA,OUT,run
from prepare_model_ab import extent

def prepare():
 corpus=Path('D:/TEMP/DLSSNR-RDNA2/fixtures/video_corpus_v2/holdout')
 selected=[]
 for prefix in ['2ND_','rp4','sJ4']:
  files=sorted(corpus.glob(f'{prefix}*/540p-3s.png'));n=len(files)
  selected += [files[i] for i in np.linspace(0,len(files)-1,n,dtype=int)]
 assert len(selected)>=18
 dest=OUT/'holdout';dest.mkdir(parents=True,exist_ok=True);w,h=960,540;nw,nh=extent(2,w,h);records=[]
 with (dest/'b.bin').open('wb') as out:
  out.write(struct.pack('<8s5I',b'NRQUAL01',len(selected)*2,h,w,nh,nw))
  for image in selected:
   im=Image.open(image).convert('RGB').resize((w,h),Image.Resampling.LANCZOS);rgba=np.ones((h,w,4),dtype='<f2');rgba[:,:,:3]=np.asarray(im,dtype='f4')/255
   for controls in [(0,1,1,-1,-1,1,0,0),(1,1.5,1.5,-1,-1,1,0,0)]:
    out.write(struct.pack('<4I8f',0,0,0,0,*controls));out.write(rgba.tobytes());out.write(bytes(h*w*12))
   records.append({'path':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest()})
 (dest/'inputs.json').write_text(json.dumps(records,indent=2));print('Holdout',len(records),flush=True)
 # The same full-resolution portrait with six controls, history, and reset.
 from model_ab_controls import CASES
 dest=OUT/'controls1080';dest.mkdir(parents=True,exist_ok=True)
 with (DATA/'ab/portrait/b.bin').open('rb') as f:
  magic,_,h,w,nh,nw=struct.unpack('<8s5I',f.read(28));f.read(48);pixels=f.read(h*w*20)
 with (dest/'b.bin').open('wb') as f:
  f.write(struct.pack('<8s5I',magic,len(CASES),h,w,nh,nw))
  for _,controls,history in CASES:f.write(struct.pack('<4I8f',history,0,0,0,*controls));f.write(pixels)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);p.add_argument('--build',default='static');a=p.parse_args()
 if a.action=='prepare':prepare()
 else:
  exe=ROOT/'build/model-ab/model_ab_fp16.exe' if a.build=='fp16' else ROOT/f'build/model-b-mixed/{a.build}/model_ab_mixed.exe'
  for scene in ['holdout','controls1080']:
   run(exe,'b',OUT/scene/'b.bin',OUT/scene/a.build,['--images-only'])
