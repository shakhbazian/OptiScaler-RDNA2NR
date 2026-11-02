"""Model B mixed research driver: immutable reference outputs, explicit execution builds."""
import argparse, json, os, subprocess, sys, struct, hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_model_ab import extent
ROOT=Path(__file__).resolve().parents[2]
DATA=Path('D:/TEMP/DLSSNR-RDNA2/research/nr3109-20261017')
OUT=DATA/'mixed-b'
SOURCE='2BC19BC09A7B89DD44F73E2AC0108FCC586E2154BDD0830F242F4F040E28789F'
HASH={'a':'5C2B4272542C4CAF3B2E16A1242F21484B3BE2F8DC31337ECCCEA93143BFAE2C','b':'2CE6F8BB07F18399A58120F9FB2F1152DD39D6B872370DA7EE61B104E4165AE7'}
def run(exe, model, corpus, dest, extra=()):
    dest.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['PATH']='C:/Program Files/AMD/ROCm/6.4/bin;'+env['PATH']
    args=[str(exe),str(DATA/f'model-{model}.nrwgt'),HASH[model],SOURCE,str(corpus),str(dest),*extra]
    with (dest/'run.log').open('w') as log:
        result=subprocess.run(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f'{dest}: exit {result.returncode}; '+(dest/'run.log').read_text()[-1600:])
    print(str(dest),flush=True)

def prepare_calibration():
    corpus=Path('D:/TEMP/DLSSNR-RDNA2/fixtures/video_corpus_v2/calibration')
    selected=[]
    for prefix,count in [('2ND_',12),('rp4',12),('sJ4',8)]:
        files=sorted(p for p in corpus.glob(f'{prefix}*/540p-3s.png') if '0410' not in str(p))
        if len(files)<count: raise RuntimeError(f'No calibration images: {prefix}, {len(files)}')
        selected += [files[i] for i in np.linspace(0,len(files)-1,count,dtype=int)]
    dest=OUT/'calibration';dest.mkdir(parents=True,exist_ok=True)
    w,h=960,540;nw,nh=extent(2,w,h)
    records=[]
    with (dest/'b.bin').open('wb') as out:
        out.write(struct.pack('<8s5I',b'NRQUAL01',len(selected)*2,h,w,nh,nw))
        for image in selected:
            im=Image.open(image).convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
            rgba=np.ones((h,w,4),dtype='<f2');rgba[:,:,:3]=np.asarray(im,dtype=np.float32)/255
            for controls in [(0,1,1,-1,-1,1,0,0),(1,1.5,1.5,-1,-1,1,0,0)]:
                out.write(struct.pack('<4I8f',0,0,0,0,*controls));out.write(rgba.tobytes());out.write(bytes(h*w*12))
            records.append({'path':str(image),'sha256':hashlib.sha256(image.read_bytes()).hexdigest()})
    (dest/'inputs.json').write_text(json.dumps(records,indent=2))
    print(f'Prepared {len(selected)} scenes, 64 frames',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','calibrate','run']);p.add_argument('--build',default='dynamic');p.add_argument('--scenes',nargs='+',default=['landscape540','portrait','street','chloe','controls','dialogue']);a=p.parse_args()
    if a.action=='prepare':prepare_calibration()
    elif a.action=='calibrate':run(ROOT/'build/model-b-mixed/reference/model_ab_fp16.exe','b',OUT/'calibration/b.bin',OUT/'calibration/reference',['--calibrate','--images-only','--first-frame-only'])
    else:
        for scene in a.scenes:
            extra=['--images-only']+([] if scene in ['dialogue','controls'] else ['--first-frame-only'])
            run(ROOT/f'build/model-b-mixed/{a.build}/model_ab_mixed.exe','b',DATA/f'ab/{scene}/b.bin',OUT/scene/a.build,extra)
