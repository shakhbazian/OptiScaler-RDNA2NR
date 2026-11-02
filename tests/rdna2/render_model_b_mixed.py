"""Build a local mixed-model comparison without replacing the earlier A/B archive."""
import argparse,json,shutil,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from model_b_mixed import ROOT,DATA,OUT
from render_model_ab import rgb,png,SCENES

def main():
 p=argparse.ArgumentParser();p.add_argument('--assets-only',action='store_true');p.add_argument('--video',action='store_true');a=p.parse_args()
 dest=DATA/'comparison/mixed';dest.mkdir(parents=True,exist_ok=True)
 variants=['dynamic','static-large','static','tail','edges','balanced']
 scenes=[]
 for name,title in SCENES:
  original=DATA/f'ab/{name}';meta=json.loads((original/'input.json').read_text());w,h=meta['width'],meta['height'];folder=dest/name;folder.mkdir(exist_ok=True)
  source=np.asarray(Image.open(original/'source.png').convert('RGB'),dtype='f4')/255
  ref=rgb(original/'b/0.rgba.f16',w,h)
  for key,path in [('source',None),('fp16',original/'b/0.rgba.f16'),('release',original/'release/0.rgba.f16')]+[(k,OUT/name/k/'0.rgba.f16') for k in variants]:
   if path is not None and not path.exists():continue
   x=source if path is None else rgb(path,w,h)
   png(x,folder/f'{key}.png');png(.5+4*(x-source),folder/f'{key}-edit.png');png(.5+8*(x-ref),folder/f'{key}-error.png')
  scenes.append(dict(id=name,title=title,width=w,height=h,controls=meta['controls']))
 if a.video:
  folder=dest/'video';folder.mkdir(exist_ok=True)
  for key,file in [('source','source'),('fp16','b')]:shutil.copyfile(DATA/f'comparison/video/{file}.mp4',folder/f'{key}.mp4')
  for variant in variants:
   if not (OUT/'dialogue'/variant/'59.rgba.f16').exists():continue
   output=folder/f'{variant}.mp4'
   if output.exists():continue
   frames=OUT/'dialogue'/variant/'png';frames.mkdir(exist_ok=True)
   for i in range(60):png(rgb(OUT/'dialogue'/variant/f'{i}.rgba.f16',1280,720),frames/f'{i:03}.png')
   subprocess.run([shutil.which('ffmpeg'),'-hide_banner','-loglevel','error','-y','-framerate','30','-i',str(frames/'%03d.png'),'-frames:v','60','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],check=True)
 # Native portrait crops; no enlargement or additional image processing.
 canvas=Image.new('RGB',(420*4,565),(15,18,24));draw=ImageDraw.Draw(canvas)
 for i,(key,label) in enumerate([('source','Source'),('fp16','B FP16'),('balanced','B mixed / balanced'),('static','B mixed / static C32')]):
  canvas.paste(Image.open(dest/f'portrait/{key}.png').crop((220,110,640,650)),(i*420,25));draw.text((i*420+8,6),label,fill='white')
 canvas.save(dest/'portrait-crops.png')
 if not a.assets_only:
  result={'scenes':scenes,'quality':json.loads((OUT/'quality.json').read_text()),'timings':json.loads((OUT/'paired/results.json').read_text())}
  (dest/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
  template=Path(__file__).with_name('model_b_gallery.html').read_text(encoding='utf-8')
  (dest/'index.html').write_text(template.replace('__DATA__',json.dumps(result,ensure_ascii=False)),encoding='utf-8')
 print(dest,flush=True)
if __name__=='__main__':main()
