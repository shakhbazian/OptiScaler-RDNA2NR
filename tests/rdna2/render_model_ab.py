"""Render lossless A/B stills, video and a local comparison page from probe outputs."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SCENES = [('portrait', 'Сохранённый портрет · 854×1080'),
          ('chloe', 'Хлоя · 2560×1440'), ('street', 'Улица · 1280×720'),
          ('landscape540', 'Рабочий размер · 960×540')]


def rgb(path, w, h):
    x = np.fromfile(path, '<f2').reshape(h, w, 4).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f'Nonfinite output: {path}')
    return x[:, :, :3]


def png(x, path):
    Image.fromarray(np.rint(np.clip(x, 0, 1)*255).astype(np.uint8)).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--reference', type=Path)
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    scenes = []
    for name, title in SCENES:
        folder = args.root/name
        meta = json.loads((folder/'input.json').read_text())
        w,h = meta['width'],meta['height']
        destination = args.output/name
        destination.mkdir(exist_ok=True)
        shutil.copyfile(folder/'source.png',destination/'source.png')
        source = np.asarray(Image.open(folder/'source.png'),dtype=float)/255
        scene = dict(id=name,title=title,width=w,height=h,controls=meta['controls'],timings={},metrics={})
        for model in ('a','b','release'):
            x = rgb(folder/model/'0.rgba.f16', w, h)
            png(x,destination/(model+'.png'))
            png((x-source)*4+0.5,destination/(model+'-edit.png'))
            scene['metrics'][model] = {'edit_rms_255':float(np.sqrt(np.mean((x-source)**2))*255)}
            with (folder/model/'timings.csv').open() as f:
                rows = list(csv.DictReader(f))[2:]
            ms = [float(row['stream_ms']) for row in rows]
            scene['timings'][model] = {'median_ms':float(np.median(ms)), 'min_ms':min(ms),'max_ms':max(ms),'samples':len(ms)}
            for tensor in ('features.f16','head.f16','history.f32'):
                file = folder/model/('0.'+tensor)
                if file.exists():
                    dtype = '<f4' if tensor.endswith('f32') else '<f2'
                    if not np.isfinite(np.fromfile(file,dtype)).all():
                        raise ValueError(f'Nonfinite tensor: {file}')
        scenes.append(scene)
    ref = {}
    if args.reference:
        source = np.asarray(Image.open(args.reference/'ngx-input.png').convert('RGB'),dtype=float)
        destination = args.output/'reference'
        destination.mkdir(exist_ok=True)
        shutil.copyfile(args.reference/'ngx-input.png',destination/'source.png')
        for model in ('a','b'):
            x = np.rint(np.clip(rgb(args.root/'ngx1080'/model/'0.rgba.f16',1920,1080),0,1)*255)
            gold = np.asarray(Image.open(args.reference/f'ngx-{model}.png').convert('RGB'),dtype=float)
            diff = x-gold
            ref[model]={'psnr_db':float(20*np.log10(255/np.sqrt(np.mean(diff**2)))),
                        'mae_255':float(np.abs(diff).mean()),
                        'edit_correlation':float(np.corrcoef((x-source).ravel(),(gold-source).ravel())[0,1])}
            png(x/255,destination/f'our-{model}.png')
            shutil.copyfile(args.reference/f'ngx-{model}.png',destination/f'ngx-{model}.png')
    regression = {}
    for suffix in ('features.f16','head.f16','rgba.f16','history.f32'):
        files=[args.root/'portrait'/model/('0.'+suffix) for model in ('a','baseline')]
        hashes=[hashlib.sha256(file.read_bytes()).hexdigest() for file in files]
        if hashes[0]!=hashes[1]:
            raise ValueError('A changed against release source: '+suffix)
        regression[suffix]=hashes[0]
    controls = json.loads((args.root/'controls/checks.json').read_text())
    result={'scenes':scenes,'reference':ref,'a_regression':regression,'controls':controls}
    (args.output/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    # Pixel crops retain the native resolution. Labels stay outside the image.
    out=Image.new('RGB',(420*4,565),(20,23,28));draw=ImageDraw.Draw(out)
    for i,(model,label) in enumerate([('source','Source'),('a','A / FP16'),('b','B / FP16'),('release','Release / mixed')]):
        im=Image.open(args.output/'portrait'/(model+'.png')).crop((220,110,640,650))
        out.paste(im,(i*420,25));draw.text((i*420+10,7),label,fill='white')
    out.save(args.output/'portrait-crops.png')
    if args.video:
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:
            raise RuntimeError('ffmpeg not found')
        destination=args.output/'video';destination.mkdir(exist_ok=True)
        for model in ('source','a','b'):
            frames=args.root/'dialogue/input' if model=='source' else args.root/'dialogue'/model/'png'
            if model!='source':
                frames.mkdir(exist_ok=True)
                for i in range(60):
                    png(rgb(args.root/'dialogue'/model/f'{i}.rgba.f16',1280,720),frames/f'{i:03}.png')
            subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-y','-framerate','30','-i',str(frames/'%03d.png'),
                            '-frames:v','60','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',
                            str(destination/(model+'.mp4'))],check=True)
    template=Path(__file__).with_name('model_ab_gallery.html').read_text(encoding='utf-8')
    (args.output/'index.html').write_text(template.replace('__DATA__',json.dumps(result,ensure_ascii=False)),encoding='utf-8')
    print(json.dumps({'timings':{s['id']:s['timings'] for s in scenes},'reference':ref,'regression':'bit exact'},indent=2))


if __name__=='__main__':
    main()
