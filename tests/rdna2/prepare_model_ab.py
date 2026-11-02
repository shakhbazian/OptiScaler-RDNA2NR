"""Prepare identical RGB inputs with each model's required working extent."""
import argparse
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def extent(version, w, h):
    up = lambda n, a: (n + a - 1) // a * a
    if version == 1:
        return up(max(320, w), 64), up(max(320, h), 64)
    ax = 128 if up((w + 1) // 2, 4) % 8 == 4 else 64
    ay = 128 if up((h + 1) // 2, 4) % 8 == 4 else 64
    nw, nh = max(320, up(w, ax)), max(320, up(h, ay))
    if nh % (ay * 4) == 0 and nw % (ax * 4) == 0:
        nw += ax
    return nw, nh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--height', type=int)
    p.add_argument('--frames', type=int, default=5)
    p.add_argument('--temporal', action='store_true')
    p.add_argument('--standard', action='store_true')
    p.add_argument('--controls', type=float, nargs=8)
    p.add_argument('--sequence', action='store_true')
    args = p.parse_args()
    inputs = sorted(args.image.glob('*.png'))[:args.frames] if args.sequence else [args.image] * args.frames
    if not inputs:
        raise ValueError('empty input sequence')
    args.frames = len(inputs)
    image = Image.open(inputs[0]).convert('RGB')
    if args.height:
        image = image.resize((round(image.width * args.height / image.height), args.height), Image.Resampling.LANCZOS)
    args.output.mkdir(parents=True, exist_ok=True)
    image.save(args.output / 'source.png')
    w, h = image.size
    rgba = np.ones((h, w, 4), dtype='<f2')
    rgba[:, :, :3] = np.asarray(image, dtype=np.float32) / 255
    motion = np.zeros((h, w, 2), dtype='<f2')
    mask = np.zeros_like(rgba)
    controls = (0, 1, 1, -1, -1, 1, 0, 0) if args.standard else (1, 1.5, 1.5, -1, -1, 1, 0, 0)
    controls = args.controls or controls
    for version, name in ((1, 'a'), (2, 'b')):
        nw, nh = extent(version, w, h)
        with (args.output / (name + '.bin')).open('wb') as out:
            out.write(struct.pack('<8s5I', b'NRQUAL01', args.frames, h, w, nh, nw))
            for frame in range(args.frames):
                if args.sequence:
                    im = Image.open(inputs[frame]).convert('RGB').resize((w, h), Image.Resampling.LANCZOS)
                    rgba[:, :, :3] = np.asarray(im, dtype=np.float32) / 255
                out.write(struct.pack('<4I8f', int(args.temporal and frame > 0), 0, 0, 0, *controls))
                out.write(rgba.tobytes())
                out.write(motion.tobytes())
                out.write(mask.tobytes())
    (args.output / 'input.json').write_text(json.dumps({
        'source': str(args.image), 'width': w, 'height': h, 'frames': args.frames,
        'temporal': args.temporal, 'controls': controls,
        'a_extent': extent(1, w, h), 'b_extent': extent(2, w, h),
    }, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
