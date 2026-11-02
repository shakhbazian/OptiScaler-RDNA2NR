"""Build/check a control sweep and temporal reset on one fixed image."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np

CASES = [
    ('standard', (0, 1, 1, -1, -1, 1, 0, 0), 0),
    ('style', (1, 1, 1, -1, -1, 1, 0, 0), 0),
    ('tone', (0, 1.5, 1, -1, -1, 1, 0, 0), 0),
    ('structure', (0, 1, 1.5, -1, -1, 1, 0, 0), 0),
    ('automatic_skin', (0, 1, 1, 0.5, 1, 1, 0, 0), 0),
    ('intensity_zero', (0, 1, 1, -1, -1, 0, 0, 0), 0),
    ('history', (0, 1, 1, -1, -1, 1, 0, 0), 1),
    ('reset', (0, 1, 1, -1, -1, 1, 0, 0), 0),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--prepare-from', type=Path)
    args = p.parse_args()
    if args.prepare_from:
        args.root.mkdir(parents=True, exist_ok=True)
        for model in ('a', 'b'):
            raw = (args.prepare_from / (model + '.bin')).read_bytes()
            _, _, h, w, nh, nw = struct.unpack('<8s5I', raw[:28])
            pixels = raw[76:76 + h*w*20]
            with (args.root / (model + '.bin')).open('wb') as out:
                out.write(struct.pack('<8s5I', b'NRQUAL01', len(CASES), h, w, nh, nw))
                for _, controls, history in CASES:
                    out.write(struct.pack('<4I8f', history, 0, 0, 0, *controls))
                    out.write(pixels)
        return
    results = {}
    for model in ('a', 'b'):
        raw = (args.root / (model + '.bin')).read_bytes()
        _, _, h, w, _, _ = struct.unpack('<8s5I', raw[:28])
        source = np.frombuffer(raw[76:76+h*w*8], '<f2').reshape(h,w,4)
        outputs = [np.fromfile(args.root/model/f'{i}.rgba.f16', '<f2').reshape(h,w,4) for i in range(len(CASES))]
        assert all(np.isfinite(x).all() for x in outputs)
        assert np.array_equal(outputs[0], outputs[-1]), 'history reset did not restore first-frame output'
        assert np.array_equal(outputs[5], source), 'zero intensity is not exact bypass'
        delta = {name: float(np.mean(np.abs(x[:,:,:3].astype(float)-outputs[0][:,:,:3].astype(float)))*255)
                 for (name, _, _), x in zip(CASES, outputs)}
        assert all(delta[name] > 0 for name in ('style','tone','structure','automatic_skin','history'))
        results[model] = {'mean_rgb_change_255': delta, 'finite': True, 'reset_bitexact': True, 'intensity_zero_bitexact': True}
    (args.root/'checks.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
