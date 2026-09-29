"""Render the source-built host's FP16 readbacks for a human A/B check."""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_half(path: Path, width: int, height: int) -> np.ndarray:
    pixels = np.fromfile(path, dtype="<f2")
    if pixels.size != width * height * 4:
        raise ValueError(f"Unexpected RGBA16F size: {path}")
    return pixels.reshape(height, width, 4).astype(np.float32)


def display_rgb(pixels: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(pixels[..., :3], 0, 1) * 255).astype(np.uint8)


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit("usage: script raw.rgba.f16 nr.rgba.f16 width height output-dir")
    raw_path, nr_path = map(Path, sys.argv[1:3])
    width, height = map(int, sys.argv[3:5])
    output = Path(sys.argv[5])
    output.mkdir(parents=True, exist_ok=True)
    raw = read_half(raw_path, width, height)
    nr = read_half(nr_path, width, height)
    difference = np.abs(nr[..., :3] - raw[..., :3])
    raw_image = Image.fromarray(display_rgb(raw))
    nr_image = Image.fromarray(display_rgb(nr))
    raw_image.save(output / "raw.png")
    nr_image.save(output / "nr.png")
    Image.fromarray(np.rint(np.clip(difference * 8, 0, 1) * 255).astype(np.uint8)).save(
        output / "difference-x8.png"
    )
    comparison = Image.new("RGB", (width * 2, height + 28), "#202020")
    comparison.paste(raw_image, (0, 28))
    comparison.paste(nr_image, (width, 28))
    labels = ImageDraw.Draw(comparison)
    labels.text((12, 7), "FSR, NR off", fill="white")
    labels.text((width + 12, 7), "FSR + migrated RDNA2 NR", fill="white")
    comparison.save(output / "comparison.png")
    # Keep the facial region at native pixels; a scaled montage hides the effect.
    face = (750, 125, 1190, 535)
    face_comparison = Image.new("RGB", ((face[2] - face[0]) * 2, face[3] - face[1]))
    face_comparison.paste(raw_image.crop(face), (0, 0))
    face_comparison.paste(nr_image.crop(face), (face[2] - face[0], 0))
    face_comparison.save(output / "face-comparison.png")
    metrics = {
        "width": width,
        "height": height,
        "meanAbsoluteRgb": float(difference.mean()),
        "p95AbsoluteRgb": float(np.percentile(difference, 95)),
        "maxAbsoluteRgb": float(difference.max()),
        "rawPath": str(raw_path),
        "nrPath": str(nr_path),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
