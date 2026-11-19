"""Measure full-resolution reconstruction error of the locked WebGL base.

It emulates src/webgl.js: the frame-0 base is never updated, while the current
reference frame replaces pixels selected by the feathered motion mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def smoothstep(low, high, x):
    t = np.clip((x - low) / (high - low), 0, 1)
    return t * t * (3 - 2 * t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--assets", type=Path, default=Path("public/assets"))
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--out", type=Path, default=Path("analysis/quality.json"))
    args = ap.parse_args()

    base = np.asarray(Image.open(args.assets / "base.webp").convert("RGB"), np.float32)
    mask = np.asarray(Image.open(args.assets / "motion-mask.webp").convert("L"), np.float32) / 255.0
    alpha = smoothstep(0.055, 0.145, mask)[..., None]
    locked = alpha[..., 0] == 0

    cap = cv2.VideoCapture(str(args.video))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    indices = np.unique(np.linspace(0, count - 1, min(args.samples, count)).round().astype(int))
    first = None
    results = []
    for frame_number in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
        ok, bgr = cap.read()
        if not ok:
            continue
        reference = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        if first is None:
            first = reference
        output = base * (1 - alpha) + reference * alpha
        error = np.abs(output - reference)
        results.append({
            "frame": int(frame_number),
            "time": round(float(frame_number) / fps, 6),
            "mae": round(float(error.mean()), 5),
            "p95": round(float(np.percentile(error, 95)), 5),
            "lockedReferenceDrift": round(float(np.abs(reference[locked] - first[locked]).mean()), 5),
        })
    cap.release()

    summary = {
        "sourceFrames": count,
        "fps": fps,
        "sampledFrames": len(results),
        "width": base.shape[1],
        "height": base.shape[0],
        "shaderLockedPixelRatio": round(float(locked.mean()), 5),
        "meanMae": round(float(np.mean([r["mae"] for r in results])), 5),
        "maxMae": round(float(np.max([r["mae"] for r in results])), 5),
        "meanLockedReferenceDrift": round(float(np.mean(
            [r["lockedReferenceDrift"] for r in results])), 5),
        "baseTextureUpdates": 0,
        "frames": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
