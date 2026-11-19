"""Fit one still, editable motion grids and local appearance modes to an MP4."""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
GRID_STEP = 0.8  # canvas pixels per stored signed unit
CORRECTION_STEP = 2
TILE_SIZE = 124


def mesh(width, height, step):
    cols, rows = math.ceil((width - 1) / step) + 1, math.ceil((height - 1) / step) + 1
    return np.meshgrid(np.linspace(0, width - 1, cols, dtype=np.float32),
                       np.linspace(0, height - 1, rows, dtype=np.float32))


def remap(image, x, y):
    return cv2.remap(image, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def reconstruct(base, grid, x, y):
    h, w = base.shape[:2]
    rows, cols = grid.shape[:2]
    offsets = remap(grid.astype(np.float32) * GRID_STEP,
                    x * ((cols - 1) / (w - 1)), y * ((rows - 1) / (h - 1)))
    return remap(base, x + offsets[:, :, 0], y + offsets[:, :, 1])


class OpticalFit:
    def __init__(self, base):
        h, w = base.shape[:2]
        self.size = w, h
        self.small_size = w // 2, h // 2
        self.x, self.y = mesh(*self.small_size, 1)
        self.reference = self.gray(base)
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def gray(self, image):
        luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.resize(luma, self.small_size, interpolation=cv2.INTER_AREA)

    def solve(self, frame):
        # Backward flow maps each output coordinate to the shared still. A second
        # solve on the first warp sharpens contours without accumulating drift.
        target = self.gray(frame)
        first = self.flow.calc(target, self.reference, None)
        warped = remap(self.reference, self.x + first[..., 0], self.y + first[..., 1])
        second = self.flow.calc(target, warped, None)
        combined = second + remap(first, self.x + second[..., 0], self.y + second[..., 1])
        full = cv2.resize(cv2.GaussianBlur(combined, (0, 0), .9),
                          self.size, interpolation=cv2.INTER_LINEAR)
        full *= np.asarray(self.size, dtype=np.float32) / self.small_size
        return full


def fit_appearance(residuals, rank, tile):
    """Local PCA on color residuals, not stored replacement frames."""
    n, h, w, channels = residuals.shape
    basis = np.empty((rank, h, w, channels), np.int8)
    coefficients = []
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            area = residuals[:, y:y+tile, x:x+tile].astype(np.float32)
            observations = area.reshape(n, -1)
            sample = area[:, ::2, ::2].copy().reshape(n, -1)
            _, vectors = np.linalg.eigh(sample @ sample.T)
            keys = np.ascontiguousarray(vectors[:, -rank:] * np.float32(math.sqrt(n)))
            modes = keys.T @ observations / n
            scale = np.maximum(np.percentile(np.abs(modes), 99.95, axis=1) / 120, .005).astype(np.float32)
            quantized = np.rint(modes / scale[:, None]).clip(-128, 127).astype(np.int8)
            basis[:, y:y+tile, x:x+tile] = quantized.reshape((rank, *area.shape[1:]))
            coefficients.append(keys * scale)
    return basis, np.stack(coefficients, axis=1).astype("<f4")


def appearance_at(basis, weights, tile):
    h, w = basis.shape[1:3]
    plane = np.empty((h, w, 3), np.float32)
    i = 0
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            part = basis[:, y:y+tile, x:x+tile].astype(np.float32)
            values = weights[i] @ part.reshape(part.shape[0], -1)
            plane[y:y+tile, x:x+tile] = values.reshape(part.shape[1:])
            i += 1
    return np.rint(plane / CORRECTION_STEP).clip(-128, 127) * CORRECTION_STEP


def write_gzip(path, *blocks):
    with path.open("wb") as output, gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as gz:
        for block in blocks:
            gz.write(block.tobytes())
    return {"file": path.name, "bytes": path.stat().st_size}


def grid_asset(assets, name, grids):
    data = np.stack(grids).view(np.uint8) ^ np.uint8(128)
    data[1:] -= data[:-1].copy()  # modulo 256 temporal prediction
    return {**write_gzip(assets / f"fit-{name}.cbfit", data),
            "rows": data.shape[1], "cols": data.shape[2], "unitPx": GRID_STEP}


def error(prediction, frame):
    return float(cv2.absdiff(prediction, frame).mean())


def summary(values):
    return {"mae": round(float(np.mean(values)), 4),
            "maxMae": round(max(values), 4),
            "perFrame": [round(v, 4) for v in values]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "public/assets",
                        help="directory containing reference.mp4, base.webp and timeline.json")
    parser.add_argument("--rank", type=int, choices=range(1, 33), metavar="1..32", default=24)
    args = parser.parse_args()
    cv2.setNumThreads(4)
    assets = args.assets
    video = assets / "reference.mp4"
    still = assets / "base.webp"
    timeline = assets / "timeline.json"
    legacy = json.loads(timeline.read_text(encoding="utf-8"))
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Cannot decode {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    base = np.array(Image.open(still).convert("RGB"))
    h, w = base.shape[:2]
    if (w, h, fps, frame_count) != (legacy["width"], legacy["height"], legacy["fps"], legacy["frameCount"]):
        raise SystemExit("Base, source video and legacy timeline must describe the same clip.")
    if frame_count < args.rank:
        raise SystemExit("Rank exceeds the number of frames.")

    x, y = mesh(w, h, 1)
    sample_grids = {"coarse": mesh(w, h, 48), "dense": mesh(w, h, 8)}
    grids = {name: [] for name in sample_grids}
    errors = {"still": [], "coarse": [], "dense": [], "refined": []}
    fit = OpticalFit(base)
    frames, predictions, residuals = [], [], []
    residual_size = ((w + 1) // 2, (h + 1) // 2)

    for index in range(frame_count):
        ok, bgr = cap.read()
        if not ok or bgr.shape[:2] != (h, w):
            raise SystemExit(f"Invalid full-resolution frame {index}")
        frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if index == 0:
            if error(frame, base) > .01:
                raise SystemExit("Use the clip's original frame-0 base.webp.")
        frames.append(frame)
        field = fit.solve(frame)
        errors["still"].append(error(base, frame))
        for name, (gx, gy) in sample_grids.items():
            grid = np.rint(remap(field, gx, gy) / GRID_STEP).clip(-128, 127).astype(np.int8)
            grids[name].append(grid)
            prediction = reconstruct(base, grid, x, y)
            errors[name].append(error(prediction, frame))
            if name == "dense":
                predictions.append(prediction)
                residual = frame.astype(np.float32) - prediction
                residuals.append(cv2.resize(residual, residual_size,
                                            interpolation=cv2.INTER_AREA).astype(np.float16))
        if index % 24 == 0:
            print(f"Fit {index + 1}/{frame_count}", flush=True)
    cap.release()

    rank = args.rank
    basis, weights = fit_appearance(np.stack(residuals), rank, TILE_SIZE)
    for frame, prediction, key in zip(frames, predictions, weights):
        correction = cv2.resize(appearance_at(basis, key, TILE_SIZE), (w, h),
                                interpolation=cv2.INTER_LINEAR)
        result = np.rint(prediction + correction).clip(0, 255).astype(np.uint8)
        errors["refined"].append(error(result, frame))

    spatial_delta = basis.view(np.uint8) ^ np.uint8(128)
    spatial_delta[:, :, 1:] -= spatial_delta[:, :, :-1].copy()
    correction_file = write_gzip(assets / "fit-correction.cbfit", weights, spatial_delta)
    details = {name: grid_asset(assets, name, controls) for name, controls in grids.items()}
    report = {
        "version": 1, "width": w, "height": h, "fps": fps,
        "frameCount": frame_count, "duration": frame_count / fps,
        "base": {"file": still.name, "bytes": still.stat().st_size},
        "reference": {"file": video.name, "bytes": video.stat().st_size, "codec": "H.264"},
        "brush": {"file": timeline.name, "bytes": timeline.stat().st_size,
                  "count": len(legacy["commands"])},
        "grids": details,
        "correction": {**correction_file, "width": residual_size[0], "height": residual_size[1],
                       "rank": rank, "tile": TILE_SIZE, "tileCount": weights.shape[1],
                       "unitRgb": CORRECTION_STEP},
        "quality": {name: summary(values) for name, values in errors.items()},
    }
    (assets / "fit.json").write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                                    encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in values.items() if k != "perFrame"}
                      for name, values in report["quality"].items()}, indent=2), flush=True)
    print("Generated", *(f"{item['file']}={item['bytes']} bytes"
                         for item in [*details.values(), correction_file]), sep="\n")


if __name__ == "__main__":
    main()
