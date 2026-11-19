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
STABLE_SPACING = 6
MIN_AREA_RATIO = .22
MAX_EDGE_STRETCH = 2.5
LIMIT_PASSES = 42


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
    def __init__(self, base, full_resolution=False):
        h, w = base.shape[:2]
        self.size = w, h
        self.tracking_size = self.size if full_resolution else (w // 2, h // 2)
        self.sigma = 1.35 if full_resolution else .9
        self.x, self.y = mesh(*self.tracking_size, 1)
        self.reference = self.gray(base)
        self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        if full_resolution:
            self.flow.setFinestScale(0)

    def gray(self, image):
        luma = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.tracking_size == self.size:
            return luma
        return cv2.resize(luma, self.tracking_size, interpolation=cv2.INTER_AREA)

    def solve(self, frame):
        # Align each frame directly to the common still, then refine the first
        # warp instead of accumulating offsets from previous frames.
        target = self.gray(frame)
        first = self.flow.calc(target, self.reference, None)
        warped = remap(self.reference, self.x + first[..., 0], self.y + first[..., 1])
        second = self.flow.calc(target, warped, None)
        combined = second + remap(first, self.x + second[..., 0], self.y + second[..., 1])
        smoothed = cv2.GaussianBlur(combined, (0, 0), self.sigma)
        if self.tracking_size == self.size:
            return smoothed
        full = cv2.resize(smoothed, self.size, interpolation=cv2.INTER_LINEAR)
        return full * (np.asarray(self.size, dtype=np.float32) / self.tracking_size)


def quantize_grid(offsets):
    return np.rint(offsets / GRID_STEP).clip(-128, 127).astype(np.int8)


def cell_geometry(grid, width, height):
    """Area minima and edge-stretch maxima over cells of the bilinear warp."""
    offsets = grid.astype(np.float32) * GRID_STEP
    rows, cols = grid.shape[:2]
    du = np.diff(offsets, axis=1) * ((cols - 1) / (width - 1))
    dv = np.diff(offsets, axis=0) * ((rows - 1) / (height - 1))
    du[..., 0] += 1
    dv[..., 1] += 1
    top, bottom = du[:-1], du[1:]
    left, right = dv[:, :-1], dv[:, 1:]

    def det(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # The Jacobian determinant is affine within a bilinear cell, so its
    # extrema lie at the four corners. Horizontal and vertical edges bound
    # the stretch of axes throughout the cell.
    area = np.minimum.reduce([det(top, left), det(top, right),
                              det(bottom, left), det(bottom, right)])
    stretch_squared = np.maximum.reduce([np.sum(v*v, axis=-1)
                                         for v in (top, bottom, left, right)])
    return area, stretch_squared


def stabilize_grid(offsets, width, height):
    """Relax only suspect cells and their neighbors; keep coherent motion."""
    for _ in range(LIMIT_PASSES):
        grid = quantize_grid(offsets)
        area, stretch = cell_geometry(grid, width, height)
        bad = (area < MIN_AREA_RATIO) | (stretch > MAX_EDGE_STRETCH**2)
        if not bad.any():
            return grid
        # Mark all four corner nodes and taper the update into adjacent cells.
        nodes = np.pad(bad, ((0, 1), (0, 1)))
        nodes[1:] |= nodes[:-1].copy()
        nodes[:, 1:] |= nodes[:, :-1].copy()
        nodes = nodes.astype(np.float32)
        feather = cv2.GaussianBlur(nodes, (5, 5), .82)
        mix = np.maximum(.52 * nodes, .94 * feather)[..., None]
        smooth = cv2.GaussianBlur(offsets, (5, 5), 1.02)
        offsets = offsets + mix * (smooth - offsets)
    return quantize_grid(offsets)


def geometry_summary(grids, width, height):
    folds, stretches = [], []
    for grid in grids:
        area, stretch = cell_geometry(grid, width, height)
        folds.append(float((area <= 0).mean()))
        stretches.append(float((stretch > MAX_EDGE_STRETCH**2).mean()))
    return {"meanFoldedCellRatio": round(float(np.mean(folds)), 8),
            "maxFoldedCellRatio": round(max(folds), 8),
            "meanOverstretchedCellRatio": round(float(np.mean(stretches)), 8),
            "maxOverstretchedCellRatio": round(max(stretches), 8)}


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


def interpolation_at(keyframes, frame):
    right = int(np.searchsorted(keyframes, frame))
    if keyframes[right] == frame:
        return right, right, 0.
    left = right - 1
    fraction = (frame - keyframes[left]) / (keyframes[right] - keyframes[left])
    return left, right, fraction


def sparse_keyframes(base, frames, grids, target_keys):
    """Pick intervals with a small cost in rendered space, not just flow space."""
    n, h, w = frames.shape[:3]
    if target_keys >= n:
        return np.arange(n)
    pw, ph = (w + 3) // 4, (h + 3) // 4
    x, y = mesh(pw, ph, 1)
    rows, cols = grids.shape[1:3]
    u = ((x + .5) * (w / pw) - .5) * ((cols - 1) / (w - 1))
    v = ((y + .5) * (h / ph) - .5) * ((rows - 1) / (h - 1))
    scale = np.array([pw / w, ph / h], np.float32) * GRID_STEP
    motion = [remap(g, u, v) * scale for g in grids]
    still = cv2.resize(base, (pw, ph), interpolation=cv2.INTER_AREA)
    source = [cv2.resize(f, (pw, ph), interpolation=cv2.INTER_AREA) for f in frames]

    def draw(flow):
        return remap(still, x + flow[..., 0], y + flow[..., 1])

    dense = [draw(flow) for flow in motion]
    dense_error = [error(fit, ref) for fit, ref in zip(dense, source)]

    # Keep the worst interval error bounded. Penalize screen-space deviation
    # from dense tracking as well, so matching the mean does not hide jumps.
    max_gap, peak_limit = 16, .7
    costs = np.full((n, max_gap + 1), np.inf)
    for left in range(n - 1):
        costs[left, 1] = 0
        for gap in range(2, min(max_gap, n - left - 1) + 1):
            right = left + gap
            cost = 0.
            for t in range(left + 1, right):
                alpha = (t - left) / gap
                prediction = draw(cv2.addWeighted(motion[left], 1-alpha, motion[right], alpha, 0))
                extra = max(0., error(prediction, source[t]) - dense_error[t])
                if extra > peak_limit:
                    break
                cost += extra + .25 * error(prediction, dense[t]) + .22 * extra**2
            else:
                costs[left, gap] = cost

    # Find the lowest-cost path with the requested key count. If the cap on
    # local error makes that infeasible, use the first larger feasible count.
    distance = np.full(n, np.inf)
    distance[0] = 0
    parents = []
    for step in range(1, n):
        follow = np.full(n, np.inf)
        parent = np.full(n, -1, np.int32)
        for right in range(step, n):
            gaps = np.arange(1, min(max_gap, right) + 1)
            lefts = right - gaps
            options = distance[lefts] + costs[lefts, gaps]
            best = int(np.argmin(options))
            parent[right] = lefts[best]
            follow[right] = options[best]
        parents.append(parent)
        distance = follow
        if step + 1 >= target_keys and np.isfinite(follow[-1]):
            path = [n - 1]
            for parents_at_step in reversed(parents):
                path.append(int(parents_at_step[path[-1]]))
            return np.array(path[::-1])
    return np.arange(n)


def add_sparse_motion(assets, report, target_keys, base=None, frames=None, grids=None):
    spec = report["grids"]["dense"]
    n, w, h = report["frameCount"], report["width"], report["height"]
    if base is None:
        base = np.array(Image.open(assets / report["base"]["file"]).convert("RGB"))
    if frames is None:
        cap = cv2.VideoCapture(str(assets / report["reference"]["file"]))
        frames = []
        for i in range(n):
            ok, bgr = cap.read()
            if not ok or bgr.shape[:2] != (h, w):
                raise SystemExit(f"Invalid source frame {i}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
    if grids is None:
        encoded = np.frombuffer(gzip.decompress((assets / spec["file"]).read_bytes()),
                                np.uint8).reshape(n, spec["rows"], spec["cols"], 2)
        grids = np.cumsum(encoded, axis=0, dtype=np.uint8) ^ np.uint8(128)
        grids = grids.view(np.int8)
    frames = np.asarray(frames)
    grids = np.asarray(grids, np.float32)
    keyframes = sparse_keyframes(base, frames, grids, target_keys)
    samples = grids[keyframes]
    packed = samples.astype(np.int8).view(np.uint8) ^ np.uint8(128)
    # Store independent grids, with horizontal differences within each row.
    packed[:, :, 1:] -= packed[:, :, :-1].copy()
    asset = write_gzip(assets / "fit-dense-sparse.cbfit", packed)
    report["grids"]["sparse"] = {
        **asset, "rows": spec["rows"], "cols": spec["cols"], "unitPx": spec["unitPx"],
        "encoding": "row-delta", "keyframes": keyframes.tolist(),
    }
    x, y = mesh(w, h, 1)
    sparse_grids, errors = [], []
    for frame, original in enumerate(frames):
        left, right, alpha = interpolation_at(keyframes, frame)
        grid = (cv2.addWeighted(samples[left], 1-alpha, samples[right], alpha, 0)
                if alpha else samples[left])
        sparse_grids.append(grid)
        errors.append(error(reconstruct(base, grid, x, y), original))
    report["quality"]["sparse"] = summary(errors)
    report["tracking"]["geometry"]["sparse"] = geometry_summary(sparse_grids, w, h)
    print(f"Sparse motion: {len(keyframes)}/{n} keys; {asset['bytes']} B; "
          f"MAE {report['quality']['sparse']['mae']}", flush=True)


def write_report(assets, report):
    (assets / "fit.json").write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n",
                                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ROOT / "public/assets",
                        help="directory containing reference.mp4, base.webp and timeline.json")
    parser.add_argument("--rank", type=int, choices=range(1, 33), metavar="1..32", default=24)
    parser.add_argument("--motion-keys", type=int, default=24,
                        help="target key count for sparse motion; more are allowed on complex motion")
    parser.add_argument("--sparse-only", action="store_true",
                        help="reuse the generated dense tracking and rebuild only sparse motion")
    args = parser.parse_args()
    if args.motion_keys < 2:
        parser.error("--motion-keys must be at least 2")
    cv2.setNumThreads(4)
    assets = args.assets
    if args.sparse_only:
        generated = json.loads((assets / "fit.json").read_text(encoding="utf-8"))
        add_sparse_motion(assets, generated, args.motion_keys)
        write_report(assets, generated)
        return
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
    sample_grids = {"coarse": mesh(w, h, 48), "previous": mesh(w, h, 8),
                    "dense": mesh(w, h, STABLE_SPACING)}
    grids = {name: [] for name in sample_grids}
    errors = {name: [] for name in ("still", *grids, "refined")}
    baseline = OpticalFit(base)
    dense_fit = OpticalFit(base, full_resolution=True)
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
        field = baseline.solve(frame)
        tracked = dense_fit.solve(frame)
        errors["still"].append(error(base, frame))
        for name, (gx, gy) in sample_grids.items():
            offsets = remap(tracked if name == "dense" else field, gx, gy)
            grid = stabilize_grid(offsets, w, h) if name == "dense" else quantize_grid(offsets)
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
    correction_file = write_gzip(assets / "fit-correction-stable.cbfit", weights, spatial_delta)
    grid_files = {"coarse": "coarse", "previous": "dense", "dense": "dense-stable"}
    details = {name: grid_asset(assets, grid_files[name], controls)
               for name, controls in grids.items()}
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
        "tracking": {
            "resolution": [w, h], "targetMinAreaRatio": MIN_AREA_RATIO,
            "maxEdgeStretch": MAX_EDGE_STRETCH,
            "geometry": {name: geometry_summary(controls, w, h)
                         for name, controls in grids.items()},
        },
    }
    add_sparse_motion(assets, report, args.motion_keys, base, frames, grids["dense"])
    write_report(assets, report)
    print(json.dumps({name: {k: v for k, v in values.items() if k != "perFrame"}
                      for name, values in report["quality"].items()}, indent=2), flush=True)
    print("Generated", *(f"{item['file']}={item['bytes']} bytes"
                         for item in [*details.values(), correction_file]), sep="\n")


if __name__ == "__main__":
    main()
