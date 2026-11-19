import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import test from "node:test";
import { decodeCorrection, decodeGrid, gridInterval, modeSizes } from "../src/fit-data.js";

const url = (file) => new URL("../public/assets/" + file, import.meta.url);
const meta = JSON.parse(await readFile(url("fit.json")));
const unzip = async (spec) => {
  const raw = new Uint8Array(gunzipSync(await readFile(url(spec.file))));
  return raw.buffer;
};

test("each independent mode counts only its actual runtime assets", async () => {
  const assets = [meta.base, meta.reference, meta.brush, meta.grids.coarse, meta.grids.previous, meta.grids.dense, meta.grids.sparse, meta.correction];
  for (const spec of assets) assert.equal((await stat(url(spec.file))).size, spec.bytes, spec.file);
  assert.deepEqual(modeSizes(meta), {
    brush: meta.base.bytes + meta.brush.bytes,
    coarse: meta.base.bytes + meta.grids.coarse.bytes,
    previous: meta.base.bytes + meta.grids.previous.bytes,
    dense: meta.base.bytes + meta.grids.dense.bytes,
    sparse: meta.base.bytes + meta.grids.sparse.bytes,
    refined: meta.base.bytes + meta.grids.dense.bytes + meta.correction.bytes,
  });
  assert.equal(meta.brush.count, 96);
});

test("motion controls decode modulo 256 deltas in time", () => {
  const encoded = new Uint8Array([128, 128, 1, 127, 129, 255, 2, 255]);
  const data = decodeGrid(encoded.buffer, { rows: 1, cols: 2 }, 2);
  assert.deepEqual([...data.data], [128, 128, 1, 127, 1, 127, 3, 126]);
});

test("grid densities are distinct and both cover every frame", async () => {
  const specs = [meta.grids.coarse, meta.grids.previous, meta.grids.dense];
  const counts = specs.map(s => s.rows * s.cols);
  assert.ok(counts[0] < counts[1] && counts[1] < counts[2]);
  for (const spec of specs) {
    const grid = decodeGrid(await unzip(spec), spec, meta.frameCount);
    assert.ok(grid.data.subarray(0, grid.stride).every(v => v === 128));
    assert.ok(grid.data.subarray(grid.stride * 22, grid.stride * 23).some(v => v !== 128));
  }
});

test("appearance modes have finite per-frame weights and RGB bases for every tile", async () => {
  const spec = meta.correction;
  const appearance = decodeCorrection(await unzip(spec), spec, meta.frameCount);
  assert.equal(spec.tileCount, Math.ceil(spec.width/spec.tile) * Math.ceil(spec.height/spec.tile));
  assert.equal(appearance.weights.length, meta.frameCount * spec.tileCount * spec.rank);
  assert.ok(appearance.weights.every(Number.isFinite));
  assert.ok(appearance.basis.some(v => v !== 128));
});

test("added fitting data improves full-resolution error over the entire clip", () => {
  assert.equal(meta.frameCount, 145);
  for (const mode of ["still", "coarse", "previous", "dense", "sparse", "refined"]) {
    const q = meta.quality[mode];
    assert.equal(q.perFrame.length, meta.frameCount);
    assert.ok(q.perFrame.every(Number.isFinite));
    const measuredMean = q.perFrame.reduce((sum, v) => sum + v, 0) / meta.frameCount;
    assert.ok(Math.abs(measuredMean - q.mae) < .0001);
  }
  const q = meta.quality;
  assert.ok(q.still.mae > q.coarse.mae && q.coarse.mae > q.previous.mae &&
    q.previous.mae > q.dense.mae && q.dense.mae > q.refined.mae);
  assert.ok(q.refined.mae < 2.5);
});

function measuredGeometry(grid, count) {
  const { cols, rows, unitPx, data, stride } = grid;
  const sx = unitPx * (cols - 1) / (meta.width - 1);
  const sy = unitPx * (rows - 1) / (meta.height - 1);
  const stretchLimit2 = meta.tracking.maxEdgeStretch ** 2;
  let folds = 0;
  let stretched = 0;
  const det = (ax, ay, bx, by) => ax * by - ay * bx;
  for (let frame = 0; frame < count; frame++) {
    for (let row = 0; row < rows - 1; row++) {
      for (let col = 0; col < cols - 1; col++) {
        const a = frame * stride + 2 * (row * cols + col);
        const b = a + 2, c = a + 2 * cols, d = c + 2;
        const tx = 1 + (data[b] - data[a]) * sx, ty = (data[b+1] - data[a+1]) * sx;
        const bx = 1 + (data[d] - data[c]) * sx, by = (data[d+1] - data[c+1]) * sx;
        const lx = (data[c] - data[a]) * sy, ly = 1 + (data[c+1] - data[a+1]) * sy;
        const rx = (data[d] - data[b]) * sy, ry = 1 + (data[d+1] - data[b+1]) * sy;
        if (Math.min(det(tx, ty, lx, ly), det(tx, ty, rx, ry),
                     det(bx, by, lx, ly), det(bx, by, rx, ry)) <= 0) folds++;
        if (Math.max(tx*tx + ty*ty, bx*bx + by*by, lx*lx + ly*ly, rx*rx + ry*ry) >
            stretchLimit2) stretched++;
      }
    }
  }
  const total = count * (cols - 1) * (rows - 1);
  return [folds / total, stretched / total];
}

test("the shipped dense mesh reduces folds and extreme stretch versus the previous grid", async () => {
  const measured = {};
  for (const name of ["previous", "dense"]) {
    const spec = meta.grids[name];
    const grid = decodeGrid(await unzip(spec), spec, meta.frameCount);
    measured[name] = measuredGeometry(grid, meta.frameCount);
    const report = meta.tracking.geometry[name];
    assert.ok(Math.abs(measured[name][0] - report.meanFoldedCellRatio) < 1e-7);
    assert.ok(Math.abs(measured[name][1] - report.meanOverstretchedCellRatio) < 1e-7);
  }
  for (const i of [0, 1]) assert.ok(measured.dense[i] < measured.previous[i] / 10);
});

test("sparse motion stores exact dense samples and interpolates across nonuniform gaps", async () => {
  const spec = meta.grids.sparse;
  const dense = decodeGrid(await unzip(meta.grids.dense), meta.grids.dense, meta.frameCount);
  const sparse = decodeGrid(await unzip(spec), spec, meta.frameCount);
  assert.equal(sparse.stride, dense.stride);
  const keys = spec.keyframes;
  assert.ok(keys.length <= Math.ceil(meta.frameCount / 5));
  assert.ok(spec.bytes < meta.grids.dense.bytes * .22);
  for (const [i, frame] of keys.entries()) {
    assert.deepEqual(gridInterval(sparse, frame), [i, i, 0]);
    assert.deepEqual(sparse.data.subarray(i*sparse.stride, (i+1)*sparse.stride),
                     dense.data.subarray(frame*dense.stride, (frame+1)*dense.stride));
  }
  for (let frame = 0; frame < meta.frameCount; frame++) {
    if (keys.includes(frame)) continue;
    const [left, right, alpha] = gridInterval(sparse, frame);
    assert.equal(right, left + 1);
    assert.ok(alpha > 0 && alpha < 1);
    assert.ok(Math.abs((1-alpha)*keys[left] + alpha*keys[right] - frame) < 1e-9);
  }
  const { dense: baseline, sparse: result } = meta.quality;
  assert.ok(result.mae - baseline.mae < .45);
  assert.ok(Math.max(...result.perFrame.map((error, i) => error - baseline.perFrame[i])) < 1);
  assert.ok(meta.tracking.geometry.sparse.meanFoldedCellRatio <=
            meta.tracking.geometry.dense.meanFoldedCellRatio);
});

test("reject a sparse grid without a strictly increasing timeline covering both endpoints", () => {
  const buffer = new Uint8Array(6).buffer;
  for (const keys of [[0,2,2], [1,3,4], [0,1,3]]) {
    assert.throws(() => decodeGrid(buffer, { rows: 1, cols: 1, keyframes: keys }, 5), /关键帧/);
  }
});
