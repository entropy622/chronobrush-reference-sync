import assert from "node:assert/strict";
import { readFile, stat } from "node:fs/promises";
import { gunzipSync } from "node:zlib";
import test from "node:test";
import { decodeCorrection, decodeGrid, modeSizes } from "../src/fit-data.js";

const url = (file) => new URL("../public/assets/" + file, import.meta.url);
const meta = JSON.parse(await readFile(url("fit.json")));
const unzip = async (spec) => {
  const raw = new Uint8Array(gunzipSync(await readFile(url(spec.file))));
  return raw.buffer;
};

test("each independent mode counts only its actual runtime assets", async () => {
  const assets = [meta.base, meta.reference, meta.brush, meta.grids.coarse, meta.grids.dense, meta.correction];
  for (const spec of assets) assert.equal((await stat(url(spec.file))).size, spec.bytes, spec.file);
  assert.deepEqual(modeSizes(meta), {
    brush: meta.base.bytes + meta.brush.bytes,
    coarse: meta.base.bytes + meta.grids.coarse.bytes,
    dense: meta.base.bytes + meta.grids.dense.bytes,
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
  const specs = [meta.grids.coarse, meta.grids.dense];
  assert.ok(specs[0].rows * specs[0].cols < specs[1].rows * specs[1].cols);
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
  for (const mode of ["still", "coarse", "dense", "refined"]) {
    const q = meta.quality[mode];
    assert.equal(q.perFrame.length, meta.frameCount);
    assert.ok(q.perFrame.every(Number.isFinite));
    const measuredMean = q.perFrame.reduce((sum, v) => sum + v, 0) / meta.frameCount;
    assert.ok(Math.abs(measuredMean - q.mae) < .0001);
  }
  const q = meta.quality;
  assert.ok(q.still.mae > q.coarse.mae && q.coarse.mae > q.dense.mae && q.dense.mae > q.refined.mae);
  assert.ok(q.refined.mae < 2.5);
});
