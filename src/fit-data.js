import { assetUrl } from "./assets.js";

async function responseFor(file) {
  const response = await fetch(assetUrl(file));
  if (!response.ok) throw new Error(`无法读取 ${file}（${response.status}）`);
  return response;
}

export async function loadJson(file) {
  return (await responseFor(file)).json();
}

export async function loadGzip(file) {
  const response = await responseFor(file);
  const stream = response.body.pipeThrough(new DecompressionStream("gzip"));
  return new Response(stream).arrayBuffer();
}

export function decodeGrid(buffer, spec, count) {
  const data = new Uint8Array(buffer);
  const stride = spec.cols * spec.rows * 2;
  const keys = spec.keyframes;
  if (keys && (keys[0] !== 0 || keys.at(-1) !== count - 1 ||
    keys.some((key, i) => !Number.isInteger(key) || (i > 0 && key <= keys[i-1])))) {
    throw new Error("网格关键帧范围无效");
  }
  if (data.length !== stride * (keys?.length ?? count)) throw new Error("网格文件长度和帧数不符");
  if (spec.encoding === "row-delta") {
    const rowBytes = spec.cols * 2;
    for (let row = 0; row < data.length; row += rowBytes) {
      for (let x = 2; x < rowBytes; x++) data[row + x] += data[row + x - 2];
    }
  } else {
    for (let i = stride; i < data.length; i++) data[i] += data[i - stride];
  }
  return { ...spec, data, stride };
}

export function gridInterval(grid, frame) {
  const keys = grid.keyframes;
  if (!keys) return [frame, frame, 0];
  let lo = 0, hi = keys.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (keys[mid] < frame) lo = mid + 1;
    else hi = mid;
  }
  if (keys[lo] === frame) return [lo, lo, 0];
  return [lo - 1, lo, (frame - keys[lo-1]) / (keys[lo] - keys[lo-1])];
}

export function decodeCorrection(buffer, spec, count) {
  const stride = spec.tileCount * spec.rank;
  const weights = new Float32Array(buffer, 0, stride * count);
  const basis = new Uint8Array(buffer, weights.byteLength);
  const rowBytes = spec.width * 3;
  if (basis.length !== rowBytes * spec.height * spec.rank) {
    throw new Error("颜色校正文件长度不符");
  }
  for (let row = 0; row < basis.length; row += rowBytes) {
    for (let x = 3; x < rowBytes; x++) basis[row + x] += basis[row + x - 3];
  }
  return { ...spec, basis, weights, stride };
}

export function modeSizes(meta) {
  const base = meta.base.bytes;
  const dense = base + meta.grids.dense.bytes;
  return {
    brush: base + meta.brush.bytes,
    coarse: base + meta.grids.coarse.bytes,
    previous: base + meta.grids.previous.bytes,
    dense,
    sparse: base + meta.grids.sparse.bytes,
    refined: dense + meta.correction.bytes,
  };
}

export function modeLabels(meta) {
  const points = (grid) => (grid.cols * grid.rows).toLocaleString("en-US");
  const dense = points(meta.grids.dense);
  return {
    brush: `原有 ${meta.brush.count} 笔`,
    coarse: `粗网格 · ${points(meta.grids.coarse)} 点`,
    previous: `旧细网格 · ${points(meta.grids.previous)} 点`,
    dense: `密追踪 · ${dense} 点`,
    sparse: `压缩 · ${meta.grids.sparse.keyframes.length} 帧`,
    refined: `${dense} 点＋${meta.correction.rank * meta.correction.tileCount} 项校正`,
  };
}
