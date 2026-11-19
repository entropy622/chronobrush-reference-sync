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
  if (data.length !== stride * count) throw new Error("网格文件长度和帧数不符");
  for (let i = stride; i < data.length; i++) data[i] += data[i - stride];
  return { ...spec, data, stride };
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
    refined: `${dense} 点＋${meta.correction.rank * meta.correction.tileCount} 项校正`,
  };
}
