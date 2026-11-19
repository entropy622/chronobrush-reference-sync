import { assetUrl } from "./assets.js";
import { renderCommands } from "./draw.js";
import { decodeCorrection, decodeGrid, loadGzip, loadJson, modeLabels, modeSizes } from "./fit-data.js";
import { FitRenderer } from "./fit-renderer.js";

const $ = (id) => document.getElementById(id);
const source = $("source");
const modes = [...document.querySelectorAll("[data-mode]")];
const edits = ["editX", "editY", "resetEdit"].map($);
const regions = {
  ring: { x: .11, y: .55, rx: .14, ry: .29, zoom: 2.42 },
  cape: { x: .31, y: .56, rx: .29, ry: .43, zoom: 1.82 },
  hair: { x: .8, y: .22, rx: .13, ry: .22, zoom: 2.42 },
  face: { x: .76, y: .43, rx: .11, ry: .23, zoom: 2.38 },
};

const state = {
  meta: null, mode: null, renderer: null, frame: 0, change: 0, grid: new Map(), pending: new Map(),
  labels: null, sizes: null, refReady: false, refAligned: false, playing: false,
  refClock: false, localTick: 0, lastMeter: -Infinity,
  edited: { dx: 0, dy: 0 }, pose: null,
};

const byteFormat = new Intl.NumberFormat("zh-CN");
const mb = (bytes) => (bytes / 1e6).toFixed(2) + " MB";
const number = (v, digits = 2) => v.toFixed(digits);

function setStatus(value, error = false) {
  $("status").textContent = value;
  $("status").classList.toggle("error", error);
}

function loadStill(file) {
  const img = new Image();
  img.src = assetUrl(file);
  return img.decode().then(() => img);
}

function once(key, task) {
  if (!state.pending.has(key)) {
    state.pending.set(key, Promise.resolve().then(task).catch(error => {
      state.pending.delete(key);
      throw error;
    }));
  }
  return state.pending.get(key);
}

function readGrid(name) {
  const spec = state.meta.grids[name];
  return once(name, async () => {
    state.grid.set(name, decodeGrid(await loadGzip(spec.file), spec, state.meta.frameCount));
  });
}

function setFitStats() {
  const mode = state.mode;
  const size = state.sizes[mode];
  const quality = state.meta.quality[mode];
  $("fitBytes").textContent = mb(size);
  const sampleCount = state.meta.grids.sparse.keyframes.length;
  $("fitSizeDetail").textContent = mode === "sparse"
    ? byteFormat.format(size) + " B · 仅存 " + sampleCount + "/" + state.meta.frameCount + " 帧"
    : byteFormat.format(size) + " B · " + number(size / state.meta.reference.bytes) + "× 原视频";
  $("meanError").textContent = quality ? number(quality.mae) : "—";
  $("meanDetail").textContent = quality
    ? state.meta.frameCount + " 帧全分辨率 · RGB MAE / 255"
    : "原有诊断笔触 · 可查逐帧误差";
}

async function chooseMode(mode) {
  if (!state.renderer) return;
  const change = ++state.change;
  setStatus("载入" + state.labels[mode]);
  try {
    if (mode === "brush") {
      await once("brush", async () => {
        state.brush = (await loadJson(state.meta.brush.file)).commands;
      });
    } else if (mode === "refined") {
      const spec = state.meta.correction;
      await Promise.all([readGrid("dense"), once("correction", async () => {
        const data = decodeCorrection(await loadGzip(spec.file), spec, state.meta.frameCount);
        state.renderer.enableCorrection(data);
      })]);
    } else {
      await readGrid(mode);
    }
    if (change !== state.change) return;
    state.mode = mode;
    for (const button of modes) button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
    $("fitLabel").textContent = state.labels[mode];
    $("brushCanvas").hidden = mode !== "brush";
    $("fitMessage").hidden = true;
    setStatus("拟合已就绪 · 左侧独立渲染");
    setFitStats();
    updateEditor();
    showFrame(state.frame, true);
    for (const id of ["play", "prev", "next", "seek"]) $(id).disabled = false;
  } catch (error) {
    if (change !== state.change) return;
    setStatus(error.message, true);
    if (!state.mode) $("fitMessage").textContent = "当前拟合载入失败，请切换拟合量";
  }
}

function updateEditor() {
  const region = regions[$("focus").value];
  const canEdit = Boolean(region && state.mode && state.mode !== "brush");
  for (const input of edits) input.disabled = !canEdit;
  const { dx, dy } = state.edited;
  $("editXValue").textContent = dx + " px";
  $("editYValue").textContent = dy + " px";
  state.pose = canEdit ? {
    cx: region.x * state.meta.width, cy: region.y * state.meta.height,
    rx: region.rx * state.meta.width, ry: region.ry * state.meta.height, dx, dy,
  } : null;
  $("editHelp").textContent = state.mode === "brush"
    ? "原有笔触保持原样。切换到网格拟合后可试改局部位移。"
    : region
      ? "滑动后只改变左侧。影响范围沿边缘渐弱，可观察衣料与背景是否一起被带动。"
      : "选择披风、发梢或脸与领口，观察局部移动时周围画面的变化。";
}

function readbackCanvas() {
  const canvas = document.createElement("canvas");
  canvas.width = 490;
  canvas.height = 210;
  return canvas.getContext("2d", { willReadFrequently: true });
}

const fitReader = readbackCanvas();
const refReader = readbackCanvas();
const diff = $("difference").getContext("2d");
const heat = diff.createImageData(490, 210);
const colors = [[12,18,32], [152,104,57], [255,116,88]];

function comparePixels(force = false) {
  if (!state.mode || !state.refAligned || $("referenceMode").value === "off") return;
  const isHeat = $("referenceMode").value === "difference";
  const now = performance.now();
  if (!force && !isHeat && state.playing && now - state.lastMeter < 200) return;
  state.lastMeter = now;
  const w = 490, h = 210;
  fitReader.drawImage($("fitCanvas"), 0, 0, w, h);
  if (state.mode === "brush") fitReader.drawImage($("brushCanvas"), 0, 0, w, h);
  refReader.drawImage(source, 0, 0, w, h);
  const fit = fitReader.getImageData(0, 0, w, h).data;
  const ref = refReader.getImageData(0, 0, w, h).data;
  let sum = 0;
  for (let i = 0; i < fit.length; i += 4) {
    const error = (Math.abs(fit[i] - ref[i]) + Math.abs(fit[i+1] - ref[i+1]) +
      Math.abs(fit[i+2] - ref[i+2])) / 3;
    sum += error;
    if (!isHeat) continue;
    const lo = error < 8 ? 0 : 1;
    const t = lo === 0 ? error / 8 : Math.min(1, (error - 8) / 16);
    for (let c = 0; c < 3; c++) heat.data[i+c] = colors[lo][c] + t * (colors[lo+1][c] - colors[lo][c]);
    heat.data[i+3] = 255;
  }
  if (isHeat) diff.putImageData(heat, 0, 0);
  $("frameError").textContent = number(sum / (w*h));
  $("frameErrorDetail").textContent = state.pose && (state.pose.dx || state.pose.dy)
    ? "编辑后 · 490 × 210" : "490 × 210 · RGB MAE / 255";
}

function showFrame(frame, force = false) {
  if (!state.meta) return;
  state.frame = Math.max(0, Math.min(state.meta.frameCount - 1, frame));
  if (state.mode && (force || state.lastFrame !== state.frame)) {
    const mode = state.mode;
    const grid = state.grid.get(mode === "refined" ? "dense" : mode);
    state.renderer.render(state.frame, grid, mode === "refined", state.pose);
    if (mode === "sparse") {
      $("fitLabel").textContent = grid.keyframes.includes(state.frame) ? "压缩追踪 · 已记录帧" : "压缩追踪 · 插值帧";
    }
    if (mode === "brush") {
      const canvas = $("brushCanvas");
      renderCommands(canvas.getContext("2d"), state.brush, state.frame / state.meta.fps, canvas.width, canvas.height);
    }
    state.lastFrame = state.frame;
  }
  $("seek").value = state.frame;
  $("frame").textContent = String(state.frame).padStart(3, "0") + " / " + (state.meta.frameCount - 1);
  $("time").textContent = number(state.frame / state.meta.fps, 3) + " s";
  comparePixels(force);
}

function refTarget() {
  // Inside the desired frame interval, clear of browser timestamp rounding.
  return (state.frame + .25) / state.meta.fps;
}

function alignReference() {
  if (!state.refReady || $("referenceMode").value === "off") return;
  const target = refTarget();
  state.refAligned = false;
  if (!source.seeking && Math.abs(source.currentTime - target) < .002) {
    state.refAligned = true;
    $("referenceMessage").hidden = true;
    comparePixels(true);
  } else {
    if (!state.playing) $("frameError").textContent = "—";
    source.currentTime = target;
  }
}

function pause() {
  state.playing = false;
  state.refClock = false;
  source.pause();
  cancelAnimationFrame(state.localTick);
  $("play").textContent = "播放";
}

function useLocalClock() {
  cancelAnimationFrame(state.localTick);
  const start = performance.now();
  const startFrame = state.frame;
  function tick(now) {
    if (!state.playing || state.refClock) return;
    const frame = (startFrame + Math.floor((now - start) * state.meta.fps / 1000)) % state.meta.frameCount;
    if (frame !== state.frame) showFrame(frame);
    state.localTick = requestAnimationFrame(tick);
  }
  state.localTick = requestAnimationFrame(tick);
}

function useReferenceClock() {
  state.refClock = true;
  cancelAnimationFrame(state.localTick);
  alignReference();
  source.play().catch(() => {
    if (!state.playing || $("referenceMode").value === "off") return;
    state.refClock = false;
    state.refAligned = false;
    $("referenceMessage").hidden = false;
    $("referenceMessage").textContent = "参考播放受限 · 左侧继续独立播放";
    useLocalClock();
  });
}

function changeReference() {
  if (!state.meta) return;
  const mode = $("referenceMode").value;
  const off = mode === "off";
  const isHeat = mode === "difference";
  $("referenceView").hidden = off;
  $("comparison").classList.toggle("solo", off);
  $("referenceLabel").textContent = isHeat ? "差异热图" : "原视频";
  $("referenceInfo").textContent = isHeat ? "同一帧 · RGB 通道差" : "H.264 · " + state.meta.width + " × " + state.meta.height;
  source.hidden = isHeat;
  $("difference").hidden = !isHeat;
  $("heatLegend").hidden = !isHeat;
  if (off) {
    source.pause();
    source.removeAttribute("src");
    source.load();
    state.refReady = state.refAligned = state.refClock = false;
    $("frameError").textContent = "—";
    $("frameErrorDetail").textContent = "已关闭参考";
    if (state.playing) useLocalClock();
  } else if (!source.getAttribute("src")) {
    state.refAligned = false;
    $("referenceMessage").hidden = false;
    $("referenceMessage").textContent = "正在载入参考";
    source.preload = "auto";
    source.src = assetUrl(state.meta.reference.file);
  } else {
    comparePixels(true);
  }
}

source.loop = true;
source.addEventListener("loadeddata", () => {
  if ($("referenceMode").value === "off") return;
  state.refReady = true;
  if (state.playing) useReferenceClock();
  else alignReference();
});
source.addEventListener("seeked", () => {
  if (!state.refReady || $("referenceMode").value === "off") return;
  const videoFrame = Math.floor(source.currentTime * state.meta.fps + .001);
  if (!state.playing && videoFrame === state.frame) {
    state.refAligned = true;
    $("referenceMessage").hidden = true;
    comparePixels(true);
  }
});
source.addEventListener("error", () => {
  if ($("referenceMode").value === "off") return;
  state.refReady = state.refAligned = state.refClock = false;
  $("referenceMessage").textContent = "参考视频不可用 · 左侧拟合可继续";
  $("referenceMessage").hidden = false;
  $("frameError").textContent = "—";
  $("frameErrorDetail").textContent = "参考未载入";
  if (state.playing) useLocalClock();
});

if ("requestVideoFrameCallback" in source) {
  function onVideoFrame(_, result) {
    if (state.playing && state.refClock && !source.seeking) {
      state.refAligned = true;
      $("referenceMessage").hidden = true;
      showFrame(Math.round(result.mediaTime * state.meta.fps));
    }
    source.requestVideoFrameCallback(onVideoFrame);
  }
  source.requestVideoFrameCallback(onVideoFrame);
} else {
  source.addEventListener("timeupdate", () => {
    if (state.playing && state.refClock && !source.seeking) {
      state.refAligned = true;
      $("referenceMessage").hidden = true;
      showFrame(Math.floor(source.currentTime * state.meta.fps));
    }
  });
}

function jump(frame) {
  pause();
  state.refAligned = false;
  showFrame(frame);
  alignReference();
}

$("play").addEventListener("click", () => {
  if (state.playing) {
    pause();
    alignReference();
  } else {
    state.playing = true;
    $("play").textContent = "暂停";
    if (state.refReady) useReferenceClock();
    else useLocalClock();
  }
});
$("prev").addEventListener("click", () => jump(state.frame - 1));
$("next").addEventListener("click", () => jump(state.frame + 1));
$("seek").addEventListener("input", (event) => jump(Number(event.target.value)));
$("referenceMode").addEventListener("change", changeReference);
$("focus").addEventListener("change", () => {
  const region = regions[$("focus").value];
  const style = $("comparison").style;
  style.setProperty("--zoom", region?.zoom ?? 1);
  style.setProperty("--focusX", (region?.x ?? .5)*100 + "%");
  style.setProperty("--focusY", (region?.y ?? .5)*100 + "%");
  updateEditor();
  showFrame(state.frame, true);
});
for (const [id, axis] of [["editX", "dx"], ["editY", "dy"]]) {
  $(id).addEventListener("input", (event) => {
    state.edited[axis] = Number(event.target.value);
    updateEditor();
    showFrame(state.frame, true);
  });
}
$("resetEdit").addEventListener("click", () => {
  state.edited = { dx: 0, dy: 0 };
  $("editX").value = $("editY").value = 0;
  updateEditor();
  showFrame(state.frame, true);
});
for (const button of modes) button.addEventListener("click", () => chooseMode(button.dataset.mode));

async function init() {
  try {
    if (new URLSearchParams(location.search).get("compare") === "off") $("referenceMode").value = "off";
    const meta = state.meta = await loadJson("fit.json?tracking=sparse");
    state.labels = modeLabels(meta);
    state.sizes = modeSizes(meta);
    $("seek").max = meta.frameCount - 1;
    $("brushCanvas").width = meta.width;
    $("brushCanvas").height = meta.height;
    for (const button of modes) {
      if (button.dataset.mode !== "refined") button.textContent = state.labels[button.dataset.mode];
    }
    $("sourceBytes").textContent = mb(meta.reference.bytes);
    $("sourceSizeDetail").textContent = byteFormat.format(meta.reference.bytes) + " B · " + meta.fps + " fps";
    showFrame(Math.round((meta.frameCount - 1) * .16));
    changeReference();
    state.renderer = new FitRenderer($("fitCanvas"), meta, await loadStill(meta.base.file));
    await chooseMode("sparse");
  } catch (error) {
    setStatus(error.message, true);
    $("fitMessage").textContent = "拟合载入失败";
  }
}

init();
