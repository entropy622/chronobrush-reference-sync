import { renderCommands } from "./draw.js";
import { frameAtTime, meanAbsError, sheetCell } from "./timeline.js";
import { loadRenderer } from "./webgl.js";

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), stage: $("stage"), gl: $("gl"), vector: $("vector"),
  reference: $("reference"), clock: $("clock"), play: $("play"),
  seek: $("seek"), time: $("time"), stepBack: $("stepBack"),
  stepForward: $("stepForward"), vectorToggle: $("vectorToggle"),
  splitToggle: $("splitToggle"), frameLabel: $("frameLabel"),
  frameValue: $("frameValue"), mae: $("mae"), locked: $("locked"),
  sync: $("sync"), displacement: $("displacement"),
  deformation: $("deformation"), motionTime: $("motionTime"),
  chart: $("chart"), cmdCount: $("cmdCount"), typeRow: $("typeRow"),
  commandList: $("commandList"), stageCaption: $("stageCaption"),
};

const state = {
  meta: null,
  renderer: null,
  sheetImage: null,
  playing: true,
  lastFrame: -1,
  lastMeterAt: 0,
};

function fitCanvas(canvas, width, height) {
  canvas.width = width;
  canvas.height = height;
}

function rounded(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function renderCommandList(meta) {
  const counts = meta.commands.reduce((acc, c) => {
    acc[c.type] = (acc[c.type] || 0) + 1;
    return acc;
  }, {});
  els.cmdCount.textContent = `${meta.commands.length} 条 · ${meta.keyframes.length} 个采样时刻`;
  els.typeRow.innerHTML = ["ellipse", "line", "bezier", "brush"]
    .map((type) => `<span>${type}<b>${counts[type] || 0}</b></span>`).join("");
  els.commandList.innerHTML = meta.commands.slice(0, 12).map((c) => `
    <div class="command" data-command="${c.id}">
      <span class="id">${c.id}</span>
      <span class="type"><b>${c.type}</b> · ${rounded(c.width, 1)}px · ${c.points.length} 控制点</span>
      <span class="motion" id="motion-${c.id}">0.00px</span>
    </div>`).join("");
}

function drawChart(meta, time) {
  const ctx = els.chart.getContext("2d");
  const w = els.chart.width;
  const h = els.chart.height;
  const pad = 16;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "#263141";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad + (h - pad * 2) * i / 4;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(w - pad, y);
    ctx.stroke();
  }
  const series = meta.motionSeries;
  const n = series.displacement.length;
  if (!n) return;
  const maxD = Math.max(...series.displacement, 0.001);
  const maxS = Math.max(...series.deformation, 0.001);
  const x = (i) => pad + i / Math.max(1, n - 1) * (w - pad * 2);
  ctx.lineWidth = 1.6;
  for (const [values, max, color] of [
    [series.displacement, maxD, "#4cc9f0"],
    [series.deformation, maxS, "#f6bd60"],
  ]) {
    ctx.beginPath();
    values.forEach((v, i) => {
      const y = h - pad - v / max * (h - pad * 2);
      if (i === 0) ctx.moveTo(x(i), y);
      else ctx.lineTo(x(i), y);
    });
    ctx.strokeStyle = color;
    ctx.stroke();
  }
  const cursor = pad + Math.min(1, time / meta.duration) * (w - pad * 2);
  ctx.beginPath();
  ctx.moveTo(cursor, pad);
  ctx.lineTo(cursor, h - pad);
  ctx.strokeStyle = "#e85d75";
  ctx.stroke();
  els.motionTime.textContent = `t = ${rounded(time, 3)}s`;
}

function smallContext(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  return ctx;
}

function measureQuality(meta, time) {
  if (els.clock.readyState < 2 || !els.clock.videoWidth) return;
  if (Math.abs(time - state.lastMeterAt) < 0.12 && state.lastFrame >= 0) return;
  state.lastMeterAt = time;
  const w = 147;
  const h = 63;
  const refCtx = smallContext(w, h);
  refCtx.drawImage(els.clock, 0, 0, w, h);
  let ref;
  try {
    ref = refCtx.getImageData(0, 0, w, h).data;
  } catch {
    return;
  }
  // One canvas reads the WebGL core, then adds the diagnostic brush layer.
  const composite = smallContext(w, h);
  composite.drawImage(els.gl, 0, 0, w, h);
  const coreData = composite.getImageData(0, 0, w, h).data;
  if (els.vectorToggle.checked) composite.drawImage(els.vector, 0, 0, w, h);
  const overlay = composite.getImageData(0, 0, w, h).data;
  const coreError = meanAbsError(coreData, ref);
  const overlayError = meanAbsError(overlay, ref);
  els.mae.textContent = `${rounded(coreError, 2)} / ${rounded(overlayError, 2)}`;
}

function render(time) {
  if (!state.meta || !state.renderer) return;
  const meta = state.meta;
  const clamped = Math.max(0, Math.min(meta.duration - 1 / meta.fps, time));
  const frame = frameAtTime(clamped, meta.fps, meta.frameCount);
  const cell = state.renderer.render(frame);
  const ctx = els.vector.getContext("2d");
  renderCommands(ctx, meta.commands, clamped, meta.width, meta.height);

  // The independently decoded HTML video is shown on the right of split view.
  const refCtx = els.reference.getContext("2d");
  if (els.clock.readyState >= 2 && els.clock.videoWidth) {
    refCtx.drawImage(els.clock, 0, 0, meta.width, meta.height);
  } else if (state.sheetImage) {
    refCtx.drawImage(state.sheetImage, cell.sx, cell.sy,
      cell.cellWidth, cell.cellHeight, 0, 0, meta.width, meta.height);
  }

  if (frame !== state.lastFrame) {
    state.lastFrame = frame;
    els.frameLabel.textContent = `FRAME ${String(frame).padStart(3, "0")}`;
    els.frameValue.textContent = `${frame} / ${meta.frameCount - 1}`;
    els.time.textContent = `${rounded(clamped, 3)} / ${rounded(meta.duration, 3)} s`;
    els.seek.value = String(clamped);
    const index = cell.index;
    els.displacement.textContent = `${rounded(meta.motionSeries.displacement[index] || 0, 2)} px`;
    els.deformation.textContent = rounded(meta.motionSeries.deformation[index] || 0, 3);
    els.sync.textContent = `${Math.abs(frame - meta.keyframes[cell.index])} f / ${rounded(
      Math.abs(clamped - els.clock.currentTime) * 1000, 1)} ms`;
    drawChart(meta, clamped);
    for (const command of meta.commands.slice(0, 12)) {
      const m = command.motion[Math.max(0, Math.min(index, command.motion.length - 1))];
      const node = $(`motion-${command.id}`);
      if (node && m) node.textContent = `${rounded(Math.hypot(m[0], m[1]), 2)}px`;
    }
  }
  measureQuality(meta, clamped);
}

function loop() {
  if (!state.meta) return;
  const video = els.clock;
  if (state.playing && video.readyState >= 2) {
    if (video.currentTime >= state.meta.duration - 0.04) video.currentTime = 0;
  }
  render(video.currentTime);
  requestAnimationFrame(loop);
}

async function init() {
  try {
    const response = await fetch("/assets/timeline.json");
    if (!response.ok) throw new Error(`timeline.json ${response.status}`);
    const meta = await response.json();
    state.meta = meta;
    fitCanvas(els.vector, meta.width, meta.height);
    fitCanvas(els.reference, meta.width, meta.height);
    els.seek.max = String(meta.duration);
    els.locked.textContent = `${rounded(meta.base.staticPixelRatio * 100, 1)}%`;
    renderCommandList(meta);
    drawChart(meta, 0);

    state.sheetImage = new Image();
    state.sheetImage.src = `/assets/${meta.sheet.file}`;
    await state.sheetImage.decode();
    state.renderer = await loadRenderer(els.gl, meta, els.clock);
    await new Promise((resolve, reject) => {
      if (els.clock.readyState >= 2) return resolve();
      els.clock.addEventListener("loadeddata", resolve, { once: true });
      els.clock.addEventListener("error", () => reject(new Error("参考视频解码失败")), { once: true });
    });
    els.status.textContent = `${meta.frameCount} 帧 · ${meta.commands.length} 笔触 · WebGL LOCK=ON`;
    els.status.classList.remove("error");
    window.__chronobrush = {
      meta,
      renderer: state.renderer,
      renderAt: (t) => {
        els.clock.currentTime = t;
        render(t);
      },
      setTime: async (t) => {
        state.playing = false;
        els.play.textContent = "▶";
        els.clock.pause();
        const target = Math.max(0, Math.min(state.meta.duration - 1 / state.meta.fps, t));
        await new Promise((resolve) => {
          els.clock.addEventListener("seeked", resolve, { once: true });
          els.clock.currentTime = target;
        });
        state.lastMeterAt = -1;
        render(els.clock.currentTime);
        return window.__chronobrush.diagnostic;
      },
      get diagnostic() {
        return {
          frame: state.lastFrame,
          keyframe: sheetCell(state.meta, state.lastFrame).index,
          baseTextureUpdates: state.renderer.baseTextureUpdates,
          referenceTextureUpdates: state.renderer.referenceTextureUpdates,
          videoTime: els.clock.currentTime,
          mae: els.mae.textContent,
          sync: els.sync.textContent,
          vectorEnabled: els.vectorToggle.checked,
          splitEnabled: els.splitToggle.checked,
        };
      },
    };
    await els.clock.play();
    loop();
  } catch (error) {
    els.status.textContent = `初始化失败：${error.message}`;
    els.status.classList.add("error");
    console.error(error);
  }
}

els.play.addEventListener("click", async () => {
  state.playing = !state.playing;
  els.play.textContent = state.playing ? "❚❚" : "▶";
  if (state.playing) await els.clock.play();
  else els.clock.pause();
});

els.seek.addEventListener("input", () => {
  els.clock.currentTime = Number(els.seek.value);
  render(els.clock.currentTime);
});

function step(delta) {
  state.playing = false;
  els.play.textContent = "▶";
  els.clock.pause();
  const meta = state.meta;
  const frame = frameAtTime(els.clock.currentTime, meta.fps, meta.frameCount);
  els.clock.currentTime = Math.max(0, Math.min(meta.duration - 1 / meta.fps,
    (frame + delta) / meta.fps));
}

els.stepBack.addEventListener("click", () => step(-1));
els.stepForward.addEventListener("click", () => step(1));
els.vectorToggle.addEventListener("change", () => {
  els.stage.classList.toggle("no-vector", !els.vectorToggle.checked);
  state.lastMeterAt = -1;
});
els.splitToggle.addEventListener("change", () => {
  els.stage.classList.toggle("split", els.splitToggle.checked);
  els.stageCaption.textContent = els.splitToggle.checked
    ? "左：重建 · 右：参考" : "重建画面";
});
els.clock.addEventListener("seeked", () => render(els.clock.currentTime));

init();
