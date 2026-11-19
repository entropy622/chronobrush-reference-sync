import { gridInterval } from "./fit-data.js";

const VERTEX = `#version 300 es
out vec2 uv;
void main() {
    uv = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAME = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 outputColor;
uniform sampler2D base;
uniform sampler2D grid;
uniform sampler2D nextGrid;
uniform float gridMix;
uniform sampler2D correction;
uniform vec2 size;
uniform vec2 gridSize;
uniform float flowStep;
uniform float correctionStep;
uniform bool hasFlow;
uniform bool hasCorrection;
uniform vec2 editCenter;
uniform vec2 editRadii;
uniform vec2 editShift;
void main() {
    vec2 pixel = vec2(uv.x, 1.0-uv.y) * size - 0.5;
    vec2 q = (pixel - editCenter) / editRadii;
    float weight = 1.0 - smoothstep(0.1, 1.0, dot(q,q));
    vec2 p = clamp(pixel - editShift * weight, vec2(0.0), size - 1.0);
    vec2 offset = vec2(0.0);
    if (hasFlow) {
        vec2 point = (p * (gridSize - 1.0) / (size - 1.0) + 0.5) / gridSize;
        vec2 motion = texture(grid, point).rg;
        if (gridMix > 0.0) motion = mix(motion, texture(nextGrid, point).rg, gridMix);
        offset = (motion * 255.0 - 128.0) * flowStep;
    }
    vec3 color = texture(base, (p + offset + 0.5) / size).rgb;
    if (hasCorrection) {
        vec3 delta = texture(correction, (p + 0.5) / size).rgb * 255.0 - 128.0;
        color += delta * (correctionStep / 255.0);
    }
    outputColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}`;

const CORRECTION = `#version 300 es
precision highp float;
precision highp sampler2DArray;
out vec4 outputColor;
uniform sampler2DArray basis;
uniform sampler2D coefficients;
uniform int tilesAcross;
uniform int tileSize;
uniform int rank;
uniform float stepRgb;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 tile = p / tileSize;
    int region = tile.x + tile.y * tilesAcross;
    vec3 residual = vec3(0.0);
    for (int i = 0; i < rank; ++i) {
        float w = texelFetch(coefficients, ivec2(i, region), 0).r;
        residual += (texelFetch(basis, ivec3(p,i), 0).rgb * 255.0 - 128.0) * w;
    }
    outputColor = vec4(clamp(round(residual / stepRgb) + 128.0, 0.0, 255.0) / 255.0, 1.0);
}`;

function shader(gl, kind, source) {
  const result = gl.createShader(kind);
  gl.shaderSource(result, source);
  gl.compileShader(result);
  if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(result) || "着色器编译失败");
  }
  return result;
}

function program(gl, fragment, uniforms) {
  const id = gl.createProgram();
  const a = shader(gl, gl.VERTEX_SHADER, VERTEX);
  const b = shader(gl, gl.FRAGMENT_SHADER, fragment);
  gl.attachShader(id, a);
  gl.attachShader(id, b);
  gl.linkProgram(id);
  gl.deleteShader(a);
  gl.deleteShader(b);
  if (!gl.getProgramParameter(id, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(id) || "着色器链接失败");
  }
  return {
    id,
    u: Object.fromEntries(uniforms.split(" ").map(name => [name, gl.getUniformLocation(id, name)])),
  };
}

function bind(gl, texture, unit, target = gl.TEXTURE_2D) {
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(target, texture);
}

function texture(gl, filter, target = gl.TEXTURE_2D) {
  const t = gl.createTexture();
  bind(gl, t, 0, target);
  gl.texParameteri(target, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(target, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(target, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(target, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return t;
}

export class FitRenderer {
  constructor(canvas, meta, still) {
    this.width = canvas.width = meta.width;
    this.height = canvas.height = meta.height;
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false,
      preserveDrawingBuffer: true });
    if (!gl) throw new Error("当前浏览器未启用 WebGL 2");
    this.gl = gl;
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    this.frame = program(gl, FRAME,
      "base grid nextGrid gridMix correction size gridSize flowStep correctionStep hasFlow hasCorrection editCenter editRadii editShift");

    this.still = texture(gl, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, still);
    this.rightFlow = { texture: texture(gl, gl.LINEAR) };
    this.leftFlow = { texture: texture(gl, gl.LINEAR) };
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, 1, 1, 0, gl.RG, gl.UNSIGNED_BYTE, new Uint8Array([128,128]));
    // A complete texture for modes that skip the correction sampler.
    this.blank = texture(gl, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array([128,128,128,255]));
  }

  enableCorrection(data) {
    const gl = this.gl;
    this.appearance = program(gl, CORRECTION, "basis coefficients tilesAcross tileSize rank stepRgb");
    this.basis = texture(gl, gl.NEAREST, gl.TEXTURE_2D_ARRAY);
    gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.RGB8, data.width, data.height, data.rank, 0,
      gl.RGB, gl.UNSIGNED_BYTE, data.basis);
    this.coefficients = texture(gl, gl.NEAREST);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, data.rank, data.tileCount, 0, gl.RED, gl.FLOAT, null);
    this.colorPlane = texture(gl, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, data.width, data.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    this.fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.colorPlane, 0);
    const complete = gl.checkFramebufferStatus(gl.FRAMEBUFFER) === gl.FRAMEBUFFER_COMPLETE;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    if (!complete || gl.getError() !== gl.NO_ERROR) throw new Error("无法初始化颜色校正纹理");
    this.correction = data;
  }

  correct(frame) {
    if (this.lastCorrection === frame) return;
    this.lastCorrection = frame;
    const gl = this.gl;
    const { id, u } = this.appearance;
    const spec = this.correction;
    gl.useProgram(id);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.viewport(0, 0, spec.width, spec.height);
    bind(gl, this.basis, 0, gl.TEXTURE_2D_ARRAY);
    bind(gl, this.coefficients, 1);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, spec.rank, spec.tileCount, gl.RED, gl.FLOAT,
      spec.weights.subarray(frame * spec.stride, (frame + 1) * spec.stride));
    gl.uniform1i(u.basis, 0);
    gl.uniform1i(u.coefficients, 1);
    gl.uniform1i(u.tilesAcross, Math.ceil(spec.width / spec.tile));
    gl.uniform1i(u.tileSize, spec.tile);
    gl.uniform1i(u.rank, spec.rank);
    gl.uniform1f(u.stepRgb, spec.unitRgb);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  uploadFlow(slot, grid, index, unit) {
    if (slot.grid === grid && slot.index === index) return;
    const gl = this.gl;
    const bytes = grid.data.subarray(index * grid.stride, (index + 1) * grid.stride);
    bind(gl, slot.texture, unit);
    if (slot.grid !== grid) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, grid.cols, grid.rows, 0, gl.RG, gl.UNSIGNED_BYTE, bytes);
    } else {
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, grid.cols, grid.rows, gl.RG, gl.UNSIGNED_BYTE, bytes);
    }
    slot.grid = grid;
    slot.index = index;
  }

  render(frame, grid, refined, edit) {
    const gl = this.gl;
    if (refined) this.correct(frame);
    let alpha = 0;
    if (grid) {
      const [left, right, mix] = gridInterval(grid, frame);
      alpha = mix;
      this.uploadFlow(this.leftFlow, grid, left, 1);
      if (alpha > 0) this.uploadFlow(this.rightFlow, grid, right, 3);
    }

    const { id, u } = this.frame;
    gl.useProgram(id);
    gl.viewport(0, 0, this.width, this.height);
    bind(gl, this.still, 0);
    bind(gl, this.leftFlow.texture, 1);
    bind(gl, alpha > 0 ? this.rightFlow.texture : this.leftFlow.texture, 3);
    bind(gl, refined ? this.colorPlane : this.blank, 2);
    gl.uniform1i(u.base, 0);
    gl.uniform1i(u.grid, 1);
    gl.uniform1i(u.nextGrid, 3);
    gl.uniform1f(u.gridMix, alpha);
    gl.uniform1i(u.correction, 2);
    gl.uniform2f(u.size, this.width, this.height);
    gl.uniform1i(u.hasFlow, Boolean(grid));
    gl.uniform1i(u.hasCorrection, refined);
    gl.uniform2f(u.gridSize, grid?.cols ?? 1, grid?.rows ?? 1);
    gl.uniform1f(u.flowStep, grid?.unitPx ?? 0);
    gl.uniform1f(u.correctionStep, this.correction?.unitRgb ?? 1);
    gl.uniform2f(u.editCenter, edit?.cx ?? 0, edit?.cy ?? 0);
    gl.uniform2f(u.editRadii, edit?.rx ?? 1, edit?.ry ?? 1);
    gl.uniform2f(u.editShift, edit?.dx ?? 0, edit?.dy ?? 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
