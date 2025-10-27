import type { InferenceSession, Tensor } from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';

declare const ort: any;


const MODEL_INPUT_SIZE: [number, number] = [512, 288];

const default_EMA = 0.55;
const default_NOISE_CUTOFF = 0.06;
const default_HIGH_THRESHOLD = 0.95;
const default_GAMMA = 0.4;
const default_USE_BILATERAL = true;
const default_BILATERAL_SIGMA_SPATIAL = 1.0;
const default_BILATERAL_SIGMA_RANGE = 12.0;

export const defaultConfig = {
  EMA: default_EMA,
  NOISE_CUTOFF: default_NOISE_CUTOFF,
  HIGH_THRESHOLD: default_HIGH_THRESHOLD,
  GAMMA: default_GAMMA,
  USE_BILATERAL: default_USE_BILATERAL,
  BILATERAL_SIGMA_SPATIAL: default_BILATERAL_SIGMA_SPATIAL,
  BILATERAL_SIGMA_RANGE: default_BILATERAL_SIGMA_RANGE,
};

export let config = { ...defaultConfig };


const maskCanvas = document.createElement('canvas');
maskCanvas.width = MODEL_INPUT_SIZE[0];
maskCanvas.height = MODEL_INPUT_SIZE[1];
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true })!;


let prevAlpha: Float32Array | null = null;


type ProcessOpts = {
  lastAffine?: { a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null;
};


export async function processFrame(
    videoElement: HTMLVideoElement,
    session: InferenceSession,
    outputCtx: CanvasRenderingContext2D,
    opts: ProcessOpts = {}
): Promise<{ inferenceTime: number; totalTime: number }> {
  const totalStartTime = performance.now();

  // Препроцесс → NCHW
  const frame = tf.browser.fromPixels(videoElement);
  const resized = tf.image.resizeBilinear(frame, [MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]]);
  const normalized = resized.div(255.0);
  const transposed = normalized.transpose([2, 0, 1]);
  const inputTensorTf = transposed.expandDims(0);

  const inputData = inputTensorTf.dataSync() as Float32Array;
  const ortInputTensor = new ort.Tensor('float32', inputData, inputTensorTf.shape);

  frame.dispose(); resized.dispose(); normalized.dispose(); transposed.dispose(); inputTensorTf.dispose();

  const feeds = { input: ortInputTensor };
  const infStart = performance.now();
  const results = await session.run(feeds);
  const infEnd = performance.now();

  const mask = results['output'] as Tensor;
  const alphaRaw = squeezeMaskTo2D(mask);
  const maskW = mask.dims[3];
  const maskH = mask.dims[2];

  let baseAlpha = alphaRaw;

  if (opts.lastAffine && prevAlpha && prevAlpha.length === baseAlpha.length) {
    const w = maskW, h = maskH;
    const warped = warpAffineNearest(
        prevAlpha, w, h,
        opts.lastAffine.a11, opts.lastAffine.a12, opts.lastAffine.tx,
        opts.lastAffine.a21, opts.lastAffine.a22, opts.lastAffine.ty
    );
    for (let i = 0; i < baseAlpha.length; i++) {
      const currentPixel = baseAlpha[i];
      const warpedPixel = warped[i];

      baseAlpha[i] = Math.max(currentPixel, warpedPixel * 0.75);
    }
  }


  const emaAlpha = temporalEMA(baseAlpha);

  const guidedAlpha = config.USE_BILATERAL
      ? jointBilateral3x3(emaAlpha, sampleGuidePixels(videoElement, maskW, maskH), maskW, maskH)
      : emaAlpha;

  const refinedAlpha = refineAlphaOnce(guidedAlpha);

  const maskImageData = alphaToImageData(refinedAlpha, maskW, maskH);
  maskCtx.putImageData(maskImageData, 0, 0);

  const outputCanvas = outputCtx.canvas;
  outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
  outputCtx.drawImage(videoElement, 0, 0, outputCanvas.width, outputCanvas.height);

  outputCtx.globalCompositeOperation = 'destination-in';
  outputCtx.drawImage(maskCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
  outputCtx.globalCompositeOperation = 'source-over';

  const totalEndTime = performance.now();
  return {
    inferenceTime: infEnd - infStart,
    totalTime: totalEndTime - totalStartTime,
  };
}

function squeezeMaskTo2D(maskTensor: Tensor): Float32Array {
  const [_, __, h, w] = maskTensor.dims;
  const total = h * w;
  if (maskTensor.type === 'uint8') {
    const u8 = maskTensor.data as Uint8Array;
    const out = new Float32Array(total);
    for (let i = 0; i < total; i++) out[i] = u8[i] / 255;
    return out;
  }
  const f32 = maskTensor.data as Float32Array;
  return f32.length === total ? f32 : new Float32Array(f32.slice(0, total));
}

function alphaToImageData(alpha: Float32Array, w: number, h: number): ImageData {
  const imageData = new ImageData(w, h);
  const pixels = imageData.data;
  for (let i = 0; i < alpha.length; i++) {
    const a = Math.max(0, Math.min(1, alpha[i]));
    const p = i * 4;
    pixels[p] = 255;       // R - используем значение маски
    pixels[p + 1] = 255;   // G
    pixels[p + 2] = 255;   // B
    pixels[p + 3] = Math.round(a * 255);
  }
  return imageData;
}

function temporalEMA(current: Float32Array): Float32Array {
  if (!prevAlpha || prevAlpha.length !== current.length) {
    prevAlpha = current.slice();
    return current;
  }
  for (let i = 0; i < current.length; i++) {
    prevAlpha[i] = config.EMA * prevAlpha[i] + (1 - config.EMA) * current[i];
  }
  return prevAlpha;
}
// function temporalEMA(current: Float32Array): Float32Array {
//   if (!prevAlpha || prevAlpha.length !== current.length) {
//     prevAlpha = current.slice();
//     return current;
//   }

//   // Порог, ниже которого мы считаем пиксель "дырой" в потенциально сплошной области
//   const HOLE_THRESHOLD = 0.1;

//   for (let i = 0; i < current.length; i++) {
//     const currentPixel = current[i];
//     const prevPixel = prevAlpha[i];

//     if (currentPixel < HOLE_THRESHOLD && prevPixel > (HOLE_THRESHOLD + 0.2)) {
//       // Если текущий пиксель - это внезапная дыра, а на прошлом кадре здесь был объект,
//       // то мы почти полностью доверяем старому значению, лишь немного его затухая.
//       // Это эффективно "затыкает" мгновенные дыры.
//       prevAlpha[i] = prevPixel * 0.90; // Коэффициент затухания, чтобы избежать "зависших" артефактов
//     } else {
//       // В остальных случаях используем стандартное EMA-сглаживание
//       prevAlpha[i] = config.EMA * prevPixel + (1 - config.EMA) * currentPixel;
//     }
//   }

//   return prevAlpha;
// }

// Совместный билатеральный фильтр 3×3 (edge-aware)
function jointBilateral3x3(
    alpha: Float32Array,
    guidePixels: Uint8ClampedArray,
    w: number,
    h: number,
    sigmaSpatial = config.BILATERAL_SIGMA_SPATIAL,
    sigmaRange = config.BILATERAL_SIGMA_RANGE
): Float32Array {
  const out = new Float32Array(w * h);
  const twoSigmaS2 = 2 * sigmaSpatial * sigmaSpatial;
  const twoSigmaR2 = 2 * sigmaRange * sigmaRange;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const r0 = guidePixels[idx * 4];
      const g0 = guidePixels[idx * 4 + 1];
      const b0 = guidePixels[idx * 4 + 2];
      let sumW = 0, sumA = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          const dr = guidePixels[j * 4] - r0;
          const dg = guidePixels[j * 4 + 1] - g0;
          const db = guidePixels[j * 4 + 2] - b0;
          const range2 = dr*dr + dg*dg + db*db;
          const spatial2 = dx*dx + dy*dy;
          const wgt = Math.exp(-spatial2 / twoSigmaS2) * Math.exp(-range2 / twoSigmaR2);
          sumW += wgt; sumA += wgt * alpha[j];
        }
      }
      out[idx] = sumW > 0 ? sumA / sumW : alpha[idx];
    }
  }
  return out;
}

// — вспомогательные функции ниже —

function refineAlphaOnce(a: Float32Array, low = config.NOISE_CUTOFF, high = config.HIGH_THRESHOLD, gamma = config.GAMMA): Float32Array {
  const out = new Float32Array(a.length);
  const denom = Math.max(1e-6, high - low);
  for (let i = 0; i < a.length; i++) {
    let v = a[i];
    if (v <= low) v = 0;
    else if (v >= high) v = 1;
    else {
      const t = (v - low) / denom;
      v = Math.pow(t, gamma);
    }
    out[i] = v;
  }
  return out;
}

function sampleGuidePixels(video: HTMLVideoElement, w: number, h: number): Uint8ClampedArray {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(video, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h).data;
}

function invertAffine(a11: number, a12: number, tx: number, a21: number, a22: number, ty: number) {
  const det = a11 * a22 - a12 * a21;
  const d = det !== 0 ? det : 1e-6;
  const ia11 =  a22 / d;
  const ia12 = -a12 / d;
  const ia21 = -a21 / d;
  const ia22 =  a11 / d;
  const itx = -(ia11 * tx + ia12 * ty);
  const ity = -(ia21 * tx + ia22 * ty);
  return { ia11, ia12, itx, ia21, ia22, ity };
}

function warpAffineNearest(
    src: Float32Array,
    w: number,
    h: number,
    a11: number, a12: number, tx: number,
    a21: number, a22: number, ty: number
): Float32Array {
  const out = new Float32Array(w * h);
  const inv = invertAffine(a11, a12, tx, a21, a22, ty);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const sx = inv.ia11 * x + inv.ia12 * y + inv.itx;
      const sy = inv.ia21 * x + inv.ia22 * y + inv.ity;
      const xi = Math.round(sx), yi = Math.round(sy);
      out[y * w + x] = (xi >= 0 && xi < w && yi >= 0 && yi < h) ? src[yi * w + xi] : 0;
    }
  }
  return out;
}
