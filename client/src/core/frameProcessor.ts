// client/src/core/frameProcessor.ts
import type { InferenceSession, Tensor } from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';

// Глобальный ORT (динамический импорт из CDN/скрипта)
declare const ort: any;

/**
 * Результат тестов Дениса:
 * Ключевые параметры подстройки:
 * - MODEL_INPUT_SIZE: [W,H] вход для MODNet. Cборка q4f16 стабильно работает с [512, 288].
 * - EMA: 0.75–0.85 — темпоральное сглаживание; выше — стабильнее, ниже — быстрее реакция.
 * - NOISE_CUTOFF/HIGH_THRESHOLD/GAMMA: мягкие пороги и гамма для волос. Диапазоны см. комментарии ниже.
 * - USE_BILATERAL и его сигмы: edge-aware фильтр для сохранения тонких волос.
 * - FACE_TRACK: включение BlazeFace-трекинга центра лица + варп предыдущей маски (убирает дрожание).
 * - WARP_GAIN: 0..1 — сила варпа; 0.5–0.8 обычно хорошо.
 */

// Настройки MODNet
const MODEL_INPUT_SIZE: [number, number] = [512, 288]; // [W,H] для MODNet q4f16

// Темпоральная стабилизация
const EMA = 0.7; // 0..1, ↑ стабильность, ↓ реактивность

// Порог/гамма (мягкая логика)
const NOISE_CUTOFF = 0.06;     // все ниже — 0 (0.04–0.08)
const HIGH_THRESHOLD = 0.95;   // все выше — 1 (0.92–0.98)
const GAMMA = 0.8;             // <1 усиливает полутона (0.7–0.9)

// Edge-aware фильтр
const USE_BILATERAL = true;          // включить совместный билатеральный 3×3
const BILATERAL_SIGMA_SPATIAL = 1.0; // радиус по пространству (пиксель)
const BILATERAL_SIGMA_RANGE   = 12.0; // чувствительность к цвету (10–18)


// BlazeFace (face tracking + варп)
const FACE_TRACK = true; // включить трекинг лица
const WARP_GAIN = 0.9;   // сила варпа 0..1

// Путь к blaze.onnx. Если используешь Vite: импортируй URL и подставь сюда.
import BLAZE_MODEL_PATH from '../assets/blaze.onnx?url';

const BLAZE_CONF_THRESHOLD = 0.5; // фильтрация по confidence
const BLAZE_IOU_THRESHOLD  = 0.3; // NMS порог IoU
const BLAZE_MAX_DETECTIONS = 5;   // верхняя граница детекций

const maskCanvas = document.createElement('canvas');
maskCanvas.width = MODEL_INPUT_SIZE[0];
maskCanvas.height = MODEL_INPUT_SIZE[1];
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true })!;

const guideCanvas = document.createElement('canvas');
guideCanvas.width = MODEL_INPUT_SIZE[0];
guideCanvas.height = MODEL_INPUT_SIZE[1];
const guideCtx = guideCanvas.getContext('2d', { willReadFrequently: true })!;

let prevAlpha: Float32Array | null = null;
let prevFaceCenter: { x: number; y: number } | null = null;

let blazeSession: InferenceSession | null = null;

async function ensureBlazeSession(): Promise<void> {
  if (blazeSession) return;
  try {
    blazeSession = await ort.InferenceSession.create(BLAZE_MODEL_PATH, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      enableMemPattern: false,
    });
  } catch (e) {
    console.warn('BlazeFace init failed; disable FACE_TRACK. Reason:', e);
    blazeSession = null;
  }
}



function makeOrtInputFromVideo(videoElement: HTMLVideoElement): Tensor {
  const frame = tf.browser.fromPixels(videoElement); // NHWC
  const resized = tf.image.resizeBilinear(frame, [MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]]);
  const normalized = resized.div(255.0);
  const transposed = normalized.transpose([2, 0, 1]); // CHW
  const inputTensorTf = transposed.expandDims(0);     // NCHW

  const inputData = inputTensorTf.dataSync() as Float32Array;
  const ortInputTensor = new ort.Tensor('float32', inputData, inputTensorTf.shape);

  frame.dispose();
  resized.dispose();
  normalized.dispose();
  transposed.dispose();
  inputTensorTf.dispose();

  return ortInputTensor;
}

function squeezeMaskTo2D(maskTensor: Tensor): Float32Array {
  const [_, __, height, width] = maskTensor.dims;
  const total = height * width;

  if (maskTensor.type === 'uint8') {
    const u8 = maskTensor.data as Uint8Array;
    const out = new Float32Array(total);
    for (let i = 0; i < total; i++) out[i] = u8[i] / 255;
    return out;
  }
  const f32 = maskTensor.data as Float32Array;
  return f32.length === total ? f32 : new Float32Array(f32.slice(0, total));
}

function warpTranslate(src: Float32Array, w: number, h: number, dx: number, dy: number): Float32Array {
  const out = new Float32Array(w * h);
  const sdx = dx | 0;
  const sdy = dy | 0;
  for (let y = 0; y < h; y++) {
    const sy = y - sdy;
    if (sy < 0 || sy >= h) continue;
    for (let x = 0; x < w; x++) {
      const sx = x - sdx;
      if (sx < 0 || sx >= w) continue;
      out[y * w + x] = src[sy * w + sx];
    }
  }
  return out;
}

function temporalEMA(current: Float32Array): Float32Array {
  if (!prevAlpha || prevAlpha.length !== current.length) {
    prevAlpha = current.slice();
    return current;
  }
  for (let i = 0; i < current.length; i++) {
    prevAlpha[i] = EMA * prevAlpha[i] + (1 - EMA) * current[i];
  }
  return prevAlpha;
}

function jointBilateral3x3(
  alpha: Float32Array,
  guidePixels: Uint8ClampedArray,
  w: number,
  h: number,
  sigmaSpatial = BILATERAL_SIGMA_SPATIAL,
  sigmaRange = BILATERAL_SIGMA_RANGE
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

      let sumW = 0;
      let sumA = 0;

      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;

        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;

          const j = yy * w + xx;

          const dr = guidePixels[j * 4] - r0;
          const dg = guidePixels[j * 4 + 1] - g0;
          const db = guidePixels[j * 4 + 2] - b0;
          const range2 = dr * dr + dg * dg + db * db;
          const spatial2 = dx * dx + dy * dy;

          const wgt = Math.exp(-spatial2 / twoSigmaS2) * Math.exp(-range2 / twoSigmaR2);
          sumW += wgt;
          sumA += wgt * alpha[j];
        }
      }
      out[idx] = sumW > 0 ? sumA / sumW : alpha[idx];
    }
  }
  return out;
}

function refineAlphaOnce(a: Float32Array, low = NOISE_CUTOFF, high = HIGH_THRESHOLD, gamma = GAMMA): Float32Array {
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
  guideCanvas.width = w;
  guideCanvas.height = h;
  guideCtx.drawImage(video, 0, 0, w, h);
  return guideCtx.getImageData(0, 0, w, h).data;
}

function alphaToImageData(alpha: Float32Array, w: number, h: number): ImageData {
  const imageData = new ImageData(w, h);
  const pixels = imageData.data;
  for (let i = 0; i < alpha.length; i++) {
    const a = Math.max(0, Math.min(1, alpha[i]));
    const p = i * 4;
    pixels[p] = 255;
    pixels[p + 1] = 255;
    pixels[p + 2] = 255;
    pixels[p + 3] = Math.round(a * 255);
  }
  return imageData;
}

function bilinearUpscale(srcData: Float32Array, srcW: number, srcH: number, dstW: number, dstH: number): Float32Array {
  const out = new Float32Array(dstW * dstH);
  for (let y = 0; y < dstH; y++) {
    const sy = y * (srcH - 1) / (dstH - 1);
    const y0 = Math.floor(sy);
    const y1 = Math.min(y0 + 1, srcH - 1);
    const wy = sy - y0;

    for (let x = 0; x < dstW; x++) {
      const sx = x * (srcW - 1) / (dstW - 1);
      const x0 = Math.floor(sx);
      const x1 = Math.min(x0 + 1, srcW - 1);
      const wx = sx - x0;

      const i00 = y0 * srcW + x0;
      const i01 = y0 * srcW + x1;
      const i10 = y1 * srcW + x0;
      const i11 = y1 * srcW + x1;

      const a =
        srcData[i00] * (1 - wx) * (1 - wy) +
        srcData[i01] * (wx) * (1 - wy) +
        srcData[i10] * (1 - wx) * (wy) +
        srcData[i11] * (wx) * (wy);

      out[y * dstW + x] = Math.min(Math.max(a, 0), 1);
    }
  }
  return out;
}

async function runBlazeFace(
  video: HTMLVideoElement,
  maskW: number,
  maskH: number
): Promise<{ x: number; y: number } | null> {
  if (!blazeSession) return null;

  // Препроцесс: ресайз кадра в 128×128, нормализация [0..1], NCHW [1,3,128,128]
  const FACE_W = 128, FACE_H = 128;

  const frame = tf.browser.fromPixels(video);                      // NHWC
  const resized = tf.image.resizeBilinear(frame, [FACE_H, FACE_W]);
  const normalized = resized.div(255.0);
  const transposed = normalized.transpose([2, 0, 1]);              // CHW
  const inputTensorTf = transposed.expandDims(0);                  // NCHW

  const imageData = inputTensorTf.dataSync() as Float32Array;
  const imageTensor = new ort.Tensor('float32', imageData, [1, 3, FACE_H, FACE_W]);

  frame.dispose();
  resized.dispose();
  normalized.dispose();
  transposed.dispose();
  inputTensorTf.dispose();

  // Формируем входы из Netron
  const confThresholdTensor = new ort.Tensor('float32', new Float32Array([BLAZE_CONF_THRESHOLD]), [1]);
  const maxDetectionsTensor = new ort.Tensor('int64',   new BigInt64Array([BigInt(BLAZE_MAX_DETECTIONS)]), [1]);
  const iouThresholdTensor  = new ort.Tensor('float32', new Float32Array([BLAZE_IOU_THRESHOLD]), [1]);

  // Имя входа 'image' из Netron
  const feeds: Record<string, Tensor> = {
    image: imageTensor,
    conf_threshold: confThresholdTensor,
    max_detections: maxDetectionsTensor,
    iou_threshold: iouThresholdTensor,
  };

  let outputs: Record<string, Tensor>;
  try {
    outputs = await blazeSession.run(feeds) as unknown as Record<string, Tensor>;
  } catch (e) {
    console.warn('BlazeFace run failed:', e);
    return null;
  }

  // Выход selectedBoxes: float32[1,896,16]
  const boxesT = outputs['selectedBoxes'] as Tensor;
  if (!boxesT || !boxesT.data) return null;

  const arr = boxesT.data as Float32Array;
  const dims = boxesT.dims;

  let ymin, xmin, ymax, xmax, conf;
  console.log('dims len: ' + dims.length + ' dims[0] ' + dims[0] + ' dims[1] ' + dims[1])
  
  if (dims.length === 3 && dims[0] === 1 && dims[2] === 16) {
      console.log("Случай [1, N, 16]")
  // Случай [1, N, 16], где N - количество найденных боксов (например, 896)
    // Просто берем самый первый бокс из списка, он обычно лучший.
    // Каждый бокс - это 16 чисел.
    const boxOffset = 0; 
    ymin = arr[boxOffset + 0];
    xmin = arr[boxOffset + 1];
    ymax = arr[boxOffset + 2];
    xmax = arr[boxOffset + 3];
    conf = arr[boxOffset + 4]; // Confidence score

  } else if (dims.length === 2 && dims[0] === 1 && dims[1] === 16) {
    // Случай [1, 16], когда модель вернула только один лучший бокс
    console.log("dims.length === 2 && dims[0] === 1 && dims[1] === 16")
    ymin = arr[0];
    xmin = arr[1];
    ymax = arr[2];
    xmax = arr[3];
    conf = arr[4]; // Confidence score

  } else {
    // Неизвестная форма, пропускаем кадр
    console.warn('Unexpected selectedBoxes shape:', dims);
    return null;
  }

  if (conf < BLAZE_CONF_THRESHOLD) {
    return null;
  }

  const cxNorm = (xmin + xmax) * 0.5;
  const cyNorm = (ymin + ymax) * 0.5;

  const faceX = Math.max(0, Math.min(maskW - 1, Math.round(cxNorm * maskW)));
  const faceY = Math.max(0, Math.min(maskH - 1, Math.round(cyNorm * maskH)));
  
   if (isNaN(faceX) || isNaN(faceY)) {
    return null;
  }

  return { x: faceX, y: faceY };
}

export async function processFrame(
  videoElement: HTMLVideoElement,
  session: InferenceSession,
  outputCtx: CanvasRenderingContext2D,
): Promise<{ inferenceTime: number, totalTime: number }> {
  const totalStartTime = performance.now();

  // BlazeFace — по требованию
  if (FACE_TRACK) {
    await ensureBlazeSession();
  }

  // 1) MODNet препроцесс и инференс
  const ortInputTensor = makeOrtInputFromVideo(videoElement);
  const feeds = { input: ortInputTensor };

  const inferenceStartTime = performance.now();
  const results = await session.run(feeds);
  const inferenceEndTime = performance.now();

  const mask = results['output'] as Tensor;
  const alphaRaw = squeezeMaskTo2D(mask);
  const maskW = mask.dims[3];
  const maskH = mask.dims[2];

  // 2) Face-tracking варп предыдущей маски
  let baseAlpha = alphaRaw;
  if (FACE_TRACK && blazeSession) {
    const center = await runBlazeFace(videoElement, maskW, maskH);
    console.log('Найден центр лица:', center);
    if (center) {
      if (prevFaceCenter && prevAlpha && prevAlpha.length === baseAlpha.length) {
        const dx = (center.x - prevFaceCenter.x) * WARP_GAIN;
        const dy = (center.y - prevFaceCenter.y) * WARP_GAIN;
        const warped = warpTranslate(prevAlpha, maskW, maskH, dx, dy);
        // лёгкое смешивание варпа и текущей маски
        for (let i = 0; i < baseAlpha.length; i++) {
          baseAlpha[i] = 0.5 * baseAlpha[i] + 0.5 * warped[i];
        }
      }
      prevFaceCenter = center;
    }
  }

  // 3) EMA
  const emaAlpha = temporalEMA(baseAlpha);

  // 4) Edge-aware фильтр
  const guidedAlpha = USE_BILATERAL
    ? jointBilateral3x3(emaAlpha, sampleGuidePixels(videoElement, maskW, maskH), maskW, maskH)
    : emaAlpha;

  // 5) Порог/гамма
  const refinedAlpha = refineAlphaOnce(guidedAlpha);

  // 6) Маска → ImageData → композит
  const maskImageData = alphaToImageData(refinedAlpha, maskW, maskH);
  maskCtx.putImageData(maskImageData, 0, 0);

  const outputCanvas = outputCtx.canvas;
  outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

  // Рендер исходного видео
  outputCtx.drawImage(videoElement, 0, 0, outputCanvas.width, outputCanvas.height);

  // Быстрый композит: destination-in (если края не нравятся — см. explicitAlphaBlend ниже)
  outputCtx.globalCompositeOperation = 'destination-in';
  outputCtx.drawImage(maskCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
  outputCtx.globalCompositeOperation = 'source-over';

  const totalEndTime = performance.now();

  return {
    inferenceTime: inferenceEndTime - inferenceStartTime,
    totalTime: totalEndTime - totalStartTime,
  };
}

function explicitAlphaBlend(
  videoElement: HTMLVideoElement,
  alpha: Float32Array,
  alphaW: number,
  alphaH: number,
  ctx: CanvasRenderingContext2D
) {
  const canvas = ctx.canvas;
  const w = canvas.width, h = canvas.height;

  const frame = ctx.getImageData(0, 0, w, h);
  const fd = frame.data;

  const up = bilinearUpscale(alpha, alphaW, alphaH, w, h);

  const bg = [20, 25, 30];
  for (let i = 0, px = 0; i < fd.length; i += 4, px++) {
    const a = up[px];
    const invA = 1 - a;
    const r = fd[i], g = fd[i + 1], b = fd[i + 2];
    fd[i]     = Math.round(r * a + bg[0] * invA);
    fd[i + 1] = Math.round(g * a + bg[1] * invA);
    fd[i + 2] = Math.round(b * a + bg[2] * invA);
    fd[i + 3] = 255;
  }
  ctx.putImageData(frame, 0, 0);
}
