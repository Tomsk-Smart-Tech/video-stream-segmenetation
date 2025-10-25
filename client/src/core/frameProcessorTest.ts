// client/src/core/frameProcessor_branch.ts
import type { InferenceSession, Tensor } from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';

declare const ort: any;

const USE_NEG_ONE_TO_ONE = false;

// Параметры
const MODEL_INPUT_SIZE: [number, number] = [512, 288]; // MODNet

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

// Для кропа и лэндмарков
const FD_INPUT = [256, 256]; // FaceDetector
const LMK_INPUT = [192, 192]; // Landmarks
const FACE_SCORE_THRESH = 0.6;

// Временные холсты
const maskCanvas = document.createElement('canvas');
maskCanvas.width = MODEL_INPUT_SIZE[0];
maskCanvas.height = MODEL_INPUT_SIZE[1];
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true })!;

const faceCanvas = document.createElement('canvas');
const faceCtx = faceCanvas.getContext('2d', { willReadFrequently: true })!;

// Состояние
let prevAlpha: Float32Array | null = null;

// Опции вызова
type ProcessOpts = {
  lastAffine?: { a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null;
  faceSession?: InferenceSession | null;
  landmarkSession?: InferenceSession | null;
  frameIdx?: number;
  lmkInterval?: number;
};

// Главная функция
// client/src/core/frameProcessor_branch.ts — обновлённая processFrame
export async function processFrame(
  videoElement: HTMLVideoElement,
  session: InferenceSession,
  outputCtx: CanvasRenderingContext2D,
  opts: {
    lastAffine?: { a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null;
    faceSession?: InferenceSession | null;
    landmarkSession?: InferenceSession | null;
    frameIdx?: number;
    lmkInterval?: number;
  } = {}
): Promise<{
  inferenceTime: number;
  totalTime: number;
  updatedAffine?: { a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null;
}> {
  const totalStartTime = performance.now();

  // 1) MODNet препроцесс → NCHW
  const frame = tf.browser.fromPixels(videoElement);
  const resized = tf.image.resizeBilinear(frame, [MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]]);
  const normalized = resized.div(255.0);
  const transposed = normalized.transpose([2, 0, 1]);
  const inputTensorTf = transposed.expandDims(0);
  const inputData = inputTensorTf.dataSync() as Float32Array;
  const ortInputTensor = new ort.Tensor('float32', inputData, inputTensorTf.shape);

  frame.dispose(); resized.dispose(); normalized.dispose(); transposed.dispose(); inputTensorTf.dispose();

  // 2) Запуск MODNet
  const infStart = performance.now();
  const results = await session.run({ input: ortInputTensor });
  const infEnd = performance.now();

  const mask = results['output'] as Tensor;
  const alphaRaw = squeezeMaskTo2D(mask);
  const maskW = mask.dims[3];
  const maskH = mask.dims[2];

  let baseAlpha = alphaRaw;

  // 3) Варп предыдущей маски по lastAffine (стабилизация)
  if (opts.lastAffine && prevAlpha && prevAlpha.length === baseAlpha.length) {
    const warped = warpAffineNearest(
      prevAlpha, maskW, maskH,
      opts.lastAffine.a11, opts.lastAffine.a12, opts.lastAffine.tx,
      opts.lastAffine.a21, opts.lastAffine.a22, opts.lastAffine.ty
    );
    const WARP_BLEND_WEIGHT = 0.3; // 30% прошлое, 70% текущий кадр
    for (let i = 0; i < baseAlpha.length; i++) {
      baseAlpha[i] = warped[i] * WARP_BLEND_WEIGHT + baseAlpha[i] * (1 - WARP_BLEND_WEIGHT);
    }
  }

  // 4) EMA по времени
  const emaAlpha = temporalEMA(baseAlpha);

  // 5) Морфологический opening для удаления мелкого шума
  const openedAlpha = morphologicalOpening(emaAlpha, maskW, maskH);

  // 6) FaceDetector → ROI → Landmarks → аффинная матрица + эллиптический face prior
  let guidePixels: Uint8ClampedArray | null = null;
  let updatedAffine: { a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null = null;
  let facePrior: Float32Array | null = null;

  const canRunLmk =
    opts.faceSession &&
    opts.landmarkSession &&
    typeof opts.frameIdx === 'number' &&
    typeof opts.lmkInterval === 'number' &&
    opts.frameIdx % opts.lmkInterval === 0;

  if (canRunLmk) {
    const det = await runFaceDetector(videoElement, opts.faceSession!);
    if (det && det.score >= FACE_SCORE_THRESH) {
      // Эллиптический prior из детекции
      facePrior = facePriorMask(det.box, videoElement.videoWidth, videoElement.videoHeight, maskW, maskH);

      // ROI для landmarks
      const roi = cropFaceROI(videoElement, det.box, 0.25);
      guidePixels = roi.imageData.data;

      const lm = await runLandmarks468(roi.imageData, opts.landmarkSession!);
      if (lm && lm.score >= 0.3) {
        updatedAffine = estimateAffineFromLandmarks(
          lm.points,
          roi.transformToFull,
          maskW,
          maskH,
          videoElement.videoWidth,
          videoElement.videoHeight
        );
      }
    }
  }

  // 7) Дополнительный локальный closing в зоне лица, чтобы вернуть края, съеденные opening
  const openedClosedAlpha = morphologicalClosingInPrior(openedAlpha, facePrior, maskW, maskH);

  // 8) Опциональный edge-aware (по умолчанию выключен)
  const guideImage = sampleGuidePixels(videoElement, maskW, maskH);
  const guidedAlpha = config.USE_BILATERAL
    ? jointBilateral3x3(openedClosedAlpha, guideImage, maskW, maskH)
    : openedClosedAlpha;

  // 9) Порог + гамма + жёсткий prior-кламп
  const refinedAlpha = refineAlphaOnce(guidedAlpha, config.NOISE_CUTOFF, config.HIGH_THRESHOLD, config.GAMMA, facePrior ?? undefined);

  // 10) Маска → Canvas композит
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
    updatedAffine
  };
}


// Сжатие [1,1,H,W] → Float32Array(H*W)
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

// Превратить плоскую маску (Float32Array H×W) в ImageData
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

function refineAlphaOnce(
  a: Float32Array,
  low = config.NOISE_CUTOFF,
  high = config.HIGH_THRESHOLD,
  gamma = config.GAMMA,
  prior?: Float32Array
): Float32Array {
  const out = new Float32Array(a.length);
  const denom = Math.max(1e-6, high - low);

  // Параметры клампа
  const minFaceFloor = 0.55;  // внутри лица альфа не должна опускаться ниже
  const maxNearBgCap = 0.35;  // вне лица в окрестности, альфа не должна превышать этот потолок
  const nearBgBlend = 0.15;   // насколько сильно ограничиваем «рядом с лицом» вне prior

  for (let i = 0; i < a.length; i++) {
    let v = a[i];

    // Base threshold+gamma
    if (v <= low) {
      v = 0;
    } else if (v >= high) {
      v = 1;
    } else {
      const t = (v - low) / denom;
      v = Math.pow(t, gamma);
    }

    if (prior) {
      const p = prior[i];

      if (p > 0.25) {
        // Внутри/на границе лица: жесткий «пол»
        v = Math.max(v, Math.min(1, minFaceFloor * p + 0.15)); // плавный рост к центру
      } else if (p > 0) {
        // В окрестности лица (тонкая зона), ограничиваем фон
        v = Math.min(v, maxNearBgCap + nearBgBlend * p);
      }
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

//FACE DETECTOR + LANDMARKS ИНТЕГРАЦИЯ 

function preprocessToNCHW(imageData: ImageData, targetH: number, targetW: number): { tensor: any } {
  const srcW = imageData.width;
  const srcH = imageData.height;
  const c = document.createElement('canvas');
  c.width = targetW; c.height = targetH;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(
    (() => {
      const tmp = document.createElement('canvas');
      tmp.width = srcW; tmp.height = srcH;
      const tctx = tmp.getContext('2d', { willReadFrequently: true })!;
      tctx.putImageData(imageData, 0, 0);
      return tmp;
    })(),
    0, 0, targetW, targetH
  );
  const img = ctx.getImageData(0, 0, targetW, targetH);
  const u8 = img.data;

  const chw = new Float32Array(1 * 3 * targetH * targetW);
  let pR = 0, pG = targetH * targetW, pB = 2 * targetH * targetW;
  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const idx = (y * targetW + x) * 4;
      const R = u8[idx] / 255;
      const G = u8[idx + 1] / 255;
      const B = u8[idx + 2] / 255;
      chw[pR++] = R;
      chw[pG++] = G;
      chw[pB++] = B;
    }
  }
  const tensor = new ort.Tensor('float32', chw, [1, 3, targetH, targetW]);
  return { tensor };
}

type DetBox = { x0: number; y0: number; x1: number; y1: number };
type FaceDetResult = { box: DetBox; score: number };

async function runFaceDetector(video: HTMLVideoElement, session: InferenceSession): Promise<FaceDetResult | null> {
  const c = document.createElement('canvas');
  c.width = video.videoWidth; c.height = video.videoHeight;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.drawImage(video, 0, 0, c.width, c.height);
  const img = ctx.getImageData(0, 0, c.width, c.height);

  // [ИЗМЕНЕНИЕ 1] Получаем не только тензор, но и функцию обратного отображения
  const { tensor, letterboxMap } = preprocessToNCHW(img, FD_INPUT[1], FD_INPUT[0]); // [H,W]=[256,256]
  
  const outputs = await session.run({ image: tensor }) as Record<string, Tensor>;
  const coordsT = outputs['box_coords'] as Tensor; // [1,896,16]
  const scoresT = outputs['box_scores'] as Tensor; // [1,896,1]

  if (!coordsT || !scoresT) return null;
  const coords = coordsT.data as Float32Array;
  const scores = scoresT.data as Float32Array;
  const num = coordsT.dims[1];

  let bestIdx = -1;
  let bestScore = -Infinity;
  for (let i = 0; i < num; i++) {
    const s = scores[i];
    if (s > bestScore) {
      bestScore = s;
      bestIdx = i;
    }
  }
  if (bestIdx < 0 || !letterboxMap) return null; // Добавили проверку на letterboxMap

  const base = bestIdx * 16;
  const x0n = coords[base + 0];
  const y0n = coords[base + 1];
  const x1n = coords[base + 2];
  const y1n = coords[base + 3];

  // [ИЗМЕНЕНИЕ 2] Применяем обратное преобразование координат
  // Сначала переводим нормализованные координаты [0,1] в пиксели пространства модели (256x256)
  const p0_letterbox = { x: x0n * FD_INPUT[0], y: y0n * FD_INPUT[1] };
  const p1_letterbox = { x: x1n * FD_INPUT[0], y: y1n * FD_INPUT[1] };
  
  // Теперь отображаем эти пиксели обратно в пространство оригинального видео
  const p0_full = letterboxMap(p0_letterbox);
  const p1_full = letterboxMap(p1_letterbox);

  const x0 = Math.max(0, Math.min(c.width, p0_full.x));
  const y0 = Math.max(0, Math.min(c.height, p0_full.y));
  const x1 = Math.max(0, Math.min(c.width, p1_full.x));
  const y1 = Math.max(0, Math.min(c.height, p1_full.y));
  
  if (x1 <= x0 || y1 <= y0) return null;

  return { box: { x0, y0, x1, y1 }, score: bestScore };
}

function cropFaceROI(video: HTMLVideoElement, box: DetBox, padRatio = 0.25): { imageData: ImageData; transformToFull: (pt: { x: number; y: number }) => { x: number; y: number } } {
  const vw = video.videoWidth, vh = video.videoHeight;
  const bw = box.x1 - box.x0, bh = box.y1 - box.y0;
  const padX = bw * padRatio, padY = bh * padRatio;
  const x0 = Math.max(0, Math.floor(box.x0 - padX));
  const y0 = Math.max(0, Math.floor(box.y0 - padY));
  const x1 = Math.min(vw, Math.ceil(box.x1 + padX));
  const y1 = Math.min(vh, Math.ceil(box.y1 + padY));
  const rw = Math.max(1, x1 - x0);
  const rh = Math.max(1, y1 - y0);

  faceCanvas.width = rw;
  faceCanvas.height = rh;
  faceCtx.clearRect(0, 0, rw, rh);
  faceCtx.drawImage(video, x0, y0, rw, rh, 0, 0, rw, rh);
  const imageData = faceCtx.getImageData(0, 0, rw, rh);

  const transformToFull = (pt: { x: number; y: number }) => ({ x: pt.x + x0, y: pt.y + y0 });
  return { imageData, transformToFull };
}

type LandmarksResult = { points: { x: number; y: number; z?: number }[]; score: number };

// client/src/core/frameProcessor_branch.ts (продолжение)
async function runLandmarks468(roiImage: ImageData, session: InferenceSession): Promise<LandmarksResult | null> {
  // Препроцесс ROI → [1,3,192,192]
  const { tensor } = preprocessToNCHW(roiImage, LMK_INPUT[1], LMK_INPUT[0]); // [H,W]=[192,192]
  const outputs = await session.run({ image: tensor }) as Record<string, Tensor>;
  const scoresT = outputs['scores'] as Tensor;     // float32 [1]
  const lmT = outputs['landmarks'] as Tensor;      // float32 [1,468,3]
  if (!scoresT || !lmT) return null;

  const score = (scoresT.data as Float32Array)[0] ?? 0;
  const lmData = lmT.data as Float32Array;
  const num = lmT.dims[1]; // 468
  const dim = lmT.dims[2]; // 3

  const pts: { x: number; y: number; z?: number }[] = new Array(num);
  // Координаты предположительно нормализованы в [0..1] по ширине/высоте входа 192×192
  for (let i = 0; i < num; i++) {
    const base = i * dim;
    const xn = lmData[base + 0];
    const yn = lmData[base + 1];
    const zn = lmData[base + 2];
    // Конвертируем в пиксели ROI, чтобы избежать накопления ошибок при последующих вычислениях
    pts[i] = { x: xn * roiImage.width, y: yn * roiImage.height, z: zn };
  }

  return { points: pts, score };
}

// Оценка аффинной матрицы (подобие: масштаб+поворот+сдвиг) из якорных лэндмарков.
// Возвращаем матрицу в координатах маски (H×W MODNet), чтобы варпать prevAlpha.
// Используем 5 якорей: правый/левый глаз, кончик носа, уголки рта.
function estimateAffineFromLandmarks(
  pointsROI: { x: number; y: number }[],
  mapToFull: (pt: { x: number; y: number }) => { x: number; y: number },
  maskW: number,
  maskH: number,
  videoW: number,
  videoH: number
): { a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null {
  if (pointsROI.length < 300) return null;

  // Индексы приближённые к MediaPipe FaceMesh:
  // 33 — правый глаз внешний, 263 — левый глаз внешний, 1 — кончик носа,
  // 13 — правый уголок рта, 14 — левый уголок рта
  const idxs = [33, 263, 1, 13, 14].filter(i => i >= 0 && i < pointsROI.length);
  if (idxs.length < 3) return null;

  // Преобразуем ROI-пиксели → координаты полного кадра (в пикселях видео)
  const dstFull: { x: number; y: number }[] = idxs.map(i => mapToFull(pointsROI[i]));

  // Эталонные точки (в нормализованных координатах лица, эмпирически подобранные):
  // Используем единичный масштаб и центр около (0.5, 0.5) для стабильности.
  const refNorm = [
    { x: 0.35, y: 0.4 }, // right eye
    { x: 0.65, y: 0.4 }, // left eye
    { x: 0.50, y: 0.55 }, // nose tip
    { x: 0.58, y: 0.70 }, // mouth right
    { x: 0.42, y: 0.70 }, // mouth left
  ];
  const ref: { x: number; y: number }[] = idxs.map((_, k) => ({
    x: refNorm[k].x * videoW,
    y: refNorm[k].y * videoH,
  }));

  // Решаем подобие через обычный Procrustes (Kabsch для 2D с масштабом):
  // 1) Центрируем
  const cxRef = avg(ref.map(p => p.x));
  const cyRef = avg(ref.map(p => p.y));
  const cxDst = avg(dstFull.map(p => p.x));
  const cyDst = avg(dstFull.map(p => p.y));

  const refC = ref.map(p => ({ x: p.x - cxRef, y: p.y - cyRef }));
  const dstC = dstFull.map(p => ({ x: p.x - cxDst, y: p.y - cyDst }));

  // 2) Оценим масштаб как отношение средних норм
  const refNormSum = sum(refC.map(p => p.x * p.x + p.y * p.y));
  const dstNormSum = sum(dstC.map(p => p.x * p.x + p.y * p.y));
  if (refNormSum < 1e-6 || dstNormSum < 1e-6) return null;

  // 3) Оценим поворот через матрицу кросс-ковариации
  // C = [ sum(rx*dx + ry*dy), sum(-ry*dx + rx*dy) ] → эквивалент 2D
  const Sxx = sum(refC.map((r, i) => r.x * dstC[i].x + r.y * dstC[i].y));
  const Sxy = sum(refC.map((r, i) => -r.y * dstC[i].x + r.x * dstC[i].y));
  // Угол поворота θ, масштаб s
  const theta = Math.atan2(Sxy, Sxx);
  const cosT = Math.cos(theta);
  const sinT = Math.sin(theta);

  // Масштаб s: dst ≈ s R ref, оценим по отношению дисперсий
  const s = Math.sqrt(dstNormSum / refNormSum);

  // 4) Трансляция
  // dst ≈ s R (ref - cRef) + cDst → tx, ty
  const tx = cxDst - (s * (cosT * cxRef - sinT * cyRef));
  const ty = cyDst - (s * (sinT * cxRef + cosT * cyRef));

  // Получили трансформацию в координатах видео: x' = a11 x + a12 y + tx; y' = a21 x + a22 y + ty
  const a11_v = s * cosT;
  const a12_v = -s * sinT;
  const a21_v = s * sinT;
  const a22_v = s * cosT;

  // Переведём матрицу в координаты маски (maskW×maskH), учитывая масштаб от видео к маске.
  // Маска растянута на размер видео при композите, но варп prevAlpha идёт в пространстве маски H×W.
  const sx = maskW / videoW;
  const sy = maskH / videoH;

  // Аффинная в пикселях маски: сначала применяем видео-переход, затем масштабируем в mask-space
  // x_mask' = sx * x_video', y_mask' = sy * y_video'
  const a11 = a11_v;
  const a12 = a12_v;
  const a21 = a21_v;
  const a22 = a22_v;
  const tx_v = tx;
  const ty_v = ty;

  // Учтём перевод из видео в маску:
  // x_m' = sx*(a11_v*x_v + a12_v*y_v + tx_v)
  // y_m' = sy*(a21_v*x_v + a22_v*y_v + ty_v)
  return {
    a11: a11_v, // применяются на индексах x_v, но warpAffineNearest уже работает в маске, поэтому конвертируем tx/ty
    a12: a12_v,
    tx: tx_v * sx,
    a21: a21_v,
    a22: a22_v,
    ty: ty_v * sy,
  };
}

function avg(arr: number[]): number {
  return arr.length ? sum(arr) / arr.length : 0;
}
function sum(arr: number[]): number {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i];
  return s;
}


function toSquareLetterbox(imageData: ImageData, target: number, fill = [0,0,0,255]): { data: ImageData, mapFromSquareToSrc: (pt:{x:number;y:number})=>{x:number;y:number} } {
  const srcW = imageData.width, srcH = imageData.height;
  const scale = Math.min(target / srcW, target / srcH);
  const drawW = Math.max(1, Math.round(srcW * scale));
  const drawH = Math.max(1, Math.round(srcH * scale));
  const offX = Math.floor((target - drawW) / 2);
  const offY = Math.floor((target - drawH) / 2);

  const c = document.createElement('canvas'); c.width = target; c.height = target;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  // fill letterbox
  const bg = ctx.createImageData(target, target);
  for (let i=0;i<bg.data.length;i+=4) {
    bg.data[i]=fill[0]; bg.data[i+1]=fill[1]; bg.data[i+2]=fill[2]; bg.data[i+3]=fill[3];
  }
  ctx.putImageData(bg,0,0);

  // draw source into letterbox
  const tmp = document.createElement('canvas'); tmp.width = srcW; tmp.height = srcH;
  const tctx = tmp.getContext('2d', { willReadFrequently: true })!;
  tctx.putImageData(imageData,0,0);
  ctx.drawImage(tmp, 0,0,srcW,srcH, offX, offY, drawW, drawH);

  const data = ctx.getImageData(0,0,target,target);
  const mapFromSquareToSrc = (pt:{x:number;y:number})=>({
    x: (pt.x - offX) / scale,
    y: (pt.y - offY) / scale
  });
  return { data, mapFromSquareToSrc };
}

function morphologicalOpening(alpha: Float32Array, w: number, h: number): Float32Array {
    const eroded = new Float32Array(alpha.length);
    const dilated = new Float32Array(alpha.length);

    // 1. Эрозия (минимальный фильтр 3x3)
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            const centerIdx = y * w + x;
            let minVal = 1.0;
            // Проходим по окну 3x3
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const idx = (y + dy) * w + (x + dx);
                    if (alpha[idx] < minVal) {
                        minVal = alpha[idx];
                    }
                }
            }
            eroded[centerIdx] = minVal;
        }
    }

    // 2. Дилэтация (максимальный фильтр 3x3) по результату эрозии
    for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
            const centerIdx = y * w + x;
            let maxVal = 0.0;
            // Проходим по окну 3x3
            for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                    const idx = (y + dy) * w + (x + dx);
                    if (eroded[idx] > maxVal) { // Используем eroded массив!
                        maxVal = eroded[idx];
                    }
                }
            }
            dilated[centerIdx] = maxVal;
        }
    }
    
    return dilated;
}


/**
 * Создает маску-карту уверенности для области лица.
 * @param box - Прямоугольник лица от детектора (в координатах видео)
 * @param videoW - Ширина видео
 * @param videoH - Высота видео
 * @param maskW - Ширина целевой маски
 * @param maskH - Высота целевой маски
 * @returns - Float32Array с мягкой маской лица
 */
function facePriorMask(
  box: DetBox,
  videoW: number,
  videoH: number,
  maskW: number,
  maskH: number
): Float32Array {
  const out = new Float32Array(maskW * maskH);

  // Перевод бокса в координаты маски
  const sx = maskW / videoW;
  const sy = maskH / videoH;
  const x0 = Math.floor(box.x0 * sx);
  const y0 = Math.floor(box.y0 * sy);
  const x1 = Math.ceil(box.x1 * sx);
  const y1 = Math.ceil(box.y1 * sy);

  // Эллипс, приближенно соответствующий голове: центр и радиусы
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const rx = (x1 - x0) * 0.56; // голова шире по горизонтали, но не сильно
  const ry = (y1 - y0) * 0.70; // по вертикали чуть больше (с учётом лба/подбородка)
  const pad = Math.max(4, Math.floor(Math.min(maskW, maskH) * 0.02)); // мягкая граница

  for (let y = 0; y < maskH; y++) {
    for (let x = 0; x < maskW; x++) {
      const dx = (x - cx) / Math.max(1e-6, rx);
      const dy = (y - cy) / Math.max(1e-6, ry);
      const d2 = dx*dx + dy*dy; // <= 1 внутри эллипса
      let v = 0;
      if (d2 <= 1) {
        // косинусная рампа от центра к границе
        // t=0 в центре, t=1 на границе
        const t = Math.sqrt(Math.max(0, Math.min(1, d2)));
        v = 0.5 - 0.5 * Math.cos(Math.PI * (1 - t));
        // небольшой soft-расширитель до pad пикселей за границу
        if (d2 > 1 - (pad / Math.max(rx, ry))) {
          v = Math.max(v, 0.25); // минимальный «порог уверенности» в краевой зоне лица
        }
      }
      out[y * maskW + x] = v;
    }
  }
  return out;
}

function morphologicalClosingInPrior(alpha: Float32Array, prior: Float32Array | null, w: number, h: number): Float32Array {
  if (!prior) return alpha;
  const dilated = new Float32Array(alpha.length);
  const closed = new Float32Array(alpha.length);

  // Дилатация 3x3, но применяем только там, где prior>0 (лицо)
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const center = y * w + x;
      if (prior[center] <= 0) {
        dilated[center] = alpha[center];
        continue;
      }
      let maxVal = 0.0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const idx = (y + dy) * w + (x + dx);
          if (alpha[idx] > maxVal) maxVal = alpha[idx];
        }
      }
      dilated[center] = maxVal;
    }
  }

  // Эрозия 3x3, тоже только в зоне prior
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const center = y * w + x;
      if (prior[center] <= 0) {
        closed[center] = dilated[center];
        continue;
      }
      let minVal = 1.0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const idx = (y + dy) * w + (x + dx);
          if (dilated[idx] < minVal) minVal = dilated[idx];
        }
      }
      closed[center] = minVal;
    }
  }

  return closed;
}
