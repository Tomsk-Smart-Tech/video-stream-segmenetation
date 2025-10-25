import type { InferenceSession, Tensor } from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';
declare const ort: any;


const MODEL_INPUT_SIZE: [number, number] = [512, 288];
const DOWNSAMPLE_RATIO = 0.25;
const USE_EMA = true;
const EMA_BLEND = 0.7;

const maskCanvas = document.createElement('canvas');
maskCanvas.width = MODEL_INPUT_SIZE[0];
maskCanvas.height = MODEL_INPUT_SIZE[1];
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true })!;

let r1: Tensor | null = null; // [1,16,dH,dW] float32
let r2: Tensor | null = null; // [1,20,dH,dW] float32
let r3: Tensor | null = null; // [1,40,dH,dW] float32
let r4: Tensor | null = null; // [1,64,dH,dW] float32


let prevAlpha: Float32Array | null = null;

export async function processFrame(
  videoElement: HTMLVideoElement,
  session: InferenceSession,
  outputCtx: CanvasRenderingContext2D
): Promise<{ inferenceTime: number; totalTime: number }> {
  const totalStart = performance.now();


  const frame = tf.browser.fromPixels(videoElement);
  const resized = tf.image.resizeBilinear(frame, [MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]]);
  const normalized = resized.div(255.0);
  const transposed = normalized.transpose([2, 0, 1]); // CHW
  const inputTensorTf = transposed.expandDims(0);      // NCHW [1,3,H,W]
  const srcData = inputTensorTf.dataSync() as Float32Array;
  const src = new ort.Tensor('float32', srcData, inputTensorTf.shape);

  frame.dispose(); resized.dispose(); normalized.dispose(); transposed.dispose(); inputTensorTf.dispose();

  const H = src.dims[2], W = src.dims[3];
  const dH = Math.max(1, Math.round(H * DOWNSAMPLE_RATIO));
  const dW = Math.max(1, Math.round(W * DOWNSAMPLE_RATIO));

  const ratio = new ort.Tensor('float32', new Float32Array([DOWNSAMPLE_RATIO]), [1]);

  if (!r1 || r1.dims[2] !== dH || r1.dims[3] !== dW) {
    r1 = zeroF32([1, 16, dH, dW]);
    r2 = zeroF32([1, 20, dH, dW]);
    r3 = zeroF32([1, 40, dH, dW]);
    r4 = zeroF32([1, 64, dH, dW]);
  }

  const feeds: Record<string, Tensor> = {
    src,
    downsample_ratio: ratio,
    r1i: r1!, r2i: r2!, r3i: r3!, r4i: r4!,
  };

  const infStart = performance.now();
  const results = await session.run(feeds);
  const infEnd = performance.now();


  const phaT = results['pha'] as Tensor;
  r1 = results['r1o'] as Tensor;
  r2 = results['r2o'] as Tensor;
  r3 = results['r3o'] as Tensor;
  r4 = results['r4o'] as Tensor;

  const alpha = toFloat32Squeezed(phaT);

  const alphaSmoothed = USE_EMA ? ema(alpha, EMA_BLEND) : alpha;

  const maskImage = alphaToImageData(alphaSmoothed, W, H);
  maskCtx.putImageData(maskImage, 0, 0);

  const canvas = outputCtx.canvas;
  outputCtx.clearRect(0, 0, canvas.width, canvas.height);
  outputCtx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  outputCtx.globalCompositeOperation = 'destination-in';
  outputCtx.drawImage(maskCanvas, 0, 0, canvas.width, canvas.height);
  outputCtx.globalCompositeOperation = 'source-over';

  const totalEnd = performance.now();
  return { inferenceTime: infEnd - infStart, totalTime: totalEnd - totalStart };
}


// Нулевой float32 тензор
function zeroF32(dims: number[]): Tensor {
  const total = dims.reduce((a, b) => a * b, 1);
  return new ort.Tensor('float32', new Float32Array(total), dims);
}

// Универсальная конверсия pha → Float32Array [H*W]
function toFloat32Squeezed(t: Tensor): Float32Array {
  const [_, __, H, W] = t.dims;
  const total = H * W;
  if (t.type === 'float32') {
    const f32 = t.data as Float32Array;
    return f32.length === total ? f32 : new Float32Array(f32.slice(0, total));
  } else if (t.type === 'float16') {
    const src = t.data as Uint16Array;
    const out = new Float32Array(total);
    for (let i = 0; i < total; i++) out[i] = f16BitsToF32(src[i]);
    return out;
  } else if (t.type === 'uint8') {
    const u8 = t.data as Uint8Array;
    const out = new Float32Array(total);
    for (let i = 0; i < total; i++) out[i] = u8[i] / 255;
    return out;
  } else {

    const any = t.data as any;
    const out = new Float32Array(total);
    for (let i = 0; i < total; i++) out[i] = any[i] as number;
    return out;
  }
}

// Альфа → ImageData
function alphaToImageData(alpha: Float32Array, w: number, h: number): ImageData {
  const img = new ImageData(w, h);
  const px = img.data;
  for (let i = 0; i < alpha.length; i++) {
    const a = Math.max(0, Math.min(1, alpha[i]));
    const p = i * 4;
    px[p] = 255; px[p + 1] = 255; px[p + 2] = 255; px[p + 3] = Math.round(a * 255);
  }
  return img;
}


function ema(current: Float32Array, k: number): Float32Array {
  if (!prevAlpha || prevAlpha.length !== current.length) {
    prevAlpha = current.slice(); return current;
  }
  const out = new Float32Array(current.length);
  for (let i = 0; i < current.length; i++) {
    prevAlpha[i] = k * prevAlpha[i] + (1 - k) * current[i];
    out[i] = prevAlpha[i];
  }
  return out;
}

function f16BitsToF32(h: number): number {
  const sign = (h >>> 15) & 1; let exp = (h >>> 10) & 0x1F; let mant = h & 0x3FF;
  if (exp === 0) { if (mant === 0) return sign ? -0 : 0; const m = mant / 1024.0; return (sign ? -1 : 1) * m * Math.pow(2, -14); }
  if (exp === 0x1F) return mant ? NaN : (sign ? -Infinity : Infinity);
  const m = 1 + mant / 1024.0; const e = exp - 15; return (sign ? -1 : 1) * m * Math.pow(2, e);
}
