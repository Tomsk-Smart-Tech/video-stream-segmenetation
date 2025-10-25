import * as tf from '@tensorflow/tfjs';


const MODEL_INPUT_SIZE: [number, number] = [320, 320];
const USE_IMAGENET_NORM = false;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];


type OrtnxTensor = InstanceType<typeof ort.Tensor>;

function ensureOrtReady() {
  if (typeof ort === 'undefined' || !ort.Tensor || !ort.InferenceSession) {
    throw new Error('onnxruntime-web (ort) не найден. Подключи скрипт onnxruntime-web перед этим модулем.');
  }
}


function preprocessToTensor(imageSource: HTMLVideoElement | HTMLImageElement): OrtnxTensor {
  ensureOrtReady();

  const tfTensor = tf.tidy(() => {
    const frame = tf.browser.fromPixels(imageSource); // HWC uint8
    const resized = tf.image.resizeBilinear(frame, [MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]], true);
    let floatImg = resized.toFloat().div(255.0); // [0..1]
    if (USE_IMAGENET_NORM) {
      const mean = tf.tensor1d(MEAN);
      const std = tf.tensor1d(STD);
      floatImg = floatImg.sub(mean).div(std);
      mean.dispose();
      std.dispose();
    }
    const nchw = floatImg.transpose([2, 0, 1]).expandDims(0); // -> [1,3,H,W]
    return nchw;
  });

  const data = tfTensor.dataSync() as Float32Array;
  const tensor = new ort.Tensor('float32', data, tfTensor.shape);
  tfTensor.dispose();
  return tensor;
}

function pickMaskOutput(results: Record<string, OrtnxTensor>, preferredNames: string[] = []): OrtnxTensor {
  for (const name of preferredNames) {
    if (results[name]) {
      const t = results[name] as OrtnxTensor;
      if (t.dims.length === 4 && t.dims[0] === 1 && t.dims[1] === 1 && t.dims[2] === MODEL_INPUT_SIZE[1] && t.dims[3] === MODEL_INPUT_SIZE[0]) {
        return t;
      }
    }
  }

  for (const key of Object.keys(results)) {
    const t = results[key] as OrtnxTensor;
    if (
      t.dims &&
      t.dims.length === 4 &&
      t.dims[0] === 1 &&
      t.dims[1] === 1 &&
      t.dims[2] === MODEL_INPUT_SIZE[1] &&
      t.dims[3] === MODEL_INPUT_SIZE[0]
    ) {
      return t;
    }
  }

  for (const key of Object.keys(results)) {
    const t = results[key] as OrtnxTensor;
    if (t.dims && t.dims.length === 4 && t.dims[0] === 1 && t.dims[1] === 1) {
      return t;
    }
  }

  throw new Error('Не удалось найти выход маски [1,1,H,W] в результатах U2Net.');
}


function composeMatteOnCanvas(
  videoElement: HTMLVideoElement,
  mask: OrtnxTensor,
  outputCtx: CanvasRenderingContext2D,
  options?: {
    threshold?: number;
    backgroundColor?: [number, number, number];
    useSoftAlpha?: boolean;
  }
) {
  const threshold = options?.threshold ?? 0.5;
  const bg = options?.backgroundColor ?? [32, 32, 32];
  const useSoftAlpha = options?.useSoftAlpha ?? true;

  const [n, c, h, w] = mask.dims;
  if (n !== 1 || c !== 1) {
    throw new Error(`Ожидался выход маски [1,1,H,W], получили [${mask.dims.join(', ')}]`);
  }
  const maskData = mask.data as Float32Array;

  const canvasW = outputCtx.canvas.width;
  const canvasH = outputCtx.canvas.height;

  const srcTemp = document.createElement('canvas');
  srcTemp.width = w;
  srcTemp.height = h;
  const srcTempCtx = srcTemp.getContext('2d')!;
  srcTempCtx.drawImage(videoElement, 0, 0, w, h);
  const srcImageData = srcTempCtx.getImageData(0, 0, w, h);
  const srcPixels = srcImageData.data;

  const outPixels = new Uint8ClampedArray(w * h * 4);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idxMask = y * w + x;
      const outIdx = (y * w + x) * 4;
      const srcIdx = outIdx;

      let a = Math.max(0, Math.min(1, maskData[idxMask]));
      if (!useSoftAlpha) {
        a = a >= threshold ? 1 : 0;
      } else {
      }

      const rSrc = srcPixels[srcIdx + 0] / 255;
      const gSrc = srcPixels[srcIdx + 1] / 255;
      const bSrc = srcPixels[srcIdx + 2] / 255;

      const r = a * rSrc + (1 - a) * (bg[0] / 255);
      const g = a * gSrc + (1 - a) * (bg[1] / 255);
      const b = a * bSrc + (1 - a) * (bg[2] / 255);

      outPixels[outIdx + 0] = Math.round(r * 255);
      outPixels[outIdx + 1] = Math.round(g * 255);
      outPixels[outIdx + 2] = Math.round(b * 255);
      outPixels[outIdx + 3] = Math.round(a * 255);
    }
  }

  const matteImage = new ImageData(outPixels, w, h);

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = w;
  tempCanvas.height = h;
  const tempCtx = tempCanvas.getContext('2d')!;
  tempCtx.putImageData(matteImage, 0, 0);

  outputCtx.clearRect(0, 0, canvasW, canvasH);
  outputCtx.drawImage(tempCanvas, 0, 0, canvasW, canvasH);
}

export async function processFrameU2NetHuman(
  videoElement: HTMLVideoElement,
  session: ort.InferenceSession,
  outputCtx: CanvasRenderingContext2D,
  opts?: {
    preferredOutputNames?: string[];
    threshold?: number;
    backgroundColor?: [number, number, number];
    useSoftAlpha?: boolean;
  }
): Promise<void> {
  ensureOrtReady();

  const srcTensor = preprocessToTensor(videoElement);

  const feeds: Record<string, OrtnxTensor> = {
    'input.1': srcTensor,
  };

  const results = await session.run(feeds);

  const mask = pickMaskOutput(results, opts?.preferredOutputNames ?? []);

  composeMatteOnCanvas(videoElement, mask, outputCtx, {
    threshold: opts?.threshold ?? 0.5,
    backgroundColor: opts?.backgroundColor ?? [32, 32, 32],
    useSoftAlpha: opts?.useSoftAlpha ?? true,
  });
}


export async function createU2NetSession(modelUrl: string): Promise<ort.InferenceSession> {
  ensureOrtReady();
  const session = await ort.InferenceSession.create(modelUrl, {
  });
  return session;
}
