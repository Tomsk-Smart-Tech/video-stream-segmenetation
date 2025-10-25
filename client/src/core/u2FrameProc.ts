// client/src/core/frameProcessor_U2NetHuman.ts
import * as tf from '@tensorflow/tfjs';

// Предпосылки:
// - onnxruntime-web подключён через <script>, доступен как глобальный `ort`.
// - tf (TensorFlow.js) уже подключён, используем для препроцессинга.
// - Модель u2net-human-seg имеет вход "input.1": float32 [1,3,320,320] и
//   несколько выходов. Нам нужен финальный выход маски: [1,1,320,320], sigmoid.

// Конфигурации
const MODEL_INPUT_SIZE: [number, number] = [320, 320]; // [W, H] как в Netron
const USE_IMAGENET_NORM = false; // если твой экспорт требует mean/std, включи и проверь
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// Типы
type OrtnxTensor = InstanceType<typeof ort.Tensor>;

function ensureOrtReady() {
  if (typeof ort === 'undefined' || !ort.Tensor || !ort.InferenceSession) {
    throw new Error('onnxruntime-web (ort) не найден. Подключи скрипт onnxruntime-web перед этим модулем.');
  }
}

/**
 * Препроцессинг видео/изображения в тензор ONNX NCHW float32.
 */
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

/**
 * Выбрать корректный выход маски из набора outputs.
 * Правила:
 * 1) Берём выход с формой [1,1,320,320], если есть несколько — берём первый.
 * 2) Если имена известны, можно указать preferredNames — приоритетный список.
 */
function pickMaskOutput(results: Record<string, OrtnxTensor>, preferredNames: string[] = []): OrtnxTensor {
  // 1) Если указаны приоритетные имена — проверим их сначала
  for (const name of preferredNames) {
    if (results[name]) {
      const t = results[name] as OrtnxTensor;
      if (t.dims.length === 4 && t.dims[0] === 1 && t.dims[1] === 1 && t.dims[2] === MODEL_INPUT_SIZE[1] && t.dims[3] === MODEL_INPUT_SIZE[0]) {
        return t;
      }
    }
  }

  // 2) Иначе ищем любой [1,1,H,W] == [1,1,320,320]
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

  // 3) Если нет точного совпадения — попробуем найти [1,1,H,W] и затем ресайз
  for (const key of Object.keys(results)) {
    const t = results[key] as OrtnxTensor;
    if (t.dims && t.dims.length === 4 && t.dims[0] === 1 && t.dims[1] === 1) {
      return t;
    }
  }

  throw new Error('Не удалось найти выход маски [1,1,H,W] в результатах U2Net.');
}

/**
 * Постобработка: приводим маску к Uint8ClampedArray (RGBA) или к альфе,
 * и компонуем с исходным кадром. Здесь делаем:
 * - Сигмоид (на всякий случай, многие экспорты уже применяют sigmoid)
 * - Бинаризацию по порогу (настраиваем порог)
 * - Композицию: matte = alpha*src + (1-alpha)*backgroundColor
 */
function composeMatteOnCanvas(
  videoElement: HTMLVideoElement,
  mask: OrtnxTensor, // [1,1,H,W]
  outputCtx: CanvasRenderingContext2D,
  options?: {
    threshold?: number;  // порог бинаризации
    backgroundColor?: [number, number, number]; // RGB фон
    useSoftAlpha?: boolean; // если true, используем мягкую альфу без жесткой бинаризации
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

  // Снимем исходный кадр в размер Canvas
  const canvasW = outputCtx.canvas.width;
  const canvasH = outputCtx.canvas.height;

  // Захватываем исходный кадр из video на временный канвас
  const srcTemp = document.createElement('canvas');
  srcTemp.width = w;
  srcTemp.height = h;
  const srcTempCtx = srcTemp.getContext('2d')!;
  // Отрисуем видео в размер модели (w,h), затем будем масштабировать на выход
  srcTempCtx.drawImage(videoElement, 0, 0, w, h);
  const srcImageData = srcTempCtx.getImageData(0, 0, w, h);
  const srcPixels = srcImageData.data; // RGBA Uint8ClampedArray

  // Создаём выход RGBA в модельном размере
  const outPixels = new Uint8ClampedArray(w * h * 4);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idxMask = y * w + x;
      const outIdx = (y * w + x) * 4;
      const srcIdx = outIdx;

      // Маска может быть уже сигмоидной; применим мягкую сигмоиду на всякий случай
      let a = Math.max(0, Math.min(1, maskData[idxMask]));
      // Бинаризация при необходимости
      if (!useSoftAlpha) {
        a = a >= threshold ? 1 : 0;
      } else {
        // Можно слегка растянуть контраст вокруг порога
        // a = 1 / (1 + Math.exp(-12*(a - threshold))); // опционально
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
      outPixels[outIdx + 3] = Math.round(a * 255); // альфа
    }
  }

  const matteImage = new ImageData(outPixels, w, h);

  // Рисуем на выходной Canvas с масштабированием до его фактического размера
  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = w;
  tempCanvas.height = h;
  const tempCtx = tempCanvas.getContext('2d')!;
  tempCtx.putImageData(matteImage, 0, 0);

  outputCtx.clearRect(0, 0, canvasW, canvasH);
  outputCtx.drawImage(tempCanvas, 0, 0, canvasW, canvasH);
}

/**
 * Основная функция: прогон u2net-human-seg и рендер результата.
 */
export async function processFrameU2NetHuman(
  videoElement: HTMLVideoElement,
  session: ort.InferenceSession,
  outputCtx: CanvasRenderingContext2D,
  opts?: {
    preferredOutputNames?: string[]; // Например: ['1959', 'D0', 'sigmoid_out']
    threshold?: number;
    backgroundColor?: [number, number, number];
    useSoftAlpha?: boolean;
  }
): Promise<void> {
  ensureOrtReady();

  // 1. Препроцессинг
  const srcTensor = preprocessToTensor(videoElement);

  // 2. Запуск модели
  // Имя входа по Netron: "input.1"
  const feeds: Record<string, OrtnxTensor> = {
    'input.1': srcTensor,
  };

  const results = await session.run(feeds);

  // 3. Выбор нужного выхода маски
  const mask = pickMaskOutput(results, opts?.preferredOutputNames ?? []);

  // 4. Постпроцесс и отрисовка на Canvas
  composeMatteOnCanvas(videoElement, mask, outputCtx, {
    threshold: opts?.threshold ?? 0.5,
    backgroundColor: opts?.backgroundColor ?? [32, 32, 32],
    useSoftAlpha: opts?.useSoftAlpha ?? true,
  });
}

/**
 * Создание сессии для u2net-human-seg (без статических импортов).
 */
export async function createU2NetSession(modelUrl: string): Promise<ort.InferenceSession> {
  ensureOrtReady();
  const session = await ort.InferenceSession.create(modelUrl, {
    // executionProviders: ['webgpu', 'webgl', 'wasm'], // раскомментируй при необходимости
    // graphOptimizationLevel: 'all',
  });
  return session;
}
