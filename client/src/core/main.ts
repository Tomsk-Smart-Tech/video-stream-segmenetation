// client/src/main.ts
import { startCamera } from './camera';
import { initializeModnet, initializeLandmarks } from './model';
import { processFrame } from './frameProcessor_branch';

// Путь к MODNet (оставляю твой import)
import modelUrl from '../assets/model_q4f16.onnx?url';
// Путь к landmarks (atksh). Если используешь Vite/Webpack: 
import landmarkUrl from '../assets/fc_patched.onnx?url' // поправь под твою структуру

// Интервал запуска landmarks в кадрах
const LANDMARK_INTERVAL = 6;

// Простой мьютекс-очередь: не даёт двум инференсам запускаться одновременно
let inferLock: Promise<void> = Promise.resolve();
function runExclusive<T>(fn: () => Promise<T>): Promise<T> {
  const task = inferLock.then(fn).catch(fn);
  inferLock = task.then(() => undefined, () => undefined);
  return task;
}

export async function run() {
  try {
    const videoElement = document.getElementById('webcam') as HTMLVideoElement;
    const outputCanvas = document.getElementById('output__mask') as HTMLCanvasElement;
    if (!videoElement || !outputCanvas) throw new Error('Не найдены video или canvas элементы');

    await startCamera(videoElement);

    // Инициализируем две сессии
    const modnetSession = await initializeModnet(modelUrl);
    const landmarksSession = await initializeLandmarks(landmarkUrl);

    // Настройка canvas
    outputCanvas.width = videoElement.videoWidth;
    outputCanvas.height = videoElement.videoHeight;
    const ctx = outputCanvas.getContext('2d');
    if (!ctx) throw new Error('Не удалось получить 2D контекст canvas');

    // Состояние для landmarks (аффинная матрица варпа из atksh)
    let lastAffine: { a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null = null;

    let frameIdx = 0;

    async function loop() {
      frameIdx++;

      // 1) MODNet инференс (WebGPU → WASM), обязательно через очередь
      const timings = await runExclusive(() =>
        processFrame(videoElement, modnetSession, ctx, {
          lastAffine, // пробрасываем текущую матрицу варпа
        })
      );

      // 2) Каждые LANDMARK_INTERVAL кадров — обновляем аффинную матрицу (WASM)
      if (frameIdx % LANDMARK_INTERVAL === 0) {
        try {
          // landmarks-инференс через очередь, чтобы не перекрывать MODNet
          const M = await runExclusive(() => runAtkshDetector(videoElement, landmarksSession));
          if (M) {
            // Смешиваем плавно для стабильности
            const WARP_GAIN = 0.7;
            if (lastAffine) {
              lastAffine = {
                a11: lastAffine.a11 * (1 - WARP_GAIN) + M.a11 * WARP_GAIN,
                a12: lastAffine.a12 * (1 - WARP_GAIN) + M.a12 * WARP_GAIN,
                tx:  lastAffine.tx  * (1 - WARP_GAIN) + M.tx  * WARP_GAIN,
                a21: lastAffine.a21 * (1 - WARP_GAIN) + M.a21 * WARP_GAIN,
                a22: lastAffine.a22 * (1 - WARP_GAIN) + M.a22 * WARP_GAIN,
                ty:  lastAffine.ty  * (1 - WARP_GAIN) + M.ty  * WARP_GAIN,
              };
            } else {
              lastAffine = M;
            }
          }
        } catch (e) {
          console.warn('Landmarks WASM failed, пропускаем кадр:', e);
        }
      }

      requestAnimationFrame(loop);
    }

    // Запуск
    loop();

  } catch (error) {
    console.error('Критическая ошибка при запуске приложения:', error);
  }
}

/**
 * Вспомогательная функция: запуск atksh детектора и извлечение матрицы M (2×3)
 * (перенесена сюда из branch-процессора для простоты интеграции main.ts)
 */
async function runAtkshDetector(
  video: HTMLVideoElement,
  session: any // InferenceSession
): Promise<{ a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null> {
  // Снимаем текущий кадр
  const c = document.createElement('canvas');
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  const cx = c.getContext('2d', { willReadFrequently: true })!;
  cx.drawImage(video, 0, 0, c.width, c.height);
  const img = cx.getImageData(0, 0, c.width, c.height);
  const u8 = img.data;

  // Формируем uint8 [H,W,3]
  const hw3 = new Uint8Array(c.width * c.height * 3);
  for (let i = 0, p = 0; i < u8.length; i += 4) {
    hw3[p++] = u8[i];
    hw3[p++] = u8[i + 1];
    hw3[p++] = u8[i + 2];
  }
  const inp = new ort.Tensor('uint8', hw3, [c.height, c.width, 3]);

  // Запуск
  const outputs = await session.run({ input: inp }) as Record<string, any>;

  const scoresT = outputs['scores'];
  const MT = outputs['M'];
  if (!scoresT || !MT) return null;

  const scores = scoresT.data as Float32Array;
  const Mdata = MT.data as Float32Array;
  const N = scoresT.dims[0];
  const stride = 6; // 2x3

  if (!N || N <= 0) return null;

  let bestIdx = 0;
  let bestScore = -Infinity;
  for (let i = 0; i < N; i++) {
    const s = scores[i];
    if (s > bestScore) {
      bestScore = s;
      bestIdx = i;
    }
  }

  const base = bestIdx * stride;
  const a11 = Mdata[base + 0];
  const a12 = Mdata[base + 1];
  const tx  = Mdata[base + 2];
  const a21 = Mdata[base + 3];
  const a22 = Mdata[base + 4];
  const ty  = Mdata[base + 5];

  return { a11, a12, tx, a21, a22, ty };
}
