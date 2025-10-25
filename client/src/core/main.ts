// client/src/main.ts
import { startCamera } from './camera';
import { initializeModnet, initializeLandmarks } from './model';
import { processFrame } from './frameProcessor_branch';

import modelUrl from '../assets/model_q4f16.onnx?url';
import landmarkUrl from '../assets/fc_patched.onnx?url';

// --- НАСТРОЙКИ ---
const LANDMARK_INTERVAL = 6; // Запускаем каждые N кадров
const L_MIN_MS = 180;        // Но не чаще чем раз в X мс (чтобы избежать "пробок" если FPS высокий)

// --- СОСТОЯНИЕ ---
let lastAffine: { a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null = null;
let landmarkInFlight = false; // Флаг, что landmarks-модель уже в работе
let lastLandmarkRunAt = 0;    // Время последнего успешного запуска

// MODNet работает быстро (WebGPU), его можно оставить в основном потоке
// для минимальной задержки маски.
let modnetLock: Promise<any> = Promise.resolve();
function runModnetExclusive<T>(fn: () => Promise<T>): Promise<T> {
  modnetLock = modnetLock.then(fn, fn);
  return modnetLock;
}


export async function run() {
  try {
    const videoElement = document.getElementById('webcam') as HTMLVideoElement;
    const outputCanvas = document.getElementById('output__mask') as HTMLCanvasElement;
    if (!videoElement || !outputCanvas) throw new Error('Не найдены video или canvas элементы');

    await startCamera(videoElement);

    const modnetSession = await initializeModnet(modelUrl);
    const landmarksSession = await initializeLandmarks(landmarkUrl);

    outputCanvas.width = videoElement.videoWidth;
    outputCanvas.height = videoElement.videoHeight;
    const ctx = outputCanvas.getContext('2d');
    if (!ctx) throw new Error('Не удалось получить 2D контекст canvas');

    let frameIdx = 0;

    async function loop() {
      // 1) MODNet инференс — он быстрый, его выполняем на каждом кадре.
      // Используем отдельную очередь, чтобы избежать гонки состояний.
      await runModnetExclusive(() =>
        processFrame(videoElement, modnetSession, ctx, { lastAffine })
      );

      // 2) Запускаем медленный landmarks-инференс АСИНХРОННО И НЕБЛОКИРУЮЩЕ
      frameIdx++;
      const now = performance.now();
      if (
        frameIdx % LANDMARK_INTERVAL === 0 && // Проверяем интервал в кадрах
        !landmarkInFlight &&                  // Проверяем, что предыдущий запуск завершился
        (now - lastLandmarkRunAt) >= L_MIN_MS // Проверяем интервал в мс
      ) {
        landmarkInFlight = true; // Поднимаем флаг
        lastLandmarkRunAt = now;

        // ЗАПУСКАЕМ БЕЗ AWAIT!
        runAtkshDetector(videoElement, landmarksSession)
          .then((M) => {
            // Этот код выполнится, когда модель отработает, НЕ блокируя рендер
            if (M) {
              const WARP_GAIN = 0.7;
              lastAffine = lastAffine
                ? {
                    a11: lastAffine.a11 * (1 - WARP_GAIN) + M.a11 * WARP_GAIN,
                    a12: lastAffine.a12 * (1 - WARP_GAIN) + M.a12 * WARP_GAIN,
                    tx:  lastAffine.tx  * (1 - WARP_GAIN) + M.tx  * WARP_GAIN,
                    a21: lastAffine.a21 * (1 - WARP_GAIN) + M.a21 * WARP_GAIN,
                    a22: lastAffine.a22 * (1 - WARP_GAIN) + M.a22 * WARP_GAIN,
                    ty:  lastAffine.ty  * (1 - WARP_GAIN) + M.ty  * WARP_GAIN,
                  }
                : M;
            }
          })
          .catch((e) => {
            console.warn('Фоновый запуск landmarks не удался:', e);
          })
          .finally(() => {
            // Опускаем флаг в любом случае, чтобы разрешить следующий запуск
            landmarkInFlight = false;
          });
      }

      // Немедленно планируем следующий кадр, не дожидаясь landmarks
      requestAnimationFrame(loop);
    }

    loop();

  } catch (error) {
    console.error('Критическая ошибка при запуске приложения:', error);
  }
}

// Функцию runAtkshDetector можно оставить без изменений
async function runAtkshDetector(
  video: HTMLVideoElement,
  session: any
): Promise<{ a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null> {
    // ... ваш код без изменений ...
}