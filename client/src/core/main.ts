// client/src/core/main.ts
import { startCamera } from './camera';
import { initializeModnet, initializeFaceDetector, initializeLandmarks } from './model';
import { processFrame } from './frameProcessorTest';

import modelUrl from '../assets/model_q4f16.onnx?url';
import faceUrl from '../assets/MediaPipeFaceDetector.onnx?url'; // имя файла детектора лица
import landmarkUrl from '../assets/MediaPipeFaceLandmarkDetector.onnx?url';

const LANDMARK_INTERVAL = 6;
const L_MIN_MS = 180;
const WARP_GAIN = 0.7; // сглаживание обновлений аффинной

let lastAffine: { a11: number; a12: number; tx: number; a21: number; a22: number; ty: number } | null = null;
let landmarkInFlight = false;
let lastLandmarkRunAt = 0;

let modnetLock: Promise<any> = Promise.resolve();
function runModnetExclusive<T>(fn: () => Promise<T>): Promise<T> {
  modnetLock = modnetLock.then(fn, fn);
  return modnetLock;
}

export async function run() {
  try {
    const videoElement = document.getElementById('webcam') as HTMLVideoElement;
    const outputCanvas = document.getElementById('output__mask') as HTMLCanvasElement;
    const fpsDisplay = document.getElementById('fps-display') as HTMLElement;
    const latencyDisplay = document.getElementById('latency-display') as HTMLElement;
    const cpuDisplay = document.getElementById('cpu-display') as HTMLElement;

    if (!videoElement || !outputCanvas || !fpsDisplay || !latencyDisplay || !cpuDisplay) {
      throw new Error('Не найдены все необходимые HTML элементы (video, canvas или оверлей производительности)');
    }

    await startCamera(videoElement);

    // Сессии
    const modnetSession = await initializeModnet(modelUrl);
    const faceSession = await initializeFaceDetector(faceUrl);
    const landmarksSession = await initializeLandmarks(landmarkUrl);

    outputCanvas.width = videoElement.videoWidth;
    outputCanvas.height = videoElement.videoHeight;
    const ctx = outputCanvas.getContext('2d');
    if (!ctx) throw new Error('Не удалось получить 2D контекст canvas');

    let frameCount = 0;
    let lastTime = performance.now();
    let frameIdx = 0;

    async function loop() {
      const now = performance.now();

      // Планировщик FD/LMK: запускаем редко и неблокирующе (без shared WebGPU конфликтов — LMK на WASM)
      const shouldRunLandmarks =
        frameIdx % LANDMARK_INTERVAL === 0 &&
        !landmarkInFlight &&
        (now - lastLandmarkRunAt) >= L_MIN_MS;

      if (shouldRunLandmarks) {
        landmarkInFlight = true;
        lastLandmarkRunAt = now;
      }

      const timings = await runModnetExclusive(() =>
        processFrame(videoElement, modnetSession, ctx, {
          lastAffine,
          faceSession,
          landmarkSession: shouldRunLandmarks ? landmarksSession : null,
          frameIdx,
          lmkInterval: LANDMARK_INTERVAL,
        })
      );

      // Если вернулась updatedAffine из processFrame — аккуратно подмешиваем
      if (timings.updatedAffine) {
        const M = timings.updatedAffine;
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
        // Завершили фоновый запуск LMK
        landmarkInFlight = false;
      } else if (shouldRunLandmarks) {
        // LMK не обновил матрицу (лицо не найдено или низкий скор), сбросим флаг
        landmarkInFlight = false;
      }

      const currentTime = performance.now();
      frameCount++;
      if (currentTime - lastTime >= 500) {
        const fps = (frameCount / (currentTime - lastTime)) * 1000;
        fpsDisplay.innerText = `FPS: ${fps.toFixed(1)}`;
        latencyDisplay.innerText = `Latency: ${timings.inferenceTime.toFixed(2)} ms`;
        cpuDisplay.innerText = `Total Frame: ${timings.totalTime.toFixed(2)} ms`;
        frameCount = 0;
        lastTime = currentTime;
      }

      frameIdx++;
      requestAnimationFrame(loop);
    }

    loop();
  } catch (error) {
    console.error('Критическая ошибка при запуске приложения:', error);
  }
}
