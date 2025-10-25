import { startCamera } from './camera';
import { initializeModnet, initializeLandmarks } from './model';

import { processFrame } from './frameProcessor_branch';

import modelUrl from '../assets/model_q4f16.onnx?url';
import landmarkUrl from '../assets/fc_patched.onnx?url';

const LANDMARK_INTERVAL = 6;
const L_MIN_MS = 180;

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

    const modnetSession = await initializeModnet(modelUrl);
    const landmarksSession = await initializeLandmarks(landmarkUrl);

    outputCanvas.width = videoElement.videoWidth;
    outputCanvas.height = videoElement.videoHeight;
    const ctx = outputCanvas.getContext('2d');
    if (!ctx) throw new Error('Не удалось получить 2D контекст canvas');

    let frameCount = 0;
    let lastTime = performance.now();
    let frameIdx = 0;

    async function loop() {

      const timings = await runModnetExclusive(() =>
          processFrame(videoElement, modnetSession, ctx, { lastAffine })
      );

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
      const now = performance.now();
      if (
          frameIdx % LANDMARK_INTERVAL === 0 &&
          !landmarkInFlight &&
          (now - lastLandmarkRunAt) >= L_MIN_MS
      ) {
        landmarkInFlight = true;
        lastLandmarkRunAt = now;

        const landmarkStartTime = performance.now();
        console.log(`[L🚀] Frame ${frameIdx}: Запускаем landmarks модель...`);
        
        // ЗАПУСКАЕМ БЕЗ AWAIT!
        // runAtkshDetector(videoElement, landmarksSession)
        //   .then((M) => {
        //     const duration = performance.now() - landmarkStartTime;
        //     if (M) {
        //       console.log(`[L✅] Landmarks модель отработала за ${duration.toFixed(1)}ms. Матрица обновлена.`);
        //       const WARP_GAIN = 0.7;
        //       lastAffine = lastAffine
        //         ? {
        //             a11: lastAffine.a11 * (1 - WARP_GAIN) + M.a11 * WARP_GAIN,
        //             a12: lastAffine.a12 * (1 - WARP_GAIN) + M.a12 * WARP_GAIN,
        //             tx:  lastAffine.tx  * (1 - WARP_GAIN) + M.tx  * WARP_GAIN,
        //             a21: lastAffine.a21 * (1 - WARP_GAIN) + M.a21 * WARP_GAIN,
        //             a22: lastAffine.a22 * (1 - WARP_GAIN) + M.a22 * WARP_GAIN,
        //             ty:  lastAffine.ty  * (1 - WARP_GAIN) + M.ty  * WARP_GAIN,
        //           }
        //         : M;
        //     } else {
        //       // <-- ЛОГ: Случай, когда модель отработала, но лицо не нашла
        //       console.log(`[L🤷] Landmarks модель отработала за ${duration.toFixed(1)}ms, но не нашла лицо.`);
        //     }
        //   })
        //   .catch((e) => {
        //     // <-- ЛОГ: Улучшаем сообщение об ошибке
        //     const duration = performance.now() - landmarkStartTime;
        //     console.warn(`[L❌] Фоновый запуск landmarks не удался после ${duration.toFixed(1)}ms:`, e);
        //   })
          // .finally(() => {
          //   landmarkInFlight = false;
          // });
      }

      requestAnimationFrame(loop);
    }

    loop();

  } catch (error) {
    console.error('Критическая ошибка при запуске приложения:', error);
  }
}

async function runAtkshDetector(
    video: HTMLVideoElement,
    session: any
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