import { startCamera } from './camera';
import { initializeModel } from './model';
import { processFrame } from './frameProcessor';

import modelUrl from '../assets/model_q4f16.onnx?url';

export async function run() {
  try {
    // Получаем все нужные HTML элементы
    const videoElement = document.getElementById('webcam') as HTMLVideoElement;
    const outputCanvas = document.getElementById('output') as HTMLCanvasElement;
    const debugPanel = document.getElementById('debug-panel') as HTMLDivElement; // <-- НАШ НОВЫЙ DIV
    if (!videoElement || !outputCanvas || !debugPanel) throw new Error('Не найдены video, canvas или debug-panel элементы');

    await startCamera(videoElement);
    const session = await initializeModel(modelUrl);

    outputCanvas.width = videoElement.videoWidth;
    outputCanvas.height = videoElement.videoHeight;
    const ctx = outputCanvas.getContext('2d');
    if (!ctx) throw new Error('Не удалось получить 2D контекст canvas');

    // --- Переменные для расчета FPS ---
    let frameCount = 0;
    let lastTime = performance.now();

    async function gameLoop() {
      // Получаем результаты замеров из processFrame
      const timings = await processFrame(videoElement, session, ctx);

      // --- Логика расчета и вывода FPS ---
      const currentTime = performance.now();
      frameCount++;
      // Обновляем текст раз в 500 мс, чтобы не так сильно мельтешил
      if (currentTime - lastTime >= 500) {
        const fps = (frameCount / (currentTime - lastTime)) * 1000;
        
        debugPanel.innerHTML = `
          FPS: ${fps.toFixed(1)} <br/>
          Inference: ${timings.inferenceTime.toFixed(2)} ms <br/>
          Total Frame: ${timings.totalTime.toFixed(2)} ms
        `;

        frameCount = 0;
        lastTime = currentTime;
      }
      
      requestAnimationFrame(gameLoop);
    }
    
    gameLoop();

  } catch (error) {
    console.error("Критическая ошибка при запуске приложения:", error);
  }
}