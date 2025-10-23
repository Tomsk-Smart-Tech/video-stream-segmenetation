// client/src/core/main.ts
import { startCamera } from './camera';
import { initializeModel } from './model';
import { processFrame } from './frameProcessor';
import ort from 'onnxruntime-web';

const MODEL_PATH = '/model/rvm_mobilenetv3_fp32.onnx';

export async function run() {
  try {
    const videoElement = document.getElementById('webcam') as HTMLVideoElement;
    const outputCanvas = document.getElementById('output') as HTMLCanvasElement;
    if (!videoElement || !outputCanvas) throw new Error('Не найдены video или canvas элементы');

    const [_, session] = await Promise.all([
      startCamera(videoElement),
      initializeModel(MODEL_PATH)
    ]);

    outputCanvas.width = videoElement.videoWidth;
    outputCanvas.height = videoElement.videoHeight;

    async function gameLoop() {
      await processFrame(videoElement, session, outputCanvas);
      // следующий кадр для обработки
      requestAnimationFrame(gameLoop);
    }
    
    // Запускаем цикл в первый раз
    gameLoop();

  } catch (error) {
    console.error("Критическая ошибка при запуске приложения:", error);
    // TODO: Вывести красивое сообщение об ошибке пользователю
  }
}