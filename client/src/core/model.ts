// client/src/core/model.ts
import type { InferenceSession } from 'onnxruntime-web';

// Объявляем, что переменная ort будет доступна глобально (из CDN скрипта)
declare const ort: any;

export async function initializeModel(modelPath: string): Promise<InferenceSession> {
  try {
    // Устанавливаем пути к wasm-файлам. Это важно для работы бэкенда wasm.
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

    const session = await ort.InferenceSession.create(modelPath, {
      // Пытаемся использовать webgpu, если не получится - откатываемся на wasm
      executionProviders: ['webgpu', 'wasm'], 
      graphOptimizationLevel: 'all',
    });
    console.log('Модель ONNX успешно загружена');
    return session;

  } catch (e) {
    console.error(`Ошибка при загрузке модели: ${e}`);
    throw new Error('Не удалось загрузить ML модель.');
  }
}