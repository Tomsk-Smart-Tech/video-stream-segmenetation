// client/src/core/model.ts
import { InferenceSession, env } from 'onnxruntime-web';

export async function initializeModel(modelPath: string): Promise<InferenceSession> {
  try {
    env.wasm.wasmPaths = '/'; // путь к wasm файлам onnxruntime

    const session = await InferenceSession.create(modelPath, {
      executionProviders: ['webgpu', 'webgl'], // Приоритет: сначала WebGPU, потом WebGL
    });
    console.log('Модель ONNX успешно загружена. Бэкенд:', session.executionProviders);
    return session;

  } catch (e) {
    console.error(`Ошибка при загрузке модели: ${e}`);
    throw new Error('Не удалось загрузить ML модель.');
  }
}