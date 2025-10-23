// client/src/core/model.ts
import type { InferenceSession } from 'onnxruntime-web';


export async function initializeModel(modelPath: string): Promise<InferenceSession> {
  try {

    const ort = await import('onnxruntime-web');

    // Включаем подробное логирование для отладки
    ort.env.debug = true;
    ort.env.logLevel = 'verbose';
    
    // Устанавливаем пути к файлам
    ort.env.wasm.wasmPaths = '/';

    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders: ['webgpu'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: true,
    });
    console.log('Модель ONNX успешно загружена');
    return session;

  } catch (e) {
    console.error(`Ошибка при загрузке модели: ${e}`);
    throw new Error('Не удалось загрузить ML модель.');
  }
}