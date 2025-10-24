// client/src/core/model.ts
import type { InferenceSession } from 'onnxruntime-web';

// Объявляем, что переменная ort будет доступна глобально (из CDN скрипта)
// model.ts — инициализация двух сессий

declare const ort: any;

// ВАЖНО: один раз перед созданием сессий
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// MODNet — WebGPU → WASM fallback
export async function initializeModnet(modnetUrl: string) {
  try {
    return await ort.InferenceSession.create(modnetUrl, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      enableMemPattern: false,
    });
  } catch {
    return await ort.InferenceSession.create(modnetUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      enableMemPattern: false,
    });
  }
}

// atksh detector (face+detection+landmarks) — строго WASM и без агрессивных оптимизаций
export async function initializeLandmarks(atkshUrl: string) {
  return await ort.InferenceSession.create(atkshUrl, {
    executionProviders: ['wasm'],             // без WebGPU
    graphOptimizationLevel: 'disabled',        // меньше фьюзинга → проще пайплайны
    enableCpuMemArena: false,
    enableMemPattern: false,
  });
}
