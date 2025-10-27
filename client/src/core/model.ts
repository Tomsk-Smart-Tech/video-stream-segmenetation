// client/src/core/model.ts
import type { InferenceSession } from 'onnxruntime-web';

declare const ort: any;

// Пути WASM — обязателен до создания сессий
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

/**
 * MODNet: приоритет WebGPU, fallback на WASM
 */
export async function initializeModnet(modnetUrl: string): Promise<InferenceSession> {
  try {
    return await ort.InferenceSession.create(modnetUrl, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      enableMemPattern: false,
    });
  } catch (e) {
    console.warn('MODNet WebGPU init failed → fallback to WASM:', e);
    return await ort.InferenceSession.create(modnetUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      enableMemPattern: false,
    });
  }
}

/**
 * FaceDetector: приоритет WebGPU, fallback на WASM
 * Вход: float32 [1,3,256,256]
 * Выход: box_coords float32 [1,896,16], box_scores float32 [1,896,1]
 */
export async function initializeFaceDetector(fdUrl: string): Promise<InferenceSession> {
  try {
    return await ort.InferenceSession.create(fdUrl, {
      executionProviders: ['webgpu', 'wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      enableMemPattern: false,
    });
  } catch (e) {
    console.warn('FaceDetector WebGPU init failed → fallback to WASM:', e);
    return await ort.InferenceSession.create(fdUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena: false,
      enableMemPattern: false,
    });
  }
}

/**
 * Landmarks (468×3): строго WASM, чтобы избежать WebGPU конфликтов.
 */
export async function initializeLandmarks(lmkUrl: string): Promise<InferenceSession> {
  const res = await fetch(lmkUrl, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Landmarks model fetch failed: ${res.status} ${res.statusText}`);
  return await ort.InferenceSession.create(lmkUrl, {
    executionProviders: ['webgpu', 'wasm'],
    graphOptimizationLevel: 'disabled',
    enableCpuMemArena: false,
    enableMemPattern: false,
  });
}
