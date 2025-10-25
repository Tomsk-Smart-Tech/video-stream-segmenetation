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
 * Landmarks (atksh): строго WASM, оптимизации отключены
 * Используем 'disabled' (правильное значение), чтобы уменьшить фьюзинг.
 */
export async function initializeLandmarks(atkshUrl: string): Promise<InferenceSession> {
  const res = await fetch(atkshUrl, { cache: 'no-store' });
  if (!res.ok) throw new Error(`Landmarks model fetch failed: ${res.status} ${res.statusText}`);
  return await ort.InferenceSession.create(atkshUrl, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'disabled',
    enableCpuMemArena: false,
    enableMemPattern: false,
  });
}
