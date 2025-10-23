// client/src/core/frameProcessor.ts
import type { InferenceSession } from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs'; // TensorFlow.js для препроцессинга

declare const ort: any;

let r1i: ort.Tensor | null = null;
let r2i: ort.Tensor | null = null;
let r3i: ort.Tensor | null = null;
let r4i: ort.Tensor | null = null;


export async function processFrame(
  videoElement: HTMLVideoElement,
  session: InferenceSession,
  outputCanvas: HTMLCanvasElement,
) {

  const inputTensor = tf.tidy(() => {
    const frame = tf.browser.fromPixels(videoElement);
    
    const inputSize = [256, 256]; 
    
    const resized = tf.image.resizeBilinear(frame, inputSize);
    
    const normalized = resized.div(255.0);
    
    const batched = normalized.expandDims(0);
    
    return batched;
  });

  // формат, понятный ONNX Runtime
  const onnxInput = new ort.Tensor('float32', inputTensor.dataSync(), inputTensor.dims);
  
const inputName = session.inputNames[0]; // Берем имя первого входа
const feeds = { [inputName]: onnxInput };
  
  const results = await session.run(feeds);

// Узнали из Netron или логов имя нужного выхода
const outputName = session.outputNames[0]; // Берем имя первого выхода
const maskTensor = results[outputName];
  

  // TODO: логика для:
  // Преобразования выходного тензора maskTensor в изображение (маску)
  // Отрисовки фона на outputCanvas
  // Наложения изображения пользователя на фон с использованием маски
  
  tf.dispose(inputTensor);
}
