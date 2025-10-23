// client/src/core/frameProcessor.ts
import { Tensor, InferenceSession } from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs'; // TensorFlow.js для препроцессинга

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
  const onnxInput = new Tensor('float32', inputTensor.dataSync(), inputTensor.dims);
  
  const feeds = { 'input': onnxInput };
  
  const results = await session.run(feeds);
  
  const maskTensor = results['output'];
  

  // TODO: логика для:
  // Преобразования выходного тензора maskTensor в изображение (маску)
  // Отрисовки фона на outputCanvas
  // Наложения изображения пользователя на фон с использованием маски
  
  tf.dispose(inputTensor);
}