// client/src/core/frameProcessor.ts
import type { InferenceSession, Tensor } from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';

declare const ort: any;


const MODEL_INPUT_SIZE: [number, number] = [512, 288]; // [ширина, высота]

// Временный canvas для маски
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d', { willReadFrequently: true });
maskCanvas.width = MODEL_INPUT_SIZE[0];
maskCanvas.height = MODEL_INPUT_SIZE[1];

export async function processFrame(
  videoElement: HTMLVideoElement,
  session: InferenceSession,
  outputCtx: CanvasRenderingContext2D,
): Promise<{ inferenceTime: number, totalTime: number }> {

  const totalStartTime = performance.now(); 

  const frame = tf.browser.fromPixels(videoElement);
  const resized = tf.image.resizeBilinear(frame, [MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]]);
  const normalized = resized.div(255.0);
  const transposed = normalized.transpose([2, 0, 1]);
  const inputTensorTf = transposed.expandDims(0);
  const inputData = inputTensorTf.dataSync();
  const ortInputTensor = new ort.Tensor('float32', inputData, inputTensorTf.shape);
  frame.dispose();
  resized.dispose();
  normalized.dispose();
  transposed.dispose();
  inputTensorTf.dispose();

  const feeds = { 'input': ortInputTensor };
  
  const inferenceStartTime = performance.now(); 
  const results = await session.run(feeds);
  const inferenceEndTime = performance.now();
  
  const mask = results['output'];

  const maskImageData = maskTensorToImageData(mask);
  maskCtx!.putImageData(maskImageData, 0, 0);
  const outputCanvas = outputCtx.canvas;
  outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
  outputCtx.drawImage(videoElement, 0, 0, outputCanvas.width, outputCanvas.height);
  outputCtx.globalCompositeOperation = 'destination-in';
  outputCtx.drawImage(maskCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
  outputCtx.globalCompositeOperation = 'source-over';

  const totalEndTime = performance.now();
  
  return {
    inferenceTime: inferenceEndTime - inferenceStartTime,
    totalTime: totalEndTime - totalStartTime
  };
}



function maskTensorToImageData(maskTensor: Tensor): ImageData {
  const [_, __, height, width] = maskTensor.dims;
  const data = maskTensor.data as Float32Array;
  
  const imageData = new ImageData(width, height);
  const pixels = imageData.data;

  const DENOISE_THRESHOLD = 0.15; 

  const HARDEN_THRESHOLD = 0.85;

  for (let i = 0; i < data.length; i++) {
    let maskValue = data[i];


    if (maskValue < DENOISE_THRESHOLD) {
      maskValue = 0;
    } 
    else if (maskValue > HARDEN_THRESHOLD) {
      maskValue = 1;
    }
  
    else {
      maskValue = (maskValue - DENOISE_THRESHOLD) / (HARDEN_THRESHOLD - DENOISE_THRESHOLD);
    }
    
    const pixelIndex = i * 4;
    
    pixels[pixelIndex] = 255;     // R
    pixels[pixelIndex + 1] = 255; // G
    pixels[pixelIndex + 2] = 255; // B
    pixels[pixelIndex + 3] = maskValue * 255; // Alpha
  }

  return imageData;
}