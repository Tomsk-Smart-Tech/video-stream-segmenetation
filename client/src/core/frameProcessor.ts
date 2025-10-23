// client/src/core/frameProcessor.ts
import type { InferenceSession} from 'onnxruntime-web';
import * as tf from '@tensorflow/tfjs';

declare const ort: any;

const INPUT_SIZE: [number, number] = [384, 384];
const DOWNSAMPLE_RATIO = 0.25;

let r1i: ort.Tensor | null = null;
let r2i: ort.Tensor | null = null;
let r3i: ort.Tensor | null = null;
let r4i: ort.Tensor | null = null;

const downsampleRatioTensor = new ort.Tensor('float32', new Float32Array([DOWNSAMPLE_RATIO]), [1]);

export async function processFrame(
  videoElement: HTMLVideoElement,
  session: InferenceSession,
  outputCanvas: HTMLCanvasElement,
) {


  const frame = tf.browser.fromPixels(videoElement);
  const resized = tf.image.resizeBilinear(frame, INPUT_SIZE);
  const normalized = resized.div(255.0);
  const transposed = normalized.transpose([2, 0, 1]);
  const inputTensor = transposed.expandDims(0); // Финальный 4D тензор


  const inputData = inputTensor.dataSync()

  const src = new ort.Tensor('float32', inputData, inputTensor.shape
);
  console.log(inputTensor.shape)
  console.log(src)

  frame.dispose();
  resized.dispose();
  normalized.dispose();
  transposed.dispose();
  inputTensor.dispose();

  if (!r1i) {
    console.log("Инициализация динамических рекуррентных состояний...");
    const batchSize = 1;
    // Рассчитываем внутренние размеры
    const h = INPUT_SIZE[0] * DOWNSAMPLE_RATIO;
    const w = INPUT_SIZE[1] * DOWNSAMPLE_RATIO;

    // Берем количество каналов из выходов r*o
    r1i = new ort.Tensor('float32', new Float32Array(batchSize * 16 * h * w).fill(0), [batchSize, 16, h, w]);
    r2i = new ort.Tensor('float32', new Float32Array(batchSize * 20 * h * w).fill(0), [batchSize, 20, h, w]);
    r3i = new ort.Tensor('float32', new Float32Array(batchSize * 40 * h * w).fill(0), [batchSize, 40, h, w]);
    r4i = new ort.Tensor('float32', new Float32Array(batchSize * 64 * h * w).fill(0), [batchSize, 64, h, w]);
  }

 const feeds = {
    'src': src,
    'r1i': r1i!,
    'r2i': r2i!,
    'r3i': r3i!,
    'r4i': r4i!,
    'downsample_ratio': downsampleRatioTensor
  };

  const results = await session.run(feeds);

  r1i = results['r1o'];
  r2i = results['r2o'];
  r3i = results['r3o'];
  r4i = results['r4o'];

  // ПОЛУЧЕНИЕ РЕЗУЛЬТАТА (МАСКИ)
  const mask = results['pha']; //(прозрачность)
  const foreground = results['fgr'];

  console.log("Маска успешно получена!", mask);
  console.log("Маска (pha) и передний план (fgr) успешно получены!");
  console.log("Размер маски:", mask.dims);

  // TODO: логика для:
  // Преобразования выходного тензора maskTensor в изображение (маску)
  // Отрисовки фона на outputCanvas
  // Наложения изображения пользователя на фон с использованием маски

}
