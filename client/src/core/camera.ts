// client/src/core/camera.ts
export async function startCamera(videoElement: HTMLVideoElement): Promise<void> {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error('Ваш браузер не поддерживает API для доступа к камере.');
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        width: { ideal: 1280 }, // Запрашиваем HD-качество
        height: { ideal: 720 },
      },
      audio: false,
    });

    videoElement.srcObject = stream;
    
    // Ждем, пока видео начнет проигрываться, чтобы получить его реальные размеры
    await new Promise((resolve) => {
      videoElement.onloadedmetadata = () => {
        videoElement.play();
        resolve(null);
      };
    });

  } catch (err) {
    console.error("Ошибка при доступе к камере:", err);
    throw new Error('Не удалось получить доступ к камере. Пожалуйста, проверьте разрешения в браузере.');
  }
}