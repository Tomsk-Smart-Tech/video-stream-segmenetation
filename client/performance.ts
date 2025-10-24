// const fpsDisplay = document.getElementById('fps-display') as HTMLElement;
// const latencyDisplay = document.getElementById('latency-display') as HTMLElement;
// const cpuDisplay = document.getElementById('cpu-display') as HTMLElement;
//
// let lastFrameTime = performance.now();
// let frameCount = 0;
// let lastFpsUpdateTime = performance.now();
//
// function monitorPerformance() {
//     const now = performance.now();
//     const deltaTime = now - lastFrameTime;
//     lastFrameTime = now;
//
//     frameCount++;
//     if (now > lastFpsUpdateTime + 1000) {
//         const fps = frameCount;
//         fpsDisplay.textContent = `FPS: ${fps}`;
//         frameCount = 0;
//         lastFpsUpdateTime = now;
//     }
//
//     latencyDisplay.textContent = `Latency: ${deltaTime.toFixed(2)} ms`;
//
//     const idealFrameTime = 1000 / 60;
//     const threadLoad = Math.min(100, (deltaTime / idealFrameTime) * 100);
//     cpuDisplay.textContent = `CPU (Thread): ${threadLoad.toFixed(0)}%`;
//
//     requestAnimationFrame(monitorPerformance);
// }
//
// monitorPerformance();