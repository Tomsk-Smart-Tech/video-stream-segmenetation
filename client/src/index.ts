// client/src/index.ts
import { run } from './core/main';

// Запускаем всю магию, когда DOM готов
document.addEventListener('DOMContentLoaded', () => {
  run();
});