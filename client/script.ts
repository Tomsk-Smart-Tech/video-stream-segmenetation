// --- 1. ИМПОРТЫ ---
import { config, defaultConfig } from './src/core/frameProcessorTest';
import { setPrivacyLevel, PrivacyLevel } from './customization.js';

// --- 2. КАРТА UI-ЭЛЕМЕНТОВ И НАСТРОЕК ---
const sliderMap = {
    'ema-slider': { displayId: 'ema-value', key: 'EMA', decimals: 2 },
    'noise-cutoff-slider': { displayId: 'noise-cutoff-value', key: 'NOISE_CUTOFF', decimals: 3 },
    'high-threshold-slider': { displayId: 'high-threshold-value', key: 'HIGH_THRESHOLD', decimals: 3 },
    'gamma-slider': { displayId: 'gamma-value', key: 'GAMMA', decimals: 2 },
    'sigma-spatial-slider': { displayId: 'sigma-spatial-value', key: 'BILATERAL_SIGMA_SPATIAL', decimals: 1 },
    'sigma-range-slider': { displayId: 'sigma-range-value', key: 'BILATERAL_SIGMA_RANGE', decimals: 0 }
};

// --- 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (остаются без изменений) ---
function setupSlider(sliderId: string, configData: typeof sliderMap[keyof typeof sliderMap]) {
    const sliderEl = document.getElementById(sliderId) as HTMLInputElement;
    const displayEl = document.getElementById(configData.displayId) as HTMLElement;
    if (!sliderEl || !displayEl) return;
    sliderEl.addEventListener('input', () => {
        const value = Number(sliderEl.value);
        (config as any)[configData.key] = value;
        displayEl.textContent = value.toFixed(configData.decimals);
    });
}

function updateUiFromConfig() {
    for (const [sliderId, configData] of Object.entries(sliderMap)) {
        const sliderEl = document.getElementById(sliderId) as HTMLInputElement;
        const displayEl = document.getElementById(configData.displayId) as HTMLElement;
        const value = (config as any)[configData.key];
        if (sliderEl && displayEl) {
            sliderEl.value = String(value);
            displayEl.textContent = Number(value).toFixed(configData.decimals);
        }
    }
    const bilateralToggle = document.getElementById('bilateral-toggle') as HTMLInputElement;
    if (bilateralToggle) {
        bilateralToggle.checked = config.USE_BILATERAL;
    }
}

function resetAllSettings() {
    Object.assign(config, defaultConfig);
    updateUiFromConfig();
}

// --- 4. ГЛАВНАЯ ФУНКЦИЯ ИНИЦИАЛИЗАЦИИ ---
// Вся логика, которая взаимодействует с HTML, теперь находится здесь.
function initializeUI() {
    // 1. Настраиваем все слайдеры
    for (const [sliderId, configData] of Object.entries(sliderMap)) {
        setupSlider(sliderId, configData);
    }

    // 2. Настраиваем переключатель
    const bilateralToggle = document.getElementById('bilateral-toggle') as HTMLInputElement;
    if (bilateralToggle) {
        bilateralToggle.addEventListener('change', () => {
            config.USE_BILATERAL = bilateralToggle.checked;
        });
    }

    // 3. Настраиваем кнопку сброса
    const resetButton = document.getElementById('reset-settings-btn');
    if (resetButton) {
        resetButton.addEventListener('click', resetAllSettings);
    }

    // 4. Настраиваем сворачиваемую панель
    const toggleButton = document.getElementById('toggle-settings-btn');
    const settingsBody = document.getElementById('settings-body-content');
    if (toggleButton && settingsBody) {
        toggleButton.addEventListener('click', () => {
            settingsBody.classList.toggle('hidden');
            toggleButton.classList.toggle('active');
            settingsBody.classList.toggle('expanded');
        });
    }

    // 5. Настраиваем кнопки приватности
    const privacySelector = document.querySelector('.privacy-selector');
    if (privacySelector) {
        const options = privacySelector.querySelectorAll<HTMLButtonElement>('.privacy-option');
        const updateButtons = (activeKey: string) => {
            options.forEach(button => {
                const buttonKey = button.dataset.key;
                if (buttonKey === activeKey) {
                    button.classList.add('active');
                    button.textContent = button.dataset.fullName || '';
                } else {
                    button.classList.remove('active');
                    button.textContent = button.dataset.shortName || '';
                }
            });
        };
        options.forEach(button => {
            button.addEventListener('click', () => {
                const clickedKey = button.dataset.key;
                if (clickedKey) {
                    updateButtons(clickedKey);
                    setPrivacyLevel(clickedKey as PrivacyLevel);
                }
            });
        });
        const initialActiveButton = privacySelector.querySelector<HTMLButtonElement>('.privacy-option.active');
        if (initialActiveButton?.dataset.key) {
            updateButtons(initialActiveButton.dataset.key);
            setPrivacyLevel(initialActiveButton.dataset.key as PrivacyLevel);
        }
    }

    // 6. Устанавливаем начальные значения в UI при первой загрузке
    updateUiFromConfig();
}

// --- 5. ЗАПУСК ПОСЛЕ ЗАГРУЗКИ СТРАНИЦЫ ---
// Этот слушатель гарантирует, что функция initializeUI() будет вызвана
// только тогда, когда все HTML-элементы точно существуют.
document.addEventListener('DOMContentLoaded', initializeUI);