import { setPrivacyLevel, PrivacyLevel } from './customization.js';

const defaultSettings = {
    ema: 0.55,
    noiseCutoff: 0.06,
    highThreshold: 0.95,
    gamma: 0.6,
    useBilateral: true,
    bilateralSigmaSpatial: 1.0,
    bilateralSigmaRange: 12.0
};

export const processingSettings = { ...defaultSettings };

const sliders = {
    ema: { slider: 'ema-slider', display: 'ema-value', key: 'ema', decimals: 2 },
    noiseCutoff: { slider: 'noise-cutoff-slider', display: 'noise-cutoff-value', key: 'noiseCutoff', decimals: 3 },
    highThreshold: { slider: 'high-threshold-slider', display: 'high-threshold-value', key: 'highThreshold', decimals: 3 },
    gamma: { slider: 'gamma-slider', display: 'gamma-value', key: 'gamma', decimals: 2 },
    sigmaSpatial: { slider: 'sigma-spatial-slider', display: 'sigma-spatial-value', key: 'bilateralSigmaSpatial', decimals: 1 },
    sigmaRange: { slider: 'sigma-range-slider', display: 'sigma-range-value', key: 'bilateralSigmaRange', decimals: 0 }
};
const bilateralToggle = document.getElementById('bilateral-toggle') as HTMLInputElement;

function updateSliderUI(config: typeof sliders[keyof typeof sliders]) {
    const sliderEl = document.getElementById(config.slider) as HTMLInputElement;
    const displayEl = document.getElementById(config.display) as HTMLElement;
    const value = processingSettings[config.key as keyof typeof processingSettings];
    if (sliderEl && displayEl) {
        sliderEl.value = String(value);
        displayEl.textContent = Number(value).toFixed(config.decimals);
    }
}

function setupSlider(config: typeof sliders[keyof typeof sliders]) {
    const sliderEl = document.getElementById(config.slider) as HTMLInputElement;
    if (!sliderEl) return;
    sliderEl.addEventListener('input', () => {
        (processingSettings as any)[config.key] = Number(sliderEl.value);
        updateSliderUI(config);
    });
}

function resetAllSettings() {
    Object.assign(processingSettings, defaultSettings);
    Object.values(sliders).forEach(updateSliderUI);
    if (bilateralToggle) {
        bilateralToggle.checked = processingSettings.useBilateral;
    }
}

Object.values(sliders).forEach(setupSlider);

if (bilateralToggle) {
    bilateralToggle.addEventListener('change', () => {
        processingSettings.useBilateral = bilateralToggle.checked;
    });
}

const resetButton = document.getElementById('reset-settings-btn');
if (resetButton) {
    resetButton.addEventListener('click', resetAllSettings);
}

const toggleButton = document.getElementById('toggle-settings-btn');
const settingsBody = document.getElementById('settings-body-content');
if (toggleButton && settingsBody) {
    toggleButton.addEventListener('click', () => {
        settingsBody.classList.toggle('hidden');
        toggleButton.classList.toggle('active');
    });
}

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

resetAllSettings();