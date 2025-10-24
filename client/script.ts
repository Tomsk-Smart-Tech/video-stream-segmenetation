import { setPrivacyLevel, PrivacyLevel, updateCanvas } from './customization.js';

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
    if (initialActiveButton && initialActiveButton.dataset.key) {
        updateButtons(initialActiveButton.dataset.key);
        setPrivacyLevel(initialActiveButton.dataset.key as PrivacyLevel);
    }
}