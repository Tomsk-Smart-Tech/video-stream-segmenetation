// --- 1. ТИПЫ ДАННЫХ (Обновленные) ---
export type PrivacyLevel = 'low' | 'medium' | 'high';
interface Shadow { color: string; blur: number; offsetX: number; offsetY: number; }
// Добавляем radius в интерфейс
interface TemplateLayer { type: string; privacy: PrivacyLevel; content: string; x: number; y: number; font?: string; color?: string; align?: CanvasTextAlign; lineHeight?: number; width?: number; height?: number; bgColor?: string; shadow?: Shadow; radius?: number; }
interface Template { name: string; layers: TemplateLayer[]; }
interface Employee { full_name: string; position: string; company: string; department: string; office_location: string; email: string; telegram: string; qr_code_url: string; company_logo_url: string; slogan: string; default_template_id: string; default_background: string; }
interface AppData { background_options: string[]; templates: { [key: string]: Template }; employees: { [key: string]: Employee }; }

// --- 2. ЭЛЕМЕНТЫ (Без изменений) ---
const canvas = document.getElementById('output') as HTMLCanvasElement, ctx = canvas.getContext('2d');
const employeeSelector = document.getElementById('employee-selector') as HTMLSelectElement;
const backgroundCarousel = document.getElementById('background-carousel') as HTMLDivElement;
const uploadInputs = {
    background: document.getElementById('background-upload-input') as HTMLInputElement,
    qr_code: document.getElementById('qr-upload-input') as HTMLInputElement,
    company_logo: document.getElementById('company-logo-upload-input') as HTMLInputElement
};
const textInputs = {
    full_name: document.getElementById('full_name-input') as HTMLInputElement,
    position: document.getElementById('position-input') as HTMLInputElement,
    department: document.getElementById('department-input') as HTMLInputElement,
    company: document.getElementById('company-input') as HTMLInputElement,
    office_location: document.getElementById('office_location-input') as HTMLTextAreaElement,
    email: document.getElementById('email-input') as HTMLInputElement,
    telegram: document.getElementById('telegram-input') as HTMLInputElement,
    slogan: document.getElementById('slogan-input') as HTMLInputElement
};

// --- 3. СОСТОЯНИЕ (Без изменений) ---
let appData: AppData, currentState: Employee, currentTemplate: Template;
let currentPrivacy: PrivacyLevel = 'medium';
const images: { [key: string]: HTMLImageElement } = { background: new Image(), qr_code: new Image(), email_icon: new Image(), telegram_icon: new Image(), company_logo: new Image() };
images.email_icon.src = './src/assets/logo/email_logo.png';
images.telegram_icon.src = './src/assets/logo/tg_logo.png';

// --- 4. ЭКСПОРТ И ОТРИСОВКА (Обновленная) ---
export function setPrivacyLevel(level: PrivacyLevel) { currentPrivacy = level; updateCanvas(); }

export function updateCanvas() {
    if (!ctx || !currentState || !currentTemplate) return;
    canvas.width = 1920; canvas.height = 1080;
    const privacyLevels = { low: 1, medium: 2, high: 3 };
    const currentLevel = privacyLevels[currentPrivacy];

    if (images.background.complete && images.background.naturalHeight) ctx.drawImage(images.background, 0, 0, canvas.width, canvas.height);
    else { ctx.fillStyle = '#000'; ctx.fillRect(0, 0, canvas.width, canvas.height); }

    currentTemplate.layers.forEach(layer => {
        if (privacyLevels[layer.privacy] > currentLevel) return;


        ctx.fillStyle = layer.color || '#FFFFFF';
        ctx.textAlign = layer.align || 'left';
        ctx.font = layer.font || '24px Rubik';

        if (layer.type === 'text') {
            if (layer.shadow) {
                ctx.shadowColor = layer.shadow.color;
                ctx.shadowBlur = layer.shadow.blur;
                ctx.shadowOffsetX = layer.shadow.offsetX;
                ctx.shadowOffsetY = layer.shadow.offsetY;
            }
            const text = layer.content === 'department_and_company' ? `${currentState.department}\n${currentState.company}` : (currentState as any)[layer.content];
            drawMultilineText(text, layer.x, layer.y, layer.lineHeight || 40);
            if (layer.shadow) {
                ctx.shadowColor = 'transparent';
                ctx.shadowBlur = 0;
                ctx.shadowOffsetX = 0;
                ctx.shadowOffsetY = 0;
            }
        } else if (layer.type === 'image') {
            const img = images[layer.content];
            if (img?.complete && img.naturalHeight) ctx.drawImage(img, layer.x, layer.y, layer.width!, layer.height!);
        }
        else if (layer.type === 'roundedRect') {
            ctx.fillStyle = layer.color!;
            ctx.beginPath();
            ctx.roundRect(layer.x, layer.y, layer.width!, layer.height!, layer.radius!);
            ctx.fill();
        }
    });
}

function drawMultilineText(text: string, x: number, y: number, lineHeight: number) { if (text) text.split('\n').forEach((line, i) => ctx!.fillText(line, x, y + i * lineHeight)); }

function updateUIFromState() {
    if (!currentState) return;
    textInputs.full_name.value = currentState.full_name;
    textInputs.position.value = currentState.position;
    textInputs.department.value = currentState.department;
    textInputs.company.value = currentState.company;
    textInputs.office_location.value = currentState.office_location;
    textInputs.email.value = currentState.email;
    textInputs.telegram.value = currentState.telegram;
    textInputs.slogan.value = currentState.slogan;
}

function updateSelectedThumbnail(selectedSrc: string | null) {
    backgroundCarousel.querySelectorAll<HTMLImageElement>('.thumbnail-img').forEach(img => {
        if (selectedSrc && img.dataset.src === selectedSrc) img.classList.add('selected');
        else img.classList.remove('selected');
    });
}

function changeResource(imgKey: keyof typeof images, src: string) {
    if (!src) return;
    images[imgKey].src = src;
    images[imgKey].onload = updateCanvas;
    images[imgKey].onerror = () => { console.error(`Ошибка загрузки: ${src}`); updateCanvas(); };
}

// --- 6. ГЛАВНАЯ ФУНКЦИЯ ИНИЦИАЛИЗАЦИИ (Без изменений) ---
async function main() {
    try { appData = await (await fetch('/data.json')).json(); }
    catch (e) { alert('Критическая ошибка: не удалось загрузить data.json. Убедитесь, что он лежит в папке public.'); return; }

    Object.keys(appData.employees).forEach(key => employeeSelector.add(new Option(appData.employees[key].full_name, key)));

    appData.background_options.forEach(src => {
        const img = document.createElement('img');
        img.src = src; img.className = 'thumbnail-img'; img.dataset.src = src;
        img.onerror = () => console.error(`Не удалось загрузить миниатюру фона: ${src}`);
        img.onclick = () => { changeResource('background', src); updateSelectedThumbnail(src); };
        backgroundCarousel.appendChild(img);
    });

    employeeSelector.onchange = () => {
        currentState = JSON.parse(JSON.stringify(appData.employees[employeeSelector.value]));
        currentTemplate = appData.templates[currentState.default_template_id];
        updateUIFromState();
        changeResource('background', currentState.default_background);
        changeResource('qr_code', currentState.qr_code_url);
        changeResource('company_logo', currentState.company_logo_url);
        updateSelectedThumbnail(currentState.default_background);
        updateCanvas();
    };

    (Object.keys(uploadInputs) as Array<keyof typeof uploadInputs>).forEach(key => {
        uploadInputs[key].onchange = e => {
            const file = (e.target as HTMLInputElement).files?.[0];
            if (file) changeResource(key, URL.createObjectURL(file));
        };
    });

    (Object.keys(textInputs) as Array<keyof typeof textInputs>).forEach(key => {
        textInputs[key].oninput = () => {
            (currentState as any)[key] = textInputs[key].value;
            updateCanvas();
        };
    });

    employeeSelector.dispatchEvent(new Event('change'));
}
main();