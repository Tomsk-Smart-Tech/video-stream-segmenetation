const slider = document.getElementById('downsample-slider') as HTMLInputElement;
const sliderValueSpan = document.getElementById('slider-value') as HTMLSpanElement;

if (slider && sliderValueSpan) {
    slider.addEventListener('input', (event) => {
        const currentValue = parseFloat(slider.value);
        sliderValueSpan.textContent = currentValue.toFixed(2);
    });

}