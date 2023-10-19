document.addEventListener('DOMContentLoaded', function () {
    let slider = document.getElementById("mySlider");
    let thumb = document.getElementById("mySliderThumb");
    let displayValue = document.getElementById("slider-value");

    slider.addEventListener('click', function (event) {
        let x = event.clientX - slider.getBoundingClientRect().left; // Get horizontal coordinate of pointer
        let width = slider.offsetWidth; // Width of the slider

        let fraction = x / width; // Calculate the fraction of the slider that's been clicked

        // Set the thumb position and display the value
        thumb.style.left = x + "px";
        displayValue.innerText = "Value of α : " + fraction.toFixed(2);
    });
});
