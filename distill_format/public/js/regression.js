// JavaScript functions for handling slider changes and image updates
let loaded_dict; // Declare the variable in the outer scope

fetch('data/nested_dict.json')
    .then((response) => response.json())
    .then((json) => {
        loaded_dict = json; // Assign the value to the variable
        console.log(loaded_dict);
    })
    .catch((error) => {
        console.error('Error loading JSON:', error);
    });

// Function to preload images
function preloadImages(array) {
    var imgList = [];
    for (var i = 0; i < array.length; i++) {
        var img = new Image();
        img.src = array[i];
        imgList.push(img);
    }
}

function changeSlider(sliderContainer, isIncrease, sliderType) {
    // Get the current value of the slider
    var slider = sliderContainer.querySelector('.slider');
    var currentValue = parseInt(slider.value);

    // Calculate the new value based on whether it's an increase or decrease
    var newValue = isIncrease ? currentValue + 2 : currentValue - 2;

    // Ensure the new value stays within the defined range
    if (newValue < 10) {
        newValue = 10;
    } else if (newValue > 20) {
        newValue = 20;
    }

    // Update the slider value
    slider.value = newValue;

    // Call the appropriate update function based on the slider type
    if (sliderType === 'n_cal') {
        updateSliderValue(newValue);
    } else {
        updateAlphaValue(newValue);
    }
}

function changeAlpha(sliderContainer, isIncrease) {
    // Get the alpha slider element
    var alphaSlider = sliderContainer.querySelector('.slider');
    
    // Get the current alpha value as a floating-point number
    var currentValue = parseFloat(alphaSlider.value);

    // Calculate the new value based on whether it's an increase or decrease
    var step = 0.1; // Step value for alpha
    var newValue = isIncrease ? currentValue + step : currentValue - step;
    newValue = parseFloat(newValue.toFixed(1));

    // Ensure the new value stays within the defined range
    if (newValue < 0.1) {
        newValue = 0.1;
    } else if (newValue > 1.0) {
        newValue = 1.0;
    }

    // Update the alpha slider value
    alphaSlider.value = newValue;

    // Call the updateAlphaValue function to update the displayed value and images
    updateAlphaValue(newValue);
}


function updateSliderValue(value) {
    document.getElementById('slider-value').textContent = value;
    updateSlidersAndImages(value, true);
}

function updateAlphaValue(value) {
    document.getElementById('alpha-slider-value').textContent = value;
    updateSlidersAndImages(value, false);
}

// Function to update the displayed value and image for both sliders
function updateSlidersAndImages(value, isNCalSlider) {
    if (isNCalSlider) {
        document.getElementById('slider-value').textContent = value;
    } else {
        document.getElementById('alpha-slider-value').textContent = value;
    }

    // Get both slider values
    var nCalSliderValue = parseInt(document.querySelector('.gif-slider .slider').value);
    var alphaSliderValue = parseFloat(document.querySelectorAll('.gif-slider .slider')[1].value).toFixed(1);

    // Update both images
    changeImages(nCalSliderValue, alphaSliderValue);
}

// Function to change both images
function changeImages(nCalValue, alphaValue) {
    // Update the image source for n_cal
    document.getElementById('calib-display').src = `images/Generated_Images/Regression_Prediction_Plot_${nCalValue}.png`;

    // Get the q value from the dictionary based on n_cal and alpha
    console.log(alphaValue, nCalValue)
    var q = loaded_dict[nCalValue][alphaValue]['q'];
    var time = loaded_dict[nCalValue][alphaValue]["y_preds_46"];

    document.getElementById('regression_qval').textContent = q.toFixed(4);

    document.getElementById("significance-level").textContent = alphaValue;

    document.getElementById('predicted-time').textContent = time.toFixed(2);
    document.getElementById('time-lower_bound').textContent = (time-q).toFixed(2);
    document.getElementById('time-upper_bound').textContent = (time+q).toFixed(2);

    // Update the image source for alpha
    document.getElementById('alpha-image-display').src = `images/Generated_Images/Regression_Histogram_plot_${nCalValue}_for_${alphaValue}.png`;

    // Preload images for smoother transitions
    var preloadArray = [];
    for (var i = 10; i <= 20; i += 2) {
        preloadArray.push(`images/Generated_Images/Regression_Prediction_Plot_${i}.png`);
    }
    var alphaValues = Array.from({ length: 10 }, (_, i) => (i + 1) / 10).map(val => val.toFixed(1));
    preloadArray = preloadArray.concat(alphaValues.map(alpha => `images/Generated_Images/Regression_Histogram_plot_${nCalValue}_for_${alpha}.png`));
    preloadImages(preloadArray);
}

// Call the preloadImages function to preload images
var preloadArray = [];
for (var i = 10; i <= 20; i += 2) {
    preloadArray.push(`images/Generated_Images/Regression_Prediction_Plot_${i}.png`);
}
preloadImages(preloadArray);