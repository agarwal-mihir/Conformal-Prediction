let loaded_dict2; // Declare the variable in the outer scope

fetch('data/classification.json')
    .then((response) => response.json())
    .then((json) => {
        loaded_dict2 = json; // Assign the value to the variable
        console.log(loaded_dict);
    })
    .catch((error) => {
        console.error('Error loading JSON:', error);
    });

function c_changeAlpha(sliderContainer, isIncrease) {
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
    c_updateAlphaValue(newValue);
}

function c_updateAlphaValue(value) {
    var alphaValue = parseFloat(value).toFixed(1);
    var imgValue = parseInt(document.getElementById("c_image-selector-value").textContent);
    c_updateImages(alphaValue, imgValue);
}

// function c_changeImage(sliderContainer, step) {
//     // Get the image selector slider element
//     var imageSlider = sliderContainer.querySelector('.slider');
//     var currentValue = parseInt(imageSlider.value);
//     var newValue = currentValue + step;

//     // Ensure the new value stays within the defined range
//     if (newValue < 0) {
//         newValue = 0;
//     } else if (newValue > 7) {
//         newValue = 7;
//     }

//     // Update the image selector slider value
//     imageSlider.value = newValue;

//     // Call the c_updateImageValue function to update the displayed value and image
//     c_updateImageValue(newValue);
// }

function c_changeImage(idx) {
    const sliderValue = parseInt(document.getElementById('c_image-selector-value').textContent);
    const imgIndex = sliderValue;
    c_updateImageValue(idx);

    // Replace 'image' with your image element or container.
    // You can change the source to match your image naming pattern.
    // const img = document.getElementById('mnist_img');
    // img.src = `images/mnist_img/img_${imgIndex}.png`;

    // Optionally, you can update the button text.
    // button.textContent = `Image ${imgIndex}`;
}

function c_updateImageValue(value) {
    var alphaValue = parseFloat(document.getElementById("c_alpha-slider-value").textContent).toFixed(1);
    c_updateImages(alphaValue, value);
}

// function c_updateImageValue(value) {
//     document.getElementById('c_image-selector-value').textContent = value;
// }

function c_updateImages(alphaValue, imageIndex) {
    // Get the alpha value
    // var alphaValue = parseFloat(document.querySelector('.gif-slider .slider').value).toFixed(1);
    // var alphaValue = parseFloat(document.getElementById("c_alpha-slider-value").textContent).toFixed(1);

    // Update the image source for the image selector
    document.getElementById('c_image-display').src = `images/Generated_Images/Classification_Prediction_${imageIndex}_for_${alphaValue}.png`;
    document.getElementById("c_alpha-image-display").src = `images/Generated_Images/Classification_Histogram_${alphaValue}.png`;

    document.getElementById("c_alpha-slider-value").textContent = alphaValue;
    // document.getElementById("c_image-selector-value").textContent = imageIndex;

    // You can set the prediction set value here based on the selected image and alpha
    // For example, update it based on some data or calculation
    document.getElementById('c_prediction-set').textContent = loaded_dict2[alphaValue][imageIndex]['pred_str'];
    document.getElementById('q_val').textContent = loaded_dict2[alphaValue]['q_val'];
    document.getElementById('avg_size').textContent = loaded_dict2[alphaValue]['avg_size'];
    document.getElementById('c_threshold-value').textContent = loaded_dict2[alphaValue]['q_thres'];

    // Preload images for smoother transitions
    var preloadArray2 = [];
    for (var i = 0; i <= 7; i++) {
        preloadArray2.push(`images/Generated_Images/Classification_Prediction_${i}_for_${alphaValue}.png`);
    }
    // Preload images for smoother transitions
    console.log(alphaValue); 
    preloadImages(preloadArray2);
}

// Preload images for the image selector
var imagePreloadArray3 = [];
for (var i = 0; i <= 7; i++) {
    imagePreloadArray3.push(`images/Generated_Images/Classification_Prediction_${i}_for_0.1.png`);
}
preloadImages(imagePreloadArray3);
