function selectImage(index) {
  var resultImage = document.getElementById("frog-result-image");
  var resultText = document.getElementById("frog-result-text");
  var imagePreviews = document.querySelectorAll(".frog-image-preview");

  var inferences = [
      "The model assigns a high probability to categorize this tennis ball as a green apple due to its resemblance in both shape and color. However, this is an incorrect classification, and it is essential to incorporate a level of uncertainty into this prediction.",
      "This image depicts an orange, but the model erroneously labels it as a green apple with high probability solely because of its green hue.",
      "The classification of this image featuring a frog as a green apple is once more the result of the predominant green color. In real life scenarios, a false classification like this may have significant implications."
  ]

  imagePreviews.forEach(function (preview, i) {
      preview.classList.remove("selected");
      if (i === index) {
          preview.classList.add("selected");
      }
  });

  resultImage.src = "images/frog_tennis/example" + index + ".svg";
  resultText.innerHTML = inferences[index];
}