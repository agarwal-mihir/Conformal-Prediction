<div style="text-align: justify;">
<h3>Understanding the Predicted Set</h3>
The 'predicted set' refers to the set of classes that the model deems probable for the given input. 
A class is included in the predicted set if its softmax score is above a predetermined threshold. This threshold 
is influenced by the selected value of alpha and the computed quantile from the calibration data. The predicted set 
gives us an indication of the model's confidence in its classification. If the set contains multiple classes, it 
indicates that the model is less certain about the true class label.
</div>

<br>

<div style="text-align: justify;">For example, select an image from the below slider. The softmax scores for the classes can be seen in the plot on the right side. If the score is above the threshold value, then the class is in the predicted set.</div>

<br>