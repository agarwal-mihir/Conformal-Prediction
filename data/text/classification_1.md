<div style="text-align: justify;">In the realm of regression analysis, the model generates continuous uncertainty bands around the predicted values, offering an interval estimation of the output variable. However, as we transition to classification problems, the nature of the output changes substantively. In classification, the model generates discrete outputs, which are class probabilities. The prediction set now takes the form of a discrete subset of the available classes, mathematically represented as:</div><br>

<div style="text-align: center;">

$\hat{C}(X_{n+1}) \subseteq \{1, \ldots, K\}$

</div>

In this formalism, $\hat{C}(X_{n+1})$ is the prediction set corresponding to a new input $X_{n+1}$, and $K$ signifies the total number of unique classes.

<div style="text-align: justify;">We will use the MNIST dataset. The 60,000 training samples are split into two parts: the training set, which consists of 55,000 images, and the calibration set, which has 5,000 images. The test set consists of 10,000 images.</div>

For training, we will use a simple Multi-layer Perceptron. The **test accuracy** of the model is: 0.9066

## How to calculate Conformity Scores?

<div style="text-align: justify;">The method of calculating conformity scores is a modeling decision. Here, we will use a simple method based on the softmax scores. The score is calculated by the following formula:</div><br>


<div style="text-align: center;">

$s_i = 1 - \hat{\pi}_{x_i}(y_i)$

</div>

for a sample $(x_i, y_i)$ from the calibration set, where $\hat{\pi}_{x_i}(y_i)$ represents the softmax score of the true class $(y_i)$. The sample score $s_i$ is equal to 1 minus the softmax output of the true class. If the softmax value of the true class is low, it means that the model is uncertain. The score in such a case will be high.

After calculating the scores from the calibration set, we choose an error rate $\alpha$. The probability that the prediction set contains the correct class will be approximately $1 - \alpha$. If $\alpha = 0.05$, then the probability that the prediction set contains the true class is 0.95.

We will get the prediction set for a test sample $(x_{n+1}, y_{n+1})$ by:

<div style="text-align: center;">

$\hat{C}(x_{n+1}) = \{y' \in K: \hat{\pi}_{x_{n+1}}(y') \ge 1 - q_{val}\}$

</div>

<div style="text-align: justify;">

The prediction set $C$ consists of all the classes for which the softmax score is above a threshold value $1-q_{val}$. $q_{val}$ is calculated as the $\frac{{\lceil (1 - \alpha) \cdot (n + 1) \rceil}}{{n}}$ quantile of the scores from the calibration set.

</div>
