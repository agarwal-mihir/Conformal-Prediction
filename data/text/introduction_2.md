This example highlights two key issues. First, if your training data is not representative or is biased, your model won't work well. Second, many models don't tell you how certain they are about their predictions, which can be risky.

Conformal Prediction is an approach in machine learning that helps address the second issue. It doesn't just give a 'best guess' prediction; it also provides a range of possible outcomes. This helps you understand how confident you should be in the model's prediction.

Not having a measure of uncertainty is a big issue, especially for neural networks which are often seen as 'black boxes.' If you don't know how sure a model is about its prediction, it's tough to make informed decisions. For this reason, understanding and measuring a model's level of certainty is important for trusting its outputs.

<div style="text-align: justify;">Conformal Prediction is both robust and computationally efficient, eliminating the need for data or model-specific assumptions. Its computational complexity can be as low as O(n), outperforming Bayesian methods like MC Dropout.</div>

<br>

For any test sample $(X_{n+1},Y_{n+1})$, the calculated uncertainty bands will satisfy the following property:

<div style="text-align: center;">

$\mathbb{P}(Y_{n+1} \in \hat{C}(X_{n+1})) \ge 1-\alpha$

</div>

This means that with probability at least $1-\alpha$, the computed uncertainty band $\hat{C}(X_{n+1})$ around the point estimate $\hat{Y}_{n+1}$ will contain the true unknown value $Y_{n+1}$. Here $\alpha$ is a user-chosen error rate.

This mathematical guarantee on prediction intervals makes it invaluable in mission-critical applications.

### How Conformal Prediction Works
1. **Data Split**: Divide data into a Training and a Calibration Set.
2. **Model Training**: Train your model on the Training Set.
3. **Scoring Rule**: Create a rule to evaluate model predictions.
4. **Calibration**: Fine-tune the model on the Calibration Set.
5. **Confidence Level**: Set a desired confidence level.
6. **Future Predictions**: Make new predictions with confidence levels.
