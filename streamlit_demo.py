# Import required libraries
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# from tqdm.auto import trange, tqdm

# Import utility functions and model classes from custom modules
from utils import get_simple_data_train, display_equation, train, get_data, get_test_preds_and_smx, get_scores, quantile, get_pred_sets, mean_set_size, get_test_accuracy, train_model, conformal_prediction_regression
from utils_plot import plot_generic, plot_predictions, histogram_plot, show_samples, plot_conformal_prediction
from model import MLP, MLP1

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)



# Define the main function to create the Streamlit app
def main():
    # Set the title and subheader of the app
    st.title("Conformal Prediction: A Brief Overview")
    st.subheader("Introduction:")
    # Provide an overview of Conformal Prediction
    # st.write("Machine learning models, especially black-box models like neural networks, have gained widespread adoption in high-risk domains like medical diagnostics, where accurate predictions are critical to avoid potential model failures. However, the lack of uncertainty quantification in these models poses significant challenges in decision-making and trust. Conformal prediction emerges as a promising solution, providing a user-friendly paradigm to quantify uncertainty in model predictions.")
    # st.write("Uncertainty is an inherent aspect of real-world data and the predictions made by machine learning models. Conformal prediction offers a principled approach to address this challenge by providing a measure of uncertainty in predictions. The prediction intervals constructed by conformal prediction not only quantify the range of potential outcomes but also assign a confidence level (alpha) to each prediction. This allows users to gauge the reliability of the model's output and make informed decisions based on the level of uncertainty associated with the predictions.")
    # st.write("The key advantage of conformal prediction lies in its distribution-free nature, making it robust to various scenarios without making strong assumptions about the underlying data distribution or the model itself. By utilizing conformal prediction with any pre-trained model, researchers and practitioners can create prediction sets that offer explicit, non-asymptotic guarantees, instilling confidence in the reliability of model predictions.")


    # st.write("Conformal Prediction is a powerful framework in machine learning that provides a measure of uncertainty in predictions. Unlike traditional point predictions, conformal prediction constructs prediction intervals that quantify the range of potential outcomes.")

    st.write("Conformal Prediction is a powerful framework in machine learning that provides a measure of uncertainty in predictions. Unlike traditional point predictions, conformal prediction constructs prediction intervals that quantify the range of potential outcomes.")
    st.write("Machine learning models, particularly neural networks, are widely used in critical areas like medical diagnostics. However, the lack of uncertainty quantification in these models poses challenges in decision-making and trust. Conformal prediction offers a user-friendly way to quantify uncertainty, providing confidence levels for each prediction. Its distribution-free nature makes it robust without strong assumptions about the data distribution or the model. This instills confidence in the reliability of model predictions.")

    st.write(r"The significance of conformal prediction lies in its ability to provide a confidence level ($\alpha$) for the predictions, allowing users to understand the reliability of the model's output. This is especially crucial in critical applications where understanding the uncertainty is essential.")
    # st.write(r"Conformal Prediction for a General Input $x$ and Output $y$:")
    # st.write("**1.** Identify a heuristic notion of uncertainty using the pre-trained model.")
    # st.write(r"    - In conformal prediction, we make use of a pre-trained model to estimate the uncertainty associated with its predictions. This uncertainty is crucial as it allows us to create prediction intervals rather than single point predictions.")

    # st.write("**2.** Define the score function $s(f(x), y) \in \mathbb{R}$. (Larger scores encode worse agreement between $f(x)$ and $y$.)")
    # st.write(r"- The score function $s(f(x), y)$ is a function that quantifies the discrepancy or disagreement between the input $x$ and the output $y$. Larger scores indicate a worse agreement between the predicted value and the true value.")

    # st.write("**3.** Compute $\\hat{q}$ as the ceiling function of $\\frac{(n+1)(1-\\alpha)}{n}$.")
    # st.write(r"    - To determine the quantile value $\hat{q}$, we calculate the $\left\lceil \frac{(n+1)(1-\alpha)}{n} \right\rceil$-th quantile of the calibration scores $s_1 = s(f(X_1), Y_1), ..., s_n = s(f(X_n), Y_n)$, where $d$ is the number of dimensions in the output space, $n$ is the number of calibration data points, and $\alpha$ is the confidence level.")

    # st.write("**4.** Use this quantile to form the prediction sets for new examples:")
    # st.write(r"    - The prediction set $C(X_{\text{test}})$ is constructed as $\{y : s(f(X_{\text{test}}), y) \leq \hat{q}\}$. It contains all the possible output values $y$ for the new input example $X_{\text{test}}$, where the score function $s(f(X_{\text{test}}), y)$ is less than or equal to the computed quantile $\hat{q}$.")

    # st.write("By following these steps, conformal prediction provides a practical way to estimate uncertainty and create prediction intervals for new examples, enabling better decision-making and trust in the model's predictions.")

    st.write("Conformal Prediction Algorithm:")
    st.write(r"1. **Uncertainty Estimation:** Use a pre-trained model to estimate uncertainty in predictions.")
    st.write(r"2. **Score Function:** Define a score function ($s(f(x), y)$) to measure prediction discrepancy.")
    st.write(r"3. **Quantile Computation:** Compute quantile ($q_{val}$) based on calibration data and confidence level.")
    st.write(r"4. **Prediction Intervals:** Form prediction intervals for new examples using quantile ($q_{val}$).")
    st.title("Conformal Predictions for Regression:")
    # Data Visualization Section
    # Sliders to control the coefficients of sine and cosine functions and noise
    custom_slider_style = """
    <style>
    /* Add your custom CSS styling for sliders here */
    /* Example: changing the color of the slider handle to blue */
    div[role="slider"] > .stSlider { background: blue; }
    </style>
    """

    # Display the custom CSS style
    st.markdown(custom_slider_style, unsafe_allow_html=True)
    

    st.write("Conformal prediction in regression provides prediction intervals that quantify the uncertainty associated with the model's predictions.")

    st.write(r"The process involves two steps. First, we train a regression model on a training dataset. Next, we use a calibration dataset to estimate the confidence level ($\alpha$) for the prediction intervals. This represents the proportion of times the intervals will contain the true target value for future test instances.")

    st.write("To compute the prediction intervals, we calculate the residuals for the calibration dataset. The quantile of these residuals based on the chosen confidence level determines the width of the prediction intervals, reflecting the uncertainty in the model's predictions.")

    st.subheader("Function")
    # Sliders with custom styles
    coef_1 = 0.3
    coef_2 = 0.3
    coef_3 = 0.1
    coef_4 = st.slider("Coefficient for noise", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
    n_cal = st.slider("Number of calibration data points", min_value=100, max_value=200, value=150, step=10)
    # Display the equation based on user-selected coefficients
    display_equation(coef_1, coef_2, coef_3, coef_4)
    x_train, y_train, x_cal, y_cal = get_simple_data_train(coef_1, coef_2, coef_3, coef_4, n_cal)
    fig, ax = plot_generic(x_train, y_train, x_cal, y_cal, coef_1=coef_1, coef_2=coef_2, coef_3=coef_3, coef_4=coef_4)
    plt.title("Plot of Training and Calibration Data", fontsize=15)
    st.pyplot(fig)
    st.subheader("Model")
    st.write(r"The model will be trained on the training data and used to generate predictions on the calibration data. The calibration data is used to estimate the confidence level ($\alpha$) for the prediction intervals.")
    # Train the model (MLP) on the generated data
    hidden_dim = 30
    n_hidden_layers = 1
    epochs = 1000
    x_test = torch.linspace(-.5, 1.5, 3000)[:, None]
    net1 = MLP(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    net1 = train(net1, (x_train, y_train), epochs=epochs)
    y_preds = net1(x_test).clone().detach().numpy()

    # Display the plot with the true function, training data, calibration data, predictive mean
    # st.subheader("Prediction Visualization")
    
    fig, ax = plot_predictions(x_train, y_train, x_cal, y_cal, x_test, y_preds, coef_1=coef_1, coef_2=coef_2, coef_3=coef_3, coef_4=coef_4)
    st.pyplot(fig)

    # Calculate residuals and estimate quantile based on user-selected alpha

    # x_test = torch.linspace(-.5, 1.5, 1000)[:, None]
    # y_preds = net(x_test).clone().detach().numpy()
    # y_cal_preds = net(x_cal).clone().detach()
    # # st.latex(score_func)
    # st.latex(r"s_i = |y_i - \hat{y}_i|")
    # st.write("The score function $s_i$ represents the absolute difference between the true output $y_i$ and the model's predicted output $\hat{y}_i$ for each calibration data point $x_i$. It measures the discrepancy between the true values and their corresponding predictions, providing a measure of model fit to the calibration data.")
    # resid = torch.abs(y_cal - y_cal_preds).numpy()
    # alpha = st.slider("Select a value for alpha:", min_value=0.05, max_value=1.0, step=0.001, value=0.06)
    # # n = len(x_cal)
    # # print(f"n = {n}")
    # # fig = plot_calibration_scores(x_cal, scores=resid)
    # # st.pyplot(fig)
    # n = len(x_cal)
    # q_val = np.ceil((1 - alpha) * (n + 1)) / n
    # print(q_val)
    # st.latex(r"q = \frac{{\lceil (1 - \alpha) \cdot (n + 1) \rceil}}{{n}} = {:.4f}".format(q_val))
    # q = np.quantile(resid, q_val, method="higher")
    st.latex(r"s_i = |y_i - \hat{y}_i|")
    st.write("The score function $s_i$ represents the absolute difference between the true output $y_i$ and the model's predicted output $\hat{y}_i$ for each calibration data point $x_i$. It measures the discrepancy between the true values and their corresponding predictions, providing a measure of model fit to the calibration data.")
    alpha = st.slider("Select a value for alpha:", min_value=0.05, max_value=1.0, step=0.001, value=0.06)

    x_test, y_preds, q, resid = conformal_prediction_regression(x_cal, y_cal, net1,alpha)

    histogram_plot(resid, q, alpha)
    st.write(r"The $q^{th}$ quantile this:")
    st.latex(r"q_{{\text{{value}}}} = {:.4f}".format(q))
    # x_true = np.linspace(-.5, 1.5, 1000)
    # y_true = coef_1 * np.sin(2 * np.pi*(x_true)) + coef_2 * np.cos(4 * np.pi *(x_true )) + coef_3 * x_true

    # # Generate plot with the true function, training data, calibration data, predictive mean, and uncertainty bands
    # fig, ax = plt.subplots(figsize=(10, 5))
    # plt.xlim([-.5, 1.5])
    # plt.ylim([-1.5, 2.5])
    # plt.xlabel("X", fontsize=30)
    # plt.ylabel("Y", fontsize=30)
    
    # ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")
    # ax.plot(x_train, y_train, 'go', markersize=4, label="training data")
    # ax.plot(x_cal, y_cal, 'ro', markersize=4, label="calibration data")
    # ax.plot(x_test, y_preds, '-', linewidth=3, color="y", label="predictive mean")
    # ax.fill_between(x_test.ravel(), y_preds - q, y_preds + q, alpha=0.6, color='y', zorder=5)
    # plt.legend(loc='best', fontsize=15, frameon=False)
    # st.write("The prediction interval is calculated as:")
    # st.latex(r"\hat{C}(X_{n+1}) = [ \hat{f}(x_{n+1}) - {q_{val}}, \, \hat{f}(x_{n+1}) + {q_{val}} ]")
    # plt.title("Plot of confidence interval for the conformal prediction", fontsize=15)
    # st.pyplot(fig)
    plot_conformal_prediction(x_train, y_train, x_cal, y_cal, x_test, y_preds, q, coef_1, coef_2, coef_3)


    st.title("Conformal Predictions in Classification")
    
    st.write("In regression, we had the predictions as continuous uncertainty bands. Now for classification, the outputs from the model are class probabilities, so the prediction sets are now discrete sets of the type:")
    st.latex(r"\hat{C}(X_{n+1})\subseteq \{1,\dots,K\}")
    st.write(r"where $K$ is the number of classes. This change in the output affects how we calculate the conformity scores.")
    
    st.write("We will use the MNIST dataset. The 60k training samples are split into two parts: the training set, which consists of 55k images, and the calibration set, which has 5k images. The test set consists of 10k images.")
    
    X_train, y_train, X_test, y_test, X_calib, y_calib = get_data()
    
    net = MLP1()
    
    train_data = (X_train, y_train)
    # calib_data = (X_calib, y_calib)
    # test_data = (X_test, y_test)
    
    net = train_model(net, train_data)
    print("Test accuracy of the model is", get_test_accuracy(X_test, y_test, net))
    
    st.write("For training, we will use a simple MLP. **Test accuracy** of the model is", get_test_accuracy(X_test, y_test, net))
    
    st.subheader("How to calculate Conformity Scores?")
    st.write("The method of calculating conformity scores is a modelling decision. Here, we will use a simple method based on the softmax scores. The score is calculated by the following formula:")
    st.write(r"$s_i=1-\hat{\pi}_{x_i}(y_i)$ for a sample $(x_i, y_i)$ from the calibration set.")
    
    st.write(r"The sample score $s_i$ is equal to 1 minus the softmax output of the true class.  If the softmax value of the true class is low, it means that the model is uncertain. The score in such a case will be high.")
    st.write(r"After calculating the scores from the calibration set, we choose an error rate $\alpha$. The probability that the prediction set contains the correct class will be approximately 1 - $\alpha$. If $\alpha$ = 0.05, then the probability that the prediction set contains the true class is 0.95.")
    st.write(r"We will get the prediction set for a test sample $(x_{n+1}, y_{n+1})$ by:")
    st.latex(r"\hat{C}(x_{n+1})=\{y'\in K:\hat{\pi}_{x_{n+1}}(y') \ge 1-{q_{val}}\}")
    st.write(r"The prediction set $C$ consists of all the classes for which the softmax score is above a threshold value 1-${q_{val}}$. ${q}$ is calculated as $\frac{{\lceil (1 - \alpha) \cdot (n + 1) \rceil}}{{n}}$ quantile of the scores from the calibration set.")
    n = len(X_calib)
    scores = get_scores(net, (X_calib, y_calib))
    alpha = st.slider("Select a value for alpha:", min_value=0.01, max_value=1.0, step=0.001, value=0.04)
    q_val = np.ceil((1 - alpha) * (n + 1)) / n
    st.latex(r"q = \frac{{\lceil (1 - \alpha) \cdot (n + 1) \rceil}}{{n}} = {:.4f}".format(q_val))
    q = np.quantile(scores, q_val, method="higher")
    histogram_plot(scores, q, alpha)
    # st.pyplot(fig)
    st.write(r"For this value of alpha, the threshold value 1-${q_{val}}$"+ " is {:.4f}".format(1 - q))
    
    st.write("For example, select an image from the below slider. The softmax scores for the classes can be seen in the plot on the right side. If the score is above the threshold value, then the class is in the predicted set.")
    
    pred_sets = get_pred_sets(net, (X_test, y_test), q, alpha)
    
    idxs = [976,300,844,149,195,619,511,112,65,658]
    show_samples(X_test, idxs, pred_sets, net, q, alpha)
    
    test_img_index = st.slider("Choose Image:", min_value=1, max_value=10, step=1, value=5)
    # test_img_index = st.selectbox("Choose Image:", options = [i for i in range(1, 10)])
    
    fig, ax, pred, pred_str = get_test_preds_and_smx(X_test, idxs[test_img_index-1], pred_sets, net, q, alpha)
    st.pyplot(fig)
    st.write("Prediction Set for this image: ", pred_str)
    st.write("The average size of prediction sets for all the images from the test set is {:.3f}".format(mean_set_size(pred_sets)))
    st.write("*What does the average size mean?* We observe that the average size of the prediction set decreases when value of alpha is increased. This is because of our method for computing conformity scores, where we only take into account the softmax scores of the correct class when calculating 𝑞̂. With increasing alpha, the softmax scores for the classes decreases and thus there are lesser scores above the threshold value.")    


if __name__ == "__main__":
    main()

