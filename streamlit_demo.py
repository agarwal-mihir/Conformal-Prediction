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
import streamlit_image_select as st_image
# from tueplots import axes, bundles

# from tqdm.auto import trange, tqdm
import utils
# Import utility functions and model classes from custom modules
from utils import get_simple_data_train, display_equation, train, get_data, get_test_preds_and_smx, get_scores, quantile, get_pred_sets, mean_set_size, get_test_accuracy, train_model, conformal_prediction_regression, tensor_to_img
from utils_plot import plot_generic, plot_predictions, histogram_plot, show_samples, plot_conformal_prediction
from model import MLP, MLP1
from PIL import Image
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
# plt.rcParams.update(bundles.icml2022())


# Define the main function to create the Streamlit app
def main():


   
    st.markdown('<h1 style="text-align: center;font-size:50px;">Conformal Prediction: <br>A Visual Introduction</h1>', unsafe_allow_html=True)

    st.markdown("""
    <div style='position: relative;text-align: center; color: grey; font-size: 18px;'>
        <b>By</b> <br>
        <div style='position: relative; left: 0px; top: 20px;display: inline-block; margin-right: 40px;font-size: 14px;'>
            <b>Lalit Chandra Routhu</b> <br>
            IIT Patna
        </div>
        <div style='position: relative; left: 0px; top: 20px;display: inline-block; margin-left: 40px;font-size: 14px;'>
            <b>Mihir Agarwal</b> <br>
            IIT Gandhinagar
        </div> <br>
        <div style='position: relative; left: 13px; top:30px;display: inline-block; margin-right: 10px;font-size: 14px;'>
            <b>Zeel B Patel</b> <br>
            IIT Gandhinagar
        </div>
        <div style='position: relative; left: 5px; top: 30px;display: inline-block; margin-left: 90px;font-size: 14px;'>
            <b>Nipun Batra</b> <br>
            IIT Gandhinagar
        </div>
    </div>
""", unsafe_allow_html=True)
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    
    st.title("Introduction:")

    

    st.markdown('<div style=\"text-align: justify;\">Understanding the nuances of uncertainty is pivotal in numerous domains, ranging from financial forecasting to healthcare diagnostics and autonomous vehicle control. The accurate quantification of uncertainty enables robust decision-making and engenders trust in machine learning models. For instance, in medical settings, a false negative could result in untreated disease progression, while a false positive might lead to unnecessary treatments—both with life-altering implications.</div>', unsafe_allow_html=True)
    st.markdown('<br>', unsafe_allow_html=True)
    st.markdown("<div style=\"text-align: justify;\">To illustrate, consider a scenario where a machine learning model was fine-tuned to classify green apples and oranges. Utilizing the fast.ai library, a ResNet18 model was deployed and fed a myriad of images containing these fruits. However, when exposed to images of other objects that were green—such as frogs, green tennis balls, and even green oranges—the model overwhelmingly classified these as 'green apples' with high confidence. You can see the examples as follows:</div>", unsafe_allow_html=True)
    image_paths = [
    "Screenshot 2023-09-29 at 3.56.30 PM.png",
    "Screenshot 2023-09-29 at 3.56.50 PM.png",
    "Screenshot 2023-09-29 at 3.56.57 PM.png"
    # Add more image paths here
]

# Read images into numpy arrays after converting to RGB and put them in a list
    all_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]

# Streamlit image selection widget
    test_img_idx = st_image.image_select(label="Select an image", images=all_images, return_value="index", use_container_width=False)
    # dict = {0: "Probability it's a apple: 0.9947, Probability it's a orange: 0.0053", 1: "Probability it's a apple: 0.9753, Probability it's a orange: 0.0247", 2: "Probability it's a apple: 0.9936, Probability it's a orange: 0.0064"}
    # st.write(dict[test_img_idx])
    dict = {
    0: {"apple": 0.9947, "orange": 0.0053},
    1: {"apple": 0.9753, "orange": 0.0247},
    2: {"apple": 0.9936, "orange": 0.0064}
}


    apple_prob = dict[test_img_idx]["apple"]    
    orange_prob = dict[test_img_idx]["orange"]

    st.markdown(f"<div style='font-family: \"Helvetica, Arial, sans-serif\"; font-size:21px;'><b><span style='color:green;'>Probability it's an apple: {apple_prob}</span><br><span style='color:orange;'>Probability it's an orange: {orange_prob}</span></b></div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style=\"text-align: justify;\">This example vividly illustrates the pitfalls of biased training and the lack of uncertainty quantification. A traditional classification model would provide point estimates—single labels with associated probabilities—that could be misleading. Such deterministic outputs can have far-reaching consequences, from flawed recommendations to incorrect automated decisions.</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div style=\"text-align: justify;\">Conformal Prediction is an efficacious framework in machine learning that delivers well-calibrated measures of uncertainty associated with predictions. It extends beyond the provision of mere point estimates, constructing prediction intervals that encapsulate the realm of plausible outcomes.</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style=\"text-align: justify;\">The absence of uncertainty quantification in many machine learning models, especially neural networks, presents obstacles in decision-making processes and undermines trustworthiness. It is imperative, therefore, to comprehend and assess the confidence level of a model’s predictions.</div>", unsafe_allow_html=True)


    
    # st.write("Conformal Prediction is notable for its distribution-free properties, which eliminate the need for rigorous assumptions about the data distribution or the model’s architectural design. This makes the method inherently robust and lends greater confidence in the reliability of the model’s predictions.")

    # st.write("Additionally, Conformal Prediction is computationally more efficient compared to Bayesian methods such as Monte Carlo Dropout (MC Dropout), Deep Ensembles, and Bootstrap techniques. While Bayesian methods necessitate multiple forward passes or resampling to estimate uncertainty, Conformal Prediction typically requires only a single forward pass, rendering it faster and more scalable. The computational complexity for Conformal Prediction can be as low as $O(n)$ for some implementations, where $n$ is the size of the calibration set.")

    # st.write(r"Mathematically, Conformal Prediction offers strong guarantees about the coverage of its prediction intervals. Given a predefined confidence level $\alpha$, the framework guarantees that the true output will lie within the prediction interval with a probability of at least $1 - \alpha$. This is formally expressed as:")
    # st.latex(r"P(y \in C(x)) \geq 1 - \alpha")
    # st.write("where $C(x)$ is the prediction interval for a new input $x$ and $y$ is the true output. These coverage guarantees are valid under fairly general conditions, providing a robust measure of uncertainty.")

    # st.write("The significance of Conformal Prediction is amplified in mission-critical applications where understanding uncertainty is of paramount importance. Its mathematical rigor and computational efficiency make it an excellent choice for real-time and resource-constrained environments.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div style=\"text-align: justify;\">Conformal Prediction is both robust and computationally efficient, eliminating the need for data or model-specific assumptions. Its computational complexity can be as low as O(n), outperforming Bayesian methods like MC Dropout.</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(r"For any test sample $(X_{n+1},Y_{n+1})$, the calculated uncertainty bands will satisfy the following property:")
    st.latex(r"\mathbb{P}(Y_{n+1} \in \hat{C}(X_{n+1})) \ge 1-\alpha,")
    st.markdown(r"This means that with probabliity at least $1-\alpha$, the computed uncertainty band $\hat{C}(X_{n+1})$ around the point estimate $\hat{Y}_{n+1}$ will contain the true unknown value $Y_{n+1}$. Here $\alpha$ is a user-chosen error rate.")
    
    st.write("This mathematical guarantee on prediction intervals makes it invaluable in mission-critical applications.")


    st.markdown("""
### How Conformal Prediction Works
1. **Data Split**: Divide data into a Training and a Calibration Set.
2. **Model Training**: Train your model on the Training Set.
3. **Scoring Rule**: Create a rule to evaluate model predictions.
4. **Calibration**: Fine-tune the model on the Calibration Set.
5. **Confidence Level**: Set a desired confidence level.
6. **Future Predictions**: Make new predictions with confidence levels.
""")

    st.title("Conformal Prediction for Regression")
    
    st.markdown("""<div style=\"text-align: justify;\">
In conformal prediction for regression, we aim to construct prediction intervals around our point estimates to offer a statistically 
valid level of confidence. The method leverages a calibration dataset to compute 'conformity scores,' which help us rank 
how well the model's predictions align with actual outcomes. These scores, in turn, guide the creation of prediction intervals 
with a desired coverage probability. Thus, conformal prediction serves as a tool for robust and interpretable prediction intervals.
</div>""", unsafe_allow_html=True)
    
    coef_1 = 0.3
    coef_2 = 0.3
    coef_3 = 0.1
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.write("Let us consider a simple regression problem with a single input variable $x$ and a single output variable $y$.")
    st.write(r"The true function $f(x)$ is given by: ")
    # Sliders with custom styles
    display_equation(coef_1, coef_2, coef_3)
    coef_4 = st.slider(r"Coefficient for noise $(\epsilon)$", min_value=0.1, max_value=1.0, value=0.3, step=0.01, format="%.2f")
    st.write(f"You can choose the number of calibration data points $(n)$ using the slider below.")
    n_cal = st.slider("Number of calibration data points $(n)$", min_value=10, max_value=20, value=10, step=2)
    # Display the equation based on user-selected coefficients
    
    x_train, y_train, x_cal, y_cal = get_simple_data_train(coef_1, coef_2, coef_3, coef_4, n_cal)
    # fig, ax = plot_generic(x_train, y_train, x_cal, y_cal, coef_1=coef_1, coef_2=coef_2, coef_3=coef_3, coef_4=coef_4)
    # plt.title("Plot of Training and Calibration Data", fontsize=15)
    # st.pyplot(fig)
    st.subheader("Model")
    st.markdown("<div style=\"text-align: justify;\">The model will be trained on the training data and used to generate predictions on the calibration data.</div>", unsafe_allow_html=True)
    st.markdown(r"The calibration data is used to estimate the quantiles ($q_{val}$) for the prediction intervals.")
    # Train the model (MLP) on the generated data
    hidden_dim = 30
    n_hidden_layers = 1
    epochs = 1000
    x_test = torch.linspace(-.5, 1.5, 3000)[:, None]
    net1 = MLP(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    net1 = train(net1, (x_train, y_train), epochs=epochs)
    y_preds = net1(x_test).clone().detach().numpy()
    
    fig, ax = plot_predictions(x_train, y_train, x_cal, y_cal, x_test, y_preds, coef_1=coef_1, coef_2=coef_2, coef_3=coef_3, coef_4=coef_4)
    st.pyplot(fig)


    st.latex(r"s_i = |y_i - \hat{y}_i|")
    
    st.markdown("<div style=\"text-align: right;\">", unsafe_allow_html=True)
    st.write("The score function $s_i$ represents the absolute difference between the true \
             output $y_i$ and the model's predicted output $\hat{y}_i$ for each calibration data point $x_i$. \
             It measures the discrepancy between the true values and their corresponding predictions, providing a measure \
             of model fit to the calibration data.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    alpha = st.slider(r"Select a value for $\alpha$:", min_value=0.15, max_value=1.0, step=0.001, value=0.16)

    x_test, y_preds, q, resid = conformal_prediction_regression(x_cal, y_cal, net1,alpha)

    histogram_plot(resid, q, alpha)
    st.write(r"The $q^{th}$ quantile is:")
    # st.latex(r"q_{{\text{{value}}}} = {:.4f}".format(q))
    st.markdown(f'<span style=" top: 2px;font-size:50px;"><center> $q_{{\\text{{value}}}} = {q:.4f}$</center></span>', unsafe_allow_html=True)


    plot_conformal_prediction(x_train, y_train, x_cal, y_cal, x_test, y_preds, q, coef_1, coef_2, coef_3, alpha)

    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################


    st.title("Conformal Predictions in Classification")
    
    st.markdown("<div style=\"text-align: justify;\">In the realm of regression analysis, the model generates continuous uncertainty bands around the predicted values, offering an interval estimation of the output variable. However, as we transition to classification problems, the nature of the output changes substantively. In classification, the model generates discrete outputs, which are class probabilities. The prediction set now takes the form of a discrete subset of the available classes, mathematically represented as:</div>", unsafe_allow_html=True)
    st.latex(r"\hat{C}(X_{n+1}) \subseteq \{1, \ldots, K\}")
   
    st.write("In this formalism, $\hat{C}(X_{n+1})$ is the prediction set corresponding to a new input $X_{n+1}$, and $K$ signifies the total number of unique classes.")
    
    st.markdown("<div style=\"text-align: justify;\">We will use the MNIST dataset. The 60,000 training samples are split into two parts: the training set, which consists of 59950 images, and the calibration set, which has 50 images. The test set consists of 10k images.</div>", unsafe_allow_html=True)
    
    X_train, y_train, X_test, y_test, X_calib, y_calib = get_data()
    
    net = MLP1()
    
    train_data = (X_train, y_train)
    # calib_data = (X_calib, y_calib)
    # test_data = (X_test, y_test)
    
    net = train_model(net, train_data)
    # print("Test accuracy of the model is", get_test_accuracy(X_test, y_test, net))
    
    st.write("For training, we will use a simple Multi-layer Perceptron. The **test accuracy** of the model is", get_test_accuracy(X_test, y_test, net))
    
    st.subheader("How to calculate Conformity Scores?")
    st.markdown("<div style=\"text-align: justify;\">The method of calculating conformity scores is a modelling decision. Here, we will use a simple \
              method based on the softmax scores. The score is calculated by the following formula:</div>", unsafe_allow_html=True)
    
    st.latex("s_i=1-\\hat{\\pi}_{x_i}(y_i)")
    st.write("for a sample $(x_i, y_i)$ from the calibration set, where $\\hat{\\pi}_{x_i}(y_i)$ represents the softmax score of the true class $(y_i)$. The sample score $s_i$ is equal to 1 minus the softmax output of the true class.  If the softmax value \
             of the true class is low, it means that the model is uncertain. The score in such a case will be high.")
    
    # st.write(r"After calculating the scores from the calibration set, we choose an error rate $\alpha$. The probability \
    #          that the prediction set contains the correct class will be approximately 1 - $\alpha$. If $\alpha$ = 0.05, \
    #          then the probability that the prediction set contains the true class is 0.95.")
    
    st.write("After calculating the scores from the calibration set, we choose an error rate", r"$\alpha$.",
         "The probability that the prediction set contains the correct class will be approximately", 
         r"$1 - \alpha$.",
        "If", r"$\alpha = 0.05$,", "then the probability that the prediction set contains the true class is 0.95.")
    
    
    st.write(r"We will get the prediction set for a test sample $(x_{n+1}, y_{n+1})$ by:")
    st.latex(r"\hat{C}(x_{n+1})=\{y'\in K:\hat{\pi}_{x_{n+1}}(y') \ge 1-{q_{val}}\}")
    
    st.markdown("<div style=\"text-align: justify;\">The prediction set C consists of all the classes for which the softmax score is above a threshold value </div>", unsafe_allow_html=True)
    st.write(r"1-${q_{val}}$.  ${q_{val}}$ is calculated as the $\frac{{\lceil (1 - \alpha) \cdot (n + 1) \rceil}}{{n}}$ quantile of the scores from the calibration set.")
    
    n = len(X_calib)
    scores = get_scores(net, (X_calib, y_calib))
    alpha = st.slider(r"Select a value for $\alpha$:", min_value=0.01, max_value=1.0, step=0.001, value=0.04)
    q_val = np.ceil((1 - alpha) * (n + 1)) / n
    # st.latex(r"q_{\text{val}} = \frac{{\lceil (1 - \alpha) \cdot (n + 1) \rceil}}{{n}} = {:.4f}".format(q_val))
    st.latex(r"q_{{\text{{val}}}} = \frac{{\lceil (1 - \alpha) \cdot (n + 1) \rceil}}{{n}} = {:.4f}".format(q_val))

    q = np.quantile(scores, q_val, method="higher")
    histogram_plot(scores, q, alpha)
    # st.pyplot(fig)
    st.markdown(r"For this value of alpha, the threshold value $1-q_{\text{val}}$ is " + f'<span style="font-size:20px;">{1-q:.4f}</span>', unsafe_allow_html=True)
    
    st.markdown("""<div style=\"text-align: justify;\">
<h3>Understanding the Predicted Set</h3>
The 'predicted set' refers to the set of classes that the model deems probable for the given input. 
A class is included in the predicted set if its softmax score is above a predetermined threshold. This threshold 
is influenced by the selected value of alpha and the computed quantile from the calibration data. The predicted set 
gives us an indication of the model's confidence in its classification. If the set contains multiple classes, it 
indicates that the model is less certain about the true class label.</div>
""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div style=\"text-align: justify;\">For example, select an image from the below slider. The softmax scores for the classes can be seen in the plot on the right side. If the score is above the threshold value, then the class is in the predicted set.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    pred_sets = get_pred_sets(net, (X_test, y_test), q, alpha)
    print("hello", pred_sets[0])
    fashion_mnist_data = utils.fashion_mnist()
    fashion_idx = [5, 18]
    idxs = [300,149,1782,195, 1, 2]

    # Get images from fashion mnist data
    fashion_images = [tensor_to_img(fashion_mnist_data, idx) for idx in fashion_idx]

    # Get images from X_test
    xtest_images = [tensor_to_img(X_test, idx) for idx in idxs]

    # Concatenate the two lists of images
    all_images = fashion_images + xtest_images

    # Let user select an image using Streamlit widget
    test_img_idx = st_image.image_select(label="Select an image", images=all_images, return_value="index", use_container_width=False)

    # If the selected index is from fashion mnist data, handle accordingly
    if test_img_idx < len(fashion_idx):
        # handle fashion mnist image
        selected_img_tensor = fashion_mnist_data[fashion_idx[test_img_idx]]
    else:
        # adjust the index to match the X_test list
        test_img_idx -= len(fashion_idx)
        selected_img_tensor = X_test[idxs[test_img_idx]]

    # Continue with the rest of your code
    fig, ax, pred, pred_str = get_test_preds_and_smx(selected_img_tensor, test_img_idx, pred_sets, net, q, alpha)
    st.pyplot(fig)
    st.write("Prediction Set for this image: ", pred_str)
    
    st.markdown("<div style=\"text-align: justify;\">In the above examples, the first 2 images are sourced from the Fashion-MNIST dataset. The model is \
             uncertain about these images, which can be seen by the larger predicted set sizes. This is a property we want,\
             as the size of the predicted set indicates the model's uncertainty. In contrast, for the last 2 images, the predicted \
             set contains only one element because the model is confident about its prediction. This is reflected by \
             the high softmax scores of the true classes.</div>", unsafe_allow_html=True)
    
    st.markdown(f"The average size of prediction sets for all the images from the test set is <span style='font-size:20px;'>{mean_set_size(pred_sets)}</span>", unsafe_allow_html=True)


    st.write(f" **What does the average size mean?**")
    st.markdown("<div style=\"text-align: justify;\">We observe that the average size of the prediction set decreases when \
             value of alpha is increased. This is because of our method for computing conformity scores, where we only \
             take into account the softmax scores of the correct class when calculating 𝑞̂. With increasing alpha, the \
             softmax scores for the classes decreases and thus there are lesser scores above the threshold value.</div>", unsafe_allow_html=True)
    
    st.title("Conclusion")
    st.markdown("""<div style=\"text-align: justify;\">
As we navigated the complexities and intricacies of Conformal Prediction, what emerged is a technique that is not just innovative but foundational to modern machine learning practices. Its distinctive characteristics—ranging from applicability in high-stakes domains to its model-agnostic nature—demonstrate its potential to revolutionize the way we think about, and apply, machine learning.
<br><br>
<h5>Implications in High-Stakes Domains</h5>
Conformal Prediction isn't merely a theoretical novelty; its implications are deeply impactful, particularly in sectors where prediction reliability is paramount, such as medical diagnostics, autonomous vehicles, and finance. Traditional machine learning methods often supply a point estimate without a gauge of prediction uncertainty. However, in life-critical applications like diagnosing illnesses, a mere point estimate is insufficient; medical practitioners require a measure of certainty accompanying each diagnosis.
<br><br>
<h5>Model-Agnostic Nature</h5>
One of the most salient features of Conformal Prediction is its model-agnosticism. This is a breakthrough for the ML community, which often struggles with integrating uncertainty quantification techniques that are specific to particular types of models. The model-agnosticism of Conformal Prediction means that it can be applied across various machine learning algorithms without needing algorithm-specific tuning. This is not only computationally efficient but also invaluable for researchers who may be dealing with multiple types of models.
<br><br>
<h5>Data Distribution Independence</h5>
Conformal Prediction doesn't make strong assumptions about the data distribution, making it robust in the face of non-ideal or 'dirty' data, often encountered in real-world applications. This characteristic facilitates its application in diverse industries without the need for extensive data preprocessing or assumption validations.
</div>
""", unsafe_allow_html = True)


    st.subheader("References:")
    st.write(r"[A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://people.eecs.berkeley.edu/~angelopoulos/publications/downloads/gentle_intro_conformal_dfuq.pdf)")
    st.write(r"[Tutorial on Conformal Predictions](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Bayesian_Neural_Networks/dl2_bnn_tut2_student_with_answers.html#Conformal-prediction)")
    st.write(r"[A Tutorial on Conformal Predictions by Glenn Shafer and Vladimir Vovk](https://www.jmlr.org/papers/volume9/shafer08a/shafer08a.pdf)")
    
if __name__ == "__main__":
    main()

