# Import required libraries
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import streamlit_image_select as st_image
from sklearn.preprocessing import StandardScaler
# from tueplots import axes, bundles
from sklearn.model_selection import train_test_split
# from tqdm.auto import trange, tqdm
import utils
# Import utility func        tions and model classes from custom modules
from utils import get_simple_data_train, display_equation, train, get_data, get_test_preds_and_smx, get_scores, quantile, get_pred_sets, mean_set_size, get_test_accuracy, train_model, conformal_prediction_regression, tensor_to_img
from utils_plot import plot_generic, plot_predictions, histogram_plot, show_samples, plot_conformal_prediction
from model import MLP, MLP1
from PIL import Image
import json

plt.rcParams['image.cmap'] = 'gray'
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
# plt.rcParams.update(bundles.icml2022())
# plt.rcParams['text.usetex'] = True
with open('data/nested_dict.json', 'r') as f:
        loaded_dict = json.load(f)
with open('data/class_dict2.json', 'r') as f:
        class_dict = json.load(f)

# Define the main function to create the Streamlit app
def main():

    references = utils.get_references()
    text_content = utils.get_text_content()
   
    st.markdown(text_content["header_text"], unsafe_allow_html=True)
    
    st.title("Introduction:")

    st.markdown(text_content['introduction_text1'].format(references['fastai'], references['resnet_demo']), unsafe_allow_html=True)

    image_paths = [
    "Images/tennis_ball.png",
    "Images/green_oranges.png",
    "Images/green_frog.png"
    # Add more image paths here
]

# Read images into numpy arrays after converting to RGB and put them in a list
    all_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]

# Streamlit image selection widget
    with st.container():
        st.write("", "", "")  # Adding some spacing at the top if needed
        col1, col2, col3 = st.columns([0.1, .7, .1])
        with col2:
            test_img_idx = st_image.image_select(label="Select an image", images=all_images, return_value="index", use_container_width=False)

    # dict = {0: "Probability it's a apple: 0.9947, Probability it's a orange: 0.0053", 1: "Probability it's a apple: 0.9753, Probability it's a orange: 0.0247", 2: "Probability it's a apple: 0.9936, Probability it's a orange: 0.0064"}
    # st.write(dict[test_img_idx])
    

    # st.markdown(f"<div style='font-family: \"Helvetica, Arial, sans-serif\"; font-size:21px;'><b><span style='color:green;'>Probability it's an green apple: {apple_prob}</span><br><span style='color:orange;'>Probability it's an orange: {orange_prob}</span></b></div>", unsafe_allow_html=True)
    # st.markdown("<br>", unsafe_allow_html=True)

    if test_img_idx == 0:
        utils.get_svg("Images/example1.svg")
        st.markdown("<div style=\"text-align: justify;\">The model assigns a high probability to categorize this tennis ball as a green apple due to its resemblance in both shape and color. However, this is an incorrect classification, and it is essential to incorporate a level of uncertainty into this prediction.</div><br>", unsafe_allow_html=True)
        
    elif test_img_idx == 1:
        utils.get_svg("Images/example2.svg")
        st.markdown("<div style=\"text-align: justify;\">This image depicts an orange, but the model erroneously labels it as a green apple with high probability solely because of its green hue.</div><br>", unsafe_allow_html=True)
    else:
        utils.get_svg("Images/example3.svg")
        st.markdown("<div style=\"text-align: justify;\">The classification of this image featuring a frog as a green apple is once more the result of the predominant green color. In real life scenarios, a false classification like this may have significant implications.</div><br>", unsafe_allow_html=True)


    st.markdown(text_content['introduction_text2'], unsafe_allow_html=True)

 ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    st.title("Conformal Prediction for Regression")
    
    st.markdown(text_content['regression_text1'].format(references['olympic_data']), unsafe_allow_html=True)
    
    st.image('Images/newspaper_cutting.png')
    st.markdown(
    '<p style="color:grey; font-size:14px; text-align:center;">Figure 1: Newspaper Clipping of Alan Turing\'s Marathon time, '
    '<a href="https://www.turing.org.uk/book/update/part6.html" style="color:grey; font-size:14px;">Source: Alan Turing Internet Scrapbook</a></p>',
    unsafe_allow_html=True  # Make sure to enable this for rendering HTML
)
    coef_1 = 0.3
    coef_2 = 0.3
    coef_3 = 0.1
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
This dataset<sup><a href='#references'>{references["olympic_data"]}</a></sup> captures the pace of Olympic Gold Medal Marathon winners 
from 1896 to the present. The 1904 outlier is due to organizational issues.
The objective is to model and understand the trend over the years.
""", unsafe_allow_html = True)
    n_cal = 14
    st.image(f'./Images/Generated_Images/Regression_Plot_{n_cal}.png')
    st.markdown(
    '<p style="color:grey; font-size:14px; text-align:center;">Figure 2: Olympic\'s Dataset, divided into training and calibration sets',
    unsafe_allow_html=True  # Make sure to enable this for rendering HTML
)
    
    coef_4 = 2
    
    st.markdown("<h4 style=' color: black;'>Model</h4>", unsafe_allow_html=True)
    st.markdown("<div style=\"text-align: justify;\">The model which is a Multi Layer Perceptron (MLP) will be trained on the training data and used to generate predictions on the calibration data.</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(r"The calibration data is used to estimate the quantiles ($q_{val}$) for the prediction intervals. You can choose the number of calibration data points $(n)$ using the slider below.")
    # Display the equation based on user-selected coefficients

    

    # x_train, y_train, x_cal, y_cal =  get_simple_data_train(n_cal)
    
    st.image(f'./Images/Generated_Images/Regression_Prediction_Plot_{n_cal}.png')
    st.markdown(
    '<p style="color:grey; font-size:14px; text-align:center;">Figure 3: Model (MLP) Predictions on the Calibration Data',
    unsafe_allow_html=True  # Make sure to enable this for rendering HTML
)
    st.markdown("<h4 style=' color: black;'>Score Function</h4>", unsafe_allow_html=True)
    st.latex(r"s_i = |y_i - \hat{y}_i|")
    
    
    st.markdown("<div style=\"text-align: right;\">", unsafe_allow_html=True)
    st.write("The score function $s_i$ represents the absolute difference between the true \
             output $y_i$ and the model's predicted output $\hat{y}_i$ for each calibration data point $x_i$. \
             It measures the discrepancy between the true values and their corresponding predictions, providing a measure \
             of model fit to the calibration data.")
    # st.image(f'./Images/Generated_Images/Regression_Score_plot_{n_cal}_for_{0.1}.png')
    image = Image.open(f'./Images/Generated_Images/Regression_Score_plot_{n_cal}_for_{0.1}.png')
    st.image(image)
    st.markdown(
    '<p style="color:grey; font-size:14px; text-align:center;">Figure 4: Score Function for the Calibration Data</div>',
    unsafe_allow_html=True  # Make sure to enable this for rendering HTML
)
    st.markdown("<h4 style=' color: black;'>Calibration</h4>", unsafe_allow_html=True)
    image = Image.open(f'./Images/Generated_Images/Regression_Histogram_plot_{n_cal}_for_{0.1}.png')
    # new_image = image.resize((800, 600))
    
    # st.image(new_image)
    st.write("We initiate the calibration by sorting the scores in the ascending order")
    with st.container():
        st.write("", "", "")  # Adding some spacing at the top if needed
        col1, col2, col3 = st.columns([0.01, 0.9, .01])
        with col2:
            st.image(image)
            st.markdown(
    '<p style="color:grey; font-size:14px; text-align:center;">Figure 5: Sorting of the Scores',
    unsafe_allow_html=True  # Make sure to enable this for rendering HTML
)
    
    st.markdown(r"Use the below slider to choose the $\alpha$. With probability 1-$\alpha$, our computed uncertainty band $\hat{C}(X_{n+1})$ will contain the true value $Y_{n+1}$.")
    
    
    alpha = st.slider(r"Select a value for $\alpha$:", min_value=0.1, max_value=1.0, step=0.1, value=0.4)

    # q, resid = conformal_prediction_regression(x_cal, y_cal_preds,alpha, y_cal)
    
    q = loaded_dict[f"{n_cal}"][f"{alpha}"]["q"]

    # st.image(f'./Images/Generated_Images/Regression_Histogram_plot_{n_cal}_for_{alpha}.png')
    
            
    # histogram_plot(resid, q, alpha)
    st.write(r"Now, we compute $q_{val}$ by calculating the $\left\lceil \frac{(n+1)(1-\alpha)}{n} \right\rceil$th quantile of the conformity scores.")
    # st.latex(r"q_{{\text{{value}}}} = {:.4f}".format(q))
    
    st.markdown(f'<span style=" top: 2px;font-size:50px;"><center> $q_{{\\text{{val}}}} = {q:.4f}$</center></span>', unsafe_allow_html=True)
    image = Image.open(f'./Images/Generated_Images/Regression_Histogram_plot_quantile_{n_cal}_for_{alpha}.png')
    new_image = image
    
    # st.image(new_image)
    
    st.image(new_image)
    st.markdown(
    '<p style="color:grey; font-size:14px; text-align:center;">Figure 6: Quantile of the Scores',
    unsafe_allow_html=True  # Make sure to enable this for rendering HTML
)
    st.write("We now compute the confidence intervals for the predictions.")
    st.image(f'./Images/Generated_Images/Regression_Coverage_plot_{n_cal}_for_{alpha}.png')
    st.markdown(
    '<p style="color:grey; font-size:14px; text-align:center;">Figure 7: Confidence interval of the Predictions',
    unsafe_allow_html=True  # Make sure to enable this for rendering HTML
)
    y_preds_46 = loaded_dict[f"{n_cal}"][f"{alpha}"]["y_preds_46"]
    # plot_conformal_prediction(x_train, y_train, x_cal, y_cal, y_cal_preds, q, alpha, scaler, net1)
    if(y_preds_46-q<=3.94 and  y_preds_46+q>=3.94):
        true_1 = "won"
    else:
        true_1 = "lost"
    

    st.markdown("""The model predicts that the time for the Olympic gold medalist in 1946 would have been {:.2f} minutes. 
With a significance level of &alpha; = {:.2f}, the uncertainty band calculated using conformal prediction ranges from {:.2f} to {:.2f} minutes. 
Therefore, based on this model, Alan Turing would have <span style='font-size:19px;'><strong>{}</strong></span> the gold medal.""".format(y_preds_46, alpha, y_preds_46 - q, y_preds_46 + q, true_1), unsafe_allow_html=True)

    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################
    ########################################################################################################################################################


    st.title("Conformal Predictions in Classification")   
    X_train, y_train, X_test, y_test, X_calib, y_calib = get_data()
  
    st.markdown(text_content['classification_text1'], unsafe_allow_html=True)
    
    alpha = st.slider(r"Select a value for $\alpha$:", min_value=0.05, max_value=0.5, step=0.05, value=0.05)
    
    q_val = class_dict[f"{alpha}"]["q"]

    st.latex(r"q_{{\text{{val}}}} = {:.4f}".format(q_val))

    # q = np.quantile(scores, q_val, method="higher")
    # histogram_plot(scores, q, alpha)
    st.image(f'./Images/Generated_Images/Classification_Histogram_{alpha}.png')
    st.markdown(
    '<p style="color:grey; font-size:14px; text-align:center;">Figure 8: Quantile of the Scores',
    unsafe_allow_html=True  # Make sure to enable this for rendering HTML
)
    # st.pyplot(fig)
    st.markdown(r"For this value of alpha, the threshold value $1-q_{\text{val}}$ is " + f'<span style="font-size:20px;">{1-q_val:.4f}</span>', unsafe_allow_html=True)
    
    st.markdown(text_content['classification_text2'], unsafe_allow_html=True)
    
    # pred_sets = get_pred_sets(net, (X_test, y_test), q, alpha)
    
    fashion_mnist_data = utils.fashion_mnist()
    fashion_idx = [5]
    idxs = [376 , 673]

    # Get images from fashion mnist data
    fashion_images = [tensor_to_img(fashion_mnist_data, idx) for idx in fashion_idx]

    # Get images from X_test
    xtest_images = [tensor_to_img(X_test, idx) for idx in idxs]

    # Concatenate the two lists of images
    all_images = fashion_images + xtest_images

    # Let user select an image using Streamlit widget
    with st.container():
        st.write("", "", "")  # Adding some spacing at the top if needed
        col1, col2, col3 = st.columns([0.1, .7, .1])
        with col2:
            test_img_idx = st_image.image_select(label="Select an image", images=all_images, return_value="index", use_container_width=False)
            
    if(test_img_idx==1 or test_img_idx==2):
         test_img_idx+=1
         
   
    pred_str = class_dict[f"{alpha}"][f"{test_img_idx}"]
    mean = class_dict[f"{alpha}"]["mean_size"]
    
    st.image(f'./Images/Generated_Images/Classification_Final_Image_image_no_{test_img_idx}_{alpha}.png')
    st.markdown("<h2 style='font-size:21px;'>Prediction Set for this image: {}</h2>".format(pred_str), unsafe_allow_html=True)
    
    # st.markdown("<div style=\"text-align: justify;\">In the above examples, the first image is sourced from the Fashion-MNIST dataset. The model is \
    #          uncertain about this image. In contrast, for the last 2 images, the predicted \
    #          set contains only one element because the model is confident about its prediction. This is reflected by \
    #          the high softmax scores of the true classes.</div>", unsafe_allow_html=True)
    
    st.markdown(f"The average size of prediction sets for all the images from the test set is <span style='font-size:20px;'>{mean}</span>", unsafe_allow_html=True)


    st.write(f" **What does the average size mean?**")
    st.markdown("<div style=\"text-align: justify;\">We observe that the average size of the prediction set decreases when \
             value of alpha is increased. This is because of our method for computing conformity scores, where we only \
             take into account the softmax scores of the correct class when calculating 𝑞̂. With increasing alpha, the \
             softmax scores for the classes decreases and thus there are lesser scores above the threshold value.</div>", unsafe_allow_html=True)
    
    st.markdown(text_content['conclusion_text'], unsafe_allow_html=True)
    
    st.markdown(text_content['references_text'], unsafe_allow_html=True)
    
if __name__ == "__main__":
    main()

