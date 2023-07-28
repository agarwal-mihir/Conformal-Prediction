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
from utils import get_simple_data_train, display_equation, train, get_data, get_test_preds_and_smx, get_scores, quantile, get_pred_sets, mean_set_size, get_test_accuracy
from utils_plot import plot_generic, plot_predictions, plot_scores_quantile
from model import MLP, CNN

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
def get_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_train, y_train = train_dataset.data.float() / 255.0, train_dataset.targets
    X_test, y_test = test_dataset.data.float() / 255.0, test_dataset.targets

    X_train = X_train.view(-1, 28*28)
    X_test = X_test.view(-1, 28*28)

    X_calib, X_train = X_train[50000:], X_train[:50000]
    y_calib, y_train = y_train[50000:], y_train[:50000]

    return X_train, y_train, X_test, y_test, X_calib, y_calib, X_test.numpy()




class MLP1(nn.Module):
    def __init__(self):
        super(MLP1, self).__init__()
        self.fc1 = nn.Linear(784, 32)
        # self.relu1 = nn.ReLU()
        self.sigmoid1 = nn.Sigmoid()
        self.fc2 = nn.Linear(32, 10)
        
    def forward(self, x):
        x = self.sigmoid1(self.fc1(x))
        x = self.fc2(x)
        return x

def accuracy(outputs, targets):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    return correct / total

def train_model(net, train_data):
    X_train, y_train = train_data
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    num_epochs = 1
    
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_accuracy += accuracy(outputs, targets)

            if batch_idx % 100 == 99:
                print(f"Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {running_loss / 100:.4f} Train Accuracy: {running_accuracy / 100:.4f}")
                running_loss = 0.0
                running_accuracy = 0.0    
    return net

def get_test_accuracy(net, test_data):
    X_test, y_test = test_data
    test_outputs = net(X_test)
    return accuracy(test_outputs, y_test)

def quantile(scores, alpha):
  # compute conformal quantiles

  n = len(scores)
  q_val = np.ceil((1 - alpha) * (n + 1)) / n
  q = np.quantile(scores, q_val, method="higher")
  return q

def mean_set_size(sets):
  # mean prediction set size
  return np.mean(np.sum(sets, axis=1), axis=0)

def emp_coverage(sets, target):
  # empirical coverage
  return sets[np.arange(len(sets)), target].mean()

def get_scores(net, calib_data):
    X_calib, y_calib = calib_data
    cal_smx = torch.functional.F.softmax(net(X_calib), dim=1).detach().numpy()
    scores = 1 - cal_smx[np.arange(len(X_calib)), y_calib.numpy()]
    return scores

def plot_scores_quantile(scores, quantile, alpha):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    ax.hist(scores, bins = 50, label = "Scores")
    # ax.hist(scores, cumulative = True, histtype = "step", label = "Cumulative")
    ax.hist(scores, cumulative = True, alpha = 0.2, label = "Cumulative")
    
    q_label = str(("{:.2f}")).format(1-alpha) + " Quantile"
    ax.axvline(x = quantile, label = q_label)
    
    plt.legend(loc = 2)
    return fig, ax

def get_pred_sets(net, test_data, q, alpha):
    X_test, y_test = test_data
    test_smx = nn.functional.softmax(net(X_test), dim=1).detach().numpy()

    pred_sets = test_smx >= (1-q)
    return pred_sets

def get_pred_str(pred):
    pred_str = ""
    
    for i in pred:
        pred_str += str(i) + ' '
    return pred_str

def get_test_preds_and_smx(X_test, index, pred_sets, net, q, alpha, X_test_unscaled):
    test_smx = nn.functional.softmax(net(X_test), dim=1).detach().numpy()
    sample_smx = test_smx[index]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    axs[0].imshow(X_test_unscaled[index].reshape(28, 28), cmap='gray', vmin=0, vmax=255)
    axs[0].set_title("Sample test image")
    
    axs[1].bar(range(10), sample_smx, label = "class scores")
    axs[1].axhline(y = q, label = 'threshold', color = "red", linestyle='dashed')
    axs[1].legend(loc = 1)
    axs[1].set_title("Class Scores")
    
    pred_set = pred_sets[index].nonzero()[0].tolist()
    
    return fig, axs, pred_set, get_pred_str(pred_set)
# Define the main function to create the Streamlit app
def main():
    # Set the title and subheader of the app
    st.title("Conformal Prediction: A Brief Overview")
    st.subheader("Introduction:")
    # Provide an overview of Conformal Prediction
    st.write("Machine learning models, especially black-box models like neural networks, have gained widespread adoption in high-risk domains like medical diagnostics, where accurate predictions are critical to avoid potential model failures. However, the lack of uncertainty quantification in these models poses significant challenges in decision-making and trust. Conformal prediction emerges as a promising solution, providing a user-friendly paradigm to quantify uncertainty in model predictions.")

    st.write("The key advantage of conformal prediction lies in its distribution-free nature, making it robust to various scenarios without making strong assumptions about the underlying data distribution or the model itself. By utilizing conformal prediction with any pre-trained model, researchers and practitioners can create prediction sets that offer explicit, non-asymptotic guarantees, instilling confidence in the reliability of model predictions.")


    st.write("Conformal Prediction is a powerful framework in machine learning that provides a measure of uncertainty in predictions. Unlike traditional point predictions, conformal prediction constructs prediction intervals that quantify the range of potential outcomes.")

    st.write("The significance of conformal prediction lies in its ability to provide a confidence level (alpha) for the predictions, allowing users to understand the reliability of the model's output. This is especially crucial in critical applications where understanding the uncertainty is essential.")
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

    # Sliders with custom styles
    coef_1 = st.slider("Coefficient for sine term", min_value=0.0, max_value=1.0, value=0.3, step=0.01, format="%.2f")
    coef_2 = st.slider("Coefficient for cosine term", min_value=0.0, max_value=1.0, value=0.3, step=0.01, format="%.2f")
    coef_3 = st.slider("Coefficient for x^3 term", min_value=0.0, max_value=1.0, value=0.1, step=0.01, format="%.2f")
    coef_4 = st.slider("Coefficient for noise", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.2f")
    # Display the equation based on user-selected coefficients
    display_equation(coef_1, coef_2, coef_3, coef_4)
    x_train, y_train, x_cal, y_cal = get_simple_data_train(coef_1, coef_2, coef_3, coef_4)
    fig, ax = plot_generic(x_train, y_train, x_cal, y_cal, coef_1=coef_1, coef_2=coef_2, coef_3=coef_3, coef_4=coef_4)
    st.pyplot(fig)

    # Train the model (MLP) on the generated data
    hidden_dim = st.slider("Hidden Dimension", min_value=1, max_value=100, value=30, step=1)
    n_hidden_layers = st.slider("Number of Hidden Layers", min_value=1, max_value=10, value=2, step=1)
    epochs = st.slider("Number of Epochs", min_value=1, max_value=10000, value=1000, step=1)
    x_test = torch.linspace(-.5, 1.5, 3000)[:, None]
    net = MLP(hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    net = train(net, (x_train, y_train), epochs=epochs)
    y_preds = net(x_test).clone().detach().numpy()

    # Display the plot with the true function, training data, calibration data, predictive mean
    st.subheader("Prediction Visualization")
    fig, ax = plot_predictions(x_train, y_train, x_cal, y_cal, x_test, y_preds, coef_1=coef_1, coef_2=coef_2, coef_3=coef_3, coef_4=coef_4)
    st.pyplot(fig)

    # Calculate residuals and estimate quantile based on user-selected alpha
    x_test = torch.linspace(-.5, 1.5, 1000)[:, None]
    y_preds = net(x_test).clone().detach().numpy()
    y_cal_preds = net(x_cal).clone().detach()
    # st.latex(score_func)
    st.write("The score function $s_i$ represents the absolute difference between the true output $y_i$ and the model's predicted output $\hat{y}_i$ for each calibration data point $x_i$. It measures the discrepancy between the true values and their corresponding predictions, providing a measure of model fit to the calibration data.")
    resid = torch.abs(y_cal - y_cal_preds).numpy()
    alpha = st.slider("Select a value for alpha:", min_value=0.01, max_value=1.0, step=0.001, value=0.03)
    n = len(x_cal)
    q_val = np.ceil((1 - alpha) * (n + 1)) / n
    q = np.quantile(resid, q_val, method="higher")
    x_true = np.linspace(-.5, 1.5, 1000)
    y_true = coef_1 * np.sin(2 * np.pi*(x_true)) + coef_2 * np.cos(4 * np.pi *(x_true )) + coef_3 * x_true

    # Generate plot with the true function, training data, calibration data, predictive mean, and uncertainty bands
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xlim([-.5, 1.5])
    plt.ylim([-1.5, 2.5])
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)
    
    ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")
    ax.plot(x_train, y_train, 'go', markersize=4, label="training data")
    ax.plot(x_cal, y_cal, 'ro', markersize=4, label="calibration data")
    ax.plot(x_test, y_preds, '-', linewidth=3, color="y", label="predictive mean")
    ax.fill_between(x_test.ravel(), y_preds - q, y_preds + q, alpha=0.6, color='y', zorder=5)
    plt.legend(loc='best', fontsize=15, frameon=False)
    st.pyplot(fig)
    # Conformal Predictions in Classification Section
    st.title("Conformal Predictions in Classification")
    st.write("In Classification, our model outputs are now class probabilities and prediction sets are discrete.")
    # Further explanations and information about the Cifar Dataset and the pre-trained CNN model
    # X_test, y_test, X_calib, y_calib = get_data()
    # net = MLP(input_dim = 1024,output_dim=1,hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers)
    # net.load_state_dict(torch.load("cifar/cifar_model", map_location=torch.device("cpu")))
    # st.write("**Test accuracy** of current model:", get_test_accuracy(X_test, y_test, net))
    # st.write("The choice of how to calculate conformity scores is a modelling decision. We will use a simple softmax based method:")
    # score_func = r"s_i=1-\hat{\pi}_{x_i}(y_i)"
    # st.latex(score_func)
    # st.write("which is 1 minus the softmax output of the true class. The prediction set is then constructed as :")
    # pred_set_latex = r"\hat{C}(x_{n+1})=\{y'\in K:\hat{\pi}_{x_{n+1}}(y') \ge 1-\hat{q}\}"
    # st.latex(pred_set_latex)
    # st.write("which collects all the classes for which the softmax score is above the threshold **1-q**.")
    # st.write("**q** is given by: ")
    # st.latex(r"\left\lceil \frac{(n+1)(1-\alpha)}{n} \right\rceil")
    # # Explanation of the score function and prediction set construction for classification
    # # Calculation of conformity scores for calibration data and visualization
    # scores = get_scores(net, (X_calib, y_calib))
    # alpha = st.slider("Select a value for alpha:", min_value=0.001, max_value=1.0, step=0.001, value=0.04)
    # q = quantile(scores, alpha)
    # fig, ax = plot_scores_quantile(scores, q, alpha)
    # st.pyplot(fig)
    
    # Display conformal quantile(1-q) of the calibration data
    # st.write("Conformal quantile(1-q) of the calibration data is: {:.3f}".format(1-q))
    
    # # Example of the prediction set for a selected test image
    # st.write("Example:")
    # test_img_index = st.slider("Choose Image:", min_value=0, max_value=1000, step=1, value=628)
    # sample_test_img = X_test[test_img_index]
    # pred_sets = get_pred_sets(net, (X_test, y_test), q, alpha)
    # fig, ax, pred, pred_str = get_test_preds_and_smx(X_test, test_img_index, pred_sets, net, q, alpha)
    # st.pyplot(fig)
    # st.write("Prediction Set for this image: ", pred_str)
    # st.write("The average size of prediction sets for the test images is {:.3f}".format(mean_set_size(pred_sets)))
    st.title("Conformal Predictions in Classification")
    
    st.write("In Classification, our model outputs are now class probabilities and prediction sets are discrete. This is in contrast to the continuous uncertainty bands we obtained before, and influences how we design the comparison of true and predicted classes to obtain our conformity scores. When the predictive uncertainty is high, size of the prediction set will be higher.")
    st.write(" We split the training samples into two parts: training set and calibration set. We take the first 50k images in training set and 10k images in the calibration set.")
    
    X_train, y_train, X_test, y_test, X_calib, y_calib, X_test_unscaled = get_data()
    
    net = MLP1()
    
    train_data = (X_train, y_train)
    calib_data = (X_calib, y_calib)
    test_data = (X_test, y_test)
    
    net = train_model(net, train_data)
    
    st.write("The choice of how to calculate conformity scores is a modelling decision. We will use a simple softmax based method:")
    score_func = r"s_i=1-\hat{\pi}_{x_i}(y_i)"
    st.latex(score_func)
    st.write("which is 1 minus the softmax output of the true class. The prediction set is then constructed as :")
    pred_set_latex = r"\hat{C}(x_{n+1})=\{y'\in K:\hat{\pi}_{x_{n+1}}(y') \ge 1-\hat{q}\}"
    st.latex(pred_set_latex)
    st.write("which collects all the classes for which the softmax score is above the threshold 1-q.")
    st.write("q is given by: ")
    st.latex(r"\left\lceil \frac{(n+1)(1-\alpha)}{n} \right\rceil")

    scores = get_scores(net, calib_data)
    
    alpha = st.slider("Select a value for alpha:", min_value=0.001, max_value=1.0, step=0.001, value=0.03)
    q = quantile(scores, alpha)
    conformal_quantile = q
    fig, ax = plot_scores_quantile(scores, q, alpha)
    st.pyplot(fig)
    
    st.write("Conformal quantile(1-q) of the calibration data is: {:.3f}".format(conformal_quantile))
    st.write("This means that for some test sample (x, y), the prediction set is constructed by collecting all classes for which the softmax score is above the threshold {:.3f}.".format(conformal_quantile))

    pred_sets = get_pred_sets(net, test_data, q, alpha)
    st.write("The average size of prediction sets for the test images is {:.3f}".format(mean_set_size(pred_sets)))
    
if __name__ == "__main__":
    main()

