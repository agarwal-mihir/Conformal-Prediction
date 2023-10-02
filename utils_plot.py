import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to plot generic data visualization

def plot_generic(x_train, y_train, x_cal, y_cal, _add_to_plot=None, coef_1=0.3, coef_2=0.02, coef_3=0.1, coef_4=0.02):

    fig, ax = plt.subplots(figsize=(10, 5))
    # plt.xlim([1975, 2020])
    # plt.ylim([2, 5])
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Pace min/km", fontsize=20)
    plt.title("Plot of Training and Calibration Data with True Function", fontsize=20)

    # Generate true function curve
    # x_true = np.linspace(-.5, 1.5, 1000)
    # y_true = coef_1 * np.sin(2 * np.pi * x_true) + coef_2 * np.cos(4 * np.pi * x_true) + coef_3 * x_true



    # Plot training data as green scatter points
    ax.scatter(x_train, y_train, c='green', s=150, label="training data")

    # Plot calibration data as red scatter points
    ax.scatter(x_cal, y_cal, c='red', s=125, label="calibration data")

    # Plot the true function as a blue line
    # ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")

    # If additional plot elements are provided, add them using the '_add_to_plot' function
    if _add_to_plot is not None:
        _add_to_plot(ax)

    # Add a legend to the plot
    plt.legend(loc='best', fontsize=15, frameon=False)

    return fig, ax


def plot_predictions(x_train, y_train, x_cal, y_cal, y_cal_preds, coef_1=0.3, coef_2=0.02, coef_3=0.1, coef_4=0.02):

    def add_predictions(ax):
        # Plot the neural network prediction curve as a line
        # ax.plot(_x_test, _y_preds, 'y-', linewidth=3, label='neural net prediction')

        ax.plot(x_cal, y_cal_preds, 'y-', linewidth=3, label='neural net prediciton')

        #Plot Score Lines
        ax.vlines(x_cal[0], min(y_cal[0], y_cal_preds[0]) ,max(y_cal[0], y_cal_preds[0]), color='black', linestyle='dashed', linewidth=2, alpha = 0.7, label = "Scores")
        for i in range(1, len(x_cal)):
            ax.vlines(x_cal[i], min(y_cal[i], y_cal_preds[i]) ,max(y_cal[i], y_cal_preds[i]), color='black', linestyle='dashed', linewidth=2, alpha = 0.7)


    fig, ax = plot_generic(x_train, y_train, x_cal, y_cal, add_predictions, coef_1, coef_2, coef_3, coef_4)
    plt.title("Plot of Training, Calibration, and Neural Network Predictions", fontsize=15)
    return fig, ax

# @st.cache_data
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

def plot_calibration_scores(x_cal, scores):
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.8
    ax.bar(x_cal.squeeze(),scores,width=bar_width, color='blue', linewidth=1, edgecolor='black')
    ax.set_xlabel('Calibration Data Points', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.set_title('Calibration Scores', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def histogram_plot(scores, q, alpha):

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    # Plot scores of calibration data
    ax[0].bar(np.arange(len(scores)), height = scores, alpha = 0.7, color = 'b')
    ax[0].set_ylabel("Score")
    ax[0].set_xlabel("Calibration Data Points")
    ax[0].set_title("Scores of Calibration Data")
    
    # Plot the histogram
    n, bins, _ = ax[1].hist(scores, bins=30, alpha=0.7, cumulative = True, color='#E94B3CFF', edgecolor='black', label='Score Frequency')
    
    # Plot the vertical line at the quantile
    # q_x = np.quantile(scores, q)
    ax[1].axvline(q, color='b', linestyle='dashed', linewidth=2, label=r"Quantile (${q_{val}}$ = " + str(("{:.2f}")).format(q) + ")")
    
    ax[1].set_xlabel('Scores')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Histogram of Scores with Quantile Line')
    plt.legend()
    # plt.grid(True)
    
    st.pyplot(fig)

def show_samples(X_test, idxs, pred_sets, net, q, alpha):

    fig, axes = plt.subplots(1, 10, figsize = (12, 2))
    axes = axes.flatten()

    # Plot each image on its corresponding subplot
    for i, ax in enumerate(axes):
        if i < len(idxs):
            ax.imshow(X_test[idxs[i]].reshape(28,28).numpy())  # Assuming the images are grayscale, change cmap as needed
            title = f"Image {i+1}"
            ax.set_title(title)
            ax.axis("off")
                
    st.pyplot(fig)


def plot_conformal_prediction(x_train, y_train, x_cal, y_cal, y_cal_preds, q, alpha, scaler, net1):

    x_true = np.arange(1891, 2020, 4)
    x_true_scale = scaler.transform(x_true.reshape(-1, 1))
    x_true_scale = torch.from_numpy(x_true_scale).float()
    y_true = net1(x_true_scale).detach().numpy()

    # y_true = coef_1 * np.sin(2 * np.pi*(x_true)) + coef_2 * np.cos(4 * np.pi *(x_true )) + coef_3 * x_true
    fig, ax = plt.subplots(figsize=(10, 5))
    # plt.xlim([-.5, 1.5])
    # plt.ylim([-1.5, 2.5])
    plt.xlabel("Year", fontsize=20)
    plt.ylabel("Pace min/km", fontsize=20)

    ax.plot(x_true, y_true, 'b-', linewidth=3, label="predicted function")
    ax.scatter(x_train, y_train, c='g', s=150, label="training data")
    ax.scatter(x_cal, y_cal, c='r', s=125, label="calibration data")
    # ax.plot(x_test, y_preds, '-', linewidth=3, color="y", label="predictive mean")
    ax.fill_between(x_true.ravel(), y_true - q, y_true + q, alpha=0.6, color='y', zorder=5)
    ax.scatter(1946, 3.94, c='black', s=150, label="Alan Turing's Speed")
    plt.legend(loc='best', fontsize=15, frameon=False)
    st.write("The prediction interval is calculated as:")
    st.latex(r"\hat{C}(X_{n+1}) = [ \hat{f}(x_{n+1}) - {q_{val}}, \, \hat{f}(x_{n+1}) + {q_{val}} ]")
    
    # cov = np.mean(((y_cal_preds - q) <= y_cal) * ((y_cal_preds + q) >= y_cal))
    # s = r"Below is the plot of the predictions with uncertainty bands. We want the uncertainty band to\
    #     contain (1-$\alpha$) = " + f"{1-alpha:.2%}" + " of the ground truth. Empirically, the prediction set\
    #     contains " + f"{cov:.2%}" + " of the ground truth."
    # st.write(s)
    # s = r"""Below is the plot of the predictions with uncertainty bands. We want the uncertainty band to
    #     contain <br>(1-$\alpha$) = <span style='font-size:20px;'>""" + f"{1-alpha:.2%}" + """</span> of the ground truth. 
    #     Empirically, the prediction set contains <span style='font-size:20px;'>""" + f"{cov:.2%}" + """</span> of the ground truth."""

    # st.markdown("<div style=\"text-align: justify;\">", unsafe_allow_html=True)
    # st.markdown(s, unsafe_allow_html=True)
    # st.markdown("</div>", unsafe_allow_html=True)

    plt.title("Plot of confidence interval for the conformal prediction", fontsize=15)
    st.pyplot(fig)