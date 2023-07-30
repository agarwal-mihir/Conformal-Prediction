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
def plot_generic(x_train, y_train, x_cal, y_cal, add_to_plot=None, coef_1=0.3, coef_2=0.02, coef_3=0.1, coef_4=0.02):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xlim([-.5, 1.5])
    plt.ylim([-1.5, 2.5])
    plt.xlabel("X", fontsize=30)
    plt.ylabel("Y", fontsize=30)
    plt.title("Plot of Training and Calibration Data with True Function", fontsize=20)

    # Generate true function curve
    x_true = np.linspace(-.5, 1.5, 1000)
    y_true = coef_1 * np.sin(2 * np.pi * x_true) + coef_2 * np.cos(4 * np.pi * x_true) + coef_3 * x_true
    print(x_train.shape, y_train.shape, x_cal.shape, y_cal.shape)

    # Plot training data as green scatter points
    ax.scatter(x_train, y_train, c='green', s=10, label="training data")

    # Plot calibration data as red scatter points
    ax.scatter(x_cal, y_cal, c='red', s=10, label="calibration data")

    # Plot the true function as a blue line
    ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")

    # If additional plot elements are provided, add them using the 'add_to_plot' function
    if add_to_plot is not None:
        add_to_plot(ax)

    # Add a legend to the plot
    plt.legend(loc='best', fontsize=15, frameon=False)

    return fig, ax


def plot_predictions(x_train, y_train, x_cal, y_cal, x_test, y_preds, coef_1=0.3, coef_2=0.02, coef_3=0.1, coef_4=0.02):
    def add_predictions(ax):
        # Plot the neural network prediction curve as a line
        ax.plot(x_test, y_preds, 'y-', linewidth=3, label='neural net prediction')
    fig, ax = plot_generic(x_train, y_train, x_cal, y_cal, add_predictions, coef_1, coef_2, coef_3, coef_4)
    plt.title("Plot of Training, Calibration, and Neural Network Predictions", fontsize=15)
    return fig, ax


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

@st.cache_data
def histogram_plot(scores, q, alpha):
    print("running histogram")
    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    # Plot scores of calibration data
    ax[0].bar(np.arange(len(scores)), height = scores, alpha = 0.7, color = 'b')
    ax[0].set_ylabel("Score")
    ax[0].set_xlabel("Calibration Data Points")
    ax[0].set_title("Scores of Calibration Data")
    
    # Plot the histogram
    n, bins, _ = ax[1].hist(scores, bins=30, alpha=0.7, cumulative = True, color='b', edgecolor='black', label='Score Frequency')
    
    # Plot the vertical line at the quantile
    # q_x = np.quantile(scores, q)
    ax[1].axvline(q, color='r', linestyle='dashed', linewidth=2, label=r"Quantile (${q_{val}}$ = " + str(("{:.2f}")).format(q) + ")")
    
    ax[1].set_xlabel('Scores')
    ax[1].set_ylabel('Frequency')
    ax[1].set_title('Histogram of Scores with Quantile Line')
    plt.legend()
    plt.grid(True)
    
    st.pyplot(fig)
