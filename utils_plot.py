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

def plot_histogram_with_quantile(scores, q, alpha):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot the histogram
    n, bins, _ = plt.hist(scores, bins=50, alpha=0.7, color='b', edgecolor='black', label='Score Frequency')
    
    # Plot the vertical line at the quantile
    q_x = np.quantile(scores, q)
    plt.axvline(q_x, color='r', linestyle='dashed', linewidth=2, label=f'Quantile (q = {q:.4f})')
    
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Histogram of Scores with Quantile Line')
    plt.legend()
    plt.grid(True)
    
    st.pyplot(fig)