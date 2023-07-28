import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    # Generate true function curve
    x_true = np.linspace(-.5, 1.5, 1000)
    y_true = coef_1 * np.sin(2 * np.pi*(x_true)) + coef_2 * np.cos(4 * np.pi *(x_true )) + coef_3 * x_true

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

def plot_multiple_predictions(x_train, y_train, x_cal, y_cal, x_test, y_preds, coef_1=0.3, coef_2=0.02, coef_3=0.1, coef_4=0.02):
    def add_multiple_predictions(ax):
        for idx in range(len(y_preds)):
            # Plot each prediction curve as a line with a unique label
            ax.plot(x_test, y_preds[idx], '-', linewidth=3, label=f"Prediction {idx+1}")

    return plot_generic(x_train, y_train, x_cal, y_cal, add_multiple_predictions, coef_1, coef_2, coef_3, coef_4)

def plot_predictions(x_train, y_train, x_cal, y_cal, x_test, y_preds, coef_1=0.3, coef_2=0.02, coef_3=0.1, coef_4=0.02):
    def add_predictions(ax):
        # Plot the neural network prediction curve as a line
        ax.plot(x_test, y_preds, 'y-', linewidth=3, label='neural net prediction')
    fig, ax = plot_generic(x_train, y_train, x_cal, y_cal, add_predictions, coef_1, coef_2, coef_3, coef_4)
    return fig, ax

def plot_uncertainty_bands(x_train, y_train, x_cal, y_cal, x_test, y_preds, coef_1=0.3, coef_2=0.02, coef_3=0.1, coef_4=0.02):
    y_preds = np.array(y_preds)
    y_mean = y_preds.mean(axis=0)
    y_std = y_preds.std(axis=0)

    def add_uncertainty(ax):
        # Plot the predictive mean as a line
        ax.plot(x_test, y_mean, '-', linewidth=3, color="#408765", label="predictive mean")
        
        # Fill the area between the upper and lower uncertainty bounds
        ax.fill_between(x_test.ravel(), y_mean - 2 * y_std, y_mean + 2 * y_std, alpha=0.6, color='#86cfac', zorder=5)

    return plot_generic(x_train, y_train, x_cal, y_cal, add_uncertainty, coef_1, coef_2, coef_3, coef_4)

# Function to plot scores and the 1 - alpha quantile
def plot_scores_quantile(scores, quantile, alpha):
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    
    # Plot histogram of scores
    ax.hist(scores, bins=50, label="Scores")
    
    # Plot cumulative histogram (cumulative distribution function)
    ax.hist(scores, cumulative=True, alpha=0.2, label="Cumulative")
    
    # Plot the 1 - alpha quantile as a vertical red line
    q_label = str(("{:.2f}")).format(1-alpha) + " Quantile"
    ax.axvline(x=quantile, label=q_label, color='red')
    
    # Add legend to the plot
    plt.legend(loc=2)
    
    return fig, ax
