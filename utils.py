import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to generate synthetic training and calibration data
def get_simple_data_train(coef_1, coef_2, coef_3, coef_4):
    # Generate data points for the custom function with some noise
    x = np.linspace(-.2, 0.2, 500)
    x = np.hstack([x, np.linspace(.6, 1, 500)])
    eps = coef_4 * np.random.randn(x.shape[0])
    y = coef_1 * np.sin(2 * np.pi*(x + eps)) + coef_2 * np.cos(4 * np.pi *(x + eps)) + coef_3 * x+ eps
    x = torch.from_numpy(x).float()[:, None]
    y = torch.from_numpy(y).float()

    # Split data into calibration and training sets
    cal_idx = np.arange(len(x), step=1/0.2, dtype=np.int64)
    mask = np.zeros(len(x), dtype=bool)
    mask[cal_idx] = True
    x_cal, y_cal = x[mask], y[mask]
    x_train, y_train = x[~mask], y[~mask]
    return x_train, y_train, x_cal, y_cal

def display_equation(coef_1, coef_2, coef_3, coef_4):
    equation = r"f(x, \varepsilon) = {:.2f} \sin(2\pi(x + \varepsilon)) + {:.2f} \cos(4\pi(x + \varepsilon)) + {:.2f}x + \varepsilon".format(coef_1, coef_2, coef_3)
    st.subheader("Function")
    st.latex(equation)


# Function to train a neural network model
def train(net, train_data, epochs=1000):
    x_train, y_train = train_data
    optimizer = torch.optim.Adam(params=net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = net(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    return net


# Function to load the CIFAR-10 test and calibration data
def get_data():
    # Define the transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
    ])

    # Download and load the CIFAR-10 test set
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)

    # Download and load the CIFAR-10 calibration set (validation set)
    calib_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    calib_loader = torch.utils.data.DataLoader(calib_set, batch_size=len(calib_set), shuffle=False)

    # Get the data and labels from the loaders
    X_test, y_test = next(iter(test_loader))
    X_calib, y_calib = next(iter(calib_loader))

    return X_test, y_test, X_calib, y_calib

# Function to get the class label based on index
def class_label(i):
    labels = {0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer", 5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
    return labels[i]

# Function to calculate the test accuracy of a neural network model
def get_test_accuracy(X_test, y_test, net):
    # Create a DataLoader for the test dataset
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test.squeeze().long())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)  # No need to shuffle for testing
    
    def calculate_accuracy(outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    # Evaluate the model on the test dataset
    net.eval()  # Set the model to evaluation mode (important for dropout and batch normalization layers)
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.permute(0, 3, 1, 2)
            outputs = net(inputs)
            accuracy = calculate_accuracy(outputs, labels)
            total_accuracy += accuracy * labels.size(0)
            total_samples += labels.size(0)

    # Calculate the overall accuracy on the test dataset
    test_accuracy = total_accuracy / total_samples
    return test_accuracy

# Function to compute the quantile for conformal prediction
def quantile(scores, alpha):
    # Compute conformal quantiles
    n = len(scores)
    q_val = np.ceil((1 - alpha) * (n + 1)) / n
    q = np.quantile(scores, q_val, method="higher")
    return q

# Function to compute the mean prediction set size
def mean_set_size(sets):
    # Compute the mean prediction set size
    return np.mean(np.sum(sets, axis=1), axis=0)

# Function to compute the scores for conformal prediction
def get_scores(net, calib_data):
    X_calib, y_calib = calib_data
    y_calib = y_calib.reshape(-1)
    
    cal_smx = torch.functional.F.softmax(net(X_calib.permute(0, 3, 1, 2)), dim=1).detach().numpy()
    scores = 1 - cal_smx[np.arange(len(X_calib)), y_calib.numpy()]
    return scores

# Function to compute the prediction sets for conformal prediction
def get_pred_sets(net, test_data, q, alpha):
    X_test, y_test = test_data
    test_smx = nn.functional.softmax(net(X_test.permute(0, 3, 1, 2)), dim=1).detach().numpy()

    pred_sets = test_smx >= (1 - q)
    return pred_sets

# Function to get the predicted class labels as a string
def get_pred_str(pred):
    pred_str = "{"
    for i in pred:
        pred_str += class_label(i) + ', '  # Use comma instead of space
    pred_str = pred_str.rstrip(', ') + "}"  # Remove the trailing comma and add closing curly brace
    return pred_str


# Function to display test predictions and class scores
def get_test_preds_and_smx(X_test, index, pred_sets, net, q, alpha):
    test_smx = nn.functional.softmax(net(X_test.permute(0, 3, 1, 2)), dim=1).detach().numpy()
    sample_smx = test_smx[index]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    axs[0].imshow(X_test[index].numpy())
    axs[0].set_title("Sample test image")
    
    axs[1].bar(range(10), sample_smx, label="class scores")
    axs[1].set_xticks(range(10))
    axs[1].set_xticklabels([class_label(i) for i in range(10)], rotation=90)
    axs[1].axhline(y=1 - q, label='threshold', color="red", linestyle='dashed')
    axs[1].legend(loc=1)
    axs[1].set_title("Class Scores")
    
    pred_set = pred_sets[index].nonzero()[0].tolist()
    
    return fig, axs, pred_set, get_pred_str(pred_set)
