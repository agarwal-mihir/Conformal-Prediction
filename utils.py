import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import PIL
import base64

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Function to generate synthetic training and calibration data
@st.cache_data
def get_simple_data_train(n_cal):
    x = np.array([1896,1900,1904,1908,1912,1920,1924,1928,1932,1936,1948,1952,1956,1960,1964,1968,1972,1976,1980,1984,1988,1992,1996,2000,2004,2008,2012])
    y = np.array([4.47083333333333,4.46472925981123,5.22208333333333,4.1546786744085,3.90331674958541,3.5695126705653,3.8245447722874,3.62483706600308,3.59284275388079,3.53880791562981,3.6701030927835,3.39029110874116,3.43642611683849,3.2058300746534,3.13275664573212,3.32819844373346,3.13583757949204,3.07895880238575,3.10581822490816,3.06552909112454,3.09357348817,3.16111703598373,3.14255243512264,3.08527866650867,3.1026582928467,2.99877552632618,3.03392977050993])
    cal_idx = np.random.choice(x.shape[0], n_cal, replace=False)
    mask = np.zeros(len(x), dtype=bool)
    mask[cal_idx] = True
    x_cal, y_cal = x[mask], y[mask]
    x_train, y_train = x[~mask], y[~mask]
    x_train, y_train, x_cal, y_cal = torch.tensor(x_train, dtype=torch.float).unsqueeze(1), torch.tensor(y_train, dtype=torch.float), torch.tensor(x_cal, dtype=torch.float).unsqueeze(1), torch.tensor(y_cal, dtype=torch.float)
    return x_train, y_train, x_cal, y_cal
# @st.cache_data
def display_equation(coef_1, coef_2, coef_3):

    equation = r"f(x, \varepsilon) = {:.2f} \sin(2\pi(x)) + {:.2f} \cos(4\pi(x)) + {:.2f}x + \varepsilon".format(coef_1, coef_2, coef_3)
    st.latex(equation)


# Function to train a neural network model

def train(_net, _train_data, epochs=1000):

    x_train, y_train = _train_data
    optimizer = torch.optim.Adam(params=_net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = _net(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()

    return _net


# Function to load the CIFAR-10 test and calibration data
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

    X_calib, X_train = X_train[59950:], X_train[:59950]
    y_calib, y_train = y_train[59950:], y_train[:59950]

    return X_train, y_train, X_test, y_test, X_calib, y_calib

# # Function to get the class label based on index
def class_label(i):
    labels = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 
                5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
    return labels[i]

# Function to calculate the test accuracy of a neural network model
# @st.cache_data
def get_test_accuracy(_X_test, _y_test, _net):
    # Create a DataLoader for the test dataset

    test_dataset = torch.utils.data.TensorDataset(_X_test, _y_test.squeeze().long())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)  # No need to shuffle for testing
    def calculate_accuracy(outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy

    # Evaluate the model on the test dataset
    _net.eval()  # Set the model to evaluation mode (important for dropout and batch normalization layers)
    total_accuracy = 0.0
    total_samples = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            # inputs = inputs
            outputs = _net(inputs)
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
    
    cal_smx = torch.functional.F.softmax(net(X_calib), dim=1).detach().numpy()
    scores = 1 - cal_smx[np.arange(len(X_calib)), y_calib.numpy()]
    return scores

# Function to compute the prediction sets for conformal prediction
def get_pred_sets(net, test_data, q, alpha):
    X_test, y_test = test_data
    test_smx = nn.functional.softmax(net(X_test), dim=1).detach().numpy()

    pred_sets = test_smx >= (1 - q)
    return pred_sets

# Function to get the predicted class labels as a string
def get_pred_str(pred):
    pred_str = "{"
    for i, j in enumerate(pred):

        if j:
            pred_str += class_label(i) + ', '  # Use comma instead of space
    pred_str = pred_str.rstrip(', ') + "}"  # Remove the trailing comma and add closing curly brace
    return pred_str




@st.cache_resource
def train_model(_net, _train_data):
    X_train, y_train = _train_data
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(_net.parameters(), lr=0.001)
    num_epochs = 1
    
    for epoch in range(num_epochs):
        _net.train()
        running_loss = 0.0
        running_accuracy = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = _net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # running_accuracy += accuracy(outputs, targets)

            if batch_idx % 100 == 99:

                running_loss = 0.0
                running_accuracy = 0.0    
    return _net

# @st.cache_data
def conformal_prediction_regression(_x_cal, _y_cal_preds, alpha,_y_cal):

    # x_test = torch.linspace(-.5, 1.5, 1000)[:, None]
    # y_preds = _net(x_test).clone().detach().numpy()
    # _y_cal_preds = _net(_x_cal).clone().detach()
    
    resid = torch.abs(_y_cal - _y_cal_preds).numpy()
    
    n = len(_x_cal)
    q_val = np.ceil((1 - alpha) * (n + 1)) / n
    q = np.quantile(resid, q_val, method="higher")

    return q, resid

def tensor_to_img(X_test, idx):
    # fig, ax = plt.subplots(figsize=(10, 5))
    # ax.plot(X_test[idx].reshape(28,28).numpy())
    # return PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plt.switch_backend('Agg')  # Set the backend to Agg
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(X_test[idx].reshape(28, 28).numpy())
    ax.set_axis_off()
    fig.canvas.draw()  # Draw the canvas
    img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    plt.close(fig)  # Close the figure to free up resources
    return img


#FashionMNIST
def fashion_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    X_test, y_test = test_dataset.data.float() / 255.0, test_dataset.targets
    X_test = X_test.view(-1, 28*28)
    
    X_test = X_test[:40]
    
    return X_test

#Fashion MNIST Predictions
def get_test_preds_and_smx(selected_img_tensor, index, pred_sets, net, q, alpha):
    # Note: Instead of passing X_test, we pass the selected_img_tensor directly

    test_smx = nn.functional.softmax(net(selected_img_tensor.unsqueeze(0)), dim=1).detach().numpy()

    sample_smx = test_smx[0]  # Since we're using only one image, get the first (and only) softmax output
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    axs[0].imshow(selected_img_tensor.reshape(28, 28).numpy())
    axs[0].set_title("Sample test image")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    axs[1].bar(range(10), sample_smx, label="class scores", color='#5B84B1FF')
    axs[1].set_xticks(range(10))
    axs[1].set_xticklabels([class_label(i) for i in range(10)])
    axs[1].axhline(y=1 - q, label='threshold', color="#FC766AFF", linestyle='dashed')
    axs[1].legend(loc=1)
    axs[1].set_title("Class Scores")
    pred_sets = sample_smx >= (1 - q)

    
    return fig, axs, pred_sets, get_pred_str(list(pred_sets))

def get_svg(img_path):
    f = open(img_path,"r")
    lines = f.readlines()
    line_string=''.join(lines)

    render_svg(line_string)

def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode('utf-8')).decode("utf-8")
    width_value = 700

    html = r'<div align="center"><img src="data:image/svg+xml;base64,%s" width="%d"/></div><br>' % (b64, width_value)
    st.write(html, unsafe_allow_html=True)
    
def get_references():
    return {
        "angelopoulos": [1],
        "uva_dl": [2],
        "vovk": [3],
        "fastai": [4],
        "olympic_data": [5],
        "resnet_demo": [6],
        "awesome-conformal": [7]
    }

def get_text_content():
    text_content = {}
    with open("data/text/references.md", "r") as f:
        text_content['references_text'] = f.read()
        
    with open("data/text/header.md", "r") as f:
        text_content['header_text'] = f.read()
        
    with open("data/text/introduction_1.md", "r") as f:
        text_content['introduction_text1'] = f.read()
        
    with open("data/text/introduction_2.md", "r") as f:
        text_content['introduction_text2'] = f.read()
        
    with open("data/text/regression_1.md", "r") as f:
        text_content['regression_text1'] = f.read()
        
    with open("data/text/classification_1.md", "r") as f:
        text_content['classification_text1'] = f.read()
        
    with open("data/text/classification_2.md", "r") as f:
        text_content['classification_text2'] = f.read()
        
    with open("data/text/conclusion.md", "r") as f:
        text_content['conclusion_text'] = f.read()
    
    return text_content