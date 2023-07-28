import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm

torch.manual_seed(42)
np.random.seed(42)

def get_data():
    X_test = np.load("cifar/npy/cifar_x_test.npy")
    y_test = np.load("cifar/npy/cifar_y_test.npy")
    X_calib = np.load("cifar/npy/cifar_x_calib.npy")
    y_calib = np.load("cifar/npy/cifar_y_calib.npy")
    
    X_test = torch.tensor(X_test, dtype = torch.float32) / 255.0
    y_test = torch.tensor(y_test, dtype = torch.long)
    
    X_calib = torch.tensor(X_calib, dtype = torch.float32) / 255.0
    y_calib = torch.tensor(y_calib, dtype = torch.long)
    
    return X_test, y_test, X_calib, y_calib

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Input has 3 channels
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)  # Assuming you have 10 classes for classification

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        # print(x.shape)
        x = x.reshape(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def class_label(i):
    labels = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
    
    return labels[i]

def get_test_accuracy(X_test, y_test, net):
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

def quantile(scores, alpha):
  # compute conformal quantiles
  n = len(scores)
  q_val = np.ceil((1 - alpha) * (n + 1)) / n
  q = np.quantile(scores, q_val, method="higher")
  return q

def mean_set_size(sets):
  # mean prediction set size
  return np.mean(np.sum(sets, axis=1), axis=0)

def get_scores(net, calib_data):
    X_calib, y_calib = calib_data
    y_calib = y_calib.reshape(-1)
    
    cal_smx = torch.functional.F.softmax(net(X_calib.permute(0, 3, 1, 2)), dim=1).detach().numpy()
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
    ax.axvline(x = quantile, label = q_label, color = 'red')
    
    plt.legend(loc = 2)
    return fig, ax

def get_pred_sets(net, test_data, q, alpha):
    X_test, y_test = test_data
    test_smx = nn.functional.softmax(net(X_test.permute(0, 3, 1, 2)), dim=1).detach().numpy()

    pred_sets = test_smx >= (1-q)
    return pred_sets

def get_pred_str(pred):
    pred_str = "{"
    for i in pred:
        pred_str += class_label(i) + ' '
    pred_str += "}"
    return pred_str

def get_test_preds_and_smx(X_test, index, pred_sets, net, q, alpha):
    test_smx = nn.functional.softmax(net(X_test.permute(0, 3, 1, 2)), dim=1).detach().numpy()
    sample_smx = test_smx[index]
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 3))
    axs[0].imshow(X_test[index].numpy())
    axs[0].set_title("Sample test image")
    
    axs[1].bar(range(10), sample_smx, label = "class scores")
    axs[1].set_xticks(range(10))
    axs[1].set_xticklabels([class_label(i) for i in range(10)], rotation = 90)
    axs[1].axhline(y = 1-q, label = 'threshold', color = "red", linestyle='dashed')
    axs[1].legend(loc = 1)
    axs[1].set_title("Class Scores")
    
    pred_set = pred_sets[index].nonzero()[0].tolist()
    
    return fig, axs, pred_set, get_pred_str(pred_set)

def main():
    st.title("Conformal Predictions in Classification")
    
    st.write("In Classification, our model outputs are now class probabilities and prediction sets are discrete. This is in contrast to the continuous uncertainty bands we obtained before, and influences how we design the comparison of true and predicted classes to obtain our conformity scores. When the predictive uncertainty is high, size of the prediction set will be higher.")
    st.write("We will use the Cifar Dataset for this demo. We split the training samples into two parts: training set and calibration set. We take the first 45k images in training set and 5k images in the calibration set.")
    
    X_test, y_test, X_calib, y_calib = get_data()
    
    net = CNN()
    net.load_state_dict(torch.load("cifar/cifar_model", map_location=torch.device("cpu")))
    st.write("Test accuracy of current model: ", get_test_accuracy(X_test, y_test, net))
    
    st.write("The choice of how to calculate conformity scores is a modelling decision. We will use a simple softmax based method:")
    score_func = r"s_i=1-\hat{\pi}_{x_i}(y_i)"
    st.latex(score_func)
    st.write("which is 1 minus the softmax output of the true class. The prediction set is then constructed as :")
    pred_set_latex = r"\hat{C}(x_{n+1})=\{y'\in K:\hat{\pi}_{x_{n+1}}(y') \ge 1-\hat{q}\}"
    st.latex(pred_set_latex)
    st.write("which collects all the classes for which the softmax score is above the threshold 1-q.")
    st.write("q is given by: ")
    st.latex(r"\left\lceil \frac{(n+1)(1-\alpha)}{n} \right\rceil")

    scores = get_scores(net, (X_calib, y_calib))
    alpha = st.slider("Select a value for alpha:", min_value=0.001, max_value=1.0, step=0.001, value=0.04)
    q = quantile(scores, alpha)
    fig, ax = plot_scores_quantile(scores, q, alpha)
    st.pyplot(fig)
    
    st.write("Conformal quantile(1-q) of the calibration data is: {:.3f}".format(1-q))
    st.write("This means that for some test sample (x, y), the prediction set is constructed by collecting all classes for which the softmax score is above the threshold {:.3f}.".format(1-q))
    
    pred_sets = get_pred_sets(net, (X_test, y_test), q, alpha)
    st.write("The average size of prediction sets for the test images is {:.3f}".format(mean_set_size(pred_sets)))
    
    st.write("Example: ")
    
    test_img_index = st.slider("Choose Image:", min_value=0, max_value=1000, step=1, value=628)
    sample_test_img = X_test[test_img_index]
    fig, ax, pred, pred_str = get_test_preds_and_smx(X_test, test_img_index, pred_sets, net, q, alpha)
    st.pyplot(fig)
    st.write("Prediction Set for this image: ", pred_str)

if __name__ == "__main__":
    main()
