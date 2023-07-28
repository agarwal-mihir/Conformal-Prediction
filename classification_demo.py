import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm.auto import trange, tqdm

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




class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
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


def main():
    st.title("Conformal Predictions in Classification")
    
    st.write("In Classification, our model outputs are now class probabilities and prediction sets are discrete. This is in contrast to the continuous uncertainty bands we obtained before, and influences how we design the comparison of true and predicted classes to obtain our conformity scores. When the predictive uncertainty is high, size of the prediction set will be higher.")
    st.write(" We split the training samples into two parts: training set and calibration set. We take the first 50k images in training set and 10k images in the calibration set.")
    
    X_train, y_train, X_test, y_test, X_calib, y_calib, X_test_unscaled = get_data()
    
    net = MLP()
    
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
    
    # st.write("Empirical Coverage: {:.3f}".format(emp_coverage(pred_sets, y_test.numpy())))
    
    st.write("Example: ")
    
    test_img_index = st.slider("Choose Image:", min_value=0, max_value=1000, step=1, value=5)
    sample_test_img = X_test[test_img_index]
    fig, ax, pred, pred_str = get_test_preds_and_smx(X_test, test_img_index, pred_sets, net, q, alpha, X_test_unscaled)
    st.pyplot(fig)
    st.write("Prediction Set for this image: ", pred_str)
    
    # st.write("What if we rotate the test image by some angle? In this case, the uncertainty in the predictions should increase.")
    
    # rotation_angle = st.slider("Select a value for rotation angle:", min_value=0, max_value=90, step=5, value=20)
    # rotated_sample = 
    
    
if __name__ == "__main__":
    main()
    
    
    