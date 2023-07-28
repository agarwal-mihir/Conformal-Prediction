import torch
import torch.nn as nn

# MLP (Multi-Layer Perceptron) class
class MLP(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=10, n_hidden_layers=1, use_dropout=False):
        super().__init__()

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.activation = nn.Tanh()

        # Dynamically define architecture: Input -> Hidden Layers -> Output
        self.layer_sizes = [input_dim] + n_hidden_layers * [hidden_dim] + [output_dim]
        layer_list = [nn.Linear(self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                      range(1, len(self.layer_sizes))]
        self.layers = nn.ModuleList(layer_list)

    def forward(self, input):
        hidden = self.activation(self.layers[0](input))
        for layer in self.layers[1:-1]:
            hidden_temp = self.activation(layer(hidden))

            if self.use_dropout:
                hidden_temp = self.dropout(hidden_temp)

            hidden = hidden_temp + hidden  # Residual connection

        output_mean = self.layers[-1](hidden).squeeze()
        return output_mean

# CNN (Convolutional Neural Network) class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Input has 3 channels (e.g., RGB images)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer to downsample spatial dimensions
        self.fc1 = nn.Linear(128 * 4 * 4, 64)  # Fully connected layer after the convolutional layers
        self.fc2 = nn.Linear(64, 10)  # Output layer for classification (10 classes assumed)

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        
        # Flatten the output from convolutional layers for the fully connected layers
        x = x.reshape(x.size(0), -1)
        
        # Fully connected layers with ReLU activation
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)  # Final output for classification
        return x
