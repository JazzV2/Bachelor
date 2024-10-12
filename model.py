import torch
import torch.nn as nn

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer
        self.conv1 = nn.Conv2d(3, 10, (10, 10))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        # Second layer
        self.conv2 = nn.Conv2d(10, 20, (10, 10))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)
        # To linear layer
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(6480, 500)
        self.relu3 = nn.ReLU()
        # Final layer
        self.linear2 = nn.Linear(500, 6)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)

        x = self.linear2(x)
        x = self.logsoftmax(x)

        return x