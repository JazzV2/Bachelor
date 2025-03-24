import torch
import torch.nn as nn

class CNN_model(nn.Module):
    def __init__(self):
        super().__init__()
        # First layer
        self.conv1 = nn.Conv2d(1, 10, (3, 3))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.25)
        # Second layer
        self.conv2 = nn.Conv2d(10, 30, (3, 3))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p=0.25)
        # To linear layer
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(15870, 500)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        # Final layer
        self.linear2 = nn.Linear(500, 6)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu3(x)
        x = self.dropout3(x)

        x = self.linear2(x)
        x = self.logsoftmax(x)

        return x