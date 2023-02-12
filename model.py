import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=20,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=2,
                stride=1,
                padding=1

            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        #self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(256, 5)
        self.fc2 = nn.Linear(50, 5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
  
        logits =self.fc1(x)
    
        #logits = self.fc2(x)

        return logits