import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
  def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(2*37*25, 120)
        self.fc2 = nn.Linear(120, 60)
        self.fc3 = nn.Linear(60, 6)

  def forward(self, x):
    
    #Conv Layers
    x = self.conv1(x)
    x = self.activation(x)
    x = self.pool(x)
    # print(x.shape)

    x = self.conv2(x)
    x = self.activation(x)
    x = self.pool(x)
    # print(x.shape)

    x = self.conv3(x)
    x = self.activation(x)
    x = self.pool(x)
    # print(x.shape)

    #Flatten
    x = torch.flatten(x, 1)
    
    x = self.fc1(x)
    x = self.activation(x)
    x = self.fc2(x)
    x = self.activation(x)
    x = self.fc3(x)
    x = nn.LogSoftmax(dim=-1)(x)

    return x