from __future__ import print_function
import torch
import torchvision.models as models
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

class MyNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.norm = nn.LayerNorm(normalized_shape= (3, 60, 200))
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 24, kernel_size=(5,5), stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 24, out_channels = 36, kernel_size=(5,5), stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 36, out_channels = 48, kernel_size=(5,5), stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels = 48, out_channels = 64, kernel_size=(3,3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=(3,3), stride=1, padding=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.25)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(670464,1164)
        self.fc2 = nn.Linear(1164, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 10)
        self.fc5 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.norm(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.fc5(x)

        return x
