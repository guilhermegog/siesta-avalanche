import torch.nn as nn
import torch


class ConvLayers (nn.Module):
    def __init__(self):
        super(ConvLayers, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(
            256, 512, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = torch.relu(x)

        x = self.bn2(self.conv2(x))
        x = torch.relu(x)

        x = self.bn3(self.conv3(x))
        x = torch.relu(x)

        x = self.bn4(self.conv4(x))
        x = torch.relu(x)

        x = self.bn5(self.conv5(x))
        x = torch.relu(x)

        x = x.view(-1, 512 * 2 * 2)
        return x
