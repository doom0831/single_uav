import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthImageCNN(nn.Module):
    def __init__(self, h, w):
        super(DepthImageCNN, self).__init__()
        self.h = h
        self.w = w
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        # 注意：下面的全连接层的输入维度取决于您的深度影像尺寸
        self.fc1 = nn.Linear(32 * (self.h//4) * (self.w//4), 128)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * (self.h//4) * (self.w//4))  # Flatten the tensor
        x = F.relu(self.fc1(x))
        return x
