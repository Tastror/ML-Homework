import torch
import torch.nn as nn


# 定义模型
class Net(nn.Module):
    def __init__(self, label_num: int = 200):
        super(Net, self).__init__()

        self.lr = 0.001
        self.weight_decay = 0
        
        self.conv_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1568, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, label_num)
        )
        
    def forward(self, x):
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
