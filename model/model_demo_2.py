import torch
import torch.nn as nn


# 定义模型
class Net(nn.Module):
    def __init__(self, label_num: int = 200):
        super(Net, self).__init__()
        self.conv_features_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )

        self.conv_features_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )

        self.conv_features_3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(288, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, label_num),
        )
        
    def forward(self, x):
        x = self.conv_features_1(x)
        x = self.conv_features_2(x)
        x = self.conv_features_3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
