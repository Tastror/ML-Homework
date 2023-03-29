import torch.nn as nn


# 定义模型
class Net(nn.Module):
    def __init__(self, label_num: int = 200):
        super(Net, self).__init__()

        self.lr = 0.0002
        self.weight_decay = 0.001

        self.conv_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.MaxPool2d(kernel_size=2),

            # nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.MaxPool2d(kernel_size=2),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(3136, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(512, label_num),
            nn.Dropout(0.2),
        )
        
    def forward(self, x):
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
