import torch.nn as nn

# 定义 ResNet 用到的 BottleNeck


def Conv1(in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_planes, out_channels=out_planes,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False
        ),
        nn.Dropout(0.1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class BottleNeck(nn.Module):
    def __init__(self, in_places, out_places, stride=1, downsampling=False, expansion=4):
        super(BottleNeck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                in_channels=in_places, out_channels=out_places,
                kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(out_places),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_places, out_channels=out_places,
                kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_places),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_places, out_channels=out_places *
                self.expansion, kernel_size=1, stride=1, bias=False
            ),
            nn.BatchNorm2d(out_places*self.expansion)
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_places, out_channels=out_places *
                    self.expansion, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# 定义模型
class Net(nn.Module):
    def __init__(self, label_num: int = 200):
        super(Net, self).__init__()
        self.conv_features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            # nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),

            nn.Upsample(scale_factor=2, mode='bilinear'),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(3136, 1024),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),

            nn.Linear(512, label_num),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.conv_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
