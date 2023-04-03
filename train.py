import os
import torch
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from model.model_demo_4 import Net

parser = argparse.ArgumentParser(description='train the model')
parser.add_argument(
    '--nred', action='store_true',
    help='use utils.dataloader_no_redundant (default: utils.dataloader)'
)
parser.add_argument(
    '--naug', action='store_true',
    help='augmentation=False (default: augmentation=True)'
)
args = parser.parse_args()
print('use Dataset in utils.dataloader{}'.format("_no_redundant" if args.nred else ""))
if args.nred:
    from utils.dataloader_no_redundant import Dataset
else:
    from utils.dataloader import Dataset

# 数据读入（如果使用 MNIST）
# from torchvision.datasets import MNIST
# train_dataset = MNIST(root='data/', train=True, transform=ToTensor(), download=True)
# test_dataset = MNIST(root='data/', train=False, transform=ToTensor(), download=True)
# train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_data = DataLoader(test_dataset, batch_size=32, shuffle=False)
# label_num = 10

# 数据读入
data_dir = 'dataset'
data_name = 'NewDataset.mat'
data_path = os.path.join(data_dir, data_name)
train_dataset = Dataset(data_path, train=True, transform=None, augmentation=not args.naug)
train_2_dataset = Dataset(data_path, train=True, transform=None)
test_dataset = Dataset(data_path, train=False, transform=None)
train_data = DataLoader(train_dataset, batch_size=32, shuffle=True)
train_2_data = DataLoader(train_2_dataset, batch_size=32, shuffle=True)
test_data = DataLoader(test_dataset, batch_size=32, shuffle=False)
label_num = train_dataset.data_shape[0]
print(train_dataset.shape)

# 保存位置处理
save_dir = 'weight'
if args.nred:
    best_with_date_save_name = "pt-torch-nred-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pt"
    last_save_name = "pt-torch-nred-last.pt"
    best_save_name = "pt-torch-nred-best.pt"
else:
    best_with_date_save_name = "pt-torch-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pt"
    last_save_name = "pt-torch-last.pt"
    best_save_name = "pt-torch-best.pt"
best_with_date_save_path = os.path.join(save_dir, best_with_date_save_name)
last_save_path = os.path.join(save_dir, last_save_name)
best_save_path = os.path.join(save_dir, best_save_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 定义模型、损失函数、优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("deivce use", device)
model = Net(label_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=model.lr, weight_decay=model.weight_decay)


class Accuracy:
    def __init__(self):
        self.clear()

    def clear(self):
        self.correct = 0
        self.total = 0

    def update(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += (predicted == targets).sum().item()

    def compute(self):
        return 100.0 * self.correct / self.total


# 定义准确率计算器
accuracy = Accuracy()

# 训练模型
num_epochs = 400
acc_best = 0.0
for epoch in range(num_epochs):

    running_loss = 0.0
    correct = 0
    total = 0

    model.train()
    for images, labels in tqdm(train_data):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        accuracy.update(outputs, labels)

    print(
        f'\033[33mEpoch {epoch + 1}/{num_epochs}'
        f', Loss: {running_loss / len(train_data):.4f}'
        # this is not true in model.train()
        f', (model.train accuracy: {accuracy.compute():.4f})\033[0m'
    )
    accuracy.clear()

    # show accuracy
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_2_data:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            accuracy.update(outputs, labels)
        print(f'\033[33mTrain Accuracy: {accuracy.compute():.4f}\033[0m')
        accuracy.clear()

    # save best
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        for images, labels in test_data:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            accuracy.update(outputs, labels)

        print(f'\033[34mTest Accuracy: {accuracy.compute():.4f}\033[0m')

        if accuracy.compute() > acc_best:
            print(f'\033[1;32m{acc_best:.2f} => {accuracy.compute():.2f}\033[0m')
            acc_best = accuracy.compute()
            torch.save(model, best_with_date_save_path)
            torch.save(model, best_save_path)
        accuracy.clear()

    torch.save(model, last_save_path)

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(test_data):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Last Test Accuracy: {correct/total:.4f}')
