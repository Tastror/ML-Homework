import os
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
from model.model_1 import Net

# 数据预处理
train_dataset = MNIST(root='data/', train=True, transform=ToTensor(), download=True)
test_dataset = MNIST(root='data/', train=False, transform=ToTensor(), download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 保存位置处理
save_dir = 'weight'
save_name = "pt-torch-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".pt"
last_save_name = "pt-torch-last.pt"
save_path = os.path.join(save_dir, save_name)
last_save_path = os.path.join(save_dir, save_name)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 定义模型、损失函数、优化器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("deivce use", device)
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

class Accuracy:
    def __init__(self):
        self.correct = 0
        self.total = 0
        
    def update(self, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += (predicted == targets).sum().item()

    def compute(self):
        return 100.0 * self.correct / self.total

accuracy = Accuracy()

# 训练模型
num_epochs = 4
for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        accuracy.update(outputs, labels)

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Acc: {accuracy.compute():.4f}')

torch.save(model.state_dict(), save_path)
torch.save(model.state_dict(), last_save_path)

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f'Test Accuracy: {correct/total:.4f}')
