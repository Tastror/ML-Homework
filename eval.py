import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataloader import Dataset

data_dir = 'dataset'
data_name = 'NewDataset.mat'
data_path = os.path.join(data_dir, data_name)
test_dataset = Dataset(data_path, train=False, transform=None)
test_data = DataLoader(test_dataset, batch_size=32, shuffle=False)

save_dir = 'weight'
last_save_name = "pt-torch-last.pt"
last_save_path = os.path.join(save_dir, last_save_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("deivce use", device)
model = torch.load(last_save_path)

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

    print(f'Test Accuracy: {correct/total:.4f}')
