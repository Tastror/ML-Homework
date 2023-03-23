import os
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.dataloader import Dataset

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    'model_name', metavar='model', type=str, nargs='?',
    default="pt-torch-last.pt",
    help='model name, save dir is weight/, default is pt-torch-last.pt'
)
args = parser.parse_args()
print('use weight/', args.model_name, sep="")

data_dir = 'dataset'
data_name = 'NewDataset.mat'
data_path = os.path.join(data_dir, data_name)

name = [
    "Train",
    "Train with Augmentation",
    "Test",
    "Test with Augmentation",
]
dataset = [
    Dataset(data_path, train=True, transform=None, augmentation=False),
    Dataset(data_path, train=True, transform=None, augmentation=True),
    Dataset(data_path, train=False, transform=None, augmentation=False),
    Dataset(data_path, train=False, transform=None, augmentation=True),
]
data = [DataLoader(x, batch_size=32, shuffle=False) for x in dataset]

save_dir = 'weight'
last_save_path = os.path.join(save_dir, args.model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("deivce use", device)
model = torch.load(last_save_path)

# 测试模型
i = -1
for x in data:
    i += 1
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in x:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'\033[34m{name[i]} Accuracy: \033[32m{correct / total * 100:.2f}%\033[0m')
