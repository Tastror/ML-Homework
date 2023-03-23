import os
from PIL import Image
from utils.dataloader import Dataset

# 数据读入
data_dir = 'dataset'
data_name = 'NewDataset.mat'
data_path = os.path.join(data_dir, data_name)
train_dataset = Dataset(data_path, train=True, transform=None)
test_dataset = Dataset(data_path, train=False, transform=None)
label_num = train_dataset.data_shape[0]
index_num = train_dataset.data_shape[1]

while True:
    i = int(input()) - 1
    len = int(input())
    index = i * index_num
    for j in range(index, index + len):
        img, lab = train_dataset[j]
        Image.fromarray(img[0] * 255).show()
