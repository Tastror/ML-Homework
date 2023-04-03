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
    print(f"\033[34minput label num (0 ~ {train_dataset.data_shape[0] - 1}):\033[33m ", end="")
    i = int(input())
    print(f"\033[34minput number of pictures (1 ~ {train_dataset.data_shape[1] - 1}):\033[33m ", end="")
    len = int(input())
    print("\033[0m", end="")
    index = i * index_num
    for j in range(index, index + len):
        print(f"showing index[{j - index}] in label = {i}")
        img, lab = train_dataset[j]
        Image.fromarray(img[0] * 255).show()
    print("done")
