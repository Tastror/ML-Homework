import numpy as np
import scipy.io as scio
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class Dataset(Dataset):
    def __init__(self, path, train):
        super(Dataset, self).__init__()

        self.data = scio.loadmat(path)
        self.data = self.data['train'] if train else self.data['test']
        self.shape = np.shape(self.data)
        self.data_shape = self.shape[:2]
        self.input_shape = self.shape[2:]
        self.length = self.data_shape[0] * self.data_shape[1]

        print('length of dataset is', self.length, end=", ")
        print('shape of dataset is', self.data_shape, end=", ")
        print('shape of image is', self.input_shape)

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        if index >= self.length or index < -self.length:
            raise IndexError(
                f"Dataset index out of boundary: {index} in 0 ~ {self.length - 1}"
            )
        if index < 0:
            index = self.length + index
        label = index // self.data_shape[1]
        index = index % self.data_shape[1]
        
        # 使用含有一个通道的灰度图
        # 添加通道维度，正则化为 [0, 1]，更换顺序（通道放在最前面）
        # 最终转变为标准的 [Sample, C, H, W] 张量
        image = self.data[label][index]
        image = np.expand_dims(image, axis=2)
        image = np.transpose(
            1 - np.array(image, dtype=np.float32) / 255, (2, 0, 1)
        )

        return image, label


# DataLoader 中 collate_fn 使用
# def dataset_collate(batch):
#     images = []
#     Label = []
#     for img, label in batch:
#         images.append(img)
#         Label.append(label)

#     images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
#     Label = torch.from_numpy(np.array(Label)).type(torch.LongTensor)
#     return images, Label
