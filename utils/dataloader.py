import torch
import numpy as np
from PIL import Image
import scipy.io as scio
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset


class Dataset(Dataset):
    def __init__(self, path, train, transform=None):
        super(Dataset, self).__init__()

        self.data = scio.loadmat(path)
        self.data = self.data['train'] if train else self.data['test']
        self.shape = np.shape(self.data)
        self.data_shape = self.shape[:2]
        self.input_shape = self.shape[2:]
        self.length = self.data_shape[0] * self.data_shape[1] * 7  # 7 为数据增强
        self.transform = transform

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

        # 增强类型
        augmentation_type = index % 7
        index = index // 7

        label = index // self.data_shape[1]
        index = index % self.data_shape[1]
        image = self.data[label][index]

        # 反色预处理
        image = 255 - np.array(image)

        # 增强
        if augmentation_type == 0:
            pass
        elif augmentation_type == 1:
            if index % 2 == 0:
                image = np.roll(image, 3, axis=0)  # 平移操作
            else:
                image = np.roll(image, -3, axis=0)  # 平移操作
        elif augmentation_type == 2:
            if index % 2 == 0:
                image = np.roll(image, 3, axis=1)  # 平移操作
            else:
                image = np.roll(image, -3, axis=1)  # 平移操作
        elif augmentation_type == 3:
            # 旋转操作
            image = Image.fromarray(image)
            image = image.rotate(10)
            image = np.array(image)
        elif augmentation_type == 4:
            # 旋转操作
            image = Image.fromarray(image)
            image = image.rotate(-10)
            image = np.array(image)
        elif augmentation_type == 5:
            # 缩放操作
            image = Image.fromarray(image)
            image = image.resize((24, 24), resample=Image.BILINEAR)
            image = np.array(image)
            image = np.pad(image, (2, 2), 'constant', constant_values=0)
        elif augmentation_type == 6:
            # 缩放操作
            image = Image.fromarray(image)
            image = image.resize((32, 32), resample=Image.BILINEAR)
            image = np.array(image)
            image = image[2:30, 2:30]

        # 归一预处理
        image = np.array(image, dtype=np.float32) / 255

        # 添加通道维度，更换顺序（通道放在最前面）
        # 最终转变为标准的 [C, H, W] 张量
        # <---            C              --->
        #   <-- H -->
        # [ [x, x, x], [x, x, x], [x, x, x] ]
        image = np.expand_dims(image, axis=2)
        image = np.transpose(image, (2, 0, 1))

        if self.transform:
            image = self.transform(image)

        return image, label
