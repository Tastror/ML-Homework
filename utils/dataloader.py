import numpy as np
import scipy.io as scio
from utils.augmentation_7 import augment, augment_type_num
from torch.utils.data.dataset import Dataset


class Dataset(Dataset):
    def __init__(self, path, train, transform=None, augmentation=False):
        super(Dataset, self).__init__()

        self.__data = scio.loadmat(path)
        self.__data = self.__data['train'] if train else self.__data['test']
        self.__shape = np.shape(self.__data)
        self.__data_shape = self.__shape[:2]
        self.__input_shape = self.__shape[2:]
        self.__length = self.__data_shape[0] * self.__data_shape[1]
        self.__augmentation = augmentation
        if self.__augmentation:
            self.__length = self.__length * augment_type_num  # augment_type_num 为数据增强
        self.__transform = transform

        print('length of dataset is', self.__length, end=", ")
        print('shape of dataset is', self.__data_shape, end=", ")
        print('shape of image is', self.__input_shape)

    @property
    def data_shape(self):
        return self.__data_shape

    @property
    def input_shape(self):
        return self.__input_shape

    def __len__(self):
        return self.__length

    def __getitem__(self, index):

        if index >= self.__length or index < -self.__length:
            raise IndexError(
                f"Dataset index out of boundary: {index} in 0 ~ {self.__length - 1}"
            )
        if index < 0:
            index = self.__length + index

        # 增强类型
        augmentation_type = 0
        if self.__augmentation:
            augmentation_type = index % augment_type_num
            index //= augment_type_num

        label = index // self.__data_shape[1]
        index = index % self.__data_shape[1]
        image = self.__data[label][index]

        # 反色预处理
        image = 255 - np.array(image)

        if self.__augmentation:
            image = augment(image, augmentation_type, index)

        # 归一预处理
        image = np.array(image, dtype=np.float32) / 255

        # 添加通道维度，更换顺序（通道放在最前面）
        # 最终转变为标准的 [C, H, W] 张量
        # <---            C              --->
        #   <-- H -->
        # [ [x, x, x], [x, x, x], [x, x, x] ]
        image = np.expand_dims(image, axis=2)
        image = np.transpose(image, (2, 0, 1))

        if self.__transform:
            image = self.__transform(image)

        return image, label
