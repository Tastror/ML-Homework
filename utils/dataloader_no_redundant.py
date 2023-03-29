import numpy as np
from PIL import Image
import scipy.io as scio
from torch.utils.data.dataset import Dataset

redundant = [
    (0, 81), (27, 39), (45, 195), (99, 114), (120, 121, 122)
]

class Dataset(Dataset):
    def __init__(self, path, train, transform=None, augmentation=False):
        super(Dataset, self).__init__()

        self.__data = scio.loadmat(path)
        self.__data = self.__data['train'] if train else self.__data['test']
        self.__shape = np.shape(self.__data)
        self.__data_shape = self.__shape[:2]
        self.__input_shape = self.__shape[2:]
        self.__length = self.__data_shape[0] * self.__data_shape[1]

        # 去除冗余
        self.__del_dict = {}  # 记录删除的下标和相似下标的关系，键值对：删除下标 - 相似下标
        self.__trans_dict = {}  # 将删除的下标映射到 >= length 的值，键值对：删除下标 - 映射的可用下标
        for i in redundant:
            k = i[0]
            for s in i[1:]:
                self.__del_dict[s] = k
        print('delete length is', len(self.__del_dict))
        self.__no_redundant_data_shape = (self.__data_shape[0] - len(self.__del_dict), self.__data_shape[1])
        self.__no_redundant_length = self.__no_redundant_data_shape[0] * self.__no_redundant_data_shape[1]
        k = self.__no_redundant_length
        for i in self.__del_dict.keys():
            self.__trans_dict[i] = k
            k += 1

        self.__augmentation = augmentation
        if self.__augmentation:
            self.__length = self.__length * 9  # 9 为数据增强
            self.__no_redundant_length = self.__no_redundant_length * 9  # 9 为数据增强
        self.__transform = transform

        print('length of dataset is', self.__no_redundant_length, end=", ")
        print('shape of dataset is', self.__no_redundant_data_shape, end=", ")
        print('shape of image is', self.__input_shape)

    @property
    def data_shape(self):
        return self.__no_redundant_data_shape

    @property
    def input_shape(self):
        return self.__input_shape

    def __len__(self):
        return self.__no_redundant_length

    def __getitem__(self, index):

        if index >= self.__no_redundant_length or index < -self.__no_redundant_length:
            raise IndexError(
                f"Dataset index out of boundary: {index} in 0 ~ {self.__no_redundant_length - 1}"
            )
        if index < 0:
            index = self.__no_redundant_length + index

        # 增强类型
        augmentation_type = 0
        if self.__augmentation:
            augmentation_type = index % 9
            index //= 9

        label = index // self.__data_shape[1]
        index = index % self.__data_shape[1]
        image = self.__data[label][index]

        # 冗余映射
        if index in self.__trans_dict.keys():
            index = self.__trans_dict[index]

        # 反色预处理
        image = 255 - np.array(image)

        if self.__augmentation:
            if augmentation_type == 0:
                pass
            elif augmentation_type == 1:
                image = Image.fromarray(image)
                image = image.resize((24, 24), resample=Image.BILINEAR)
                image = np.array(image)
                image = np.pad(image, (2, 2), 'constant', constant_values=0)
                if index % 2 == 0:
                    image = np.roll(image, 5, axis=0)  # 缩放并平移
                else:
                    image = np.roll(image, -5, axis=0)  # 缩放并平移
            elif augmentation_type == 2:
                image = Image.fromarray(image)
                image = image.resize((24, 24), resample=Image.BILINEAR)
                image = np.array(image)
                image = np.pad(image, (2, 2), 'constant', constant_values=0)
                if index % 2 == 0:
                    image = np.roll(image, 5, axis=1)  # 缩放并平移
                else:
                    image = np.roll(image, -5, axis=1)  # 缩放并平移
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
                # 旋转操作
                image = Image.fromarray(image)
                image = image.rotate(20)
                image = np.array(image)
            elif augmentation_type == 6:
                # 旋转操作
                image = Image.fromarray(image)
                image = image.rotate(-20)
                image = np.array(image)
            elif augmentation_type == 7:
                # 缩放操作
                image = Image.fromarray(image)
                image = image.resize((24, 24), resample=Image.BILINEAR)
                image = np.array(image)
                image = np.pad(image, (2, 2), 'constant', constant_values=0)
            elif augmentation_type == 8:
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

        if self.__transform:
            image = self.__transform(image)

        return image, label
