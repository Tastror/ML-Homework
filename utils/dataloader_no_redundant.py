import numpy as np
import scipy.io as scio
from typing import Optional, Any, Callable
from utils.augmentation_7 import augment, augment_type_num
from torch.utils.data.dataset import Dataset

redundant = [
    (0, 81), (27, 39), (45, 195), (99, 114), (120, 121, 122)
]


class Dataset(Dataset):
    def __init__(
        self, path, train,
        transform: Optional[Callable] = None,
        augmentation: bool = False,
        picture_dim_type: str = "torch"
    ):
        """
        Args:
            picture_dim_type (str): "svm" image.shape = (inputshape[0] * inputshape[1],), "torch" image.shape = (1, inputshape[0], inputshape[1])
        """
        super(Dataset, self).__init__()

        self.__data = scio.loadmat(path)
        self.__data = self.__data['train'] if train else self.__data['test']
        self.__shape = np.shape(self.__data)
        self.__data_shape = self.__shape[:2]
        self.__image_shape = self.__shape[2:]
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

        k = self.__no_redundant_data_shape[0]
        while k in self.__del_dict.keys():
            k += 1
        for i in self.__del_dict.keys():
            if i >= self.__no_redundant_data_shape[0]:
                self.__trans_dict[i] = i
            else:
                self.__trans_dict[i] = k
                k += 1
                while k in self.__del_dict.keys():
                    k += 1

        self.__augmentation = augmentation
        if self.__augmentation:
            self.__length = self.__length * augment_type_num  # augment_type_num 为数据增强
            self.__no_redundant_length = self.__no_redundant_length * augment_type_num
        self.__transform = transform
        self.__picture_dim_type = picture_dim_type

        print('length of dataset is', self.__no_redundant_length, end=", ")
        print('shape of dataset is', self.__no_redundant_data_shape, end=", ")
        print('shape of image is', self.__image_shape)

    @property
    def shape(self):
        """
        __len__(), *image_shape
        """
        return self.__len__(), *self.image_shape

    @property
    def data_shape(self):
        """
        (label, num of picture of each label)

        data_shape[0] * data_shape[1] == __len__()
        """
        return self.__no_redundant_data_shape

    @property
    def image_shape(self):
        if self.__picture_dim_type == "torch":
            return 1, *self.__image_shape
        elif self.__picture_dim_type == "svm":
            return (self.__image_shape[0] * self.__image_shape[1],)
        else:
            return self.__image_shape
        
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
            augmentation_type = index % augment_type_num
            index //= augment_type_num

        label = index // self.__no_redundant_data_shape[1]
        index = index % self.__no_redundant_data_shape[1]

        # 冗余映射
        if label in self.__trans_dict.keys():
            image = self.__data[self.__trans_dict[label]][index]
        else:
            image = self.__data[label][index]

        # 反色预处理
        image = 255 - np.array(image)

        if self.__augmentation:
            image = augment(image, augmentation_type, index)

        # 归一预处理
        image = np.array(image, dtype=np.float32) / 255

        if self.__picture_dim_type == "torch":
            # 添加通道维度，更换顺序（通道放在最前面）
            # 最终转变为标准的 [C, H, W] 张量
            # <---            C              --->
            #   <-- H -->
            # [ [x, x, x], [x, x, x], [x, x, x] ]
            image = np.expand_dims(image, axis=2)
            image = np.transpose(image, (2, 0, 1))
        elif self.__picture_dim_type == "svm":
            image = image.reshape(self.__image_shape[0] * self.__image_shape[1])
        else:
            pass

        if self.__transform:
            image = self.__transform(image)

        return image, label
