# 运行前先按下面方法安装sklearn库
# pip install -U scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple/

import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from skimage.feature import hog
from skimage.measure import block_reduce
from sklearn.feature_extraction import image
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


parser = argparse.ArgumentParser(description='train the model')
parser.add_argument(
    '--nred', action='store_true',
    help='use utils.dataloader_no_redundant (default: utils.dataloader)'
)
parser.add_argument(
    '--naug', action='store_true',
    help='augmentation=False (default: augmentation=True)'
)
args = parser.parse_args()
print(
    'use Dataset in utils.dataloader{}'.format(
        "_no_redundant" if args.nred else ""
    )
)
if args.nred:
    from utils.dataloader_no_redundant import Dataset
else:
    from utils.dataloader import Dataset

data_dir = 'dataset'
data_name = 'NewDataset.mat'
data_path = os.path.join(data_dir, data_name)
train_dataset = Dataset(
    data_path, train=True,
    picture_dim_type="HWC", augmentation=not args.naug
)
val_dataset = Dataset(data_path, train=False, picture_dim_type="HWC")

# 获取图片和标签
X_train = train_dataset.data
y_train = train_dataset.targets
X_test = val_dataset.data
y_test = val_dataset.targets


# 将图像转换为特征向量
def change(image_list):
    res_list = []
    patch_time = 0
    hog_time = 0
    for i in tqdm(image_list):

        t = time.time()

        # 3 * 3 卷积池化，50 个特征点
        patches_1 = image.extract_patches_2d(
            i, (3, 3), max_patches=70, random_state=42
        )
        pooled_patches_1 = np.zeros((patches_1.shape[0], 1, 1))
        for j, patch in enumerate(patches_1):
            pooled_patches_1[j] = block_reduce(patch, 3, np.max)
        
        # 5 * 5 卷积池化，50 个特征点
        patches_2 = image.extract_patches_2d(
            i, (5, 5), max_patches=70, random_state=42
        )
        pooled_patches_2 = np.zeros((patches_2.shape[0], 1, 1))
        for j, patch in enumerate(patches_2):
            pooled_patches_2[j] = block_reduce(patch, 5, np.max)

        patch_time += time.time() - t

        t = time.time()

        hog_image = hog(
            i[:, :, 0], orientations=8, pixels_per_cell=(3, 3),
            cells_per_block=(1, 1)
        )

        hog_time += time.time() - t

        res = np.concatenate((
            # i.ravel(),
            pooled_patches_1.ravel(),
            pooled_patches_2.ravel(),
            hog_image.ravel(),
        ))
        res_list.append(res)
    
    print(f"Time use: patch = {patch_time}s, hog = {hog_time}s")
    return res_list


print("preprocess")
X_train = change(X_train)
X_test = change(X_test)
print(np.shape(X_train))

# 训练 SVM 分类器
print('Start Train')
model = SVC(kernel='rbf', C=5, gamma=0.01)
print(np.shape(X_train), np.shape(X_test))
model.fit(X_train, y_train)

# 预测测试集并评估准确性
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification report:", report)
