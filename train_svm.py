# 运行前先按下面方法安装sklearn库
# pip install -U scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple/

import os
import argparse
import numpy as np
from sklearn.svm import SVC
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.feature_selection import VarianceThreshold
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
print('use Dataset in utils.dataloader{}'.format("_no_redundant" if args.nred else ""))
if args.nred:
    from utils.dataloader_no_redundant import Dataset
else:
    from utils.dataloader import Dataset

data_dir = 'dataset'
data_name = 'NewDataset.mat'
data_path = os.path.join(data_dir, data_name)
train_dataset = Dataset(data_path, train=True, picture_dim_type="svm", augmentation=not args.naug)
val_dataset = Dataset(data_path, train=False, picture_dim_type="svm")
print(train_dataset.shape)

num_workers = 0
shuffle = False
train_gen = DataLoader(
    train_dataset, shuffle=shuffle, batch_size=len(train_dataset), num_workers=num_workers,
    pin_memory=True, drop_last=True, sampler=None
)
val_gen = DataLoader(
    val_dataset, shuffle=shuffle, batch_size=len(val_dataset), num_workers=num_workers,
    pin_memory=True, drop_last=True, sampler=None
)


def fit_one_epoch(train_gen, val_gen):
    print('Start Train')

    model = SVC(kernel='linear', C=1.0, gamma='scale')

    for iteration, batch in enumerate(train_gen):
        if iteration >= 1:
            break

        train_images, train_label = batch[0], batch[1]
        print(np.shape(train_images), np.shape(train_label))
        model.fit(train_images, train_label)

    print('Start test')

    for iteration, batch in enumerate(val_gen):
        if iteration >= 1:
            break

        val_images, val_label = batch[0], batch[1]
        val_pred  = model.predict(val_images)

        accuracy = accuracy_score(val_label, val_pred)
        report = classification_report(val_label, val_pred)
        print("Accuracy:", accuracy)
        print("Classification report:", report)


fit_one_epoch(train_gen, val_gen)
