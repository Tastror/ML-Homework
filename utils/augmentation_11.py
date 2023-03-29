import numpy as np
from PIL import Image

augment_type_num = 11

def augment(image, augmentation_type, rand_source):

    if augmentation_type == 0:
        pass
    elif augmentation_type == 1:
        # 缩放并平移
        image = Image.fromarray(image)
        image = image.resize((24, 24), resample=Image.BILINEAR)
        image = np.array(image)
        image = np.pad(image, (2, 2), 'constant', constant_values=0)
        if rand_source % 2 == 0:
            image = np.roll(image, 5, axis=0)
        else:
            image = np.roll(image, -5, axis=0)
    elif augmentation_type == 2:
        # 缩放并平移
        image = Image.fromarray(image)
        image = image.resize((24, 24), resample=Image.BILINEAR)
        image = np.array(image)
        image = np.pad(image, (2, 2), 'constant', constant_values=0)
        if rand_source % 2 == 0:
            image = np.roll(image, 5, axis=1)
        else:
            image = np.roll(image, -5, axis=1)
    elif augmentation_type == 3:
        # 缩放并平移
        image = Image.fromarray(image)
        image = image.resize((24, 24), resample=Image.BILINEAR)
        image = np.array(image)
        image = np.pad(image, (2, 2), 'constant', constant_values=0)
        if rand_source % 2 == 0:
            image = np.roll(image, -5, axis=0)
            image = np.roll(image, -5, axis=1)
        else:
            image = np.roll(image, 5, axis=0)
            image = np.roll(image, 5, axis=1)
    elif augmentation_type == 4:
        # 缩放并平移
        image = Image.fromarray(image)
        image = image.resize((24, 24), resample=Image.BILINEAR)
        image = np.array(image)
        image = np.pad(image, (2, 2), 'constant', constant_values=0)
        if rand_source % 2 == 0:
            image = np.roll(image, 5, axis=0)
            image = np.roll(image, -5, axis=1)
        else:
            image = np.roll(image, -5, axis=0)
            image = np.roll(image, 5, axis=1)
    elif augmentation_type == 5:
        # 旋转操作
        image = Image.fromarray(image)
        image = image.rotate(10)
        image = np.array(image)
    elif augmentation_type == 6:
        # 旋转操作
        image = Image.fromarray(image)
        image = image.rotate(-10)
        image = np.array(image)
    elif augmentation_type == 7:
        # 旋转操作
        image = Image.fromarray(image)
        image = image.rotate(15)
        image = np.array(image)
    elif augmentation_type == 8:
        # 旋转操作
        image = Image.fromarray(image)
        image = image.rotate(-15)
        image = np.array(image)
    elif augmentation_type == 9:
        # 缩放操作
        image = Image.fromarray(image)
        image = image.resize((24, 24), resample=Image.BILINEAR)
        image = np.array(image)
        image = np.pad(image, (2, 2), 'constant', constant_values=0)
    elif augmentation_type == 10:
        # 缩放操作
        image = Image.fromarray(image)
        image = image.resize((32, 32), resample=Image.BILINEAR)
        image = np.array(image)
        image = image[2:30, 2:30]
    
    return image
