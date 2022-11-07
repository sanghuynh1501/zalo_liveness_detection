import random
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
from imgaug import augmenters as iaa
from torch.utils.data import Dataset

from config import FRAME, IMAGE_HEIGHT, IMAGE_WIDTH

warnings.resetwarnings()
warnings.simplefilter('ignore')


def shuffle_three_list(a, b, c):
    d = list(zip(a, b, c))
    random.shuffle(d)
    a, b, c = zip(*d)
    return a, b, c


class Normalize(object):
    def __init__(self, mode='train'):
        self.mode = mode
        self.mean = [0.485, 0.456, 0.406, 0]
        self.std = [0.229, 0.224, 0.225, 1]
        self.tranform = iaa.Sequential([
            iaa.Affine(rotate=(-25, 25)),
            iaa.Crop(percent=(0, 0.2))
        ])

    def __call__(self, image):
        mask = np.zeros((image.shape[0], image.shape[1], 1))
        mask[-int(image.shape[0] / 10):image.shape[0], :, :] = 255
        mask[:int(image.shape[0] / 10), :, :] = 255
        mask[:, :int(image.shape[1] / 5), :] = 255
        mask[:, -int(image.shape[1] / 5):, :] = 255
        image = np.concatenate([image, mask], -1)

        image = image.astype(np.float32)/255
        image -= self.mean
        image /= self.std
        image = image.transpose([2, 0, 1])

        return torch.Tensor(image)


class ImageDataset(Dataset):
    def __init__(self, split, df, shuffle=False):

        self.split = split
        self.names = df['image_link'].to_list()
        self.labels = df['label'].to_list()
        self.folders = df['folder_name'].to_list()

        if shuffle:
            self.names, self.labels, self.folders = shuffle_three_list(
                self.names, self.labels, self.folders)

        self.normalize = Normalize(split)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_paths = self.names[idx].split(',')
        label = float(round(float(self.labels[idx])))
        folder = int(self.folders[idx])

        flip = False
        if self.split == 'train':
            flip = np.random.randint(4) == 1
        rand = np.random.rand() > .5

        imgs = []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if flip:
                if rand > .5:
                    img = img[:, ::-1]
                else:
                    img = cv2.flip(img, 0)

            img = self.normalize(img)
            img = img.unsqueeze(1)
            imgs.append(img)

        imgs = torch.cat(imgs, 1)

        if imgs.shape[1] < FRAME:
            padding = torch.zeros(
                4, FRAME - imgs.shape[1], IMAGE_HEIGHT, IMAGE_WIDTH)
            imgs = torch.cat(
                [imgs, padding], 1)

        return imgs, torch.Tensor([label]), torch.Tensor([folder])
