"""
    We use the same processing strategy as process_[DATA].py.
"""
import numpy as np
import torchvision
import random
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchvision.transforms import RandomCrop, Normalize
import os
from collections import defaultdict
import sys, re
import pandas as pd
from PIL import Image
import math
from types import *
import jieba
import os.path


def read_image(path):
    image_list = {}
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for i, filename in enumerate(os.listdir(path)):
        try:
            im = Image.open(path + filename).convert('RGB')
            im = data_transforms(im)
            image_list[filename.split('/')[-1].split(".")[0].lower()] = im
        except:
            pass
    return image_list


def read_image_notrans(path):
    image_list = {}
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    for i, filename in enumerate(os.listdir(path)):
        try:
            im = Image.open(path + filename).convert('RGB')
            im = data_transforms(im)
            image_list[filename.split('/')[-1].split(".")[0].lower()] = im
        except:
            pass
    return image_list

class Forgery_Dataset(Dataset):
    def __init__(self):
        path_au = '../Data/casia/Au/'
        path_tp = '../Data/casia/Tp/'
        images_au = read_image(path_au)
        images_tp = read_image(path_tp)
        print("{} authentic data, {} tampered data.".format(len(images_au), len(images_tp)))
        self.data = []
        for id in images_au.keys():
            self.data.append([images_au[id], torch.tensor([0])])  # [image, label]
        for id in images_tp.keys():
            self.data.append([images_tp[id], torch.tensor([1])])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]


class Patch():
    def __init__(self):
        path_tp = '../Data/casia/Tp/'
        path_gt = '../Data/casia/Gt/'
        images_gt = read_image_notrans(path_gt)
        images_tp = read_image_notrans(path_tp)
        randomcrop = RandomCrop(224)
        normal = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.images_list = []
        for id in images_gt.keys():
            id_tp = id[:-3]
            try:
                self.images_list.append(normal(randomcrop(images_gt[id] * images_tp[id_tp])).cuda())
            except:
                print(id)
        self.len = len(self.images_list)

    def patching(self, image):
        patchs = torch.stack([self.images_list[random.randint(0, self.len - 1)] for _ in range(image.shape[0])])
        return image + patchs

if __name__ == '__main__':
    patch = Patch()
    dataset = Forgery_Dataset()
    print('finish!')
