import torch
import numpy as np
from torchvision import datasets, transforms
import torch.utils.data as data
from PIL import Image
import random
import os
import cv2
import random

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        raise(RuntimeError('No Module named accimage'))
    else:
        return pil_loader(path)

class ImageNetData(data.Dataset):
    def __init__(self, img_root, img_file, is_training=False, transform=None, target_transform=None, loader=default_loader):
        self.root = img_root

        self.imgs = []
        with open(img_file, 'r', encoding='utf-8') as fd:
            for i, _line in enumerate(fd.readlines()):
                infos = _line.replace('\n', '').split('\t')
                if 2 != len(infos) :                                       # Notice
                    continue
                if is_training:
                    real_path = os.path.join(self.root, 't256', infos[0])
                else:
                    real_path = os.path.join(self.root, 'v256', infos[0])
                class_id  = int(infos[-1])
                self.imgs.append((real_path, class_id))
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, class_id = self.imgs[index]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            class_id = torch.LongTensor([class_id])

        return img, class_id
    
    def __len__(self):
        return len(self.imgs)

