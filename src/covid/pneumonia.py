import os
from enum import Enum
from glob import glob

import torch
from torchvision.transforms import transforms
from mpl_toolkits.mplot3d.axes3d import Axes3D
import pandas as pd
from torch import datasets as dest


class Mode(Enum):
    TRAIN = 0
    VALID = 1
    TEST = 2


device:str=None
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

TEST = 'test'
TRAIN = 'train'
VAL ='val'

def data_prepare():
    data_dir = "./datasets/chest_xray/chest_xray/"
    TEST = 'test'
    TRAIN = 'train'
    VAL = 'val'
    image_datasets = {x: dest.ImageFolder(os.path.join(data_dir, x), data_transforms(x))
                      for x in [TEST, TRAIN, VAL]
                      }
    print(image_datasets[TEST])


def data_transforms(mode:Mode)->transforms.Compose:
    if mode == Mode.TRAIN:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif mode == Mode.VALID:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transform

def main():
    transforms = data_transforms(Mode.VALID)
    data_prepare()

    print(transforms)



if __name__ == '__main__':
    main()