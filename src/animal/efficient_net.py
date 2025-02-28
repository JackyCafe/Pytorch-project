import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, models
from torchvision.transforms import transforms, Resize, RandomHorizontalFlip, ColorJitter, RandomRotation, ToTensor, \
    Normalize, CenterCrop


class EfficientNet:
    all_data:datasets
    train_data:datasets =None
    val_data:datasets =None
    test_data:datasets =None
    batch_size:int = 32
    train_loader:DataLoader = None
    val_loader:DataLoader = None
    test_loader:DataLoader = None

    def __init__(self, path:str):
        self.test_loader = None
        self.val_loader = None
        self.train_loader = None
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.path = path
        self.transform(True)
        loader = self.get_loader(self.batch_size)

    def transform(self, train = False):
        if train:
            transform = transforms.Compose([
                RandomHorizontalFlip(),
                Resize(256),
                CenterCrop(224),
                ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                RandomRotation(10),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                Resize(256),
                CenterCrop(224),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transform

    def get_loader(self):
        all_data = datasets.ImageFolder(self.path, transform=self.transform)
        train_data_len = int(len(all_data) * 0.75)
        valid_data_len = int((len(all_data) - train_data_len) / 2)
        test_data_len = int(len(all_data) - train_data_len - valid_data_len)
        self.train_data, self.val_data, self.test_data = random_split(all_data, [train_data_len, valid_data_len, test_data_len])
        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False)
        return self.train_loader, self.val_loader, self.test_loader

    def get_model(self):

if __name__ == '__main__':
    dir = './datasets/animal151/'
    efficient_net = EfficientNet(dir)