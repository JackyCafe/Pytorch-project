import torch
from torchvision import transforms
# from torchvision.transforms import 


class LogScaleTransform:
    def __call__(self, tensor, *args, **kwds):
        epsilon = 1e-6
        return torch.log(tensor + epsilon)


class GaussianNoiseTransform:    
    def __call__(self, tensor, *args, **kwds):
        noise = torch.randn(tensor.size()) * 0.1
        return tensor + noise


class Cutout:
    def __init__(self, size=50):
        self.size = size

    def __call__(self, img):
        x = torch.randint(0, img.size(1) - self.size, (1,))
        y = torch.randint(0, img.size(2) - self.size, (1,))
        img[:, x: x + self.size, y: y + self.size] = 0
        return img


class MosaicTransform:
    def __call__(self, img):
        img = transforms.Resize((32, 32))(img)
        return transforms.Resize((224, 224))(img)
