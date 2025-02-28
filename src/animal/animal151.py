from pkgutil import get_loader

import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, models
from torchvision.transforms import transforms


class Resnet18:
    train_transforms:transforms
    val_transforms:transforms
    full_dataset:datasets
    train_dataset:datasets
    val_dataset:datasets
    val_loader:DataLoader =None
    train_loader:DataLoader = None
    model:nn.Module = None

    def __init__(self, path:str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.path = path
        self.transforms()
        self.get_loader()
        self.model = self.models(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)
        trained_model = self.train_model(self.model, criterion, optimizer,
                                    self.train_loader, self.val_loader
                                    , epochs=25, device=self.device)
        # Compute accuracy on the validation set
        val_accuracy = self.compute_accuracy(trained_model, self.val_loader, self.device)
        print(f"Validation Accuracy: {val_accuracy:.2f}%")


    def transforms(self):
        self.train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
        ])

        self.val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_loader(self):
        full_dataset = datasets.ImageFolder(self.path, transform=self.train_transforms)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Apply the transformations
        train_dataset.dataset.transform = self.train_transforms
        val_dataset.dataset.transform = self.val_transforms
        batch_size = 32
        self.full_dataset = full_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def models(self,device='cpu'):
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_classes = len(self.full_dataset.classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model

    def train_model(self,model, criterion, optimizer,
                                    train_loader, val_loader
                                    , epochs, device):
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

            print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_loader.dataset):.4f}')

        return model

    def compute_accuracy(self, model, data_loader, device):
        """Compute accuracy of the model on the provided data loader.

        Args:
        - model (torch.nn.Module): Trained classification model.
        - data_loader (torch.utils.data.DataLoader): Data loader.
        - device (torch.device): Computation device (CPU or CUDA).

        Returns:
        - float: Accuracy of the model on the provided data.
        """
        model.eval()
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = outputs.max(1)

                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        accuracy = 100.0 * correct_predictions / total_samples
        return accuracy

if __name__ == '__main__':
    dir = './datasets/animal151/'
    resnet18 = Resnet18(dir)