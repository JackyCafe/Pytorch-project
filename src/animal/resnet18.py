import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, models
from torchvision.transforms import transforms


def compute_accuracy(model, data_loader, device):
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






def train_model(model, criterion, optimizer, train_loader, val_loader, epochs=10, device='cpu'):
    """Train the model on provided data and evaluate on validation data.

    Args:
    - model (torch.nn.Module): The neural network model to be trained.
    - criterion (torch.nn.Module): The loss function.
    - optimizer (torch.optim.Optimizer): The optimization algorithm.
    - train_loader (torch.utils.data.DataLoader): DataLoader for training data.
    - val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
    - epochs (int, optional): Number of training epochs. Defaults to 10.
    - device (torch.device, optional): Device ("cpu" or "cuda") where the model should be moved to. Defaults to "cpu".

    Returns:
    - model (torch.nn.Module): Trained model.
    - history (dict): Dictionary containing training loss, validation loss, and validation accuracy for each epoch.
    """
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


def main():
    # path = './datasets/animal151/translation.json'
    # Data transformations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the full dataset without any transforms
    full_dataset = datasets.ImageFolder('./datasets/animal151/', transform=train_transforms)

    # Split the dataset into training and validation
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply the transformations
    train_dataset.dataset.transform = train_transforms
    val_dataset.dataset.transform = val_transforms

    # Create data loaders
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = models.resnet18(pretrained=True)

    # Freeze the parameters of the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Modify the final layer
    num_classes = len(full_dataset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
    trained_model = train_model(model, criterion, optimizer, train_loader, val_loader, epochs=25, device=device)
    # Compute accuracy on the validation set
    val_accuracy = compute_accuracy(trained_model, val_loader, device)
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

if __name__ == '__main__':
    main()