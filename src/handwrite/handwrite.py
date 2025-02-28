import torch
import torchvision
import torchvision.datasets as dest
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from matplotlib import pyplot as plt
from tqdm import tqdm

# 檢查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ConvNet(nn.Module):
    def __init__(self, depth, image_size):
        super(ConvNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, depth[0], 5, padding=2)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(depth[0], depth[1], 5, padding=2)
        self.fc1 = torch.nn.Linear(image_size // 4 * image_size // 4 * depth[1], 512)
        self.fc2 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, image_size // 4 * image_size // 4 * depth[1])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, training=self.training)
        x = F.log_softmax(x, dim=1)
        return x

def datapreprocess():
    batch_size = 32

    train_dataset = dest.MNIST(root='./data', train=True, transform=ToTensor(), download=True)
    test_dataset = dest.MNIST(root='./data', train=False, transform=ToTensor())

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    indices = range(len(test_dataset))
    indices_val = list(indices)[:5000]
    indices_test = list(indices)[5000:]
    sample_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
    sample_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

    val_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=sample_val)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=sample_test)

    return train_loader, val_loader, test_loader

def rightness(prediction, labels):
    pred = torch.max(prediction, dim=1)[1]
    rights = pred.eq(labels).sum().item()
    total = len(labels)
    return rights, total

if __name__ == '__main__':
    image_size = 28
    num_epochs = 20

    # 資料載入
    train_loader, validation_loader, test_loader = datapreprocess()
    depth = [4, 8]
    net = ConvNet(depth, image_size).to(device)  # 將模型移至 GPU

    criterion = nn.CrossEntropyLoss()  # 定義交叉熵損失
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    record = []

    for epoch in range(num_epochs):
        net.train()
        train_rights = []
        train_progress = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}",colour='cyan')

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)  # 將資料移至 GPU

            output = net(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            right = rightness(output, target)
            train_rights.append(right)

            if batch_idx % 100 == 0:
                net.eval()
                val_rights = []
                with torch.no_grad():  # 禁用梯度計算，加速推理
                    for val_data, val_target in validation_loader:
                        val_data, val_target = val_data.to(device), val_target.to(device)
                        output = net(val_data)
                        right = rightness(output, val_target)
                        val_rights.append(right)

                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                # print(f'訓練週期: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\t'
                #       f'Loss: {loss.item():.6f}\t訓練正確率: {100. * train_r[0] / train_r[1]:.2f}%\t'
                #       f'驗證正確率: {100. * val_r[0] / val_r[1]:.2f}%')
                train_progress.set_postfix({
                    "Loss": f"{loss.item():.6f}",
                    "Train Acc": f"{100. * train_r[0] / train_r[1]:.2f}%",
                    "Val Acc": f"{100. * val_r[0] / val_r[1]:.2f}%"
                })
                record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * val_r[0] / val_r[1]))

    # 繪製誤差曲線
    plt.figure(figsize=(10, 7))
    plt.plot(record)
    plt.xlabel('Steps')
    plt.ylabel('Error rate')
    plt.legend(['Train', 'Validation'])
    plt.show()

    # 測試模型
    net.eval()
    vals = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            val = rightness(output, target)
            vals.append(val)

    rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
    right_rate = 100. * rights[0] / rights[1]
    print(f'Test Accuracy: {right_rate:.2f}%')
