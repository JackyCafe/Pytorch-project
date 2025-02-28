from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
import  torchvision
import torch.nn as nn
import torch.optim as optim
import copy


train_loader:DataLoader
train_dataset:Dataset

def imshow(inp, title=None):
    # 将一张图打印显示出来，inp为一个张量，title为显示在图像上的文字

    # 一般的张量格式为：channels*image_width*image_height
    # 而一般的图像为image_width*image_height*channels所以，需要将channels转换到最后一个维度
    inp = inp.numpy().transpose((1, 2, 0))

    # 由于在读入图像的时候所有图像的色彩都标准化了，因此我们需要先调回去
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    # 将图像绘制出来
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 暂停一会是为了能够将图像显示出来。

def rightness(predictions, labels):
    """
    計算預測錯誤率的函數，
    predictions:是模型給出的一組預測結果, batch_size行,10列的矩陣
    labels:labels是數據中的正確答案
    """
    pred = torch.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    # rights 裝在cpu 中
    rights = rights.cpu() if rights.is_cuda else rights
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素


def main():
    data_dir = './datasets/animal'
    image_size = 224
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                         transforms.Compose([
                                             transforms.RandomResizedCrop(image_size),
                                             transforms.RandomHorizontalFlip(),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                         ])
                                         )
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                       transforms.Compose([
                                           transforms.Resize(256),
                                           transforms.CenterCrop(image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                       ])
                                       )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4,
                                               shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=4,
                                             shuffle=True, num_workers=4)

    num_classes = len(train_loader.dataset.classes)
    use_cuda = torch.cuda.is_available()
    # 当可用GPU的时候，将新建立的张量自动加载到GPU中
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    itype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    images, labels = next(iter(train_loader))
    out = torchvision.utils.make_grid(images)
    # imshow(out, title  = [train_dataset.classes[x] for x in labels])
    net = models.resnet18(pretrained=True)
    net = net.cuda() if use_cuda else net
    num_ftrs = net.fc.in_features
    net.fc = nn.Linear(num_ftrs, 3)
    net.fc = net.fc.cuda() if use_cuda else net.fc
    criterion = nn.CrossEntropyLoss()  # Loss函数的定义
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    record = []  # 记录准确率等数值的容器
    # 开始训练循环
    num_epochs = 20
    net.train(True)  #  在訓練集上做訓練
    best_model = net
    best_r = 0.0
    for epoch in range(num_epochs):
        # optimizer = exp_lr_scheduler(optimizer, epoch)
        train_rights = []  #紀錄準確率的容器
        train_losses = []  #紀錄損失率的容器

        for batch_idx, (data, target) in enumerate(train_loader):  # 針對每個批次進行循環
            data, target = data.clone().detach().requires_grad_(True), target.clone().detach()  # data為圖像，target為標籤
            # 如果存在GPU则將變量載到GPU中
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            output = net(data)  # 完成一次预测
            loss = criterion(output, target)  # 計算誤差
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向傳播
            optimizer.step()  # 隨機梯度下降
            right = rightness(output, target)  # 計算準確度
            train_rights.append(right)  # 将计算结果装到列表容器中
            loss = loss.cpu() if use_cuda else loss
            train_losses.append(loss.data.numpy())

            # if batch_idx % 20 == 0: #每间隔100个batch执行一次
        # train_r为一个二元组，分别记录训练集中分类正确的数量和该集合中总的样本数
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))

        # 在测试集上分批运行，并计算总的正确率
        net.eval()  # 标志模型当前为运行阶段
        test_loss = 0
        correct = 0
        vals = []
        # 对测试数据集进行循环
        for data, target in val_loader:
            # 如果存在GPU则将变量加载到GPU中
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data.clone().detach().requires_grad_(False), target.clone().detach()
            output = net(data)  # 将特征数据喂入网络，得到分类的输出
            val = rightness(output, target)  # 获得正确样本数以及总样本数
            vals.append(val)  # 记录结果

        # 计算准确率
        val_r = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
        val_ratio = 1.0 * val_r[0].numpy() / val_r[1]

        if val_ratio > best_r:
            best_r = val_ratio
            best_model = copy.deepcopy(net)
        # 打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
        print('训练周期: {} \tLoss: {:.6f}\t训练正确率: {:.2f}%, 校验正确率: {:.2f}%'.format(
            epoch, np.mean(train_losses), 100. * train_r[0].numpy() / train_r[1], 100. * val_r[0].numpy() / val_r[1]))
        record.append([np.mean(train_losses), train_r[0].numpy() / train_r[1], val_r[0].numpy() / val_r[1]])

if __name__ == '__main__':
    main()
