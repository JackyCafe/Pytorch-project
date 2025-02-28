import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--input", type=str, help="input csv file",default="datasets/bike/hour.csv")
    args = parser.parse_args()
    return args

def draw(counts):
    x = torch.tensor(np.arange(len(counts)), dtype=torch.double,requires_grad=True,device=device)
    y = torch.tensor(np.array(counts), dtype=torch.double,requires_grad=True,device=device)
    plt.plot(x.cpu().data,y.cpu().data, 'o-')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def data_process(counts):
    x = torch.tensor(np.arange(len(counts)), dtype=torch.double,requires_grad=True,device=device)
    y = torch.tensor(np.array(counts), dtype=torch.double,requires_grad=True,device=device)
    sz = 10
    weights = torch.randn((1,sz),requires_grad=True,device=device)
    bias = torch.randn((1,sz),requires_grad=True,device=device)
    weights2 = torch.randn((1,sz),requires_grad=True,device=device)
    learning_rate = 0.001
    losses = []
    x = x.view(50,-1)
    y = y.view(50,-1)

    hidden = x * weights + bias
    hidden = torch.sigmoid(hidden)
    for i in range(10000):
        hidden = x * weights + bias
        hidden = torch.sigmoid(hidden)
        prediction = hidden.mm(weights2)
        loss = torch.mean((prediction - y).pow(2))
        losses.append(loss.item())
    print()


def main():
    args = getArgs()
    data_path = args.input
    rides = pd.read_csv(data_path)
    counts = rides['cnt'][:50]
    # draw(counts)
    data_process(counts)
    # print(counts)

if __name__ == '__main__':
    main()