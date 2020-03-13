import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import argparse

from model import IWAE
from util import make_z_grid, show_img_grid, plot_loss
from tensorboardX import SummaryWriter


# Training Helpers

def get_data_loader(batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader


def train_epoch(net, loader, k, optimizer, device):
    running_loss = 0.0
    running_n = 0
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        x_probs, y = batch
        x = torch.distributions.Bernoulli(x_probs).sample()  # binarize
        x = x.to(device)
        B = x.size(0)
        x = x.view(B, -1)
        batch_loss = net.loss(x, k)
        batch_loss.backward()
        optimizer.step()
        running_n += B
        running_loss += B*batch_loss.detach().item()
    loss = running_loss/running_n
    return loss


def test_epoch(net, loader, k, device):
    running_loss = 0.0
    running_n = 0
    for batch_idx, batch in enumerate(loader):
        x_probs, y = batch
        x = torch.distributions.Bernoulli(x_probs).sample()  # binarize
        x = x.to(device)
        B = x.size(0)
        x = x.view(B, -1)
        batch_loss = net.loss(x, k)
        running_n += B
        running_loss += B*batch_loss.detach().item()
    loss = running_loss/running_n
    return loss


def evaluate_wo(net, threshold=0.01):
    params = list(net.named_parameters())
    print(len(params))
    for name, param in params:
        print(name, param.size())
    w_norm = np.sum(np.square(params[4][1]), 1)  # dec.layers.0.weight torch.Size([100, 10])
    w_norm_max = np.max(w_norm)
    return np.sum(w_norm > threshold * w_norm_max)


# Main routine

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_data_loader(args.batch_size)
    net = IWAE(28 * 28, args.d_latent)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adagrad(net.parameters(), lr=args.lr)

    train_losses = []
    test_losses = []
    writer = SummaryWriter()

    if args.visualize:
        plt.ion()
        plt.show()
        z_grid = make_z_grid(args.d_latent, 8, limit=1.0)

    for epoch in range(args.epochs):
        train_loss = train_epoch(net, trainloader, args.train_k, optimizer, device)
        print("(Epoch %d) train loss : %.3f" % (epoch + 1, train_loss))
        if args.visualize:
            show_img_grid(net, z_grid, device, epoch, args.train_k, writer)
            plt.draw()
            plt.pause(0.001)

        test_loss = test_epoch(net, testloader, args.test_k, device)
        print("(Epoch %d) Estimated test NLL : %.3f" % (epoch + 1, test_loss))
        # print("(This evaluation is not valid until you implement IWAE)")

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        # evaluate_wo(net)

    if args.visualize:
        plot_loss(train_losses, test_losses, args.train_k)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IWAE')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_latent', type=int, default=10)
    parser.add_argument('--train_k', type=int, default=1)  # set train_k > 1 for IWAE
    parser.add_argument('--test_k', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--visualize', type=bool, default=False)
    args = parser.parse_args()
    main(args)

"""Results
k=1::
(Epoch 3) train loss : 152.010
(Epoch 3) Estimated test NLL : 172.812
(Epoch 4) train loss : 133.816
(Epoch 4) Estimated test NLL : 183.185
(Epoch 30) train loss : 92.471
(Epoch 30) Estimated test NLL : 286.436
k=5::
(Epoch 30) train loss : 115.888
(Epoch 30) Estimated test NLL : 112.213
k=10::
(Epoch 30) train loss : 114.586
(Epoch 30) Estimated test NLL : 110.956
k=50::
(Epoch 30) train loss : 111.753
(Epoch 30) Estimated test NLL : 110.096
"""
