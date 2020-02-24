import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import argparse

from model import IWAE
from util import make_z_grid, show_img_grid


# Training Helpers

def get_data_loader(batch_size):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


def train_epoch(net, loader, k, optimizer, device):
    running_loss = 0.0
    running_n = 0
    for batch_idx, batch in enumerate(loader):
        optimizer.zero_grad()
        x_probs, y = batch
        x = torch.distributions.Bernoulli(x_probs).sample() # binarize
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
        x = torch.distributions.Bernoulli(x_probs).sample() # binarize
        x = x.to(device)
        B = x.size(0)
        x = x.view(B, -1)
        batch_loss = net.loss(x, k)
        running_n += B
        running_loss += B*batch_loss.detach().item()
    loss = running_loss/running_n
    return loss


# Main routine

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainloader, testloader = get_data_loader(args.batch_size)
    net = IWAE(28 * 28, args.d_latent)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.visualize:
        plt.ion()
        plt.show()
        z_grid = make_z_grid(args.d_latent, 8, limit=1.0)
    for epoch in range(args.epochs):
        train_loss = train_epoch(net, trainloader, args.train_k, optimizer, device)
        print("(Epoch %d) train loss : %.3f" % (epoch + 1, train_loss))
        if args.visualize:
            show_img_grid(net, z_grid, device)
            plt.draw()
            plt.pause(0.001)
    test_loss = test_epoch(net, testloader, args.test_k, device)
    print("Estimated test NLL : %.3f" % (test_loss))
    print("(This evaluation is not valid until you implement IWAE)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='IWAE')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d_latent', type=int, default=10)
    parser.add_argument('--train_k', type=int, default=1)  # set train_k > 1 for IWAE
    parser.add_argument('--test_k', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--visualize', type=bool, default=True)
    args = parser.parse_args()
    main(args)
