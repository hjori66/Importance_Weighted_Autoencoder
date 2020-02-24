import torch
import numpy as np
import matplotlib.pylab as plt


# Visualizer

def make_z_grid(Dz, N, limit=1.0):
    if Dz == 2:
        # coordinate grid
        z_grid = torch.zeros(N, N, Dz)
        linsp = torch.linspace(-limit, limit, N)
        z_grid[:, :, 0] = linsp.view(-1, 1)
        z_grid[:, :, 1] = linsp.view(1, -1)
    else:
        # sample randomly
        z_grid = torch.randn(N, N, Dz)

    return z_grid


def show_img_grid(net, z_grid, device, epoch, k, writer):
    N, _, Dz = z_grid.size()
    z_batch = z_grid.view(N*N, Dz).to(device)
    dec_img = net.dec(z_batch).cpu().detach().view(N*N, 28, 28)
    img_grid = dec_img.view(N, N, 28, 28).numpy()
    img_cat = np.concatenate(np.concatenate(img_grid, axis=2), axis=0)
    if (epoch + 1) % 10 == 0 or epoch == 0:
        plt.imshow(img_cat)
    # writer.add_image('IWAE_{}_samples', img_cat, epoch)
    plt.savefig("result/IWAE_{}_samples_{}_episode".format(k, epoch+1))


def plot_loss(train_losses, test_losses, k):
    plt.figure(figsize=(18, 6))
    plt.title('IWAE {} samples with MNIST dataset'.format(k))
    plt.plot(train_losses, label='train_loss')
    # plt.plot(test_losses, label='test_loss')
    plt.grid(b=True, color='0.60', linestyle='--')
    plt.legend(fontsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig('IWAE_{}_samples.png'.format(k))

    plt.figure(figsize=(18, 6))
    plt.title('IWAE {} samples with MNIST dataset'.format(k))
    plt.plot(test_losses, label='Estimated_test_NLL')
    plt.grid(b=True, color='0.60', linestyle='--')
    plt.legend(fontsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig('IWAE_{}_samples_valloss.png'.format(k))
