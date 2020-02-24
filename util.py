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


def show_img_grid(net, z_grid, device):
    N, _, Dz = z_grid.size()
    z_batch = z_grid.view(N*N, Dz).to(device)
    dec_img = net.dec(z_batch).cpu().detach().view(N*N, 28, 28)
    img_grid = dec_img.view(N, N, 28, 28).numpy()
    img_cat = np.concatenate(np.concatenate(img_grid, axis=2), axis=0)
    plt.imshow(img_cat)
    # plt.imshow(img_cat)


