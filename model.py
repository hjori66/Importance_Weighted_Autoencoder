import torch
import torch.nn as nn
import numpy as np


# Model

class IWAE(nn.Module):
    def __init__(self, d_data, d_latent, d_hidden=100):
        super(IWAE, self).__init__()
        self.d_latent = d_latent
        self.enc = Encoder(d_data, d_latent, d_hidden)
        self.dec = Decoder(d_data, d_latent, d_hidden)

    def loss(self, x, k):

        ######################################
        # TODO: implement k-sample IWAE loss #
        ######################################

        if "you just started, here's an autoencoder loss for you":
            mu, sigma = self.enc(x)
            z = mu
            x_r = self.dec(z)
            objective = torch.mean(bernoulliLL(x, x_r))
            loss = -objective

        return loss


def bernoulliLL(x, x_r):
    '''
    x: [B, Dx]
    x_r: [B, Dx]
    ---
    LL: [B]
    '''
    LL = torch.sum(x*torch.log(x_r)+(1-x)*torch.log(1-x_r), dim=1)
    return LL


class Encoder(nn.Module):
    def __init__(self, d_data, d_latent, d_hidden):
        super(Encoder, self).__init__()
        self.d_latent = d_latent
        self.layers = nn.Sequential(
            nn.Linear(d_data, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, 2*d_latent)
        )

    def forward(self, x):
        out = self.layers(x)
        mu = torch.tanh(out[:, :self.d_latent])
        sigma = torch.exp(out[:, :self.d_latent])
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, d_data, d_latent, d_hidden):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_data),
            nn.Sigmoid()
        )

    def forward(self, z):
        x_r = self.layers(z)
        return x_r



