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
        if k == 1:
            mu, sigma = self.enc(x)
            z = mu
            x_r = self.dec(z)
            objective = torch.mean(bernoulliLL(x, x_r))
            loss = -objective

        else:
            mu, sigma = self.enc(x)
            objective = []
            for i in range(k):
                # noise = torch.cuda.randn(mu.size())
                # noise = torch.autograd.Variable(mu.data.new(mu.size()).normal_())
                noise = torch.cuda.FloatTensor(mu.size()).normal_()
                z = noise * sigma + mu
                x_r = self.dec(z)
                loss_p_xz = bernoulliLL(x, x_r)
                loss_p_z = NormalGaussianLL(z)
                loss_q_zx = GaussianLL(noise, sigma)
                loss_logaddvar = loss_p_xz + loss_p_z - loss_q_zx
                objective.append(loss_logaddvar.view(1, -1))
            loss_logaddvars = torch.cat(objective)
            max_logaddvar = torch.max(loss_logaddvars, dim=0)[0]
            max_logaddvar_k = max_logaddvar.repeat(k).view(k, -1)
            loss_pq = torch.exp(loss_logaddvars - max_logaddvar)
            # loss_pq = torch.mean(loss_pq, dim=0) + 1e-7
            loss_pq = torch.mean(loss_pq[loss_pq == loss_pq], dim=0) + 1e-7  # to ignore nan
            loss_pq = max_logaddvar + torch.log(loss_pq)
            # loss = -torch.mean(loss_pq)
            loss = -torch.mean(loss_pq[loss_pq == loss_pq])  # to ignore nan
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


def NormalGaussianLL(z):
    '''
    z: [B, Dz]
    ---
    LL: [B]
    '''
    # LL = -torch.sum(0.5*torch.log(2*torch.pi) + torch.log(sigma) + 0.5*((z-mu))**2)/2*(sigma**2), dim=1)
    LL = torch.sum(-0.5*(z**2), dim=1)
    return LL


def GaussianLL(noise, sigma):
    '''
    noise: [B, Dz] (= (z - mu)/sigma)
    sigma: [B, Dz]
    ---
    LL: [B]
    '''
    # LL = -torch.sum(0.5*torch.log(2*torch.pi) + torch.log(sigma) + 0.5*((z-mu))**2)/2*(sigma**2), dim=1)
    LL = torch.sum(-0.5*(noise**2) -torch.log(sigma), dim=1)
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
        sigma = torch.exp(out[:, self.d_latent:])
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



