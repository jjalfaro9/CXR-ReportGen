'''Autoencoder.'''

import argparse
import os
import settings
import torch
import torch.utils.data

from torch import nn, optim
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        densenet = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121',
                                  pretrained=True)
        self.encode = densenet

    def forward(self, x):
        return self.encode(x)


class Decoder(nn.Module):
    def __init__(self, output_size, dim):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.fc1 = nn.Linear(dim, 1024)
        self.deconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=256,
                                          kernel_size=5, stride=2)
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                          kernel_size=5, stride=2)
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                          kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                          kernel_size=6, stride=2)
        self.deconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                          kernel_size=6, stride=2)
        self.deconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                          kernel_size=2, stride=2)
        self.deconv6 = nn.ConvTranspose2d(in_channels=16, out_channels=3,
                                          kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = x.view(-1, x.shape[1], 1, 1)
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        x = self.deconv6(x)
        x = x.view(-1, *self.output_size)
        x = self.sigmoid(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_size, dim):
        super(Autoencoder, self).__init__()
        self.channels, self.img_width, self.img_height = input_size
        self.encoder = Encoder(dim=dim)
        self.decoder = Decoder(output_size=input_size, dim=dim)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def loss_function(self, recon_x, x):
        mse = F.mse_loss(recon_x, x, reduction='sum')
        return mse

    def step(self, data):
        data = data.to(settings.device)
        self.optimizer.zero_grad()
        recon_batch = self.forward(data)
        loss = self.loss_function(recon_batch, data)
        loss.backward()
        train_loss = loss.item()
        self.optimizer.step()
        return loss
