'''encoder.py'''

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, dim):
        super(Encoder, self).__init__()
        self.densenet = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121',
                                       pretrained=True).features
        self.densenet.requires_grad = False
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, dim),
            nn.Dropout(inplace=True)
        )

    def forward(self, x):
        x = self.densenet(x)
        y = torch.mean(x, dim=(2, 3))
        x = self.projection(y)
        return x
